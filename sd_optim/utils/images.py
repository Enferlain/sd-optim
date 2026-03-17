from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

COLS = [[-1, 1 / 3, 2 / 3], [1, 1, 0], [0, -1, -1], [1, 0, 1]]
COLSXL = [[0, 0, 1], [1, 0, 0], [-1, -1, 0], [-1, 1, 0]]

LAYER_MAPPING = {
    0: "model.diffusion_model.input_blocks.0.0.weight",
    1: "model.diffusion_model.input_blocks.0.0.bias",
    2: "model.diffusion_model.out.0.weight",
    3: "model.diffusion_model.out.0.bias",
    4: "model.diffusion_model.out.2.weight",
    5: "model.diffusion_model.out.2.bias",
}


def colorcalc(cols, isxl):
    """Compute color adjustment deltas for layer tuning."""
    colors = COLSXL if isxl else COLS
    outs = [[value * cols[index] * 0.02 for value in row] for index, row in enumerate(colors)]
    return [sum(row) for row in zip(*outs)]


def fineman(fine, isxl):
    """Normalize fine-adjustment input into the internal adjustment list."""
    if isinstance(fine, str) and fine.find(",") != -1:
        tmp = [token.strip() for token in fine.split(",")]
        fines = [0.0] * 8
        for index, value in enumerate(tmp[0:8]):
            try:
                fines[index] = float(value)
            except ValueError:
                logger.warning(
                    "Could not convert '%s' to float. Using 0.0 instead.", value
                )
                fines[index] = 0.0
        fine = fines
    elif not isinstance(fine, list):
        logger.error(
            "Invalid input type for 'fine'. Expected a comma-separated string or a list."
        )
        return None

    return [
        1 - fine[0] * 0.01,
        1 + fine[0] * 0.02,
        1 - fine[1] * 0.01,
        1 + fine[1] * 0.02,
        1 - fine[2] * 0.01,
        [fine[3] * 0.02] + colorcalc(fine[4:8], isxl),
    ]


def weighttoxl(weights):
    """Convert a layer-adjust weight list into the expected SDXL shape."""
    if len(weights) >= 22:
        weights = weights[:9] + weights[12:22] + [0]
    return weights


def modify_state_dict(
    state_dict: dict, adjustments: dict, is_xl_model: bool
) -> dict:
    """Apply layer-adjustment values to a loaded model state dict."""
    fine_adjustments = fineman(",".join(map(str, adjustments.values())), is_xl_model)

    if fine_adjustments is None:
        raise ValueError("Error: Invalid 'fine' string format for fineman function.")

    modified_state_dict = state_dict.copy()
    if is_xl_model:
        fine_adjustments = weighttoxl(fine_adjustments)

    for index, layer_name in LAYER_MAPPING.items():
        if layer_name in state_dict:
            if index < 5:
                modified_state_dict[layer_name] = (
                    state_dict[layer_name] * fine_adjustments[index]
                )
            else:
                modified_state_dict[layer_name] = state_dict[layer_name] + torch.tensor(
                    fine_adjustments[index],
                    dtype=state_dict[layer_name].dtype,
                    device=state_dict[layer_name].device,
                )
        else:
            logger.warning("Layer '%s' not found in the state_dict.", layer_name)

    return modified_state_dict


def get_summary_images(
    log_file: Path, imgs_dir: Path, top_iterations: int
) -> list[tuple[str, float, Path]]:
    """Select the highest-scoring image for each payload in top iterations."""
    try:
        with open(log_file, encoding="utf-8") as file:
            log_data = [json.loads(line) for line in file]
    except (FileNotFoundError, json.JSONDecodeError) as error:
        logger.error("Error loading log file: %s", error)
        return []

    sorted_iterations = sorted(
        log_data, key=lambda item: item["target"], reverse=True
    )[:top_iterations]

    summary_images = []
    for _iteration_data in sorted_iterations:
        iteration_num = len(summary_images)
        payload_images = {}
        for file_name in os.listdir(imgs_dir):
            if file_name.startswith(f"{iteration_num:03}-"):
                parts = file_name[:-4].split("-")
                image_index = parts[1]
                payload = "-".join(parts[2:-1])
                score = parts[-1]
                payload_images.setdefault(payload, []).append(
                    (image_index, score, Path(imgs_dir, file_name))
                )

        for index, image_set in enumerate(payload_images.values()):
            highest_scoring_image = max(image_set, key=lambda item: item[1])
            summary_images.append(
                (
                    f"iter {iteration_num:03} - {index}",
                    float(highest_scoring_image[1]),
                    highest_scoring_image[2],
                )
            )

    return summary_images


def update_log_scores(log_file: Path, summary_images, new_scores):
    """Rewrite the saved targets with manually updated summary scores."""
    try:
        with open(log_file, "r+", encoding="utf-8") as file:
            log_data = [json.loads(line) for line in file]

            for index in range(len(summary_images)):
                log_data[index]["target"] = new_scores[index]

            file.seek(0)
            json.dump(log_data, file, indent=4)
            file.truncate()
    except Exception as error:  # noqa: BLE001 - preserve broad helper guard.
        logger.error("Error updating log file: %s", error)
