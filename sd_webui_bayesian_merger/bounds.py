import warnings
from typing import Dict, List, Tuple
import logging

from omegaconf import DictConfig, OmegaConf
import sd_mecha

logger = logging.getLogger(__name__)

class Bounds:
    @staticmethod
    def set_block_bounds(
        block_name: str, lb: float = 0.0, ub: float = 1.0
    ) -> Tuple[float, float]:
        return (lb, ub)

    @staticmethod
    def default_bounds(
        custom_ranges: Dict[str, Tuple[float, float]] = None,
        cfg=None,  # Add cfg as argument
    ) -> Dict:
        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        unet_block_identifiers = [
            key for key in model_arch.user_keys()
            if "_unet_block_" in key
        ]
        block_count = len(unet_block_identifiers)
        print(f"Default bounds - model_arch: {cfg.model_arch}, block_count: {block_count}")

        block_names = unet_block_identifiers + ["base_alpha", "base_beta"]  # Include base parameters
        ranges = {b: (0.0, 1.0) for b in block_names} | OmegaConf.to_object(
            custom_ranges
        )

        return {b: Bounds.set_block_bounds(b, *ranges[b]) for b in block_names}

    @staticmethod
    def freeze_bounds(bounds: Dict, frozen: Dict[str, float] = None) -> Dict:
        return {b: r for b, r in bounds.items() if b not in frozen}

    @staticmethod
    def group_bounds(bounds: Dict, groups: List[List[str]] = None) -> Dict:
        for group in groups:
            if not group:
                continue
            ranges = {bounds[b] for b in group if b in bounds}
            if len(ranges) > 1:
                w = (
                    f"different values for the same group: {group}"
                    + f" we're picking {group[0]} range: {bounds[group[0]]}"
                )
                warnings.warn(w)
                group_range = bounds[group[0]]
            elif ranges:
                group_range = ranges.pop()
            else:
                # all frozen
                continue

            group_name = "-".join(group)
            bounds[group_name] = group_range
            for b in group:
                if b in bounds:
                    del bounds[b]
        return bounds

    @staticmethod
    def get_bounds(
            frozen_params: Dict[str, float] = None,
            custom_ranges: Dict[str, Tuple[float, float]] = None,
            groups: List[List[str]] = None,
            cfg=None,
    ) -> Dict:
        if frozen_params is None:
            frozen_params = {}
        if custom_ranges is None:
            custom_ranges = DictConfig({})
        if groups is None:
            groups = []

        logger.debug("Input Parameters:")
        logger.debug(f"Frozen Params: {frozen_params}")
        logger.debug(f"Custom Ranges: {custom_ranges}")
        logger.debug(f"Groups: {groups}")

        bounds = Bounds.default_bounds(custom_ranges, cfg)
        not_frozen_bounds = Bounds.freeze_bounds(bounds, frozen_params)
        grouped_bounds = Bounds.group_bounds(not_frozen_bounds, groups)

        logger.debug(f"Bounds After Default Bounds: {bounds}")
        return Bounds.freeze_groups(grouped_bounds, groups, frozen_params)

    @staticmethod
    def freeze_groups(bounds, groups, frozen):
        for group in groups:
            group_name = "-".join(group)
            if group_name in bounds and any(b in frozen for b in group):
                del bounds[group_name]
        return bounds

    @staticmethod
    def get_value(params, block_name, frozen, groups, model_arch="sd1") -> float:  # Replace sdxl with model_arch
        if block_name in params:
            return params[block_name]
        if groups is not None:
            for group in groups:
                if block_name in group:
                    group_name = "-".join(group)
                    if group_name in params:
                        return params[group_name]
                    if group[0] in frozen:
                        return frozen[group[0]]
                    if group[0] in params:
                        return params[group[0]]
        return frozen[block_name]

    @staticmethod
    def assemble_params(
            params: Dict,
            frozen: Dict,
            groups: List[List[str]],
            cfg,
    ) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
        unet_block_identifiers = [
            key for key in sd_mecha.extensions.model_arch.resolve(cfg.model_arch).user_keys()
            if "_unet_block_" in key
        ]

        # Sort unet_block_identifiers using natural_sort_key
        unet_block_identifiers.sort(key=sd_mecha.hypers.natural_sort_key)

        block_count = len(unet_block_identifiers)
        if frozen is None:
            frozen = {}
        if groups is None:
            groups = []

        print(f"Assemble params - model_arch: {cfg.model_arch}, block_count: {block_count}")
        weights_list = {}
        base_values = {}

        # Get the expected number of Greek letters from the merging method
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        expected_greek_letters = len(mecha_merge_method.get_default_hypers())

        # Iterate over the expected Greek letters
        for i in range(expected_greek_letters):
            greek_letter = list(mecha_merge_method.get_default_hypers().keys())[i]

            weights_list[greek_letter] = []
            for block_id in unet_block_identifiers:
                value = Bounds.get_value(params, block_id, frozen, groups)
                weights_list[greek_letter].append(value)

            assert len(weights_list[greek_letter]) == block_count
            print(
                f"Assemble params - model_arch: {cfg.model_arch}, greek_letter: {greek_letter}, num_weights: {len(weights_list[greek_letter])}")

            base_name = f"base_{greek_letter}"
            base_values[greek_letter] = [Bounds.get_value(params, base_name, frozen, groups)]

        assert len(weights_list) == expected_greek_letters
        print(f"Assembled Weights List: {weights_list}")
        print(f"Assembled Base Values: {base_values}")
        return weights_list, base_values