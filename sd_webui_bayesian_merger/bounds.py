import warnings
from typing import Dict, List, Tuple
import logging

from omegaconf import DictConfig, OmegaConf
import sd_mecha

logger = logging.getLogger(__name__)

class Bounds:
    @staticmethod
    def set_block_bounds(
            lb: float = 0.0, ub: float = 1.0
    ) -> Tuple[float, float]:
        return (lb, ub)

    @staticmethod
    def default_bounds(
        custom_ranges: Dict[str, Tuple[float, float]] = None,
        cfg=None,
    ) -> Dict:
        model_arch = sd_mecha.extensions.model_arch.resolve(cfg.model_arch)
        unet_block_identifiers = [
            key for key in model_arch.user_keys()
            if "_unet_block_" in key
        ]
        block_count = len(unet_block_identifiers)
        print(f"Default bounds - model_arch: {cfg.model_arch}, block_count: {block_count}")

        # Get the default hyperparameters from the merging method
        mecha_merge_method = sd_mecha.extensions.merge_method.resolve(cfg.merge_mode)
        default_hypers = mecha_merge_method.get_default_hypers()

        # Construct block names, including base parameters dynamically
        block_names = unet_block_identifiers + [f"base_{name}" for name in default_hypers]
        ranges = {b: (0.0, 1.0) for b in block_names} | OmegaConf.to_object(
            custom_ranges
        )

        return {b: Bounds.set_block_bounds(float(r[0]), float(r[1])) for b, r in ranges.items()}  # Convert to floats

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
    def get_value(params, block_name, frozen, groups) -> float:
        if block_name in params:
            return params[block_name]
        if groups is not None:
            for group in groups:
                if block_name in group:
                    group_name = "-".join(group)
                    return params.get(group_name, frozen.get(group[0], params.get(group[0])))
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
        expected_param_name = len(mecha_merge_method.get_default_hypers())

        # Iterate over the expected parameter names
        for i in range(expected_param_name):
            param_name = list(mecha_merge_method.get_default_hypers().keys())[i]

            weights_list[param_name] = []
            for block_id in unet_block_identifiers:
                value = Bounds.get_value(params, block_id, frozen, groups)
                weights_list[param_name].append(value)

            assert len(weights_list[param_name]) == block_count
            print(
                f"Assemble params - model_arch: {cfg.model_arch}, param_name: {param_name}, num_weights: {len(weights_list[param_name])}")

            base_name = f"base_{param_name}"
            base_values[param_name] = [Bounds.get_value(params, base_name, frozen, groups)]

        assert len(weights_list) == expected_param_name
        print(f"Assembled Weights List: {weights_list}")
        print(f"Assembled Base Values: {base_values}")
        return weights_list, base_values