from __future__ import annotations

import ast
import contextlib
import inspect
import json
import logging
import re
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Any

import sd_mecha

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sd_mecha import recipe_nodes
from .methods import resolve_merge_method
from .config import normalize_target_node_refs
from .recipes import serialize_recipe_text

if TYPE_CHECKING:
    from sd_optim.merger import Merger

logger = logging.getLogger(__name__)

def serialize_nodes_for_rewrite(
    nodes_dict: dict[str, sd_mecha.recipe_nodes.RecipeNode],
) -> tuple[list[str], dict[str, str]]:
    """
    Serialize nodes for recipe rewriting and return replacement tokens.

    Replacement tokens are either `&N` references to prepended nodes or inline
    literals for scalar-only literal nodes.
    """
    all_new_lines: list[str] = []
    param_to_replacement: dict[str, str] = {}
    current_offset = 0

    for param_name, node in sorted(nodes_dict.items()):
        serialized_text = serialize_recipe_text(node, finalize=False)
        node_lines = serialized_text.strip().split("\n")[1:]

        if not node_lines:
            if isinstance(node, recipe_nodes.LiteralRecipeNode):
                param_to_replacement[param_name] = _to_mecha_inline_literal(
                    _get_inline_literal_value(node)
                )
                continue
            raise ValueError(
                f"Recipe node for '{param_name}' produced no serializable lines."
            )

        def shift_ref(match, offset=current_offset):
            original_idx = int(match.group(1))
            return f"&{original_idx + offset}"

        shifted_lines = [re.sub(r"&(\d+)", shift_ref, line) for line in node_lines]
        all_new_lines.extend(shifted_lines)
        param_to_replacement[param_name] = f"&{current_offset + len(node_lines) - 1}"
        current_offset += len(node_lines)

    return all_new_lines, param_to_replacement


def _get_inline_literal_value(node: recipe_nodes.LiteralRecipeNode) -> Any:
    """Extract the scalar payload from a singleton literal node."""
    if len(node.value_dict) != 1:
        raise TypeError(
            "Cannot inline a literal recipe node with multiple values during recipe rewrite."
        )
    return next(iter(node.value_dict.values()))


def _to_mecha_inline_literal(value: Any) -> str:
    """Serialize a Python scalar into a valid inline .mecha literal token."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, (int, float)):
        return str(value)
    raise TypeError(
        f"Unsupported inline literal type for recipe rewrite: {type(value)}"
    )


def rewrite_recipe_text(
    original_recipe_text: str,
    new_node_strings: list[str],
    param_to_replacement: dict[str, str],
    target_node_idx: int | None = None,
    target_node_indices: list[int] | set[int] | tuple[int, ...] | None = None,
) -> str:
    """Prepend nodes to a recipe text and patch the targeted merge line."""
    original_lines = original_recipe_text.strip().split("\n")[1:]
    num_new_nodes = len(new_node_strings)
    normalized_target_indices = set(target_node_indices or [])
    if target_node_idx is not None:
        normalized_target_indices.add(target_node_idx)

    def shift_old_ref(match):
        original_idx = int(match.group(1))
        return f"&{original_idx + num_new_nodes}"

    rewritten_lines = []
    for index, line in enumerate(original_lines):
        line_base = line.split("#", 1)[0].strip()
        if not line_base:
            continue

        shifted_line = re.sub(r"&(\d+)", shift_old_ref, line_base)

        if index in normalized_target_indices:
            line_to_append = shifted_line
            for param_name, replacement_token in param_to_replacement.items():
                pattern = re.compile(f"({re.escape(param_name)}=)([^ ]+)")
                line_to_append = pattern.sub(
                    lambda match, token=replacement_token: f"{match.group(1)}{token}",
                    line_to_append,
                )
        else:
            line_to_append = shifted_line

        rewritten_lines.append(line_to_append)

    logger.info(
        "Successfully shifted original recipe references and patched target line."
    )
    return (
        "version 0.1.0\n"
        + "\n".join(new_node_strings)
        + "\n"
        + "\n".join(rewritten_lines)
    )


class ModelVisitor(recipe_nodes.RecipeVisitor):
    """Find all unique model nodes reachable from a recipe graph."""

    def __init__(self):
        self.models: list[recipe_nodes.ModelRecipeNode] = []
        self.visited: set[recipe_nodes.RecipeNode] = set()

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        if node not in self.visited:
            self.models.append(node)
            self.visited.add(node)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        if node in self.visited:
            return
        self.visited.add(node)
        for arg in node.bound_args.args:
            arg.accept(self)
        for kwarg in node.bound_args.kwargs.values():
            kwarg.accept(self)

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode):
        if node in self.visited:
            return
        self.visited.add(node)
        for nested in node.value_dict.values():
            if isinstance(nested, recipe_nodes.RecipeNode):
                nested.accept(self)


def get_info_from_target_node(
    root_node: recipe_nodes.RecipeNode, target_node_ref: str
) -> dict[str, Any]:
    """Extract the merge-method identifier and ancestor model paths for a target."""

    def find_node_by_ref(start_node, ref_str):
        target_line_num = int(ref_str.strip("&"))

        def get_all_nodes(node_to_serialize):
            text = serialize_recipe_text(node_to_serialize, finalize=False)
            lines = text.strip().split("\n")
            node_map = {}
            for index in range(1, len(lines)):
                node_map[index - 1] = sd_mecha.deserialize(lines[: index + 1])
            return node_map

        node_map = get_all_nodes(start_node)
        return node_map.get(target_line_num)

    target_node = find_node_by_ref(root_node, target_node_ref)
    if not isinstance(target_node, recipe_nodes.MergeRecipeNode):
        return {}

    model_visitor = ModelVisitor()
    target_node.accept(model_visitor)
    model_names = [model.path for model in model_visitor.models]
    return {
        "method_name": target_node.merge_method.identifier,
        "model_names": model_names,
    }


def save_merge_artifacts(
    cfg: DictConfig,
    merger: Merger,
    final_recipe_node: recipe_nodes.RecipeNode,
    model_path: Path,
    iteration: int,
):
    """Write a standalone Python script that can reproduce a merge iteration."""
    try:
        scripts_dir = Path(HydraConfig.get().runtime.output_dir) / "merge_artifacts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_file_path = scripts_dir / f"{model_path.stem}_run.py"

        original_recipe_text = ""
        if cfg.optimization_mode == "recipe":
            recipe_path = Path(cfg.recipe_optimization.recipe_path)
            if recipe_path.exists():
                original_recipe_text = recipe_path.read_text(encoding="utf-8")

        main_method_name = get_method_names(cfg, original_recipe_text)
        method_imports, methods_code = get_source_code_for_methods({main_method_name})
        converter_names = find_used_converters(final_recipe_node)
        converter_imports, converters_code = get_source_code_for_methods(
            converter_names
        )
        custom_config_name = cfg.optimization_guide.get("custom_block_config_id")
        yaml_content = get_yaml_content(custom_config_name, cfg.configs_dir)
        fallback_path_str = get_fallback_model_path_str(cfg, merger)

        recipe_python_code = MechaToPythonConverter(
            serialize_recipe_text(
                final_recipe_node,
                model_dirs_to_add=[Path(merger.models_dir)],
                finalize=False,
            )
        ).convert()

        all_imports = method_imports | converter_imports
        final_script_content = build_reproducible_script(
            cfg=cfg,
            merger=merger,
            output_filename=model_path.name,
            iteration=iteration,
            transpiled_recipe=recipe_python_code,
            yaml_name=custom_config_name,
            yaml_content=yaml_content,
            all_imports=all_imports,
            methods_code_block=methods_code,
            converters_code_block=converters_code,
            fallback_model_path_str=fallback_path_str,
        )

        script_file_path.write_text(final_script_content, encoding="utf-8")
        logger.info("Saved reproducible script to: %s", script_file_path)
    except Exception as error:  # noqa: BLE001 - preserve current broad artifact guard.
        logger.error("Failed to save runnable script: %s", error, exc_info=True)


def get_method_names(cfg: DictConfig, original_recipe_text: str) -> str:
    """Return the primary merge method name for the current optimization mode."""
    if cfg.optimization_mode == "merge":
        return cfg.merge_method
    if cfg.optimization_mode == "recipe":
        target_nodes_raw = cfg.recipe_optimization.get("target_nodes")
        if not target_nodes_raw:
            logger.warning("Recipe mode selected but no target_nodes defined.")
            return "unknown_recipe_method_no_target"

        try:
            target_ref = normalize_target_node_refs(target_nodes_raw)[0]
            all_lines = original_recipe_text.strip().split("\n")
            target_node_idx = int(target_ref.strip("&"))
            recipe_slice_to_parse = all_lines[: target_node_idx + 2]
            target_node = sd_mecha.deserialize(recipe_slice_to_parse)

            if isinstance(target_node, sd_mecha.recipe_nodes.MergeRecipeNode):
                return target_node.merge_method.identifier
            logger.error("Target node %s is not a merge method.", target_ref)
            return "unknown_recipe_method_not_merge"
        except Exception as error:  # noqa: BLE001 - preserve legacy behavior.
            logger.error(
                "Failed to parse recipe to get main method name: %s", error
            )
            return "unknown_recipe_method_parse_fail"

    return "unknown_recipe_method_other_mode"


class ConverterFinder(recipe_nodes.RecipeVisitor):
    """Find custom converter identifiers used by a recipe graph."""

    def __init__(self):
        self.converter_names: set[str] = set()
        self.visited: set[recipe_nodes.RecipeNode] = set()
        self.known_converters = sd_mecha.extensions.merge_methods.get_all_converters()
        try:
            self.sd_mecha_path = Path(inspect.getfile(sd_mecha)).parent.resolve()
        except TypeError:
            self.sd_mecha_path = None
            logger.warning(
                "Could not determine sd-mecha library path. Converter filtering might be inaccurate."
            )

    def visit(self, node):
        if node not in self.visited:
            self.visited.add(node)
            node.accept(self)

    def visit_merge(self, node: recipe_nodes.MergeRecipeNode):
        method_obj = node.merge_method
        if method_obj in self.known_converters:
            try:
                unwrapped_func = inspect.unwrap(method_obj)
                source_file_path = Path(inspect.getfile(unwrapped_func)).resolve()
                if (
                    (
                        self.sd_mecha_path
                        and self.sd_mecha_path not in source_file_path.parents
                    )
                    or self.sd_mecha_path is None
                    and "sd_mecha" not in str(source_file_path)
                ):
                    self.converter_names.add(method_obj.identifier)
            except (TypeError, OSError):
                pass

        for arg in node.bound_args.args:
            self.visit(arg)
        for kwarg in node.bound_args.kwargs.values():
            self.visit(kwarg)

    def visit_model(self, node: recipe_nodes.ModelRecipeNode):
        return None

    def visit_literal(self, node: recipe_nodes.LiteralRecipeNode):
        for value in node.value_dict.values():
            if isinstance(value, recipe_nodes.RecipeNode):
                self.visit(value)


def find_used_converters(root_node: recipe_nodes.RecipeNode) -> set[str]:
    """Return all custom converter identifiers reachable from a recipe graph."""
    finder = ConverterFinder()
    finder.visit(root_node)
    return finder.converter_names


def get_yaml_content(config_name: str | None, configs_dir_path: str) -> str | None:
    """Read the YAML source for a specific custom config identifier."""
    if not config_name:
        return None
    config_file = Path(configs_dir_path) / f"{config_name}.yaml"
    if config_file.is_file():
        return config_file.read_text(encoding="utf-8")
    logger.warning("Could not find source for custom config: %s", config_name)
    return None


def get_fallback_model_path_str(cfg: DictConfig, merger: Merger) -> str:
    """Return a quoted fallback model path literal for generated scripts."""
    fallback_index = cfg.get("fallback_model_index", -1)
    if (
        fallback_index is not None
        and fallback_index != -1
        and fallback_index < len(merger.models)
    ):
        fallback_node = merger.models[fallback_index]
        return f'"{fallback_node.path}"'
    return "None"


def _format_and_deduplicate_imports(imports: set[str]) -> str:
    """Normalize import statements for generated reproducibility scripts."""
    direct_imports = {}
    from_imports = {}

    for imp_line in sorted(imports):
        imp_line = imp_line.strip()
        try:
            tree = ast.parse(imp_line)
            node = tree.body[0]

            if isinstance(node, ast.Import):
                for alias in node.names:
                    direct_imports[alias.name] = alias.asname or alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0:
                    logger.warning(
                        "Skipping relative import, it cannot be made portable: '%s'",
                        imp_line,
                    )
                    continue

                module_name = node.module
                if module_name not in from_imports:
                    from_imports[module_name] = {}

                for alias in node.names:
                    from_imports[module_name][alias.name] = alias.asname
        except (SyntaxError, IndexError):
            logger.warning("Could not parse import line: '%s'. Skipping.", imp_line)
            continue

    for module in list(direct_imports.keys()):
        if module in from_imports:
            del direct_imports[module]

    final_import_lines = []
    for module, alias in sorted(direct_imports.items()):
        if module == alias:
            final_import_lines.append(f"import {module}")
        else:
            final_import_lines.append(f"import {module} as {alias}")

    for module, names in sorted(from_imports.items()):
        name_parts = []
        for name, alias in sorted(names.items()):
            if name == alias or alias is None:
                name_parts.append(name)
            else:
                name_parts.append(f"{name} as {alias}")

        import_list_str = ", ".join(name_parts)
        line = f"from {module} import {import_list_str}"
        if len(line) > 88:
            line = (
                f"from {module} import (\n    "
                + ",\n    ".join(name_parts)
                + "\n)"
            )
        final_import_lines.append(line)

    return "\n".join(final_import_lines)


def build_reproducible_script(
    cfg: DictConfig,
    merger: Merger,
    output_filename: str,
    iteration: int,
    transpiled_recipe: str,
    yaml_name: str | None,
    yaml_content: str | None,
    all_imports: set[str],
    methods_code_block: str,
    converters_code_block: str,
    fallback_model_path_str: str,
) -> str:
    """Build the standalone Python reproduction script for a merge artifact."""
    models_dir_str = str(merger.models_dir.resolve())
    merge_device = cfg.get("device", "cpu")

    dtype_map = {
        "fp16": "float16",
        "fp32": "float32",
        "fp64": "float64",
        "bf16": "bfloat16",
    }
    merge_dtype_name = dtype_map.get(cfg.get("merge_dtype", "float64"), "float64")
    save_dtype_name = dtype_map.get(cfg.get("save_dtype", "float16"), "float16")
    merge_dtype_str = f"torch.{merge_dtype_name}"
    save_dtype_str = f"torch.{save_dtype_name}"
    threads_value = cfg.get("threads")

    embedded_yamls = {yaml_name: yaml_content} if yaml_name and yaml_content else {}
    embedded_yamls_str = repr(embedded_yamls)
    import_block = _format_and_deduplicate_imports(all_imports)

    return f'''
# =================================================================
#  Auto-generated by sd-optim for Full Reproducibility
#  Iteration: {iteration}
#  Output Model: {output_filename}
# =================================================================

import sd_mecha
import torch
import yaml
import inspect
import textwrap
import re
import logging
from pathlib import Path
from torch import Tensor
from typing import Optional, Dict, Tuple, Set, List, Any
from sd_optim.utils.recipes import merge_with_model_dirs
from sd_mecha.recipe_nodes import RecipeNode
from sd_mecha import Parameter, Return, merge_method, StateDict, recipe_nodes, extensions
from sd_mecha.extensions import merge_methods

# -----------------------------------------------------------------
#  Discovered Imports for Custom Functions
# -----------------------------------------------------------------
{import_block}

# -----------------------------------------------------------------
#  Environment Setup Function
# -----------------------------------------------------------------
def setup_custom_configs():
    """Parses and registers the embedded YAML configs with sd-mecha."""
    logger = logging.getLogger(__name__)
    embedded_yamls = {embedded_yamls_str}
    if not embedded_yamls:
        logger.info("No custom YAML configs to register.")
        return
    for name, yaml_str in embedded_yamls.items():
        if not name or not yaml_str: continue
        logger.info("Registering custom config: %s", name)
        try:
            config_data = yaml.safe_load(yaml_str)
            sd_mecha.extensions.model_configs.register_aux(
                sd_mecha.extensions.model_configs.ModelConfigImpl(**config_data)
            )
        except Exception as e:
            logger.exception("Could not register config '%s': %s", name, e)
            
# Run setup
setup_custom_configs()

# -----------------------------------------------------------------
#  Custom Merge Methods
# -----------------------------------------------------------------
{methods_code_block}

# -----------------------------------------------------------------
#  Custom Converters
# -----------------------------------------------------------------
{converters_code_block}

# -----------------------------------------------------------------
#  Transpiled sd-mecha Recipe
# -----------------------------------------------------------------
def get_recipe() -> RecipeNode:
    """This function contains the transpiled .mecha recipe."""
{textwrap.indent(transpiled_recipe, "    ")}
    return final_recipe

# -----------------------------------------------------------------
#  Main Execution Blocl
# -----------------------------------------------------------------
def main():
    logger = logging.getLogger(__name__)
    # Configuration from the original run
    MODELS_DIR = Path(r"{models_dir_str}")
    OUTPUT_FILENAME = "{output_filename.replace(".safetensors", "_external.safetensors")}"
    MERGE_DEVICE = "{merge_device}"
    MERGE_DTYPE = {merge_dtype_str}
    SAVE_DTYPE = {save_dtype_str}
    THREADS = {threads_value}
    FALLBACK_MODEL_PATH = {fallback_model_path_str}

    # Get and execute recipe
    logger.info("Building recipe...")
    recipe_to_run = get_recipe()
    output_path = Path(MODELS_DIR) / OUTPUT_FILENAME

    logger.info("Executing merge and saving to %s...", output_path)
    merge_with_model_dirs(
        model_dirs_to_add=[MODELS_DIR],
        recipe=recipe_to_run,
        output=output_path,
        fallback_model=sd_mecha.model(FALLBACK_MODEL_PATH) if FALLBACK_MODEL_PATH != "None" else None,
        merge_device=MERGE_DEVICE,
        merge_dtype=MERGE_DTYPE,
        output_dtype=SAVE_DTYPE,
        threads=THREADS,
        strict_mandatory_keys=False,
    )
    logger.info("Merge complete.")

if __name__ == "__main__":
    main()
'''


class _CodeParser(ast.NodeVisitor):
    """Find names and local helper calls used inside a function AST."""

    def __init__(self, local_method_names: set[str]):
        self.used_names: set[str] = set()
        self.called_local_methods: set[str] = set()
        self.local_method_names = local_method_names

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        curr_node = node
        while isinstance(curr_node, ast.Attribute):
            curr_node = curr_node.value
        if isinstance(curr_node, ast.Name):
            self.used_names.add(curr_node.id)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in self.local_method_names:
            self.called_local_methods.add(func.id)
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id in ("self", "cls")
            and func.attr in self.local_method_names
        ):
            self.called_local_methods.add(func.attr)
        self.generic_visit(node)


def get_source_code_for_methods(method_names: set[str]) -> tuple[set[str], str]:
    """Collect source code and imports required to reproduce named methods."""
    if not method_names:
        return set(), ""

    source_cache: dict[str, str] = {}
    all_source_code_blocks = []
    all_used_names = set()
    files_to_scan = set()
    top_level_code_to_add = set()

    class FullNameFinder(ast.NodeVisitor):
        def __init__(self):
            self.names = set()

        def visit_Name(self, node):
            self.names.add(node.id)
            self.generic_visit(node)

        def visit_Attribute(self, node):
            with contextlib.suppress(Exception):
                self.names.add(ast.unparse(node))
            self.generic_visit(node)

    for method_name in sorted(method_names):
        for method_name in sorted(method_names):
            try:
                method_obj = resolve_merge_method(method_name)
                source_text = inspect.getsource(method_obj.__wrapped__)
                source_cache[method_name] = textwrap.dedent(source_text)

                source_code = source_cache[method_name]
                all_source_code_blocks.append(source_code)

                module = inspect.getmodule(inspect.unwrap(method_obj))
                if module and hasattr(module, "__file__"):
                    files_to_scan.add(Path(module.__file__))

                parser = FullNameFinder()
                parser.visit(ast.parse(source_code))
                all_used_names.update(parser.names)
            except (ValueError, TypeError) as error:
                logger.error(
                    "Could not get or parse source for '%s': %s",
                    method_name,
                    error,
                )
                all_source_code_blocks.append(
                    f"# ERROR: Could not get source for {method_name}"
                )
            except SystemExit:
                logger.error(
                    "FATAL: resolve_merge_method could not find '%s'. Halting artifact generation for this method.",
                    method_name,
                )
                all_source_code_blocks.append(
                    f"# ERROR: Could not resolve merge method '{method_name}'."
                )

    relevant_imports = set()
    for file_path in files_to_scan:
        try:
            file_source = file_path.read_text(encoding="utf-8")
            file_tree = ast.parse(file_source)

            for node in file_tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for alias in node.names:
                        potential_names = {
                            alias.name,
                            alias.asname,
                            alias.name.split(".")[0],
                        }
                        if not all_used_names.isdisjoint(potential_names):
                            relevant_imports.add(ast.unparse(node))
                            break
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id in all_used_names
                        ):
                            top_level_code_to_add.add(ast.unparse(node))
                            break
        except Exception as error:  # noqa: BLE001 - preserve broad scanning guard.
            logger.error(
                "Could not parse file %s for imports/variables: %s",
                file_path,
                error,
            )

    final_code_str = (
        "\n".join(sorted(top_level_code_to_add))
        + "\n\n"
        + "\n\n".join(all_source_code_blocks)
    )
    return relevant_imports, final_code_str


class MechaToPythonConverter:
    """Translate a `.mecha` recipe string into standalone Python code."""

    def __init__(self, recipe_str: str):
        self.mecha_lines = recipe_str.strip().split("\n")[1:]
        self.python_lines = []

    def convert(self) -> str:
        for index, line in enumerate(self.mecha_lines):
            var_name = f"var_{index}"
            parts = self._parse_line(line)
            command = parts[0]
            python_line = f"# Original: {line}"

            if command == "dict":
                _args, kwargs = self._extract_args_kwargs(parts[1:])
                dict_items = [f'"{key}": {self._remap_ref(value)}' for key, value in kwargs.items()]
                python_line += f"\n{var_name} = {{{', '.join(dict_items)}}}"
            elif command in ["model", "literal"]:
                args, kwargs = self._extract_args_kwargs(parts[1:])
                if "model_config" in kwargs:
                    kwargs["config"] = kwargs.pop("model_config")
                remapped_args = [self._remap_ref(arg) for arg in args]
                remapped_kwargs = {
                    key: self._remap_ref(value) for key, value in kwargs.items()
                }
                call_args = ", ".join(remapped_args)
                if remapped_kwargs:
                    if call_args:
                        call_args += ", "
                    call_args += ", ".join(
                        f"{key}={value}" for key, value in remapped_kwargs.items()
                    )
                python_line += f"\n{var_name} = sd_mecha.{command}({call_args})"
            elif command == "merge":
                method_identifier_str = parts[1]
                args, kwargs = self._extract_args_kwargs(parts[2:])
                remapped_args = [self._remap_ref(arg) for arg in args]
                remapped_kwargs = {
                    key: self._remap_ref(value) for key, value in kwargs.items()
                }
                call_args = ", ".join(remapped_args)
                if remapped_kwargs:
                    if call_args:
                        call_args += ", "
                    call_args += ", ".join(
                        f"{key}={value}" for key, value in remapped_kwargs.items()
                    )
                python_line += (
                    "\n"
                    f"{var_name} = sd_mecha.extensions.merge_methods.resolve({method_identifier_str})({call_args})"
                )
            else:
                python_line += f"\n# SKIPPED UNKNOWN COMMAND: {line}"

            if command in ["dict", "model", "literal", "merge"]:
                self.python_lines.append(python_line)
            else:
                self.python_lines.append(f"# SKIPPED UNKNOWN COMMAND: {line}")

        final_var_name = f"var_{len(self.mecha_lines) - 1}"
        self.python_lines.append("\n# The final recipe is held in this variable")
        self.python_lines.append(f"final_recipe = {final_var_name}")
        return "\n\n".join(self.python_lines)

    def _parse_line(self, line: str) -> list[str]:
        return re.findall(r'"[^"]*"|\S+', line)

    def _extract_args_kwargs(self, parts: list[str]) -> tuple[list[str], dict[str, str]]:
        args = []
        kwargs = {}
        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                kwargs[key] = value
            else:
                args.append(part)
        return args, kwargs

    def _remap_ref(self, ref_str: str) -> str:
        if ref_str.startswith("&") and ref_str[1:].isdigit():
            index = int(ref_str[1:])
            return f"var_{index}"
        return ref_str
