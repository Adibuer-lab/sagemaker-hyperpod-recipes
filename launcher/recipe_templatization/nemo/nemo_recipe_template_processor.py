#!/usr/bin/env python3
import json
from collections import OrderedDict
from typing import Optional

from omegaconf import OmegaConf

from ..base_recipe_template_processor import (
    BaseRecipeTemplateProcessor,
    ServerlessMeteringType,
)
from ...paths import (
    get_recipe_templatization_path,
    get_recipe_yaml_path,
    resolve_project_path,
)


class NemoRecipeTemplateProcessor(BaseRecipeTemplateProcessor):
    """NeMo 2.0 recipe template processor."""

    def __init__(
        self,
        staging_cfg: dict,
        template_path: Optional[str] = None,
        platform: str = "k8s",
    ):
        if template_path is None:
            template_path = get_recipe_templatization_path(
                "nemo", "nemo_recipe_template_parameters.json"
            )
        self.template_path = resolve_project_path(template_path)
        self.platform = platform
        super().__init__(staging_cfg)

    def _load_template(self):
        with open(self.template_path) as f:
            self.template_data = json.load(f)
        with open(get_recipe_templatization_path("jumpstart_model-id_map.json"), "r") as f:
            self.recipe_jumpstart_model_id_mapping = json.load(f)
        with open(
            get_recipe_templatization_path("nemo", "nemo_regional_parameters.json"), "r"
        ) as f:
            self.regional_parameters = json.load(f)

    def get_recipe_template(self, yaml_data: dict, template: dict, recipe_file_path: str = None) -> Optional[dict]:
        if recipe_file_path is None:
            raise ValueError("recipe_file_path is required to get recipe template")

        recipe_name = self.get_recipe_name_from_path(recipe_file_path)
        if "nemo2" in recipe_name:
            return template.get("nemo2_sft_fft")

        raise ValueError(f"Invalid NeMo template for recipe: {recipe_name}")

    def get_recipe_metadata(self, recipe_file_path: str) -> OrderedDict:
        metadata = OrderedDict()
        recipe_cfg = OmegaConf.load(get_recipe_yaml_path(recipe_file_path))

        recipe_file_name = self.get_recipe_name_from_path(recipe_file_path)
        metadata["Name"] = recipe_file_name
        metadata["RecipeFilePath"] = "recipes/" + recipe_file_path + ".yaml"

        metadata["DisplayName"] = recipe_cfg.get("display_name")
        metadata["Type"] = "FineTuning"
        metadata["Framework"] = "NeMo"
        metadata["CustomizationTechnique"] = "SFT"

        # Model ID (prefer explicit mapping; fallback to HF model id if present)
        run_name = OmegaConf.select(recipe_cfg, "run.name")
        if run_name and run_name in self.recipe_jumpstart_model_id_mapping:
            metadata["Model_ID"] = self.recipe_jumpstart_model_id_mapping[run_name]
        else:
            hf_model_id = self._extract_script_arg(recipe_cfg, "--hf-model-id")
            if hf_model_id:
                metadata["Model_ID"] = hf_model_id

        # Hardware and instance types
        if "gpu" in recipe_file_name.lower():
            metadata["Hardware"] = "GPU"
        elif "trn" in recipe_file_name.lower() or "trainium" in recipe_file_name.lower():
            metadata["Hardware"] = "TRN"
        else:
            metadata["Hardware"] = "GPU"

        metadata["InstanceTypes"] = OmegaConf.to_container(recipe_cfg["instance_types"], resolve=True)

        # Versioning
        metadata["Versions"] = [recipe_cfg.get("version")]
        metadata["OutputConfig"] = {"SageMakerInferenceRecipeName": "default"}

        # Instance count
        num_nodes = OmegaConf.select(recipe_cfg, "trainer.num_nodes")
        if num_nodes is None:
            num_nodes = OmegaConf.select(recipe_cfg, "run.nodes")
        metadata["InstanceCount"] = num_nodes

        # Sequence length
        seq_length = self.extract_sequence_length(recipe_file_name)
        if seq_length is None:
            seq_arg = self._extract_script_arg(recipe_cfg, "--seq-length")
            if seq_arg is not None:
                try:
                    seq_length = self.format_sequence_length(int(seq_arg))
                except Exception:
                    seq_length = None
        if seq_length is not None:
            metadata["SequenceLength"] = seq_length

        # Peft
        peft_scheme = self._extract_script_arg(recipe_cfg, "--peft-scheme")
        if peft_scheme and str(peft_scheme).lower() == "lora":
            metadata["Peft"] = "LoRA"

        metadata["ServerlessMeteringType"] = ServerlessMeteringType.TOKEN_BASED.value

        return metadata

    @staticmethod
    def _extract_script_arg(recipe_cfg, arg_name: str) -> Optional[str]:
        script_args = OmegaConf.select(recipe_cfg, "script_args")
        if not script_args:
            return None
        for arg in list(script_args):
            if arg_name in arg:
                return arg[arg_name]
        return None
