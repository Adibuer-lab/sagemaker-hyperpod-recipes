# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import omegaconf
from nemo_launcher.utils.job_utils import JobPaths
from omegaconf import OmegaConf

from ..accelerator_devices import get_num_accelerator_devices
from .constants import (
    HPCT_ENV_VARS,
    HPCT_MODEL_TASK_TO_CODE_PATH,
    NEMO_REPO,
    NEMO_REPO_TAG,
    NEURONX_CONF_PATH,
    NEURONX_REPO_TAG,
    NEURONX_REPO_URI,
    ROOT_DIR,
    SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH,
    SM_ADAPTER_MODEL_TYPE_TO_CONFIG,
    SM_ADAPTER_REPO,
)
from .stages import SMTraining, get_num_nodes, set_multinode_envs


class SMTrainingGPURecipe(SMTraining):
    """
    Stage used to run our GPU recipes
    """

    @property
    def _default_repo(self):
        use_default = self.cfg.get("git", {}).get("use_default", True)
        return SM_ADAPTER_REPO if use_default else None

    @property
    def _entry_script_path(self) -> Path:
        # Use Git entry_script config if available
        cfg_git_entry_script = self.cfg.get("git", {}).get("entry_script", None)
        if cfg_git_entry_script != None:
            return cfg_git_entry_script

        # [TODO] Handle generate the script path from github
        choice_model_type, _ = self.get_stage_config_choice()
        choice_model_type = choice_model_type.split("/")[1]
        # predefined model
        if choice_model_type in SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH:
            return Path(SM_ADAPTER_MODEL_TYPE_TO_CODE_PATH[choice_model_type])

        # custom model
        return Path("examples/custom_model/custom_pretrain.py")

    @property
    def _entry_module(self):
        if self.cfg.get("entry_module", None):
            return self.cfg.entry_module
        return None

    def get_stage_config_choice(self):
        # [TODO] check if need to override
        return super().get_stage_config_choice()

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        super()._copy_k8s_helm_chart(template_root, job_path)

        # Copy any adapter config files so they can be mounted at /config.
        hydra_config_path = Path(job_path.folder / "k8s_template" / "config")
        for path in job_path.folder.glob("*_config.yaml"):
            shutil.copy(path, hydra_config_path)

    def _maybe_write_additional_k8s_configs(self, job_path: JobPaths, stage_cfg_path: Path) -> None:
        """
        For LLMFT recipes on k8s, generate an adapter-compatible Hydra config
        (e.g., smp_llama_config.yaml) so the adapter entrypoint can load it.
        """
        model_type = OmegaConf.select(self.cfg, "recipes.run.model_type", default=None)
        if model_type not in {"llm_finetuning_aws", "hf"}:
            return

        model_key = self._get_adapter_model_key()
        if model_key is None:
            return

        cfg_info = SM_ADAPTER_MODEL_TYPE_TO_CONFIG.get(model_key)
        if not cfg_info:
            return

        base_cfg_path = self._resolve_adapter_config_path(cfg_info["config_path"])
        adapter_cfg = OmegaConf.load(base_cfg_path)

        self._apply_llmft_recipe_overrides(adapter_cfg, self.cfg.recipes)

        out_path = job_path.folder / f"{cfg_info['config_name']}.yaml"
        OmegaConf.save(config=adapter_cfg, f=out_path)

    def _get_adapter_model_key(self) -> Optional[str]:
        """
        Infer adapter model key from the recipe choice (e.g., llama, deepseek).
        """
        try:
            choice_model_type, _ = self.get_stage_config_choice()
        except Exception:
            return None
        if not choice_model_type:
            return None
        # Expected format: "fine-tuning/llama/..."
        parts = choice_model_type.split("/")
        if len(parts) >= 2:
            return parts[1]
        return None

    def _resolve_adapter_config_path(self, rel_path: str) -> Path:
        """
        Resolve adapter config path using an optional env override and common repo layout.
        """
        env_root = os.environ.get("HYPERPOD_ADAPTER_ROOT", "").strip()
        candidates = []
        if env_root:
            candidates.append(Path(env_root))
        # Common container location (image build clones here)
        candidates.append(Path("/opt/hyperpod-adapter"))
        # Default: adapter repo is a sibling of the recipes repo root
        candidates.append(ROOT_DIR.parent / "sagemaker-hyperpod-training-adapter-for-nemo")
        candidates.append(ROOT_DIR / "sagemaker-hyperpod-training-adapter-for-nemo")
        # Fallback: search from current working directory parents
        cwd = Path.cwd()
        candidates.extend(parent / "sagemaker-hyperpod-training-adapter-for-nemo" for parent in (cwd, *cwd.parents))
        for root in candidates:
            cfg_path = root / rel_path
            if cfg_path.exists():
                return cfg_path
        raise FileNotFoundError(
            "Unable to locate adapter config. Set HYPERPOD_ADAPTER_ROOT or ensure "
            "sagemaker-hyperpod-training-adapter-for-nemo is present alongside the recipes repo."
        )

    @staticmethod
    def _apply_llmft_recipe_overrides(adapter_cfg: OmegaConf, recipe_cfg: OmegaConf) -> None:
        """
        Map LLMFT recipe fields into adapter schema.
        Only updates known adapter fields to avoid schema violations.
        """

        def _get(path: str, default=None):
            return OmegaConf.select(recipe_cfg, path, default=default)

        def _set(path: str, value):
            if value is None:
                return
            if isinstance(value, str) and not value.strip():
                return
            OmegaConf.update(adapter_cfg, path, value, merge=False)

        # Trainer
        _set("trainer.devices", _get("trainer.devices"))
        _set("trainer.num_nodes", _get("trainer.num_nodes"))
        _set("trainer.max_steps", _get("training_config.training_args.max_steps"))
        _set("trainer.log_every_n_steps", _get("training_config.training_args.logging_steps"))
        eval_steps = _get("training_config.training_args.eval_steps")
        if eval_steps is not None:
            _set("trainer.val_check_interval", eval_steps)

        # Run/exp_manager
        _set("run.name", _get("run.name"))
        _set("run.results_dir", _get("run.results_dir"))
        exp_dir = _get("training_config.training_args.training_dir")
        if exp_dir:
            _set("exp_manager.exp_dir", exp_dir)
        elif _get("run.results_dir"):
            _set("exp_manager.exp_dir", _get("run.results_dir"))

        # Model basics
        _set("model.hf_model_name_or_path", _get("training_config.model_config.model_name_or_path"))
        # LLMFT implies finetune
        trainer_type = _get("training_config.training_args.trainer_type")
        pretrain_mode = _get("training_config.training_args.pretrain_mode")
        if pretrain_mode:
            _set("model.do_finetune", False)
        else:
            _set("model.do_finetune", True)
        _set("model.seed", _get("training_config.training_args.seed"))
        _set("model.multi_modal", _get("training_config.model_config.multimodal"))

        attn_impl = _get("training_config.model_config.attn_implementation")
        if isinstance(attn_impl, str):
            attn_norm = attn_impl.strip().lower()
            if "flash" in attn_norm:
                _set("model.use_flash_attention", True)
            elif attn_norm in {"sdpa", "eager"}:
                _set("model.use_flash_attention", False)

        # Architecture overrides (only if present in recipe)
        arch_map = {
            "training_config.model_config.num_hidden_layers": "model.num_hidden_layers",
            "training_config.model_config.hidden_size": "model.hidden_size",
            "training_config.model_config.num_attention_heads": "model.num_attention_heads",
            "training_config.model_config.intermediate_size": "model.intermediate_size",
            "training_config.model_config.initializer_range": "model.initializer_range",
            "training_config.model_config.layernorm_epsilon": "model.layernorm_epsilon",
            "training_config.model_config.vocab_size": "model.vocab_size",
            "training_config.model_config.num_key_value_heads": "model.num_key_value_heads",
            "training_config.model_config.rope_theta": "model.rope_theta",
            "training_config.model_config.rope_scaling": "model.rope_scaling",
        }
        for src, dst in arch_map.items():
            _set(dst, _get(src))

        # Context length
        max_len = _get("training_config.training_args.max_len")
        if max_len is not None:
            _set("model.max_context_width", max_len)

        # Batch sizes
        micro_bs = _get("training_config.training_args.micro_train_batch_size")
        train_bs = _get("training_config.training_args.train_batch_size")
        _set("model.train_batch_size", micro_bs if micro_bs is not None else train_bs)

        # Gradient clipping
        grad_clip = _get("training_config.training_args.gradient_clipping_threshold")
        if grad_clip is None:
            grad_clip = _get("training_config.training_args.max_norm")
        if _get("training_config.training_args.gradient_clipping") is False:
            grad_clip = 0
        _set("model.grad_clip", grad_clip)

        # Optimizer
        _set("model.optim.lr", _get("training_config.training_args.learning_rate"))
        _set("model.optim.weight_decay", _get("training_config.training_args.weight_decay"))
        _set("model.optim.betas", _get("training_config.training_args.adam_betas"))

        lr_sched = _get("training_config.training_args.lr_scheduler")
        if lr_sched:
            if str(lr_sched).lower() == "cosine":
                _set("model.optim.sched.name", "CosineAnnealing")

        # Warmup steps from ratio if possible
        warmup_ratio = _get("training_config.training_args.lr_warmup_ratio")
        if warmup_ratio is not None:
            try:
                max_steps = OmegaConf.select(adapter_cfg, "trainer.max_steps", default=None)
                if max_steps:
                    warmup_steps = int(float(warmup_ratio) * int(max_steps))
                    _set("model.optim.sched.warmup_steps", warmup_steps)
            except Exception:
                pass

        # Logging/checkpoint intervals
        save_steps = _get("training_config.training_args.save_steps")
        if save_steps is not None:
            _set("exp_manager.checkpoint_callback_params.every_n_train_steps", save_steps)

        # Data paths
        train_path = _get("training_config.datasets.train_data.file_path")
        val_path = _get("training_config.datasets.val_data.file_path")
        if train_path:
            _set("model.data.train_dir", train_path)
        if val_path:
            _set("model.data.val_dir", val_path)
        if train_path or val_path:
            _set("model.data.use_synthetic_data", False)

        # FSDP/strategy config (map common recipe enums)
        fsdp_cfg = _get("training_config.training_args.strategy.fsdp_config")
        if isinstance(fsdp_cfg, omegaconf.DictConfig):
            def _norm_enum(value: Optional[str]) -> Optional[str]:
                if value is None:
                    return None
                if not isinstance(value, str):
                    return value
                return value.strip().lower()

            shard_map = {
                "no_shard": "no_shard",
                "shard_grad_op": "shard_grad_op",
                "hybrid_shard": "hybrid_shard",
                "hybrid_shard_zero2": "_hybrid_shard_zero2",
                "full_shard": "full_shard",
            }
            wrap_map = {
                "transformer_based_wrap": "transformer_auto_wrap_policy",
                "size_based_wrap": "size_based_auto_wrap_policy",
            }
            backward_map = {
                "backward_pre": "backward_pre",
                "backward_post": "backward_post",
            }

            sharding = shard_map.get(_norm_enum(fsdp_cfg.get("sharding_strategy")))
            _set("model.sharding_strategy", sharding)
            wrap = wrap_map.get(_norm_enum(fsdp_cfg.get("auto_wrap_policy")))
            _set("model.auto_wrap_policy", wrap)
            back_prefetch = backward_map.get(_norm_enum(fsdp_cfg.get("backward_prefetch")))
            _set("model.backward_fetch_policy", back_prefetch)
            _set("model.forward_prefetch", fsdp_cfg.get("forward_prefetch"))
            _set("model.limit_all_gathers", fsdp_cfg.get("limit_all_gathers"))
            _set("model.use_orig_param", fsdp_cfg.get("use_orig_params"))

        # PEFT / LoRA
        peft_type = _get("training_config.model_config.peft_config.peft_type")
        if peft_type:
            _set("model.peft.peft_type", peft_type)
            _set("model.peft.rank", _get("training_config.model_config.peft_config.r"))
            _set("model.peft.alpha", _get("training_config.model_config.peft_config.lora_alpha"))
            _set("model.peft.dropout", _get("training_config.model_config.peft_config.lora_dropout"))
            targets = _get("training_config.model_config.peft_config.target_modules")
            if isinstance(targets, str):
                targets = [targets]
            _set("model.peft.target_modules", targets)
            _set("use_smp_model", False)

        # DPO
        if isinstance(trainer_type, str) and trainer_type.strip().lower() == "dpo":
            _set("model.dpo.enabled", True)
            _set("model.dpo.beta", _get("training_config.training_args.beta"))


class SMTrainingGPURecipeElastic(SMTrainingGPURecipe):
    """
    Stage used to run elastic training jobs
    """

    @staticmethod
    def save_stage_hydra_config(stage_cfg: OmegaConf, job_path: JobPaths, cfg: OmegaConf) -> Path:
        default_cfg_path = SMTrainingGPURecipe.save_stage_hydra_config(stage_cfg, job_path, cfg)

        remove_keys = {"scale_config", "elastic_policy"}
        basic_config = {key: value for key, value in stage_cfg.items() if key not in remove_keys}
        basic_config = OmegaConf.create(basic_config)

        scale_configs = stage_cfg.get("scale_config", {})
        scale_space = scale_configs.keys()

        for nnodes in scale_space:
            new_stage_cfg = OmegaConf.merge(basic_config, scale_configs[nnodes])
            cfg_save_path = job_path.folder / f"train_config_n{nnodes}.yaml"
            omegaconf.OmegaConf.save(new_stage_cfg, cfg_save_path)

        return default_cfg_path

    def _copy_k8s_helm_chart(self, template_root: str, job_path: JobPaths):
        super()._copy_k8s_helm_chart(template_root, job_path)

        train_config_paths = job_path.folder.glob("train_config_n*.yaml")
        hydra_config_path = Path(job_path.folder / "k8s_template" / "config")
        for path in train_config_paths:
            shutil.copy(path, hydra_config_path)

    def generate_default_k8s_value_template(self, template_root, cluster_parameters, stage_cfg_path=None):
        values_template = super().generate_default_k8s_value_template(template_root, cluster_parameters, stage_cfg_path)

        elastic_policy = self.cfg.training.elastic_policy
        scale_config = self.cfg.training.get("scale_config", None)

        values_template.trainingConfig.elastic_policy.is_elastic = elastic_policy.is_elastic
        values_template.trainingConfig.elastic_policy.min_nodes = elastic_policy.min_nodes
        values_template.trainingConfig.elastic_policy.max_nodes = elastic_policy.max_nodes

        if scale_config:
            scale_space = list(scale_config.keys())
            values_template.trainingConfig.elastic_policy.replica_space = scale_space
        else:
            assert (
                elastic_policy.get("replica_increment_step", None) is not None
            ), "Either scale_config or replica_increment_step need to be defined"
            values_template.trainingConfig.elastic_policy.replica_increment_step = elastic_policy.replica_increment_step

        use_graceful_shutdown = elastic_policy.get("use_graceful_shutdown", True)
        values_template.trainingConfig.elastic_policy.use_graceful_shutdown = use_graceful_shutdown
        if use_graceful_shutdown:
            values_template.trainingConfig.envVars.HYPERPOD_SIGNAL_COORDINATION = "distributed"

        if elastic_policy.get("scaling_timeout", None) is not None:
            values_template.trainingConfig.elastic_policy.scaling_timeout = elastic_policy.scaling_timeout
        if elastic_policy.get("graceful_shutdown_timeout", None) is not None:
            values_template.trainingConfig.elastic_policy.graceful_shutdown_timeout = (
                elastic_policy.graceful_shutdown_timeout
            )
        if elastic_policy.get("faulty_timeout", None) is not None:
            values_template.trainingConfig.elastic_policy.faulty_timeout = elastic_policy.faulty_timeout

        return values_template

    def get_script_args_str(self, stage_cfg_path: Path) -> str:
        """
        Based on https://github.com/NVIDIA/NeMo-Framework-Launcher/blob/23.11/launcher_scripts/nemo_launcher/core/stages.py#L608
        """
        if self.cluster == "k8s":
            model_type = OmegaConf.select(self.cfg, "recipes.run.model_type", default=None)
            if model_type in {"llm_finetuning_aws", "hf"}:
                return "--config-path=/config"
            return "--config-path=/config --config-name=config.yaml"
        return f"--config-path={stage_cfg_path.parents[0]} --config-name={stage_cfg_path.name}"


class NeMoTraining(SMTraining):
    """
    Stage to run NeMo recipes
    """

    @property
    def _nemo_code_path(self) -> Path:
        return Path("")

    @property
    def _default_repo(self):
        return NEMO_REPO

    @property
    def _default_branch(self):
        return NEMO_REPO_TAG

    @property
    def _entry_script_path(self) -> Path:
        choice_model_type, _ = self.get_stage_config_choice()
        choice_model_type = choice_model_type.split("/")[1]
        code_path = self._get_nemo_code_path(choice_model_type)
        return Path(code_path)

    @property
    def _entry_module(self):
        if self.cfg.get("entry_module", None):
            return self.cfg.entry_module
        return None


class SMTrainingTrainiumRecipe(SMTraining):
    """
    Stage to run our Trainium recipes
    """

    DEFAULT_TRAIN_SCRIPT_PATH = "examples/train.sh"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.device = "trainium"

        # Used by Slurm and K8s. Example: "llama/megatron_llama_7B_config"
        self._training_filename = self.cfg.training_config.rsplit("/", 1)[-1]
        self._temp_training_conf_file = ROOT_DIR / f"tmp/training/{self._training_filename}.yaml"

        if not self._temp_training_conf_file.parent.exists():
            self._temp_training_conf_file.parent.mkdir(parents=True)

    @property
    def _default_repo(self):
        return NEURONX_REPO_URI

    @property
    def _default_branch(self):
        return NEURONX_REPO_TAG

    @property
    def _entry_script_path(self) -> Path:
        cfg_git_entry_script = self.cfg.get("git", {}).get("entry_script")
        entry_script_path = cfg_git_entry_script or self.DEFAULT_TRAIN_SCRIPT_PATH
        return Path(entry_script_path)

    def _make_custom_call_string(self, stage_cfg_path=None):
        """
        Create the command that runs the training script
        """
        compile = OmegaConf.select(self.cfg, "recipes.run.compile", default=0)

        commands: List[str] = [
            "# copy the resolved training config file into the cloned Neuronx repo",
            f"cp -f {self._temp_training_conf_file} {NEURONX_CONF_PATH}",
            "",
            "# training script depends on other files invoked with relative paths, so must cd into it",
            f'cd "$(dirname {self._entry_script_path})"',
            "",
            "# run training script but first define its arguments",
            f"export CONF_FILE={self._training_filename}",
            f"export COMPILE={compile}",
            f'bash ./"$(basename {self._entry_script_path})"',
            "",
        ]
        return "\n".join(commands)

    def update_stage_specific_k8s_values(self, values_template):
        """
        training specifc k8s values for trainum
        """
        super().update_stage_specific_k8s_values(values_template)
        values_template.trainingConfig.numNeuronDevices = get_num_accelerator_devices(self.instance_type)
        return values_template

    def get_env_vars(self) -> Dict:
        """
        Set up dictionary for environment variables
        By default injecting the EFA env variable when doing multi-node training
        The environment variables from hydra config will be set inside the job scripts.
        For Example:
            Set `env_vars.NVTE_BIAS_DROPOUT_FUSION=1` while calling nemo_launcherlauncher-scripts,
            `NVTE_BIAS_DROPOUT_FUSION=1` will be set while running the job.

        :return: a dictionary of env vars while running the job.
        :rtype: Dict
        """
        env_vars = super().get_env_vars()
        stage_cfg = self.stage_cfg
        nodes = get_num_nodes(stage_cfg)
        if int(nodes) > 1:
            env_vars = set_multinode_envs(env_vars, self.instance_type)
        return env_vars


class SMTrainingHPCTRecipe(SMTraining):
    """
    Stage used to run HyperPod Checkpoint-less Training recipes
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        cluster_type = cfg.get("cluster_type")
        if cluster_type != "k8s":
            raise ValueError(
                f"HyperPod checkpointless training recipes only support K8s cluster type, got: {cluster_type}"
            )

    @property
    def _default_repo(self):
        return None  # No repo needed, scripts are in container

    @property
    def _entry_module(self):
        return None

    @property
    def _entry_script_path(self) -> Path:
        training_config = self.cfg.get("training_config")

        # Extract model name from path: training/llama/... -> llama
        path_parts = training_config.split("/")
        model_name = path_parts[1]
        if "fine_tuning" in training_config.lower():
            task = "fine_tuning"
        elif "lora" in training_config.lower():
            task = "lora"
        elif "pretrain" in training_config.lower():
            task = "pretrain"
        else:
            raise ValueError(f"Cannot determine task type from training config: {training_config}")

        model_task_key = f"{model_name}_{task}"

        if model_task_key not in HPCT_MODEL_TASK_TO_CODE_PATH:
            supported_types = list(HPCT_MODEL_TASK_TO_CODE_PATH.keys())
            raise ValueError(
                f"HyperPod checkpointless model-task combination '{model_task_key}' is not supported. "
                f"Supported types: {supported_types}"
            )

        entry_script_path = HPCT_MODEL_TASK_TO_CODE_PATH[model_task_key]
        return Path(entry_script_path)

    def get_env_vars(self) -> Dict:
        """
        Set up HCT-specific environment variables
        Handle all logic without calling parent to avoid conflicts
        """
        return HPCT_ENV_VARS.copy()
