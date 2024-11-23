# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications:
# This file has been adapted and modified from its original version by Gabriel Y. Arteaga.
# Have modified the file to accomodate the custom BatchModel, which is a custom model which uses BatchEnsemble implementation.

from __future__ import annotations

import collections
import inspect
import json
import os
import re
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import packaging
import torch
import transformers
from accelerate import dispatch_model, infer_auto_device_map
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from accelerate.utils import get_balanced_memory, named_module_tensors
from huggingface_hub import ModelCard, ModelCardData, hf_hub_download
from peft import PeftConfig, PeftType, TaskType, PEFT_TYPE_TO_CONFIG_MAPPING
from peft.tuners import (
    AdaLoraModel,
    AdaptionPromptModel,
    IA3Model,
    LoHaModel,
    LoKrModel,
    LoraModel,
    MultitaskPromptEmbedding,
    OFTModel,
    PolyModel,
    PrefixEncoder,
    PromptEmbedding,
    PromptEncoder,
)
from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
from peft.utils.constants import (
    SAFETENSORS_WEIGHTS_NAME,
    TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING,
    WEIGHTS_NAME,
)
from peft.utils.other import (
    _get_batch_size,
    _prepare_prompt_learning_config,
    _set_adapter,
    _set_trainable,
    id_tensor_storage,
    infer_device,
    shift_tokens_right,
)
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    load_peft_weights,
    set_peft_model_state_dict,
)
from safetensors import safe_open
from safetensors.torch import save_file as safe_save_file
from transformers import PreTrainedModel, GenerationConfig
from transformers.utils import PushToHubMixin
from transformers.utils import logging
from transformers.quantizers import HfQuantizer
from transformers.modeling_utils import (
    _get_tied_weight_keys,
    _find_disjoint,
    _add_variant,
    shard_checkpoint,
)
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    WEIGHTS_INDEX_NAME,
)
from .batch_lora_model import BatchLoraModel

logger = logging.get_logger(__name__)

# Don't like this, will try to find a neater approach...
class PeftModel(PushToHubMixin, torch.nn.Module):
    """
    Base model encompassing various Peft methods.

    Args:
        model ([`~transformers.PreTrainedModel`]): The base transformer model used for Peft.
        peft_config ([`PeftConfig`]): The configuration of the Peft model.
        adapter_name (`str`,  *optional*): The name of the adapter, defaults to `"default"`.

    **Attributes**:
        - **base_model** ([`torch.nn.Module`]) -- The base transformer model used for Peft.
        - **peft_config** ([`PeftConfig`]) -- The configuration of the Peft model.
        - **modules_to_save** (`list` of `str`) -- The list of sub-module names to save when
            saving the model.
        - **prompt_encoder** ([`PromptEncoder`]) -- The prompt encoder used for Peft if
            using [`PromptLearningConfig`].
        - **prompt_tokens** (`torch.Tensor`) -- The virtual prompt tokens used for Peft if
            using [`PromptLearningConfig`].
        - **transformer_backbone_name** (`str`) -- The name of the transformer
            backbone in the base model if using [`PromptLearningConfig`].
        - **word_embeddings** (`torch.nn.Embedding`) -- The word embeddings of the transformer backbone
            in the base model if using [`PromptLearningConfig`].
    """

    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__()
        self.modules_to_save = None
        self.active_adapter = adapter_name
        self.peft_type = peft_config.peft_type
        # These args are special PEFT arguments that users can pass. They need to be removed before passing them to
        # forward.
        self.special_peft_forward_args = {"adapter_names"}

        self._is_prompt_learning = peft_config.is_prompt_learning
        if self._is_prompt_learning:
            self._peft_config = {adapter_name: peft_config}
            self.base_model = model
            self.add_adapter(adapter_name, peft_config)
        else:
            self._peft_config = None
            # Overwrote this part, rest
            self.base_model = BatchLoraModel(model, {adapter_name: peft_config}, adapter_name)
            self.set_additional_trainable_modules(peft_config, adapter_name)

        if getattr(model, "is_gradient_checkpointing", True):
            model = self._prepare_model_for_gradient_checkpointing(model)

        # the `pretraining_tp` is set for some models to simulate Tensor Parallelism during inference to avoid
        # numerical differences, https://github.com/pytorch/pytorch/issues/76232 - to avoid any unexpected
        # behavior we disable that in this line.
        if hasattr(self.base_model, "config") and hasattr(self.base_model.config, "pretraining_tp"):
            self.base_model.config.pretraining_tp = 1

        # Copied from PeftModelForCausalLM
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    @property
    def peft_config(self) -> dict[str, PeftConfig]:
        if self._is_prompt_learning:
            return self._peft_config
        return self.base_model.peft_config

    @property
    def active_adapters(self) -> list[str]:
        try:
            adapters = self.base_model.active_adapters
        except AttributeError:
            adapters = self.active_adapter
            if isinstance(adapters, str):
                adapters = [adapters]
        return adapters

    @peft_config.setter
    def peft_config(self, value: dict[str, PeftConfig]):
        if self._is_prompt_learning:
            self._peft_config = value
        else:
            self.base_model.peft_config = value

    ###########################
    # This part is mostly copied from PreTrainedModel, made some changes to merge the lora matrices with pretrained matrices and make the names 
    # match when loading the model.
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = False,
        **kwargs,
    ) -> None:
        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        hf_quantizer = getattr(self, "hf_quantizer", None)
        quantization_serializable = (
            hf_quantizer is not None and isinstance(hf_quantizer, HfQuantizer) and hf_quantizer.is_serializable
        )

        if hf_quantizer is not None and not _hf_peft_config_loaded and not quantization_serializable:
            raise ValueError(
                f"The model is quantized with {hf_quantizer.quantization_config.quant_method} and is not serializable - check out the warnings from"
                " the logger on the traceback to understand the reason why the quantized model is not serializable."
            )
        
        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        os.makedirs(save_directory, exist_ok=True)
        
        # Merge the LoRA weights with the pretrained weights
        self.merge_adapter()

        # This part copied from PreTrainedModel class
        if is_main_process:
            if not _hf_peft_config_loaded:
                self.base_model.model.config.save_pretrained(save_directory)
            if self.base_model.model.can_generate():
            # generation config built from the model config + the model config holds generation kwargs -> generate
            # may revert to legacy behavior if the two don't match
                if (
                    self.base_model.model.generation_config._from_model_config
                    and self.base_model.model.config._has_non_default_generation_parameters()
                ):
                    new_generation_config = GenerationConfig.from_model_config(self.base_model.model.config)
                    if new_generation_config != self.base_model.model.generation_config:
                        logger.warning(
                            "Your generation config was originally created from the model config, but the model "
                            "config has changed since then. Unless you pass the `generation_config` argument to this "
                            "model's `generate` calls, they will revert to the legacy behavior where the base "
                            "`generate` parameterization is loaded from the model config instead. "
                            "To avoid this behavior and this warning, we recommend you to overwrite the generation "
                            "config model attribute before calling the model's `save_pretrained`, preferably also "
                            "removing any generation kwargs from the model config. This warning will be raised to an "
                            "exception in v4.41."
                        )
            self.base_model.model.generation_config.save_pretrained(save_directory)
        
        # Get the state dict of the model
        state_dict = self.state_dict()
        # Keep all the states corresponding to the original archtiecture, before adding lora
        state_dict = {k: state_dict[k] for k in state_dict if 'base_layer' in k or 'RMS' in k or 'lm' in k and 'lora' not in k}
        
        for key in list(state_dict.keys()):
            # We remove the appended 'base_model.model' and '.base_layer' part so that the names are recognized when 
            # we load the model again
            new_key_name = re.sub(r'^base_model\.model\.|\.base_layer', '', key)
            state_dict[new_key_name] = state_dict.pop(key)

        if safe_serialization:
            # Safetensors does not allow tensor aliasing.
            # We're going to remove aliases before saving
            ptrs = collections.defaultdict(list)
            for name, tensor in state_dict.items():
                # Sometimes in the state_dict we have non-tensor objects.
                # e.g. in bitsandbytes we have some `str` objects in the state_dict
                if isinstance(tensor, torch.Tensor):
                    ptrs[id_tensor_storage(tensor)].append(name)
                else:
                    # In the non-tensor case, fall back to the pointer of the object itself
                    ptrs[id(tensor)].append(name)
                    
                # These are all the pointers of shared tensors.
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}
                error_names = []
                to_delete_names = set()
                # Recursively descend to find tied weight keys
                _tied_weights_keys = _get_tied_weight_keys(self.base_model.model)
                for names in shared_ptrs.values():
                    # Removing the keys which are declared as known duplicates on
                    # load. This allows to make sure the name which is kept is consistent.
                    if _tied_weights_keys is not None:
                        found = 0
                        for name in sorted(names):
                            matches_pattern = any(re.search(pat, name) for pat in _tied_weights_keys)
                            if matches_pattern and name in state_dict:
                                found += 1
                                if found < len(names):
                                    to_delete_names.add(name)
                # We are entering a place where the weights and the transformers configuration do NOT match.
                shared_names, disjoint_names = _find_disjoint(shared_ptrs.values(), state_dict)
                # Those are actually tensor sharing but disjoint from each other, we can safely clone them
                # Reloaded won't have the same property, but it shouldn't matter in any meaningful way.
                for name in disjoint_names:
                    state_dict[name] = state_dict[name].clone()


            # Shard the model if it is too big.
            if not _hf_peft_config_loaded:
                weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
                weights_name = _add_variant(weights_name, variant)
            else:
                weights_name = ADAPTER_SAFE_WEIGHTS_NAME if safe_serialization else ADAPTER_WEIGHTS_NAME

            shards, index = shard_checkpoint(state_dict, max_shard_size=max_shard_size, weights_name=weights_name)

            # Clean the folder from a previous save
            for filename in os.listdir(save_directory):
                full_filename = os.path.join(save_directory, filename)
                # If we have a shard file that is not going to be replaced, we delete it, but only from the main process
                # in distributed settings to avoid race conditions.
                weights_no_suffix = weights_name.replace(".bin", "").replace(".safetensors", "")

                # make sure that file to be deleted matches format of sharded file, e.g. pytorch_model-00001-of-00005
                filename_no_suffix = filename.replace(".bin", "").replace(".safetensors", "")
                reg = re.compile(r"(.*?)-\d{5}-of-\d{5}")

                if (
                    filename.startswith(weights_no_suffix)
                    and os.path.isfile(full_filename)
                    and filename not in shards.keys()
                    and is_main_process
                    and reg.fullmatch(filename_no_suffix) is not None
                ):
                    os.remove(full_filename)

            # Save the model
            for shard_file, shard in shards.items():
                if safe_serialization:
                    # At some point we will need to deal better with save_function (used for TPU and other distributed
                    # joyfulness), but for now this enough.
                    safe_save_file(shard, os.path.join(save_directory, shard_file), metadata={"format": "pt"})
                else:
                    save_function(shard, os.path.join(save_directory, shard_file))

            if index is None:
                path_to_weights = os.path.join(save_directory, weights_name)
                logger.info(f"Model weights saved in {path_to_weights}")
            else:
                save_index_file = SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
                save_index_file = os.path.join(save_directory, _add_variant(save_index_file, variant))
                # Save the index as well
                with open(save_index_file, "w", encoding="utf-8") as f:
                    content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                    f.write(content)
                logger.info(
                    f"The model is bigger than the maximum size per checkpoint ({max_shard_size}) and is going to be "
                    f"split in {len(shards)} checkpoint shards. You can find where each parameters has been saved in the "
                    f"index located at {save_index_file}."
                )
        # Unmerge the adapter after saving to continue training
        self.unmerge_adapter()
    @classmethod
    def from_pretrained(
        cls,
        model: torch.nn.Module,
        model_id: Union[str, os.PathLike],
        adapter_name: str = "default",
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ) -> PeftModel:
        r"""
        Instantiate a PEFT model from a pretrained model and loaded PEFT weights.

        Note that the passed `model` may be modified inplace.

        Args:
            model ([`torch.nn.Module`]):
                The model to be adapted. For 🤗 Transformers models, the model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`].
            model_id (`str` or `os.PathLike`):
                The name of the PEFT configuration to use. Can be either:
                    - A string, the `model id` of a PEFT configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a PEFT configuration file saved using the `save_pretrained`
                      method (`./my_peft_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuration. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific PEFT configuration class.
        """
        from peft import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING

        # load the config
        if config is None:
            config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    subfolder=kwargs.get("subfolder", None),
                    revision=kwargs.get("revision", None),
                    cache_dir=kwargs.get("cache_dir", None),
                    use_auth_token=kwargs.get("use_auth_token", None),
                    token=kwargs.get("token", None),
                )
            ].from_pretrained(model_id, **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        if hasattr(model, "hf_device_map"):
            weight_map = dict(named_module_tensors(model, recurse=True))

            # recreate the offload_index for disk-offloaded modules: we need to know the location in storage of each weight
            # before the offload hook is removed from the model
            disk_modules = set()
            index = None
            for name, module in model.named_modules():
                if hasattr(module, "_hf_hook") and hasattr(module._hf_hook, "original_devices"):
                    if hasattr(module._hf_hook.weights_map, "dataset"):
                        index = module._hf_hook.weights_map.dataset.index
                    for key in module._hf_hook.original_devices.keys():
                        if module._hf_hook.original_devices[key] == torch.device("meta"):
                            disk_modules.add(str(name) + "." + str(key))

            if disk_modules and not kwargs.get("use_safetensors", True):
                raise ValueError("Disk offloading currently only supported for safetensors")

            if index:
                offload_index = {
                    p: {
                        "safetensors_file": index[p]["safetensors_file"],
                        "weight_name": p,
                        "dtype": str(weight_map[p].dtype).replace("torch.", ""),
                    }
                    for p in weight_map.keys()
                    if p in disk_modules
                }
                kwargs["offload_index"] = offload_index

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if config.is_prompt_learning and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        if config.task_type not in MODEL_TYPE_TO_PEFT_MODEL_MAPPING.keys():
            model = cls(model, config, adapter_name)
        else:
            model = MODEL_TYPE_TO_PEFT_MODEL_MAPPING[config.task_type](model, config, adapter_name)
        model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
        return model

    def _setup_prompt_encoder(self, adapter_name: str):
        config = self.peft_config[adapter_name]
        if not hasattr(self, "prompt_encoder"):
            self.prompt_encoder = torch.nn.ModuleDict({})
            self.prompt_tokens = {}
        transformer_backbone = None
        for name, module in self.base_model.named_children():
            for param in module.parameters():
                param.requires_grad = False
            if isinstance(module, PreTrainedModel):
                # Make sure to freeze Tranformers model
                if transformer_backbone is None:
                    transformer_backbone = module
                    self.transformer_backbone_name = name
        if transformer_backbone is None:
            transformer_backbone = self.base_model

        if config.num_transformer_submodules is None:
            config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

        for named_param, value in list(transformer_backbone.named_parameters()):
            # for ZeRO-3, the tensor is sharded across accelerators and deepspeed modifies it to a tensor with shape [0]
            # the actual unsharded shape is stored in "ds_shape" attribute
            # special handling is needed in case the model is initialized in deepspeed.zero.Init() context or HfDeepSpeedConfig
            # has been called before
            # For reference refer to issue: https://github.com/huggingface/peft/issues/996
            deepspeed_distributed_tensor_shape = getattr(value, "ds_shape", None)

            if value.shape[0] == self.base_model.config.vocab_size or (
                deepspeed_distributed_tensor_shape is not None
                and deepspeed_distributed_tensor_shape[0] == self.base_model.config.vocab_size
            ):
                self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
                break

        if config.peft_type == PeftType.PROMPT_TUNING:
            prompt_encoder = PromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompt_encoder = MultitaskPromptEmbedding(config, self.word_embeddings)
        elif config.peft_type == PeftType.P_TUNING:
            prompt_encoder = PromptEncoder(config)
        elif config.peft_type == PeftType.PREFIX_TUNING:
            prompt_encoder = PrefixEncoder(config)
        else:
            raise ValueError("Not supported")

        prompt_encoder = prompt_encoder.to(self.device)
        self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
        self.prompt_tokens[adapter_name] = torch.arange(
            config.num_virtual_tokens * config.num_transformer_submodules
        ).long()

    def _prepare_model_for_gradient_checkpointing(self, model: PreTrainedModel):
        r"""
        Prepares the model for gradient checkpointing if necessary
        """
        if not (
            getattr(model, "is_loaded_in_8bit", False)
            or getattr(model, "is_loaded_in_4bit", False)
            or getattr(model, "is_quantized", False)
        ):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            elif hasattr(model, "get_input_embeddings"):

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        return model

    def get_prompt_embedding_to_save(self, adapter_name: str) -> torch.Tensor:
        """
        Returns the prompt embedding to save when saving the model. Only applicable when using a prompt learning
        method.
        """
        prompt_encoder = self.prompt_encoder[adapter_name]
        prompt_tokens = (
            self.prompt_tokens[adapter_name].unsqueeze(0).expand(1, -1).to(prompt_encoder.embedding.weight.device)
        )
        if self.peft_config[adapter_name].peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : self.peft_config[adapter_name].num_virtual_tokens]

        if self.peft_config[adapter_name].peft_type == PeftType.MULTITASK_PROMPT_TUNING:
            prompt_embeddings = super(MultitaskPromptEmbedding, prompt_encoder).forward(prompt_tokens)
        else:
            prompt_embeddings = prompt_encoder(prompt_tokens)

        return prompt_embeddings[0].detach().cpu()

    def get_prompt(self, batch_size: int, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Returns the virtual prompts to use for Peft. Only applicable when using a prompt learning method.
        """
        peft_config = self.active_peft_config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                peft_config.num_transformer_submodules * 2
            )
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if peft_config.peft_type == PeftType.MULTITASK_PROMPT_TUNING:
                prompts = prompt_encoder(prompt_tokens, task_ids)
            else:
                if peft_config.inference_mode:
                    prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
                else:
                    prompts = prompt_encoder(prompt_tokens)
            return prompts

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        r"""
        Returns the number of trainable parameters and the number of all parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                if hasattr(param, "element_size"):
                    num_bytes = param.element_size()
                elif not hasattr(param, "quant_storage"):
                    num_bytes = 1
                else:
                    num_bytes = param.quant_storage.itemsize
                num_params = num_params * 2 * num_bytes

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    def print_trainable_parameters(self) -> None:
        """
        Prints the number of trainable parameters in the model.

        Note: print_trainable_parameters() uses get_nb_trainable_parameters() which is different from
        num_parameters(only_trainable=True) from huggingface/transformers. get_nb_trainable_parameters() returns
        (trainable parameters, all parameters) of the Peft Model which includes modified backbone transformer model.
        For techniques like LoRA, the backbone transformer model is modified in place with LoRA modules. However, for
        prompt tuning, the backbone transformer model is unmodified. num_parameters(only_trainable=True) returns number
        of trainable parameters of the backbone transformer model which can be different.
        """
        trainable_params, all_param = self.get_nb_trainable_parameters()

        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
        )

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.base_model, name)

    @contextmanager
    def _enable_peft_forward_hooks(self, *args, **kwargs):
        # If the base model has a method called _enable_peft_forward_hooks, it is invoked as a context. Otherwise, this
        # runs without any changes
        if hasattr(self.base_model, "_enable_peft_forward_hooks"):
            with self.base_model._enable_peft_forward_hooks(*args, **kwargs):
                yield
            return
        else:
            # nothing to enable
            yield
            return

    def forward(self, *args: Any, **kwargs: Any):
        """
        Forward pass of the model.
        """
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.get_base_model()(*args, **kwargs)

    def generate(self, *args, **kwargs):
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.get_base_model().generate(*args, **kwargs)

    def _get_base_model_class(self, is_prompt_tuning=False):
        """
        Returns the base model class.
        """
        if not is_prompt_tuning:
            return self.base_model.model.__class__
        return self.base_model.__class__

    @contextmanager
    def disable_adapter(self):
        """
        Context manager that disables the adapter module. Use this to run inference on the base model.

        Example:

        ```py
        >>> with model.disable_adapter():
        ...     model(inputs)
        ```
        """
        if self.peft_config[self.active_adapter].is_prompt_learning:
            try:
                # TODO: consider replacing this patching of methods with a more robust mechanism: setting a flag and
                # letting the underlying methods deal with it, same as how LoRA does it.
                old_forward = self.forward
                self.forward = self.base_model.forward
                old_prepare_inputs_for_generation = self.prepare_inputs_for_generation
                self.prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation
                yield
            finally:
                self.forward = old_forward
                self.prepare_inputs_for_generation = old_prepare_inputs_for_generation

        elif self.peft_config[self.active_adapter].is_adaption_prompt:
            try:
                self.base_model.disable_adapter_layers()
                yield
            finally:
                self.base_model.enable_adapter_layers()

        else:  # LoRA, LoHa, etc.
            model_status = self.get_model_status()
            if model_status.enabled == "irregular":
                warnings.warn(
                    "The model contains some adapter layers that are enabled and others that are disabled. "
                    "This is most likely unintentional. After exiting the disable_adapter context, all adapters "
                    "will be enabled"
                )
            try:
                self.base_model.disable_adapter_layers()
                yield
            finally:
                if model_status.enabled is not False:
                    # model_status.enabled is `True` or `"irregular"`
                    self.base_model.enable_adapter_layers()

    def get_base_model(self) -> torch.nn.Module:
        """
        Returns the base model.
        """
        return (
            self.base_model
            if (self.active_peft_config.is_prompt_learning or self.peft_type == PeftType.POLY)
            else self.base_model.model
        )

    def add_adapter(self, adapter_name: str, peft_config: PeftConfig) -> None:
        """
        Add an adapter to the model based on the passed configuration.

        This adapter is not trained. To load a trained adapter, check out [`PeftModel.load_adapter`].

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
        """
        if peft_config.peft_type != self.peft_type:
            raise ValueError(
                f"Cannot combine adapters with different peft types. "
                f"Found {self.peft_type} and {peft_config.peft_type}."
            )

        try:
            if peft_config.is_prompt_learning:
                self.peft_config[adapter_name] = peft_config
                if hasattr(self.config, "to_dict"):
                    dict_config = self.config.to_dict()
                else:
                    dict_config = self.config

                peft_config = _prepare_prompt_learning_config(peft_config, dict_config)
                self._setup_prompt_encoder(adapter_name)
            elif peft_config.is_adaption_prompt:
                self.base_model.add_adapter(adapter_name, peft_config)
            else:
                self.peft_config[adapter_name] = peft_config
                self.base_model.inject_adapter(self.base_model.model, adapter_name)
        except Exception:  # something went wrong, roll back
            if adapter_name in self.peft_config:
                del self.peft_config[adapter_name]
            raise

        self.set_additional_trainable_modules(peft_config, adapter_name)

    def set_additional_trainable_modules(self, peft_config, adapter_name):
        if getattr(peft_config, "modules_to_save", None) is not None:
            if self.modules_to_save is None:
                self.modules_to_save = set(peft_config.modules_to_save)
            else:
                self.modules_to_save.update(peft_config.modules_to_save)
            _set_trainable(self, adapter_name)  # this may add a new ModulesToSaveWrapper

    def get_layer_status(self) -> list[TunerLayerStatus]:
        """Get the status of each adapter layer in the model.

        This method returns a list of `TunerLayerStatus` dataclass instances, each of which contains the following
        attributes:

        - `name` (`str`):
           The name of the adapter layer, e.g. `model.encoder.block.0.layer.0.SelfAttention.q`.
        - `module_type` (`str`):
           The type of the adapter layer, e.g. `lora.Linear`.
        - `enabled` (`bool`):
           Whether the adapter layer is enabled.
        - `active_adapters` (`list[str]`):
           The names of the active adapters, if any, e.g. `["default"]`.
        - `merged_adapters` (`list[str]`):
           The names of the merged adapters, if any, e.g. `["default"]`.
        - `available_adapters` (`list[str]`):
           The names of the available adapters, e.g. `["default"]`.

        Args:
            model ([`~PeftModel`]):
                The model to get the adapter layer status from.

        Returns:
            list[`peft.peft_model.TunerLayerStatus`]:
                A list of dataclasses, each containing the status of the corresponding adapter layer.

        """
        return get_layer_status(self)

    def get_model_status(self) -> TunerModelStatus:
        """Get the status of tuners of the model.

        This method returns a `TunerModelStatus` dataclass instance, which contains the following attributes:

        - `base_model_type` (`str`):
           The type of the base model, e.g. `T5Model`.
        - `adapter_model_type` (`str`):
           The type of the adapter model, e.g. `LoraModel`.
        - `peft_types` (`dict[str, str]`):
           The mapping of adapter name to adapter type, e.g. `{"default": "LORA"}`.
        - `trainable_params` (`int`):
           The number of trainable parameters in the model.
        - `total_params` (`int`):
           The total number of parameters in the model.
        - `num_adapter_layers` (`int`):
           The number of adapter layers in the model.
        - `enabled` (`bool`, `Literal["irregular"]`):
           Whether all adapter layers are enabled. If some are enabled and some are not, this will be `"irregular"`.
           This means that your model is in an inconsistent state and might not work as expected.
        - `active_adapters` (`list[str]`, `Literal["irregular"]`):
           The names of the active adapters. If the active adapters are not consistent across all layers, this will be
           `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
        - `merged_adapters` (`list[str]`, `Literal["irregular"]`):
           The names of the merged adapters. If the merged adapters are not consistent across all layers, this will be
           `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
        - `available_adapters` (`list[str]`):
           The names of the available adapters, e.g. `["default"]`.

        Args:
            model ([`~PeftModel`]):
                The model to get the adapter layer status from.

        Returns:
            `peft.peft_model.TunerModelStatus`:
                A dataclass containing the status of the model.

        """
        return get_model_status(self)

    @classmethod
    def _split_kwargs(cls, kwargs: dict[str, Any]):
        _kwargs_not_in_hf_hub_download_signature = ("use_auth_token",)
        hf_hub_download_kwargs = {}
        other_kwargs = {}

        for key, value in kwargs.items():
            if key in inspect.signature(hf_hub_download).parameters or key in _kwargs_not_in_hf_hub_download_signature:
                hf_hub_download_kwargs[key] = value
            else:
                other_kwargs[key] = value

        return hf_hub_download_kwargs, other_kwargs

    def _update_offload(self, offload_index: dict[str, dict[str, str]], adapters_weights: dict[str, torch.tensor]):
        """
        Update the offload_index and safetensors files for loading and mergine PeftModels with disk-offloaded modules.

        Args:
            offload_index (Dict[str: str]):
                Dictionary of disk-offloaded modules with their metadata and safetensors filenames
            adapters_weights (Dict[str: torch.tensor]):
                Dictionary of Peft adapter module names and weights
        """

        if not offload_index:
            return offload_index

        prefix = "base_model.model."
        # rename offload index weight and model names
        adapter_names = list(self.peft_config.keys())
        for adapter_name in adapter_names:
            keys = list(offload_index.keys())
            block_id = keys[0].split(".")[0] + "."  # for writing safetensors key,

            # replace original offload index keys with PeftModel keys
            for key in keys:
                suffix_pos = key.rfind(".")
                extended_prefix = prefix + key[:suffix_pos]
                module = dict(self.named_modules())[extended_prefix]
                if isinstance(module, BaseTunerLayer):
                    new_key = prefix + key[:suffix_pos] + ".base_layer" + key[suffix_pos:]
                else:
                    new_key = prefix + key
                offload_index[key]["weight_name"] = new_key
                offload_index[new_key] = offload_index[key]
                del offload_index[key]

            files_seen = set()
            # rename safetensors for dispatch
            for new_key in list(offload_index.keys()):
                fname = offload_index[new_key]["safetensors_file"]

                # make a new file name
                new_fname_list = list(fname.split(os.sep))
                for i, name in enumerate(new_fname_list):
                    if "--" in name:
                        new_fname_list[i] += "-peft"
                        break
                new_fname = os.path.join(*new_fname_list)

                if fname in files_seen:
                    continue
                safe_dict = {}
                with safe_open(fname, framework="pt") as f:
                    for safe_key in f.keys():
                        safe_tensor = f.get_tensor(safe_key)
                        metadata = f.metadata()
                        suffix_pos = safe_key.rfind(".")
                        extended_prefix = prefix + block_id + safe_key[:suffix_pos]
                        safe_module = dict(self.named_modules())[extended_prefix]
                        if isinstance(safe_module, BaseTunerLayer):
                            final_key = extended_prefix + ".base_layer" + safe_key[suffix_pos:]
                            lora_dict = {key: val for key, val in adapters_weights.items() if extended_prefix in key}

                            # add LoRA keys and values to disk offload
                            for lora_key, lora_val in lora_dict.items():
                                divide = lora_key.rfind(".")
                                new_key = lora_key[:divide] + f".{adapter_name}" + lora_key[divide:]
                                safe_dict[new_key] = lora_val
                        else:
                            final_key = prefix + block_id + safe_key
                        safe_dict[final_key] = safe_tensor
                    files_seen.add(new_fname)

                    # avoid overwriting original safetensors
                    for key in safe_dict.keys():
                        offload_index[key] = {"safetensors_file": new_fname, "weight_name": key}

                    base_name = os.path.dirname(new_fname)
                    if not os.path.exists(base_name):
                        os.makedirs(base_name)
                    safe_save_file(safe_dict, new_fname, metadata=metadata)

    def load_adapter(
        self,
        model_id: str,
        adapter_name: str,
        is_trainable: bool = False,
        torch_device: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        Load a trained adapter into the model.

        The name for the new adapter should be unique.

        The new adapter is not automatically set as the active adapter. Use [`PeftModel.set_adapter`] to set the active
        adapter.

        Args:
            adapter_name (`str`):
                The name of the adapter to be added.
            peft_config ([`PeftConfig`]):
                The configuration of the adapter to be added.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            torch_device (`str`, *optional*, defaults to None):
                The device to load the adapter on. If `None`, the device will be inferred.
            kwargs: (`optional`):
                Additional arguments to modify the way the adapter is loaded, e.g. the token for Hugging Face Hub.
        """

        hf_hub_download_kwargs, kwargs = self._split_kwargs(kwargs)
        if torch_device is None:
            torch_device = infer_device()

        if adapter_name not in self.peft_config:
            # load the config
            peft_config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig._get_peft_type(
                    model_id,
                    **hf_hub_download_kwargs,
                )
            ].from_pretrained(
                model_id,
                **hf_hub_download_kwargs,
            )
            if peft_config.is_prompt_learning and is_trainable:
                raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
            else:
                peft_config.inference_mode = not is_trainable
            self.add_adapter(adapter_name, peft_config)

        adapters_weights = load_peft_weights(model_id, device=torch_device, **hf_hub_download_kwargs)

        # load the weights into the model
        ignore_mismatched_sizes = kwargs.get("ignore_mismatched_sizes", False)
        load_result = set_peft_model_state_dict(
            self, adapters_weights, adapter_name=adapter_name, ignore_mismatched_sizes=ignore_mismatched_sizes
        )
        if (
            (getattr(self, "hf_device_map", None) is not None)
            and (len(set(self.hf_device_map.values()).intersection({"cpu", "disk"})) > 0)
            and len(self.peft_config) == 1
        ):
            device_map = kwargs.get("device_map", "auto")
            max_memory = kwargs.get("max_memory", None)
            offload_dir = kwargs.get("offload_folder", None)
            offload_index = kwargs.get("offload_index", None)

            dispatch_model_kwargs = {}
            # Safety checker for previous `accelerate` versions
            # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
            if "offload_index" in inspect.signature(dispatch_model).parameters:
                dispatch_model_kwargs["offload_index"] = offload_index

            no_split_module_classes = self._no_split_modules

            if device_map != "sequential":
                max_memory = get_balanced_memory(
                    self,
                    max_memory=max_memory,
                    no_split_module_classes=no_split_module_classes,
                    low_zero=(device_map == "balanced_low_0"),
                )

            if isinstance(device_map, str):
                device_map = infer_auto_device_map(
                    self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
                )

            self._update_offload(offload_index, adapters_weights)
            dispatch_model_kwargs["offload_index"] = offload_index

            dispatch_model(
                self,
                device_map=device_map,
                offload_dir=offload_dir,
                **dispatch_model_kwargs,
            )

            hook = AlignDevicesHook(io_same_device=True)
            if self.peft_config[adapter_name].is_prompt_learning:
                remove_hook_from_submodules(self.prompt_encoder)
            add_hook_to_module(self.get_base_model(), hook)

        # Set model in evaluation mode to deactivate Dropout modules by default
        if not is_trainable:
            self.eval()
        return load_result

    def set_adapter(self, adapter_name: str) -> None:
        """
        Sets the active adapter.

        Only one adapter can be active at a time.

        Additionally, this function will set the specified adapter to trainable (i.e., requires_grad=True). If this is
        not desired, use the following code.

        ```py
        >>> for name, param in model_peft.named_parameters():
        ...     if ...:  # some check on name (ex. if 'lora' in name)
        ...         param.requires_grad = False
        ```

        Args:
            adapter_name (`str`):
                The name of the adapter to be set as active. The adapter must be loaded first.
        """
        if adapter_name not in self.peft_config:
            raise ValueError(f"Adapter {adapter_name} not found.")
        self.active_adapter = adapter_name
        if not self.peft_config[adapter_name].is_prompt_learning:
            self.base_model.set_adapter(adapter_name)
        _set_adapter(self, adapter_name)

    @property
    def base_model_torch_dtype(self):
        return getattr(self.base_model, "dtype", None)

    @property
    def active_peft_config(self):
        return self.peft_config[self.active_adapter]

    def create_or_update_model_card(self, output_dir: str):
        """
        Updates or create model card to include information about peft:
        1. Adds `peft` library tag
        2. Adds peft version
        3. Adds base model info
        4. Adds quantization information if it was used
        """

        filename = os.path.join(output_dir, "README.md")

        card = ModelCard.load(filename) if os.path.exists(filename) else ModelCard.from_template(ModelCardData())

        card.data["library_name"] = "peft"

        model_config = getattr(self, "config", None)
        if hasattr(model_config, "to_dict"):
            model_config = model_config.to_dict()
        if model_config is not None and "_name_or_path" in model_config:
            card.data["base_model"] = model_config["_name_or_path"]

        lines = card.text.splitlines()

        quantization_config = None
        if hasattr(model_config, "quantization_config"):
            quantization_config = self.config.quantization_config.to_dict()
        training_config_text = ""
        quantization_prefix = "The following `bitsandbytes` quantization config was used during training:"
        # Adds quantization information if it was used
        if quantization_config is not None:
            training_config_text += f"\n{quantization_prefix}\n"
            training_config_text += "\n".join([f"- {name}: {value}" for name, value in quantization_config.items()])
            training_config_text += "\n"

        training_procedure_heading = "## Training procedure"
        if quantization_prefix not in lines and bool(training_config_text):
            if training_procedure_heading in lines:
                lines.insert(lines.index(training_procedure_heading) + 2, training_config_text)
            else:
                lines.append(f"{training_procedure_heading}\n{training_config_text}")

        # Adds peft version
        framework_block_heading = "### Framework versions"
        if f"- PEFT {__version__}" not in lines:
            if framework_block_heading in lines:
                lines.insert(lines.index(framework_block_heading) + 2, f"- PEFT {__version__}")
            else:
                lines.append(f"{framework_block_heading}\n\n- PEFT {__version__}")

        card.text = "\n".join(lines)
        card.save(filename)

@dataclass
class TunerLayerStatus:
    name: str
    module_type: str
    enabled: bool
    active_adapters: list[str]
    merged_adapters: list[str]
    requires_grad: dict[str, bool | Literal["irregular"]]
    available_adapters: list[str]


def get_layer_status(model: torch.nn.Module) -> list[TunerLayerStatus]:
    """Get the status of each adapter layer in the model.

    This function returns a list of `TunerLayerStatus` dataclass instances, each of which contains the following
    attributes:

    - `name` (`str`):
       The name of the adapter layer, e.g. `model.encoder.block.0.layer.0.SelfAttention.q`.
    - `module_type` (`str`):
       The type of the adapter layer, e.g. `lora.Linear`.
    - `enabled` (`bool`):
       Whether the adapter layer is enabled.
    - `active_adapters` (`list[str]`):
       The names of the active adapters, if any, e.g. `["default"]`.
    - `merged_adapters` (`list[str]`):
       The names of the merged adapters, if any, e.g. `["default"]`.
    - requires_grad : dict[str, bool | Literal["irregular"]]
       The requires_grad status of the parameters for each adapter module. Ideally, it should be either `True` or
       `False`. If the requires_grad status is not consistent across all parameters, the value will be set to
       `"irregular"`.
    - `available_adapters` (`list[str]`):
       The names of the available adapters, e.g. `["default"]`.

    Args:
        model ([Union[`~PeftModel`, `~transformers.PreTrainedModel`, `nn.Module`]]):
            The model to get the adapter layer status from.

    Returns:
        list[`peft.peft_model.TunerLayerStatus`]:
            A list of dataclasses, each containing the status of the corresponding adapter layer.

    """
    if isinstance(model, PeftModel):
        base_model = model.base_model
        if not isinstance(base_model, BaseTuner):
            raise TypeError(
                "get_layer_status() got an invalid PeftModel instance; prefix tuning and adaption prompt are not "
                "supported."
            )
    else:
        base_model = model

    layer_status: list[TunerLayerStatus] = []
    for name, module in base_model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue

        # determine if all submodules/parameters if this module require grad or not
        mapping_requires_grad_list: dict[str, list[bool]] = collections.defaultdict(list)
        for adapter_module_name in module.adapter_layer_names:
            adapter_module = getattr(module, adapter_module_name)
            if isinstance(adapter_module, torch.nn.ModuleDict):
                for key, submodule in adapter_module.items():
                    for param in submodule.parameters():
                        mapping_requires_grad_list[key].append(param.requires_grad)
            elif isinstance(adapter_module, torch.nn.ParameterDict):
                for key, param in adapter_module.items():
                    mapping_requires_grad_list[key].append(param.requires_grad)
            else:
                # strange, we don't know how to handle this, ignore for now
                pass

        def check_irrgular(vals: list[bool]) -> bool | Literal["irregular"]:
            if all(vals):
                return True
            if not any(vals):
                return False
            return "irregular"

        requires_grad = {key: check_irrgular(vals) for key, vals in mapping_requires_grad_list.items()}

        status = TunerLayerStatus(
            name=name,
            module_type=repr(module).partition("(")[0],
            enabled=not module.disable_adapters,
            active_adapters=module.active_adapters,
            merged_adapters=module.merged_adapters,
            requires_grad=requires_grad,
            available_adapters=sorted(module._get_available_adapters()),
        )
        layer_status.append(status)

    if not layer_status:
        raise ValueError(
            "No adapter layers found in the model, please ensure that it's a PEFT model or that you have PEFT adapters "
            "injected in the model."
        )

    return layer_status


@dataclass
class TunerModelStatus:
    base_model_type: str
    adapter_model_type: str
    peft_types: dict[str, str]
    trainable_params: int
    total_params: int
    num_adapter_layers: int
    enabled: bool | Literal["irregular"]
    active_adapters: list[str] | Literal["irregular"]
    merged_adapters: list[str] | Literal["irregular"]
    requires_grad: dict[str, bool | Literal["irregular"]]
    available_adapters: list[str]


def get_model_status(model: torch.nn.Module) -> TunerModelStatus:
    """Get the status of tuners of the model.

    This function returns a `TunerModelStatus` dataclass instance, which contains the following attributes:

    - `base_model_type` (`str`):
       The type of the base model, e.g. `T5Model`.
    - `adapter_model_type` (`str`):
       The type of the adapter model, e.g. `LoraModel`.
    - `peft_types` (`dict[str, str]`):
       The mapping of adapter name to adapter type, e.g. `{"default": "LORA"}`.
    - `trainable_params` (`int`):
       The number of trainable parameters in the model.
    - `total_params` (`int`):
       The total number of parameters in the model.
    - `num_adapter_layers` (`int`):
       The number of adapter layers in the model.
    - `enabled` (`bool`, `Literal["irregular"]`):
       Whether all adapter layers are enabled. If some are enabled and some are not, this will be `"irregular"`. This
       means that your model is in an inconsistent state and might not work as expected.
    - `active_adapters` (`list[str]`, `Literal["irregular"]`):
       The names of the active adapters. If the active adapters are not consistent across all layers, this will be
       `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
    - `merged_adapters` (`list[str]`, `Literal["irregular"]`):
       The names of the merged adapters. If the merged adapters are not consistent across all layers, this will be
       `"irregular"`, which means that your model is in an inconsistent state and might not work as expected.
    - `requires_grad` (`dict[str, bool | Literal["irregular"]]`):
       Whether for the given adapter, all adapter layers have `requires_grad` set to `True` or `False`. If there is a
       mix, this will be set to `"irregular"`, which means that your model is in an inconsistent state and might not
       work as expected.
    - `available_adapters` (`list[str]`):
       The names of the available adapters, e.g. `["default"]`.

    Args:
        model ([Union[`~PeftModel`, `~transformers.PreTrainedModel`, `nn.Module`]]):
            The model to get the adapter layer status from.

    Returns:
        `peft.peft_model.TunerModelStatus`:
            A dataclass containing the status of the model.

    """
    if isinstance(model, PeftModel):
        if not isinstance(model.base_model, BaseTuner):
            raise TypeError(
                "get_model_status() got an invalid PeftModel instance; prefix tuning and adaption prompt are not "
                "supported."
            )
        base_model_type = model.get_base_model().__class__.__name__
        trainable_params, total_params = model.get_nb_trainable_parameters()
        base_model = model.base_model
        peft_types = {key: str(config.peft_type).partition(".")[-1] for key, config in base_model.peft_config.items()}
        adapter_model_type = base_model.__class__.__name__
    elif isinstance(model, PreTrainedModel):
        base_model_type = model.__class__.__name__
        trainable_params, total_params = PeftModel.get_nb_trainable_parameters(model)
        base_model = model
        peft_types = {}
        adapter_model_type = "None"
    else:
        base_model_type = "other"
        trainable_params, total_params = PeftModel.get_nb_trainable_parameters(model)
        base_model = model
        peft_types = {}
        adapter_model_type = "None"

    layer_status = get_layer_status(model)
    num_adapter_layers = len(layer_status)

    enabled_set: set[bool] = {status.enabled for status in layer_status}  # must be {True}, {False}, or {True, False}
    enabled: bool | Literal["irregular"]
    if len(enabled_set) == 1:
        enabled = enabled_set.pop()
    else:
        enabled = "irregular"

    available_adapters: list[str] = sorted(set().union(*(status.available_adapters for status in layer_status)))

    # ideally, active adapters should be consistent across all layers of the model, but we cannot guarantee it
    all_active_adapters: set[tuple[str, ...]] = {tuple(status.active_adapters) for status in layer_status}
    active_adapters: list[str] | Literal["irregular"]
    if not all_active_adapters:
        active_adapters = []
    elif len(all_active_adapters) == 1:
        active_adapters = list(all_active_adapters.pop())
    else:
        active_adapters = "irregular"

    # Here we determine what adapters are merged. This is not trivial because multiple adapters can be merged or not at
    # the same time. Some layers may only have adapter A, some only adapter B, so it's not as easy as just checking
    # which adapters are merged on each layer.

    # First, determine all adapters that are merged on at least on module.
    merged_all: set[str] = set()
    for status in layer_status:
        merged_all.update(status.merged_adapters)

    # Next, check if on any layer, on of these adapters is not merged.
    merged_adapters: list[str] | Literal["irregular"] = sorted(merged_all)
    for status in layer_status:
        unmerged = set(status.available_adapters) - set(status.merged_adapters)
        if unmerged & merged_all:
            # there is overlap between unmerged adapters and adapters that should be merged
            merged_adapters = "irregular"
            break

    # check status of requires_grad
    # first, merge the values for all layers
    requires_grad_all: dict[str, list[bool | Literal["irregular"]]] = collections.defaultdict(list)
    for status in layer_status:
        for key, val in status.requires_grad.items():
            requires_grad_all[key].append(val)

    # then, check if the values are consistent
    def check_irrgular(vals: list[bool | Literal["irregular"]]) -> bool | Literal["irregular"]:
        if all(val is True for val in vals):
            return True
        if all(val is False for val in vals):
            return False
        return "irregular"

    requires_grad = {key: check_irrgular(vals) for key, vals in requires_grad_all.items()}

    adapter_model_status = TunerModelStatus(
        base_model_type=base_model_type,
        adapter_model_type=adapter_model_type,
        peft_types=peft_types,
        trainable_params=trainable_params,
        total_params=total_params,
        num_adapter_layers=num_adapter_layers,
        enabled=enabled,
        active_adapters=active_adapters,
        merged_adapters=merged_adapters,
        requires_grad=requires_grad,
        available_adapters=available_adapters,
    )
    return adapter_model_status
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# Will add BatchLora as its basemodel rest intact.
class BatchPeftModel(PeftModel):
    def __init__(self, model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = "default") -> None:
        super().__init__(model, peft_config, adapter_name)
        self.base_model_prepare_inputs_for_generation = self.base_model.prepare_inputs_for_generation

    ### Forward, Generate and Prepare inputs for generation straight up copied from PeftModelForCausalLM class
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_ids=None,
        **kwargs,
    ):
        peft_config = self.active_peft_config
        if not peft_config.is_prompt_learning:
            if self.base_model.config.model_type == "mpt":
                if inputs_embeds is not None:
                    raise AssertionError("forward in MPTForCausalLM does not support inputs_embeds")
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

            if peft_config.peft_type == PeftType.POLY:
                kwargs["task_ids"] = task_ids

            with self._enable_peft_forward_hooks(**kwargs):
                kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                return self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    **kwargs,
                )

        batch_size = _get_batch_size(input_ids, inputs_embeds)
        if attention_mask is not None:
            # concat prompt attention mask
            prefix_attention_mask = torch.ones(batch_size, peft_config.num_virtual_tokens).to(attention_mask.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        if kwargs.get("position_ids", None) is not None:
            warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
            kwargs["position_ids"] = None
        if kwargs.get("token_type_ids", None) is not None:
            warnings.warn("Token type ids are not supported for parameter efficient tuning. Ignoring token type ids")
            kwargs["token_type_ids"] = None
        kwargs.update(
            {
                "attention_mask": attention_mask,
                "labels": labels,
                "output_attentions": output_attentions,
                "output_hidden_states": output_hidden_states,
                "return_dict": return_dict,
            }
        )

        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            past_key_values = self.get_prompt(batch_size)
            return self.base_model(
                input_ids=input_ids, inputs_embeds=inputs_embeds, past_key_values=past_key_values, **kwargs
            )
        else:
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            # concat prompt labels
            if labels is not None:
                prefix_labels = torch.full((batch_size, peft_config.num_virtual_tokens), -100).to(labels.device)
                kwargs["labels"] = torch.cat((prefix_labels, labels), dim=1)
            prompts = self.get_prompt(batch_size=batch_size, task_ids=task_ids)
            prompts = prompts.to(inputs_embeds.dtype)
            inputs_embeds = torch.cat((prompts, inputs_embeds), dim=1)
            return self.base_model(inputs_embeds=inputs_embeds, **kwargs)

    def generate(self, *args, **kwargs):
        peft_config = self.active_peft_config
        self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation
        if hasattr(self.base_model, "model"):
            self.base_model.model.generation_config = self.generation_config
        else:
            self.base_model.generation_config = self.generation_config
        try:
            if not peft_config.is_prompt_learning:
                with self._enable_peft_forward_hooks(*args, **kwargs):
                    kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
                    outputs = self.base_model.generate(*args, **kwargs)
            else:
                outputs = self.base_model.generate(**kwargs)
        except:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            raise
        else:
            self.base_model.prepare_inputs_for_generation = self.base_model_prepare_inputs_for_generation
            return outputs

    def prepare_inputs_for_generation(self, *args, task_ids: Optional[torch.Tensor] = None, **kwargs):
        peft_config = self.active_peft_config
        model_kwargs = self.base_model_prepare_inputs_for_generation(*args, **kwargs)

        # https://github.com/huggingface/transformers/pull/26681/ introduced new cache format
        # for some architectures which requires a special fix for prompt tuning etc.
        # TODO: starting with transformers 4.38, all architectures should support caching.
        uses_transformers_4_38 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.38.0")
        uses_transformers_4_36 = packaging.version.parse(transformers.__version__) >= packaging.version.parse("4.36.0")
        transformers_new_cache_archs = ["llama", "mistral", "persimmon", "phi"]
        uses_cache = uses_transformers_4_38 or (
            uses_transformers_4_36 and self.base_model.config.model_type in transformers_new_cache_archs
        )

        if peft_config.peft_type == PeftType.POLY:
            model_kwargs["task_ids"] = task_ids
        if peft_config.is_prompt_learning:
            if uses_cache and (model_kwargs["past_key_values"] is not None):
                # change in the logic of `prepare_inputs_for_generation` makes the below code necessary
                # In prompt learning methods, past key values are longer when compared to the `input_ids`.
                # As such only consider the last input ids in the autogressive generation phase.
                if model_kwargs["past_key_values"][0][0].shape[-2] >= model_kwargs["input_ids"].shape[1]:
                    model_kwargs["input_ids"] = model_kwargs["input_ids"][:, -1:]

            if model_kwargs.get("attention_mask", None) is not None:
                size = model_kwargs["input_ids"].shape[0], peft_config.num_virtual_tokens
                prefix_attention_mask = torch.ones(size).to(model_kwargs["input_ids"].device)
                model_kwargs["attention_mask"] = torch.cat(
                    (prefix_attention_mask, model_kwargs["attention_mask"]), dim=1
                )

            if model_kwargs.get("position_ids", None) is not None:
                warnings.warn("Position ids are not supported for parameter efficient tuning. Ignoring position ids.")
                model_kwargs["position_ids"] = None

            if kwargs.get("token_type_ids", None) is not None:
                warnings.warn(
                    "Token type ids are not supported for parameter efficient tuning. Ignoring token type ids"
                )
                kwargs["token_type_ids"] = None

            if model_kwargs["past_key_values"] is None and peft_config.peft_type == PeftType.PREFIX_TUNING:
                past_key_values = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0])
                model_kwargs["past_key_values"] = past_key_values
            else:
                if model_kwargs["past_key_values"] is None:
                    inputs_embeds = self.word_embeddings(model_kwargs["input_ids"])
                    prompts = self.get_prompt(batch_size=model_kwargs["input_ids"].shape[0], task_ids=task_ids)
                    prompts = prompts.to(inputs_embeds.dtype)
                    model_kwargs["inputs_embeds"] = torch.cat((prompts, inputs_embeds), dim=1)
                    model_kwargs["input_ids"] = None

        # For transformers>=4.38.0 - for some architectures such as Llama, `cache_position` is
        # passed in the forward pass to keep track of the position ids of the cache. We have to
        # pop that from `model_kwargs` as `cache_position` is properly created by the model, using the passed
        # `inputs_embeds`: https://github.com/huggingface/transformers/blob/593230f0a1150ea9c0477b9d859f25daf73c8c33/src/transformers/models/llama/modeling_llama.py#L956
        _ = model_kwargs.pop("cache_position", None)

        return model_kwargs



