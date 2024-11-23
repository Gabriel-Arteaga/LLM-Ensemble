# Copyright 2023 The HuggingFace Team. All rights reserved.
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
#
# Modifications:
# This file has been adapted and modified from its original version by Gabriel Y. Arteaga.
# Changes include adjustments to account for the BatchEnsemble architecture.
# In addition, code includes my own implementation of the Anchored Regularization first introduced by Pearce et al. (2021).
from trl import SFTTrainer
import os
from transformers.utils import logging
from transformers import PreTrainedModel
from .peft_model import BatchPeftModel
import torch
import safetensors
from typing import Optional
from peft.utils.constants import WEIGHTS_NAME
from transformers.dynamic_module_utils import custom_object_save
from transformers.utils import SAFE_WEIGHTS_NAME
from einops import einsum
from torch.autograd import Variable
import math
from transformers import is_safetensors_available
import json 
from packaging import version 
from safetensors.torch import load_file as safe_load_file
from functools import partial
import gc 

parsed_torch_version_base = version.parse(version.parse(torch.__version__).base_version)
is_torch_greater_or_equal_than_1_13 = parsed_torch_version_base >= version.parse("1.13")
WEIGHTS_INDEX_NAME = "pytorch_model.bin.index.json"
SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"
TRAINING_ARGS_NAME = "training_args.bin"
logger = logging.get_logger(__name__)


def compute_anchored_gradients(layer, anchors, N, device, training: bool):
    """
    Compute the anchored gradients for a given layer.
    We will update the gradients of B and A matrices of LoRA, the fast weights r and s.

    Parameters:
        layer (BatchLoraLayer): The layer for which anchored gradients are computed.
        anchors (torch.Tensor): Anchors for each ensemble member.
        N (int): Number of samples in the dataset.
        device (torch.device)

    Returns:
        None
    """
    if training:
        # Intiate the regularization term
        reg_term = Variable(torch.empty(1), requires_grad=True).to(device)
    else:
        reg_term = 0
    
    # Draw anchors for each ensemble member, we use a distinct seed for each ensemble member
    for i in range(layer.base_layer.ensemble_size):
        # Set distinct random seed for ensemble member i
        torch.manual_seed(layer.base_layer.seeds[i])
        # Draw anchors for the ith member
        anchors[i] = anchors[i].normal_(layer.base_layer.mean_prior, layer.base_layer.std_prior)
    # Retrieve the LoRA matrix (BA)
    delta = layer.get_delta_weight(layer._active_adapter[0])

    # Compute the tau term from Anchored Ensembling Equations
    # Should be 1/(2*variance)
    tau = math.sqrt((1 / (2 * layer.base_layer.std_prior**2)))
    # L is the amount layers in our network, probably nicer to have it as an attribute in the Transformer, in case we choose to only train some layers, keeping it here meanwhile
    L = 226 
    # The parameter dimensions of our weight matrix
    n_in, n_out = layer.base_layer.weight.shape[0], layer.base_layer.weight.shape[1]
    # We introduce this new scalar alpha as large networks makes the regularization term to explode otherwise
    alpha = math.sqrt(1/ (n_in*n_out))

    # Compute the regularization term according to Equation (9) of Anchored ensemble
    # (with the BatchEnsemble and LoRA modification)
    reg_term = (1 / (N+L)) * (alpha*tau * (einsum(
        (layer.base_layer.weight + delta),  # We add the LoRA matrix with our pretrained weights
        (layer.base_layer.r),
        (layer.base_layer.s),
        'I O, E I, E O -> E I O'
    ) - anchors)).pow(2).sum()
    
    if training:
        # Compute the gradients for this layer's weight matrices
        reg_term.backward()
        reg_term = reg_term.item()
    else:
        reg_term = reg_term.item()

    # Return the reg_term scalar to keep track of regularization and its impact on the loss
    return reg_term


def compute_anchored_regularization(model, N: int, training: bool):
    """
    Compute the anchored regularization for the entire model.

    Parameters:
        model (BatchPeftModel): The model for which anchored regularization is computed.
        N (int): Number of samples in the dataset.

    Returns:
        None
    """
    # Unwrap the model so we have an easier time accessing its modules
    tmp_model = model.base_model.model
    
    # Initialize the reg_term, will be a scalar to keep track of regularzation impact on loss
    reg_term = 0
    # We start with computing the embedding layer
    # First lets initiate the anchors
    anchors = torch.empty((tmp_model.embed_tokens.base_layer.ensemble_size,
                           tmp_model.embed_tokens.base_layer.num_embeddings,
                           tmp_model.embed_tokens.base_layer.embedding_dim), dtype=model.dtype).to(model.device)
    reg_term += compute_anchored_gradients(tmp_model.embed_tokens, anchors, N, model.device, training)
    
    # We iterate through each decoder layer, where each decoder layer contains an self_attn and MLP layer
    for layer in tmp_model.layers:
        # We start with the MLP layer of the decoder layer
        for name, batch_linear_layer in layer.MLP.named_children():
            # w1 and w2 have the same dimensions so we initiate the same shape anchors for these
            if name == 'w1':
                anchors = torch.empty((batch_linear_layer.base_layer.ensemble_size,
                                      batch_linear_layer.base_layer.in_features,
                                      batch_linear_layer.base_layer.out_features), dtype=model.dtype).to(model.device)
            
            # w3 has different shaped so we need to reinitiate the anchors matrix
            elif name == 'w3':
                anchors = torch.empty((batch_linear_layer.base_layer.ensemble_size,
                                      batch_linear_layer.base_layer.in_features,
                                      batch_linear_layer.base_layer.out_features), dtype=model.dtype).to(model.device)
            # We only compute the regularization term when its one of the weight matrices
            # (we will also loop the activation function, don't want to compute an extra iteration by mistake)
            if name in ['w1', 'w2', 'w3']:
                # Compute the regularization for each MLP weight and accumlate
                reg_term += compute_anchored_gradients(batch_linear_layer, anchors, N, model.device, training)
        
        # We proceed with the self attention layer
        for name, batch_linear_layer in layer.self_attn.named_children():
            # weight wq have unique dimensions, initiate anchors
            if name == 'wq':
                anchors = torch.empty((batch_linear_layer.base_layer.ensemble_size,
                                      batch_linear_layer.base_layer.in_features,
                                      batch_linear_layer.base_layer.out_features), dtype=model.dtype).to(model.device)
                
            # Weights wk and wv have the same dimensions, only need to initalize anchors once for them
            if name == 'wk':
                anchors = torch.empty((batch_linear_layer.base_layer.ensemble_size,
                                      batch_linear_layer.base_layer.in_features,
                                      batch_linear_layer.base_layer.out_features), dtype=model.dtype).to(model.device)
            
            # Weight wo have a unique dimension, initialize anchors for it
            if name == 'wo':
                anchors = torch.empty((batch_linear_layer.base_layer.ensemble_size,
                                      batch_linear_layer.base_layer.in_features,
                                      batch_linear_layer.base_layer.out_features), dtype=model.dtype).to(model.device)
                
            # Rotary embeddings is also a "child" of self attention, we only want to compute reg_term
            # for wq, wk, wv & wo
            if name in ['wq', 'wk', 'wv', 'wo']:
                reg_term += compute_anchored_gradients(batch_linear_layer, anchors, N, model.device, training)
        
    # We finish by computing the reg term for the lm head
    # Initiate anchors
    anchors = torch.empty((tmp_model.lm_head.base_layer.ensemble_size,
                           tmp_model.lm_head.base_layer.in_features,
                           tmp_model.lm_head.base_layer.out_features), dtype=model.dtype).to(model.device)
    
    reg_term += compute_anchored_gradients(tmp_model.lm_head, anchors, N, model.device, training)
    
    return reg_term


class BatchSFTTrainer(SFTTrainer):
    # We need to add BatchLoRA as a supported class so that we can properly save our model
    # This funciton is inherited from the Trainer class
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")


        supported_classes = (PreTrainedModel, BatchPeftModel)
        custom_object_save(self.model.base_model.model, output_dir, config=self.model.base_model.model.config)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        self.model.save_pretrained(
            output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

class AnchoredSFTTrainer(BatchSFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # There's no reason why we can't compute the regularization before doing our forward pass, we do this to have more memory available for the gradient updates
        # We only update the gradients during training
        #if model.training == True:
        # Save the seed state before calculating the regularization term
        initial_seed_state = torch.random.get_rng_state()

        # Update the gradients based on regularization from Equation 9 Pearce et al. (Anchored Ensemble)
        reg_term = compute_anchored_regularization(model=model, N=inputs['input_ids'].shape[0], training= model.training) # Note, we don't add the reg term to the loss, instead we comput the gradients directly
        # This saves a lot of memory
        

        # Reset to the initial seed state
        torch.random.set_rng_state(initial_seed_state)

        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            # Removed every namecheck as we only compute loss with this function
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        
        # Reg term is simply a scalar and won't affect the backprop of the loss, however, it will ensure that when we save our model it is 
        # in consideration of the entire loss, not simply the NLL, which would've been the case o.w. disregarding the regularzation
        if model.training == True:
            # We want to log the NLL and regularization term's impact seperately
            self.log({"NLL loss":loss.item(), "Anchored Regularization": reg_term})
        else:
            self.log({"Eval_NLL loss":loss.item(), "Eval_Anchored Regularization": reg_term})
        # We only perform regularization during training, during evaluation we only evaluate NLL loss
        loss += reg_term
            
        return (loss, outputs) if return_outputs else loss


############# UTILITY ################################
def load_sharded_checkpoint(model, folder, strict=True, prefer_safe=True):
    """
    This is a modified function from the trainer class file in Hugginface.

    Args:
        model (`torch.nn.Module`): The model in which to load the checkpoint.
        folder (`str` or `os.PathLike`): A path to a folder containing the sharded checkpoint.
        strict (`bool`, *optional`, defaults to `True`):
            Whether to strictly enforce that the keys in the model state dict match the keys in the sharded checkpoint.
        prefer_safe (`bool`, *optional*, defaults to `False`)
            If both safetensors and PyTorch save files are present in checkpoint and `prefer_safe` is True, the
            safetensors files will be loaded. Otherwise, PyTorch files are always loaded when possible.

    Returns:
        `NamedTuple`: A named tuple with `missing_keys` and `unexpected_keys` fields
            - `missing_keys` is a list of str containing the missing keys
            - `unexpected_keys` is a list of str containing the unexpected keys
    """
    # Load the index
    index_file = os.path.join(folder, WEIGHTS_INDEX_NAME)
    safe_index_file = os.path.join(folder, SAFE_WEIGHTS_INDEX_NAME)

    index_present = os.path.isfile(index_file)
    safe_index_present = os.path.isfile(safe_index_file)

    if not index_present and not (safe_index_present and is_safetensors_available()):
        filenames = (
            (WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_INDEX_NAME) if is_safetensors_available() else (WEIGHTS_INDEX_NAME,)
        )
        raise ValueError(f"Can't find a checkpoint index ({' or '.join(filenames)}) in {folder}.")

    load_safe = False
    if safe_index_present:
        if prefer_safe:
            if is_safetensors_available():
                load_safe = True  # load safe due to preference
            else:
                logger.warning(
                    f"Cannot load sharded checkpoint at {folder} safely since safetensors is not installed!"
                )
        elif not index_present:
            load_safe = True  # load safe since we have no other choice

    load_index = safe_index_file if load_safe else index_file

    with open(load_index, "r", encoding="utf-8") as f:
        index = json.load(f)

    shard_files = list(set(index["weight_map"].values()))

    # If strict=True, error before loading any of the state dicts.
    loaded_keys = index["weight_map"].keys()
    model_keys = model.state_dict().keys()
    missing_keys = [key for key in model_keys if key not in loaded_keys]
    unexpected_keys = [key for key in loaded_keys if key not in model_keys]
    if strict and (len(missing_keys) > 0 or len(unexpected_keys) > 0):
        error_message = f"Error(s) in loading state_dict for {model.__class__.__name__}"
        if len(missing_keys) > 0:
            str_missing_keys = ",".join([f'"{k}"' for k in missing_keys])
            error_message += f"\nMissing key(s): {str_missing_keys}."
        if len(unexpected_keys) > 0:
            str_unexpected_keys = ",".join([f'"{k}"' for k in unexpected_keys])
            error_message += f"\nMissing key(s): {str_unexpected_keys}."
        raise RuntimeError(error_message)

    weights_only_kwarg = {"weights_only": True} if is_torch_greater_or_equal_than_1_13 else {}
    loader = safe_load_file if load_safe else partial(torch.load, map_location="cpu", **weights_only_kwarg)

    for shard_file in shard_files:
        state_dict = loader(os.path.join(folder, shard_file))
        model.load_state_dict(state_dict, strict=False)

        # Make sure memory is freed before we load the next state dict.
        del state_dict
        gc.collect()

    # Return the same thing as PyTorch load_state_dict function.
    return torch.nn.modules.module._IncompatibleKeys(missing_keys, unexpected_keys)