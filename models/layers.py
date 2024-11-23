import torch
from torch.nn import functional as F
import math
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Tuple
from transformers.activations import ACT2FN 
from transformers import PreTrainedModel
from utils import RotaryEmbedding, apply_rotary_pos_emb, SampleBasedGenerationMixin, SampleBasedTransformerConfig
from transformers.modeling_outputs import CausalLMOutput
from typing import Optional

_CONFIG_FOR_DOC = "SampleBasedTransformerConfig"

# There were differences in forward pass of RMSNorm, copied mistral's implementation instead
class RMSNorm(nn.Module):
    # Mistral's using eps=1e-5 when loaded in... So we change it to the same to compare
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class Attention(nn.Module):
    def __init__(self,
                 config: SampleBasedTransformerConfig,
                 device = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads # Number of heads for queries
        self.n_kv_heads = config.num_key_value_heads # Number of heads for key and values
        
        # The dimension of each head
        self.head_dim = self.hidden_size//self.n_heads
        
        # Amount of times key and values should be repeated to match the head of the queries
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # SLiding window size for attention mask
        self.window_size = config.sliding_window

        # Precompute the rotary embedding matrix
        self.rotary_embeddings = RotaryEmbedding(config, device=device)
        
        # Weight matrices for Query, Key and Value
        self.wq = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias = False) # Weight matrix for querys 
        self.wk = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias = False) # Weight matrix for keys 
        self.wv = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias = False) # Weight matrix for values
        
        # Weight matrix which we multiply to get the final attention score
        self.wo = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias = False)
    
    
    def _repeat_kv(self,
                   x: torch.Tensor,
                    ) -> torch.Tensor:
        """
        Repeat key and value tensors across a specified number of heads.

        Args:
            x (torch.Tensor): Key tensor with shape (batch_size, seq_len, n_kv_heads, head_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the repeated key tensor 
            and the repeated value tensor, both with shape (batch_size, seq_len, n_kv_heads*n_rep, head_dim).
        """
        if self.n_rep == 1:
            return x
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        return (
            x[:,:,:, None, :]
            .expand(batch_size, seq_len, n_kv_heads, self.n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads*self.n_rep, head_dim)
            )
    


    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor]= None,):
        # (batch_size, seqlen, dim)
        batch_size, seq_len, _ = x.shape
        # Get query, keys and values
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # Reshape the matrices into heads
        xq = xq.view(batch_size, seq_len, self.n_heads, self.head_dim) # Note different amount of heads for q and k,v
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        cos, sin = self.rotary_embeddings(xv, seq_len=seq_len)
        # Apply positional embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, seq_len) # Made a change to the unsqueeze dimension i
        #xq = self.rotary_embeddings(xq,seq_len, x.device) 
        #xk = self.rotary_embeddings(xk,seq_len, x.device)
        

        # Repeat the key and value heads n_rep times
        xk = self._repeat_kv(xk) 
        xv = self._repeat_kv(xv) 
        
        # (batch_size,seq_len, n_heads, head_dim) -> (batch_size, n_heads, seq_len, head_dim)
        xq = xq.transpose(1,2)
        # (batch_size,seq_len, n_kv_heads*n_rep, head_dim) -> (batch_size, n_kv_heads*n_rep, seq_len, head_dim)
        # Note n_kv_heads*n_rep = n_heads
        xk, xv = xk.transpose(1,2), xv.transpose(1,2)

        # Perform forward pass
        # Step 1. Q@K^T/sqrt(head_dim)
        #(batch_size, n_heads, seq_len, head_dim)@ (batch_size, n_heads, head_dim, seq_len)
        # --> (batch_size, n_heads, seq_len, seq_len)
        # Step 2. Add attention mask
        # (batch_size, n_heads, seq_len, seq_len) + (batch_size, 1, seq_len, seq_len)
        # Step 3. 
        # Apply softmax to result from step 2. --> (batch_size, n_heads, seq_len, seq_len) 
        # Step 4. Scores@Values
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, head_dim)
        # (batch_size, n_heads, seq_len, head_dim)
        attn_weights =  (xq@xk.transpose(2,3))/math.sqrt(self.head_dim)+attention_mask

        # upcast the attention to fp32 
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
        scores = attn_weights@xv
        #scores = F.softmax(((xq@xk.transpose(2,3))/math.sqrt(self.head_dim)+attn_mask).float(), dim=-1)@xv
        
        # Reshape the scores and squeeze (n_heads, head_dim) dimensions
        # (batch_size, n_heads, seq_len, head_dim) --> (batch_size, seq_len, n_heads*head_dim)
        scores = scores.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        # Produce output --> (batch_size, seq_len, n_heads*head_dim)
        return self.wo(scores)
        

class MLP(nn.Module):
    def __init__(self, 
                 config: SampleBasedTransformerConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # Architecture originally from LLama model then used in Mistral with different activation function
        self.w1 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # ACT2FN receives a string of act_fn name and returns pytorch's implementation of the fn
        # Mistral used SiLU, LLama2 used SwiGlu instead, we'll be using SiLU for now
        self.hidden_act = ACT2FN[config.hidden_act]
    
    def forward(self, x):
        return self.w3(self.hidden_act(self.w1(x))*self.w2(x))
    
class DecoderLayer(nn.Module):
    def __init__(self, config: SampleBasedTransformerConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config)
        self.MLP = MLP(config)
        self.input_RMSNorm = RMSNorm(dim=config.hidden_size, eps= config.rms_norm_eps)
        self.post_attn_RMSNorm = RMSNorm(dim=config.hidden_size, eps= config.rms_norm_eps)
        
    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor]= None,):
        # Normalize the input and send it in to the self_attention layer, then add the input by skip connection
        h = x + self.self_attn(self.input_RMSNorm(x), attention_mask=attention_mask)
        # Send the attention score to the FFN and add the residual connection, send it as output
        return h + self.MLP(self.post_attn_RMSNorm(h))


class SampleBasedPreTrainedModel(PreTrainedModel, SampleBasedGenerationMixin):
    # Not sure what this does:
    base_model_prefix = "model"
    # We want to save memory so checkpointing allows us to "checkpoint" our gradients to the recompute
    supports_gradient_checkpointing = True
    # If we use sevral GPUs this ensures that the decoder layer is not split during parallelization (follow HF mistral implementation)
    _no_split_modules = ["DecoderLayer"]
    # These are not implemented in our version
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = False

class SampleBasedTransformer(SampleBasedPreTrainedModel):
    def __init__(self, config: SampleBasedTransformerConfig):
        super().__init__(config)
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.n_layers = config.num_hidden_layers
        self.window_size = config.sliding_window
        
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for layer in range(config.num_hidden_layers)]
        )
        self.RMSNorm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # These parts has to do with transformer PreTrainedModel
        # Need to revise what's going on inside of these 
        self.gradient_checkpointing = False
        self.post_init()
    
    ###
        # These functions are needed for gradient checkpointing
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def get_output_embeddings(self):
        return self.lm_head
    
    # This is required to be overwritten to enable generate() function
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {'input_ids': input_ids, "attention_mask": attention_mask}


    # Returns the module names to be trained w/Lora
    def retrieve_target_modules(self):
        return ['wk', 'wq', 'wv', 'wo', 'w1', 'w2', 'w3']
    
    def _create_mask(self,
                    seq_len: int,
                    batch_size: int,
                    dtype: torch.dtype,
                    attention_mask: Optional[torch.Tensor] = None,
                    device: torch.device = None):
        # Create Causal Mask (Copied from transformers modeling_attn_mask_utils.py)
        mask=torch.full((seq_len, seq_len), torch.finfo(dtype).min, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)

        # Apply sliding window
        context_mask = torch.tril(torch.ones_like(mask, dtype=torch.bool), diagonal=-self.window_size)
        mask.masked_fill_(context_mask, torch.finfo(dtype).min)

        # Expand each mask to batch size such that we can apply padding
        mask = mask[None, None, :, :].expand(batch_size, 1, seq_len, seq_len).to(device)

        if attention_mask is not None:
            # Now we expand the attention mask and then utilize it's inverse for masking
            expanded_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, seq_len)
            inverted_mask = 1.0 - expanded_mask
            inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

            # Apply padding
            mask = mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)

        # mask will be of size (batch_size, 1, seq_len, seq_len)
        return mask
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, 
                **kwargs):
        batch_size, seq_len = input_ids.shape

        # Create token embeddings
        h = self.embed_tokens(input_ids)

        attn_mask = self._create_mask(seq_len, batch_size, h.dtype, attention_mask, h.device)

        # Iterate through each decoder layer
        for layer in self.layers:
            # We use the gradient checkpointing function 
            # It can be found in modelling_utils.py transformers file
            # This is the functions:
            # functools.partial(torch.utils.checkpoint, **gradient_checkpointing_kwargs)
            # We only do this during training to "checkpoint" our gradients
            if self.gradient_checkpointing and self.training:
                h = self._gradient_checkpointing_func(
                    layer.__call__, # Do the forward pass
                    h, # Pass the arguments to the forward pass of the decoder - Hidden states
                    attn_mask, # Attention mask when doing padding
                )
            else:
                h = layer(h, attn_mask)
            
        # After final decoder layer normalize the outpu
        h = self.RMSNorm(h)
        
        # Feed in the output from the decoder layers into a dense layer
        logits = self.lm_head(h)
        logits = logits.float()

        # For trainer class, code taken from modeling_mistral.py 
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
            # return as a dict for TrainerClass
            #return {"loss": loss, "logits": logits}
        
        return CausalLMOutput(loss=loss,
                              logits=logits)
        #return {"logits": logits}
        # Use softmax to compute probabilities of each token in the output
        #return F.softmax(logits, dim=-1) We will apply softmax outside of the forward pass 