import torch
from torch import Tensor, nn
from torch.nn import Module, init
from torch.nn.parameter import Parameter
import math
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from typing import Tuple
from transformers.activations import ACT2FN 
from utils import RotaryEmbedding, apply_rotary_pos_emb, BatchTransformerConfig, BatchPreTrainedModel
from .BatchEmbedding import BatchEmbedding
from einops import einsum, rearrange, repeat
from transformers.utils import ModelOutput
from typing import Optional
from dataclasses import dataclass
from math import log

@dataclass 
class CustomLMOutput(ModelOutput):
    """
    Modified Huggingface's CausalLMOutput to also incorporate the uncertainty measurements.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    predictive_entropy: Optional[torch.FloatTensor] = None
    mutual_information: Optional[torch.FloatTensor] = None
    aleatoric_uncertainty: Optional[torch.FloatTensor] = None
    least_uncertain_member: Optional[int] = None

class BatchLinear(nn.Module):
    def __init__(self,
                 ensemble_size: int,
                 in_features: int,
                 out_features: int,
                 bias: bool = False,
                 mode: str = 'fan_in',
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.ensemble_size = ensemble_size
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode

        # We initialize this as emtpy since we will load on the naivemodels weight.
        self.weight = Parameter(torch.empty((in_features, out_features)))
        
        # The fast weights
        # Might need to initialize it as ones when loading pretrained weights... 
        self.r = Parameter(torch.empty((ensemble_size, in_features), **factory_kwargs)) 
        self.s = Parameter(torch.empty((ensemble_size, out_features), **factory_kwargs)) 
        
        # In attention we usually skip to include a bias
        if bias:
            self.bias = Parameter(torch.empty((ensemble_size, out_features), **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Calculate the standard deviation 
        # Torch expects weight to be of shape (out_featuers, in_features) thus we send in the transpose
        fan = init._calculate_correct_fan(self.weight.T, mode=self.mode)
        gain = init.calculate_gain(nonlinearity='relu')
        std = gain / math.sqrt(fan)

        # Initialize bias parameter
        if self.bias is not None:
            with torch.no_grad():
                self.bias.normal_(0,std)

        # Initialize the weights
        with torch.no_grad():
            self.weight.normal_(0, std)
            # We change mean to 1 to not ruin pretrained weights
            self.r.normal_(1, std)
            self.s.normal_(1, std)
    
    def reset_fast_weights(self,
                            mean: float = 0.0,
                            std: float = 1.0,
                            init_strategy: str = None):
        """
        Reset the fast weights of the network.

        Args:
            mean (float, optional): Mean of the normal distribution for weight initialization. Default is 0.0.
            std (float, optional): Standard deviation of the normal distribution for weight initialization. Default is 1.0.
            init_strategy (str, optional): Initialization strategy for the fast weights. 
                Supported strategies: 'xavier_normal', 'he_normal', or None (random initialization).
                Default is None.

        Note:
            If 'init_strategy' is None, weights are initialized randomly from a normal distribution 
            with mean 'mean' and standard deviation 'std'. Otherwise, initialization is based on the chosen strategy.
            Supported initialization strategies:
            - 'xavier_normal': Initializes weights using Xavier/Glorot normal initialization.
            - 'he_normal': Initializes weights using He normal initialization.

        References:
            - He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification.
                Proceedings of the IEEE international conference on computer vision, 2015-Decem, 1026-1034.
            - Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. 
                Proceedings of the thirteenth international conference on artificial intelligence and statistics, 9-16.

        """
        if init_strategy == None:
            with torch.no_grad():
                self.r.normal_(mean, std)
                self.s.normal_(mean, std)
        else:
            # Torch expects weight to be of shape (out_featuers, in_features) thus we send in the transpose
            fan_in, fan_out = init._calculate_fan_in_and_fan_out(self.weight.T)
            # Mistral utilizes SiLU which is similar to the ReLU funciton, hence we utilize the gain usually used for ReLU
            gain = math.sqrt(2.0)
            if init_strategy == 'xavier_normal':
                std = gain*math.sqrt(float(fan_in + fan_out))
                with torch.no_grad():
                    self.r.normal_(0, std)
                    self.s.normal_(0, std)

            elif init_strategy == 'he_normal':
                if self.mode == 'fan_in':
                    std = gain / math.sqrt(fan_in)
                elif self.mode == 'fan_out':
                    std = gain / math.sqrt(fan_out)
                with torch.no_grad():
                    self.r.normal_(0, std)
                    self.s.normal_(0, std)

            elif init_strategy == 'xavier_mean_1':
                std = gain*math.sqrt(float(fan_in + fan_out))
                with torch.no_grad():
                    self.r.normal_(1, std)
                    self.s.normal_(1, std)
            
            elif init_strategy == 'he_mean_1':
                if self.mode == 'fan_in':
                    std = gain / math.sqrt(fan_in)
                elif self.mode == 'fan_out':
                    std = gain / math.sqrt(fan_out)

                with torch.no_grad():
                    self.r.normal_(1, std)
                    self.s.normal_(1, std)

    def forward(self, x):
        # x should be of shape (ensemble_size, batch_size, seq_len, dim)
        
        # E = Ensemble_size, B=Batch_Size, I=Input_dim, O = Output_dim
        output = einsum(x, self.r, self.weight, self.s,
                        'B E ... I, E I, I O, E O -> B E ... O')
        if self.bias:
            output = einsum(output, self.bias, 'B E ... O, E O -> B E ... O')
        return output
    

class RMSNorm(nn.Module):
    # Mistral's using eps=1e-5 when loaded in... So we change it to the same to compare
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class BatchAttention(nn.Module):
    def __init__(self,
                 config: BatchTransformerConfig,
                 device = None):
        super().__init__()
        self.ensemble_size = config.ensemble_size
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        # The dimension of each head
        self.head_dim = self.hidden_size//self.n_heads
        # Amount of times key and values should be repeated to match the head of the queries
        self.n_rep = self.n_heads // self.n_kv_heads
        
        # Precompute the rotary embedding matrix
        self.rotary_embeddings = RotaryEmbedding(config,
                                                 device=device
                                                 )
        
        # Weight matrices for Query Key and Value
        self.wq = BatchLinear(self.ensemble_size ,self.hidden_size, self.n_heads * self.head_dim, bias = False)
        self.wk = BatchLinear(self.ensemble_size ,self.hidden_size, self.n_kv_heads * self.head_dim, bias = False)
        self.wv = BatchLinear(self.ensemble_size ,self.hidden_size, self.n_kv_heads * self.head_dim, bias = False)
        
        # Final projection weight matrix
        self.wo = BatchLinear(self.ensemble_size,self.n_heads * self.head_dim, self.hidden_size, bias = False)

    def reset_fast_weights(self,
                            mean: float = 0.0,
                            std: float = 1.0,
                            init_strategy: str = None):
        self.wq.reset_fast_weights(mean, std, init_strategy)
        self.wk.reset_fast_weights(mean, std, init_strategy)
        self.wv.reset_fast_weights(mean, std, init_strategy)
        self.wo.reset_fast_weights(mean, std, init_strategy)

    
    def _repeat_kv(self, x):
        # x (B, E, S, n_kv_heads, head_dim)
        # -> (B, E, S, n_kv_heads * n_rep, head_dim)
        return repeat(x,
                      'B ... n_kv_heads head_dim -> B ... (n_kv_heads n_rep) head_dim',
                       n_rep=self.n_rep)
    
    def forward(self,
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None):
        # x shape (ensemble_size, batch_size, seq_len, in_dim)
        batch_size, ensemble_size, seq_len,_ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        # x (B, E, S, I)
        # Q (B, E, S, N_heads*Head_dim)
        # K (B, E, S, N_kv_heads*head_dim)
        # V (B, E, S, N_kv_heads*head_dim)
        
        # Reshape the matrices into heads
        xq = rearrange(xq, 'B ... (n_heads head_dim) -> B ... n_heads head_dim',
                  n_heads=self.n_heads, head_dim=self.head_dim)
        xk = rearrange(xk, 'B ... (n_kv_heads head_dim) -> B ... n_kv_heads head_dim', 
                  n_kv_heads=self.n_kv_heads, head_dim=self.head_dim)
        xv = rearrange(xv, 'B ... (n_kv_heads head_dim) -> B ... n_kv_heads head_dim', 
                  n_kv_heads=self.n_kv_heads, head_dim=self.head_dim)
        # -->
        # Q (B, E, S, n_heads, head_dim)
        # K (B, E, S, n_kv_heads, head_dim)
        # V (B, E, S, n_kv_heads, head_dim)
        # Apply positional rotary embeddings
        cos, sin = self.rotary_embeddings(xv, seq_len=seq_len)
        # Think this should stay untouched even though we change (E B) -> (B E)
        xq, xk = apply_rotary_pos_emb(xq,xk,cos,sin,seq_len)

        # Repeat the key and value heads n_rep times
        xk = self._repeat_kv(xk)
        xv = self._repeat_kv(xv) 
        # Reshape all tensors 
        #desired outcome:
        # (B , E, n_heads, S, S)
        # Query @ Keys
        attn_weights = einsum(xq,xk,
                              'B E SE n_heads head_dim, B E S n_heads head_dim -> B E n_heads SE S')/math.sqrt(self.head_dim)

        #Apply mask
        attn_weights = attn_weights+attention_mask

        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(xq.dtype)
        #attn_weights = F.softmax(attn_weights, dim=-1)
        # Attn_weights @ Value
        attn_weights = einsum(attn_weights, xv,
                      'B E n_heads S SS, B E SS n_heads head_dim -> B E n_heads S head_dim')
        # Reshape
        attn_weights = rearrange(attn_weights,
                         'B E n_heads S head_dim -> B E S (n_heads head_dim)')
        # Projection
        return self.wo(attn_weights)

class BatchMLP(nn.Module):
    def __init__(self,
                 config: BatchTransformerConfig):
        super().__init__()
        self.ensemble_size = config.ensemble_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # Architecture originally from LLama model then used in Mistral with different activation function
        self.w1 = BatchLinear(self.ensemble_size,self.hidden_size, self.intermediate_size, bias=False)
        self.w2 = BatchLinear(self.ensemble_size,self.hidden_size, self.intermediate_size, bias=False)
        self.w3 = BatchLinear(self.ensemble_size,self.intermediate_size, self.hidden_size, bias=False)
        # ACT2FN receives a string of act_fn name and returns pytorch's implementation of the fn
        # Mistral used SiLU, LLama2 used SwiGlu instead, we'll be using SiLU for now
        self.hidden_act = ACT2FN[config.hidden_act]
    
    def reset_fast_weights(self,
                            mean: float = 0.0,
                            std: float = 1.0,
                            init_strategy: str = None):
        self.w1.reset_fast_weights(mean, std, init_strategy)
        self.w2.reset_fast_weights(mean, std, init_strategy)
        self.w3.reset_fast_weights(mean, std, init_strategy)

    def forward(self, x):
        return self.w3(self.hidden_act(self.w1(x))*self.w2(x))

class BatchDecoderLayer(nn.Module):
    def __init__(self,
                 config: BatchTransformerConfig):
        super().__init__()
        self.ensemble_size = config.ensemble_size
        self.hidden_size = config.hidden_size
        self.self_attn = BatchAttention(config)
        self.MLP = BatchMLP(config)
        self.input_RMSNorm = RMSNorm(dim=config.hidden_size, eps= config.rms_norm_eps)
        self.post_attn_RMSNorm = RMSNorm(dim=config.hidden_size, eps= config.rms_norm_eps)
        
    def forward(self, 
                x: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Normalize the input and send it in to the self_attention layer, then add the input by skip connection
        h = x + self.self_attn(self.input_RMSNorm(x), attention_mask)
        # Send the attention score to the FFN and add the residual connection, send it as output
        return h + self.MLP(self.post_attn_RMSNorm(h))

class BatchTransformer(BatchPreTrainedModel):
    config_class = BatchTransformerConfig

    def __init__(self,
                  config: BatchTransformerConfig):
        super().__init__(config)
        self.ensemble_size = config.ensemble_size
        self.pad_token_id = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = BatchEmbedding(self.ensemble_size,
                                           config.vocab_size,
                                           config.hidden_size,
                                           config.pad_token_id)
        self.n_layers = config.num_hidden_layers
        self.window_size = config.sliding_window
        
        self.layers = nn.ModuleList(
            [BatchDecoderLayer(config) for layer in range(config.num_hidden_layers)]
        )
        self.RMSNorm = RMSNorm(dim=config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = BatchLinear(self.ensemble_size,
                                   config.hidden_size,
                                   config.vocab_size,
                                   bias=False)


        self.gradient_checkpointing = False
        self.post_init()
    
    def reset_fast_weights(self,
                            mean: float = 0.0,
                            std: float = 1.0,
                            init_strategy: str = None):
        # We do not want to destroy the pretrained weights, thus
        # we can initalize the fast weights with different strategies
        # and evaluate the outcome
        self.embed_tokens.reset_fast_weights(mean, std, init_strategy)
        for layer in self.layers:
            layer.self_attn.reset_fast_weights(mean, std, init_strategy)
            layer.MLP.reset_fast_weights(mean, std, init_strategy)
        self.lm_head.reset_fast_weights(mean, std, init_strategy)

    def get_input_embeddings(self):
        return self.embed_tokens
    
    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **kwargs):
        return {'input_ids': input_ids, "attention_mask": attention_mask}
    
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
        mask = mask[None, None, None, :, :].expand(self.ensemble_size, batch_size, 1, seq_len, seq_len).to(device)

        if attention_mask is not None:
            # Now we expand the attention mask and then utilize it's inverse for masking
            expanded_mask = attention_mask[None, :, None, None, :].expand(self.ensemble_size, batch_size, 1, seq_len, seq_len)
            inverted_mask = 1.0 - expanded_mask
            inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)

            # Apply padding
            mask = mask.masked_fill(inverted_mask.bool(), torch.finfo(dtype).min)
        # (ensemble_size, batch_size, 1, seq_len, seq_len) -> (batch_size, ensemble_size, 1, seq_len, seq_len)
        mask = mask.transpose(0,1)
        # mask will be of size (batch_size, ensemble_size, 1, seq_len, seq_len)
        return mask
    
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                labels: torch.Tensor = None, 
                compute_uncertainty: bool = False,
                emphasize_uncertainty: bool = False,
                **kwargs):
        #batch_size, seq_len = input_ids.shape[-2:]
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[-1] 
        # Create token embeddings
        h = self.embed_tokens(input_ids)

        # Create mask
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
            
        # After final decoder layer normalize the output
        h = self.RMSNorm(h)
        
        # Feed in the output from the decoder layers into a dense layer
        logits = self.lm_head(h)
        logits = logits.float()
        # Logits used to be (batch_size, seq_len, vocab_size)
        # now it is (batch_size, ensemble_size,seq_len, vocab_size)

        # Initiate return values
        predictive_entropy, mutual_information, aleatoric_uncertainty, least_uncertain_member = None, None, None, None
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_logits = shift_logits.mean(dim=1)  # Take mean of the ensembles to match numbero of targets in loss fct
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Ensure tensors are on the same device
            shift_labels = shift_labels.to(shift_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits, shift_labels)
        
        # We are making inference, hence we return the ensembled prediction
        elif compute_uncertainty and logits.dim() == 4:
            if emphasize_uncertainty == False:
                # If we're not interested in the prodsum of entropy of each model, then calculating these entropies are unnecessary
                # We still want to return the logits, thus, we create a temporary logits variable with only the latest generated token
                # The latest token is at -1, shape: (batch_size, ensemble_size, seq_len, vocab_size) -> (batch_size, ensemble_size, vocab_size)
                tmp_logits = logits[:,:,-1, :]
            else:
                tmp_logits = logits
            # We calculate the average conditional entropy (aleatoric uncertainty)
            # (batch_size, ensemble_size, vocab_size) -> (batch_size, ensemble_size)
            # or (batch_size, ensemble_size, seq_len, vocab_size) -> (batch_size, ensemble_size, seq_len) if emphasize_uncertainty =True
            aleatoric_uncertainty = (torch.special.entr(tmp_logits.softmax(-1))/log(2)).sum(dim=-1)
            if emphasize_uncertainty:
                # We do the prodsum, the argmin is the most probable ensemble member, we use its prediction
                # We first do the product along the vocabulary dimension, we then sum up the batches, then we do the argmin to retrieve
                # the ensemble member with least amount of entropy in its prediction
                least_uncertain_member = aleatoric_uncertainty.prod(dim=2).sum(0).argmin().item()
                # Now we can select the last generated token's entropy 
                # The latest token is at -1, shape: (batch_size, ensemble_size, seq_len) -> (batch_size, ensemble_size)
                aleatoric_uncertainty = aleatoric_uncertainty[:,:,-1]
            # We should sum over the ensemble members and divide by M
            # Compute the mean entropy over ensemble members
            # Shape from (batch_size, ensemble_size) --> (batch_size)
            aleatoric_uncertainty = aleatoric_uncertainty.mean(dim=1) # <--

            # Predictive Entropy
            if emphasize_uncertainty:
                tmp_logits = logits[:,:,-1, :]
            # We first calculate the average prediction 
            # Shape from (batch_size, ensemble_size, vocab_size) -> (batch_size, vocab_size)
            tmp_logits = tmp_logits.softmax(-1).mean(dim=1) # <-- The timing of when we do this calculation is what differs Predictive entropy from aleatoric
            # We then compute the  the entropy for the average prediction
            # Shape from (batch_size, vocab_size) -> (batch_size)
            # entropy function and divide with log(2)
            predictive_entropy = (torch.special.entr(tmp_logits)/log(2)).sum(dim=1)
            
            # We are done with tmp_logits, we can delete it
            del tmp_logits

            # Finally use these two terms to compute the mutual information (epistemic uncertainty)
            mutual_information = predictive_entropy-aleatoric_uncertainty

        if self.training == False and emphasize_uncertainty == False:
            # We want to return the combined prediction
            # If we emphasize uncertainty, we get better uncertainty estimates at the expense predictive performance
            logits = logits.mean(dim=1)

        return CustomLMOutput(loss=loss,
                              logits=logits,
                              predictive_entropy=predictive_entropy,
                              mutual_information=mutual_information,
                              aleatoric_uncertainty=aleatoric_uncertainty,
                              least_uncertain_member=least_uncertain_member)