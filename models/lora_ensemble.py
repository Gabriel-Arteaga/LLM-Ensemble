from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, StoppingCriteriaList, MaxLengthCriteria
from transformers.generation import EosTokenCriteria
import torch
from torch import nn
from datasets import load_dataset
from peft import LoraConfig
from copy import copy
from torch.nn import functional as F
from math import log
import torch.distributed as dist

class LoRAEnsemble(nn.Module):
    def __init__(self,
                 ensemble_size,
                 base_model_id,
                 path_directory,
                 adapter_name
                ):
        super(LoRAEnsemble, self).__init__()
        # Load the Base Model weights
        self.model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.bfloat16)
        
        # We trained and saved all models as <adapter_name>_<member_i>
        self.ensemble_size = ensemble_size
        self.vocab_size = self.model.config.vocab_size
        self.adapters = [f"{adapter_name}_{str(member)}" for member in range(1, ensemble_size+1)]
        # Path to the directory containing all ensemble members LoRA weights
        self.path_directory = path_directory
        # Load all the adapter configs
        self.load_adapters()
    
    def load_adapters(self):
        # Iterate through all the adapter names
        for adapter_name in self.adapters:
            # We have saved each adapter in a directory with the same name as the adapter name
            self.model.load_adapter(peft_model_id=self.path_directory+'/'+adapter_name+'/final',
                                    adapter_name=adapter_name)
    
    # Copied from transformers.generation.utils
    def _has_unfinished_sequences(self, this_peer_finished: bool, synced_gpus: bool, device: torch.device) -> bool:
        """
        Returns whether there are still unfinished sequences in the device. The existence of unfinished sequences is
        fed through `this_peer_finished`. ZeRO stage 3-friendly.
        """
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                return False
        elif this_peer_finished:
            return False
        return True
    
    def generate_greedy(self,
                        input_ids: torch.LongTensor,
                        attention_mask: torch.LongTensor,
                        generation_config: GenerationConfig,
                        compute_uncertainty: bool = False,
                        synced_gpus: bool = False):
        """
            Args:
                input (torch.LongTensor): Contains a tokenized input of shape (batch_size, 2, seq_len), the input has been padded, contains both the input ids and attention mask.
                max_new_tokens (int): Maximum number of new tokens to generate.
                compute_uncertainty (bool): Whether to compute the uncertainty measures or not.
                decoding_strategy (str): Which type of decoding strategy to utilize, defaults to greedy decoding.
        """
        pad_token_id = generation_config.pad_token_id

        # After first iteration we need to do argmax of mean logits and re-tokenize the sequence
        # When we do this the max_new_tokens should be reduced with 1 so that we don't generate infinitely many tokens
        # We need to define stopping criterias, think if generation_config.max_new_tokens=0 or last generated token is EOS token should suffice.

        # We will change the max_new_tokens attribute of the generation config, we do not want this to affect the generation config outside this function hence we make a local copy
        generation_config = copy(generation_config)
        # We need to output logits for making our ensembled prediction
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True

        # Initialize our stoppining criterias to determine when to stop generating new tokens
        stopping_criteria = StoppingCriteriaList()
        # We want to stop generating when we produce an EOS token in our sequence
        stopping_criteria.append(EosTokenCriteria(generation_config.eos_token_id))
        # We want to stop when we reached our predetermined max length
        # This is usually carried out in generate function and then the call to _prepare_generated_length (huggingface)
        # We do it manually instead
        stopping_criteria.append(MaxLengthCriteria(input_ids.shape[-1]+generation_config.max_length))

        
        # Implementation from transformers generation file
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        # Initialize the uncertainties
        predictive_entropy = None
        aleatoric_uncertainty = None 
        epistemic_uncertainty = None
       
        # Runs until EoS token is reached or max length is reached. Note, that this is for the entire batch of data points
        while self._has_unfinished_sequences(this_peer_finished=this_peer_finished,
                                             synced_gpus=synced_gpus,
                                             device=input_ids.device):
            # Each model should only make 1 prediction
            generation_config.max_new_tokens = 1
            # Initialize raw logits variable
            raw_logits = None
            for adapter_name in self.adapters:
                # Use the i:th ensemble member for prediction
                self.model.set_adapter(adapter_name)
                output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)
                # Logits have shape (batch_size, vocab_size), it is sent out as a tuple, thus we collect the tensor
                logits = output.logits[0] # Get the logits for the last token
                if raw_logits is None:
                    # We are only interested in the latest generated token
                    raw_logits = logits.unsqueeze(1)
                    # We want it to have shape (batch_size, ensemble_size, vocab_size)
                else:
                    # Concatenate along the ensemble size dimension
                    raw_logits = torch.cat((raw_logits, logits.unsqueeze(1)), dim=1)
            ### Implement Uncertainty calculations here ###
            if compute_uncertainty:
                # We will conduct manipulations on these logits for computing uncertainty, we do not want this to affet our
                # inference
                tmp_logits = raw_logits.detach().clone()
                # If None it is the first uncertainty in the sequence
                if predictive_entropy == None:
                    # (batch_size, ensemble_size, vocab_size) -> (batch_size, ensemble_size)
                    aleatoric_uncertainty = (torch.special.entr(tmp_logits.softmax(-1))/log(2)).sum(dim=-1)
                    # Compute the mean entropy over ensemble members
                    # Shape from (batch_size, ensemble_size) --> (batch_size, 1)
                    aleatoric_uncertainty = aleatoric_uncertainty.mean(dim=1).unsqueeze(1)
                    # Shape from (batch_size, ensemble_size, vocab_size) -> (batch_size, vocab_size)
                    tmp_logits = tmp_logits.softmax(-1).mean(dim=1) 
                    # Shape from (batch_size, vocab_size) -> (batch_size, 1)
                    predictive_entropy = (torch.special.entr(tmp_logits)/log(2)).sum(dim=1).unsqueeze(1)
                    # Compute epistemic uncertainty
                    epistemic_uncertainty = predictive_entropy-aleatoric_uncertainty
                
                # Otherwise we concatenate each new token's uncertainty so that we have a uncertainty of sequence length
                else:
                    # (batch_size, ensemble_size, vocab_size) -> (batch_size, ensemble_size)
                    tmp_aleatoric_uncertainty = (torch.special.entr(tmp_logits.softmax(-1))/log(2)).sum(dim=-1)
                    # Compute the mean entropy over ensemble members
                    # Shape from (batch_size, ensemble_size) --> (batch_size, 1)
                    tmp_aleatoric_uncertainty = tmp_aleatoric_uncertainty.mean(dim=1).unsqueeze(1)
                    # Shape from (batch_size, ensemble_size, vocab_size) -> (batch_size, vocab_size)
                    tmp_logits = tmp_logits.softmax(-1).mean(dim=1)
                    # Shape from (batch_size, vocab_size) -> (batch_size, 1)
                    tmp_predictive_entropy = (torch.special.entr(tmp_logits)/log(2)).sum(dim=1).unsqueeze(1)
                    # Compute epistemic uncertainty
                    tmp_epistemic_uncertainty = predictive_entropy-aleatoric_uncertainty
                    
                    # Concatenate the sequence's uncertainties
                    predictive_entropy = torch.cat((predictive_entropy,
                                                   tmp_predictive_entropy), dim=1)
                    aleatoric_uncertainty = torch.cat((aleatoric_uncertainty,
                                                       tmp_aleatoric_uncertainty), dim=1)
                    epistemic_uncertainty = torch.cat((epistemic_uncertainty, 
                                                      tmp_epistemic_uncertainty), dim=1)
                    # We are done with the temporary logits, remove them to not take up unneccessary memory
                    del tmp_predictive_entropy
                    del tmp_aleatoric_uncertainty
                    del tmp_epistemic_uncertainty
                del tmp_logits
            
            # We want the ensembled prediction, take the mean along the ensemble dimension
            raw_logits = raw_logits.mean(dim=1) 
            # Compute the softmax scores
            probs = F.softmax(raw_logits, dim=-1)
            # Do greedy decoding
            next_tokens = torch.argmax(probs, dim=-1)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # Add the generated tokens to the input for an additional timestep of generation
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            # Extend the attention mask
            attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                    )
            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores=None)
            this_peer_finished = unfinished_sequences.max() == 0
            
        return {'input_ids': input_ids,
                'predictive_entropy': predictive_entropy,
                'epistemic_uncertainty': epistemic_uncertainty,
                'aleatoric_uncertainty': aleatoric_uncertainty}

    def generate(self,
                 input_ids: torch.LongTensor,
                 attention_mask: torch.LongTensor,
                 generation_config: GenerationConfig,
                 compute_uncertainty: bool = False,
                 decoding_strategy: str = 'greedy'):
        """
            Args:
                input (torch.LongTensor): Contains a tokenized input of shape (batch_size, 2, seq_len), the input has been padded, contains both the input ids and attention mask.
                generation_config (transformers.GenerationConfig): The config containing instructions of how each ensemble member should generate.
                compute_uncertainty (bool): Whether to compute the uncertainty measures or not.
                decoding_strategy (str): Which type of decoding strategy to utilize, defaults to greedy decoding.
        """
        if decoding_strategy == 'greedy':
            output = self.generate_greedy(input_ids=input_ids,
                                          attention_mask=attention_mask,
                                 generation_config=generation_config,
                                 compute_uncertainty=compute_uncertainty)
        else:
            raise Exception("Only greedy decoding has been implemented so far.")
        return output