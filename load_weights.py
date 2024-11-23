from utils import BatchTransformerConfig, AnchoredBatchTransformerConfig, SampleBasedTransformerConfig
from models import BatchTransformer, AnchoredBatchTransformer, SampleBasedTransformer
import argparse
from transformers import AutoModelForCausalLM
import torch

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="General training script")
    
    # Model and Dataset configuration
    parser.add_argument('--model_type',
                        type=str,
                        required=True,
                        choices=['batch_ensemble', 'anchored_batch', 'sample_based'], 
                        help="Type of model: 'batch_ensemble', or 'anchored_batch' or 'sample_based'")
    parser.add_argument('--base_model_id',
                        type=str,
                        default='mistralai/Mistral-7B-Instruct-v0.2',
                        help="ID of the pre-trained model which will act as shared initial weights for the Batch Ensemble.")
    parser.add_argument('--model_name',
                        type=str,
                        required=True,
                        help="Directory name where the model will be saved.")
    parser.add_argument('--ensemble_size',
                        type=int,
                        default=4,
                        help="Size of the ensemble for the batch model configurations.")
    parser.add_argument('--device',
                        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Device to run the model on, e.g., 'cuda' or 'cpu'.")
    return parser.parse_args()


def load_base_model_weights(batch_ensemble_model,
                            base_model_id: str):
    # Load the base model's weights
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)

    # We iterate through all the names of the layers and store all the slow weight's names into a list
    slow_weight_names = []
    for (name, _) in batch_ensemble_model.state_dict().items():
        # We iterate through all names, if it contains .r or .s it belongs to the fast weights, we don't want these
        if not name.endswith(('.r', '.s')):
            slow_weight_names.append(name)
    
    # Our names differs so we need to create a dict which maps the names correctly
    name_mapping = {}
    for (name_base_model, _), name in zip(base_model.state_dict().items(), slow_weight_names):
        name_mapping[name_base_model] = name

    # Store base model's state dict
    base_model_state_dict = base_model.state_dict()

    # We create a new state dict with the slow weight parameters
    slow_weight_dict = {name_mapping[base_model_name]: base_model_param for base_model_name,
                        base_model_param in base_model_state_dict.items() if base_model_name in name_mapping}
    
    # The BatchLinear layer uses an mxn shape while torch's Linear uses nxm, hence we need to transpose the matrices
    for name in slow_weight_dict:
        if 'embed_tokens' not in name:
            slow_weight_dict[name] = slow_weight_dict[name].T
    
    # Update the slow weights
    batch_ensemble_model_state_dict = batch_ensemble_model.state_dict()
    batch_ensemble_model_state_dict.update(slow_weight_dict)

    # Load the slow weights to the batch ensemble model
    batch_ensemble_model.load_state_dict(batch_ensemble_model_state_dict)

def initiate_batch_model(args):
    # Initiate config
    if args.model_type == 'sample_based':
        config = SampleBasedTransformerConfig()
        model = SampleBasedTransformer(config)
    if args.model_type == 'batch_ensemble':
        config = BatchTransformerConfig(ensemble_size=args.ensemble_size)
        model = BatchTransformer(config)
    else:
        config = AnchoredBatchTransformerConfig(ensemble_size=args.ensemble_size)
        model = AnchoredBatchTransformer(config)
    model = model.to(args.device)

    # Load slow weights
    load_base_model_weights(model,
                            base_model_id=args.base_model_id)
    
    if args.model_type == 'anchored_batch':
        # Set the mean prior to the pretrained weight's mean
        model.set_mean_prior()
    
    return model, config

if __name__ == "__main__":
    args = parse_args()
    model, config = initiate_batch_model(args)
    # Register for auto class
    config.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")
    # Save model
    model.save_pretrained(args.model_name)