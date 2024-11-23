import json
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from models import LoRAEnsemble
from datasets import load_dataset
from torch.utils.data import DataLoader
from utils import extract_generated_tokens

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="General training script")
    
    # Model and Dataset configuration
    parser.add_argument('--model_type', type=str, required=True, choices=['sample_based', 'lora_ensemble', 'batch_ensemble', 'anchored_batch'], 
                        help="Type of model: 'sample_based', 'lora_ensemble', 'batch_ensemble', or 'anchored_batch'")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name (e.g., squad, squad_v2, cais/mmlu)")
    parser.add_argument('--test_size', type=int, default=5000, help="How many samples to perform evaluation on.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save results.")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Path to the checkpoint for loading the saved model.")
    parser.add_argument('--tokenizer_id', type=str, default = "mistralai/Mistral-7B-Instruct-v0.2", help="Tokenizer to use for the model.")

    ## LoRA Ensemble Arguments
    parser.add_argument('ensemble_size', type=int, default=4, help="Amount of seperate LoRA Ensemble members. Each member is it's own set of LoRA adapters.")
    parser.add_argument('path_directory', type=str, default=None, help="Path to the directory containing the LoRA adapters.")
    parser.add_argument('base_model_id', type=str, default='mistralai/Mistral-7B-Instruct-v0.2 ', help="The base model id for the LoRA Ensemble.")
    parser.add_argument('--adapter_name', type=str, 
                        help="""The adapter name for the LoRA Ensemble. Note, each member is expected to be saved in the following format:
                             "<path_directory>/<adapter_name>/final. Where <adapter_name> is the name of the adapter. For instance, each member could be saved as:
                             "<path_directory>/adapter1/final", "<path_directory>/adapter2/final", etc. where adapter1, adapter2, etc. are the names of the adapters.""")
    
    # Sample Based Model Arguments
    parser.add_argument('--amount_of_samples', type=int, default=4,
                        help="Amount of samples used by the model to generate uncertainty estimates. This simulates ensemble size of other ensembles methods.")
    

    # Model configuration
    parser.add_argument('--bf16', type=bool, default=True, help="Whether to use bf16 precision, leads to faster inference.")
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size for evaluation.")


    return parser.parse_args()

# TODO: Add all preprocessing functions for the different datasetsty 
def preprocess_squad(dataset):
    # Define special tokens
    instruct_start = '[INST]'
    instruct_end = '[/INST]'
    instruction = f"""Answer the question based only on the given context. Keep the answer short. If the answer is not in the context or if you are unsure, respond with 'I don't know'.
    """ 
    dataset["prompt"] = f"{instruct_start}\n{instruction}\n# Context\n{dataset['context']}\n# Question\n{dataset['question']} {instruct_end}"
    dataset["labels"] = dataset['answers']['text'][0]
    return dataset

def preprocess_mmlu(dataset):
    # Define special tokens
    instruct_start = '[INST]'
    instruct_end = '[/INST]'
    instruction = f"""You will be given a question followed by four options: A, B, C, and D. Your response should be either A, B, C, or D.
    """ 
    # The question to the model
    question = dataset['question']
    # The four options of which the model can respond
    options = dataset['choices']
    option_A =  options[0]
    option_B = options[1]
    option_C = options[2]
    option_D = options[3]

    dataset["prompt"] = f"{instruct_start}\n### Instruction\n{instruction}\n### Question\n{question}\n### Options \n A) {option_A} \n B) {option_B} \n C) {option_C} \n D) {option_D} {instruct_end}"
    dataset["labels"] = dataset['answer']
    return dataset

# We want to keep the unanswerable questions
def is_unanswerable(example):
    return not bool(example['answers']['text'])

# The unanswerable questions should have "I don't know" as its answer
def replace_unanswerable_answers(example):
    # we add the extra dimension for the processing stage. 
    example['answers']['text'] = ["I don't know"]
    return example

def preprocess_dataset(args):
    if args.dataset_name == 'squad':
        data = load_dataset("squad", split="validation")
        data = data.shuffle(seed=42).select(range(args.test_size))
        data = data.map(preprocess_squad)
        data = data.remove_columns(['title','context','question', 'answers'])
    
    elif args.dataset_name == 'squad_v2':
        data = load_dataset("squad_v2", split="validation")
        # Filter unanswerable questions
        data = data.filter(is_unanswerable)
         # Add a label of "I don't know" so that we can differtiate the unanswerable questions from the answerable ones
        data = data.map(replace_unanswerable_answers)
        data = data.shuffle(seed=42).select(range(args.test_size))
        data = data.map(preprocess_squad)
        data = data.remove_columns(['title','context','question', 'answers'])
    
    elif args.dataset_name == 'cais/mmlu':
        data = load_dataset("cais/mmlu", 'all', split="validation")
        data = data.shuffle(seed=42).select(range(args.test_size))
        data = data.map(preprocess_mmlu)
    
    else:
        raise ValueError("Invalid dataset name. Please choose from 'squad', 'squad_v2', 'cais/mmlu'.")
    
    return data

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id) 
    tokenizer.pad_token = tokenizer.eos_token

    if args.model_type == 'lora_ensemble':
        model = LoRAEnsemble(ensemble_size=args.ensemble_size,
                     path_directory=args.path_directory,
                     base_model_id=args.base_model_id,
                     adapter_name=args.adapter_name)
    else:
        if args.bf16:
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(args.checkpoint_dir, trust_remote_code=True).to(device)

    # Evaluation mode
    model.eval()

    test_data = preprocess_dataset(args)

    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    if args.model_type == 'sample_based':
        # Following Kuhn et al. we opt for a temperature of 0.5 (Semantic entropy paper)
        # We additionally opt for top-p to 0.99 and top-k to 5 according to Chen et al (INSIDE LLMs paper)
        generation_config = GenerationConfig(max_new_tokens=50,
                                             do_sample=True,
                                             top_k=5,
                                             top_p=0.99,
                                             temperature=0.5)
    elif args.model_type == 'lora_ensemble':
        generation_config = GenerationConfig(max_new_tokens=50,
                                             eos_token_id=tokenizer.eos_token_id,
                                             pad_token_id=tokenizer.pad_token_id)
    
    if args.dataset_name == 'mmlu':
        # The label is given as an integer, we convert it to its A-D string equivalence
        answer_to_str_mapping = {
        0: 'A',
        1: 'B',
        2: 'C',
        3: 'D'
    }
    

    with open(args.output_dir, 'a') as json_file:
        # Create a unique key for each question and answer, we increment this with +1 for each datapoint
        # Only used for MMLU as it does not have a specific ID for each question
        id_counter = 0

        # Iterate over the test data
        for batch in test_loader:
            # Pad each batch to the same length
            padded_batch = tokenizer(batch['prompt'], padding=True, return_tensors="pt").to(device)

            if args.model_type == 'sample_based':
                # Generate the output
                outputs = model.generate(**padded_batch, generation_config=generation_config, amount_of_samples=args.amount_ofsamples)
            elif args.model_type == 'lora_ensemble':
                outputs = model.generate(**padded_batch, generation_config=generation_config, compute_uncertainty=True)
            else:
                outputs = model.generate(**padded_batch, max_new_tokens=50, compute_uncertainty = True)
            
            # Decode the answers to strings
            decoded_output = tokenizer.batch_decode(outputs['input_ids'], skip_special_tokens=True)

            if args.dataset_name == 'mmlu':
                for generated_answer,tokens, prompt, label,entropy, MI, aleatoric_uncertainty in zip(
                                                                                            decoded_output,
                                                                                            outputs['input_ids'],
                                                                                            batch['prompt'],
                                                                                            batch['labels'],
                                                                                            outputs['predictive_entropy'],
                                                                                            outputs['mutual_information'], 
                                                                                            outputs['aleatoric_uncertainty']):
                    # We are only interested in what the model has generated, not the input
                    instruction_idx = generated_answer.rfind("[/INST]") + len("[/INST]")
                    # Extract the encoded tokens containing the answer
                    generated_tokens = extract_generated_tokens(tokens, pad_token_id = tokenizer.pad_token_id)
                    # Prepare the answer entry
                    answer_entry = {
                        id_counter:
                        {
                            'answer': generated_answer[instruction_idx:],
                            'generated_tokens': generated_tokens.tolist(),
                            'first_token_entropy': entropy[0].item(),
                            'predictive_entropy': entropy.tolist(),
                            'mutual_information': MI.tolist(),
                            'aleatoric_uncertainty': aleatoric_uncertainty.tolist(),
                            'average_predictive_entropy': entropy.mean().item(),
                            'aleatoric_prop_to_pred': aleatoric_uncertainty.sum().item()/entropy.sum().item(),
                            'prompt': prompt,
                            'label': answer_to_str_mapping[label.item()],
                            }
                        }
                    # Write the answer entry to the file
                    json_file.write(json.dumps(answer_entry) + '\n')
                    json_file.flush()

                    # Increment the ID counter
                    id_counter += 1
                
            else:
                for answer_id, generated_answer,tokens, prompt, label,entropy, MI, aleatoric_uncertainty in zip(batch['id'],
                                                                                                                decoded_output,
                                                                                                                outputs['input_ids'],
                                                                                                                batch['prompt'],
                                                                                                                batch['labels'],
                                                                                                                outputs['predictive_entropy'],
                                                                                                                outputs['mutual_information'],
                                                                                                                outputs['aleatoric_uncertainty']):
                    # We are only interested in what the model has generated, not the input
                    instruction_idx = generated_answer.rfind("[/INST]") + len("[/INST]")
                    # Extract the encoded tokens containing the answer
                    generated_tokens = extract_generated_tokens(tokens, pad_token_id = tokenizer.pad_token_id)
                    # Prepare the answer entry
                    answer_entry = {
                        answer_id:
                        {
                            'answer': generated_answer[instruction_idx:],
                            'generated_tokens': generated_tokens.tolist(),
                            'first_token_entropy': entropy[0].item(),
                            'predictive_entropy': entropy.tolist(),
                            'mutual_information': MI.tolist(),
                            'aleatoric_uncertainty': aleatoric_uncertainty.tolist(),
                            'average_predictive_entropy': entropy.mean().item(),
                            'aleatoric_prop_to_pred': aleatoric_uncertainty.sum().item()/entropy.sum().item(),
                            'prompt': prompt,
                            'label': label,
                            }
                        }
                    # Write the answer entry to the file
                    json_file.write(json.dumps(answer_entry) + '\n')
                    json_file.flush()


    




if __name__ == "__main__":
    args = parse_args()
    main(args)
