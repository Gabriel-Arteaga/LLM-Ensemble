import sys
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from torch.optim import AdamW
from utils import BatchPeftModel, AnchoredSFTTrainer, BatchSFTTrainer, load_sharded_checkpoint
                                  
# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="General training script")
    
    # Model and Dataset configuration
    parser.add_argument('--model_type', type=str, required=True, choices=['single', 'lora_ensemble', 'batch_ensemble', 'anchored_batch'], 
                        help="Type of model: 'single', 'lora_ensemble', 'batch_ensemble', or 'anchored_batch'")
    parser.add_argument('--model_name', type=str, required=True, help="Model ID or path")
    parser.add_argument('--dataset_name', type=str, required=True, help="Dataset name (e.g., squad, squad_v2, mmlu)")
    parser.add_argument('--train_size', type=int, default=40000, help="Training set size")
    parser.add_argument('--val_size', type=int, default=2000, help="Validation set size")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save results and weights")
    parser.add_argument('--checkpoint_dir', type=str, default=None, help="Path to the checkpoint for loading a saved model.")
    
    # LoRA configuration
    parser.add_argument('--lora_rank', type=int, default=8, help="Rank for LoRA")
    parser.add_argument('--lora_alpha', type=int, default=32, help="Scaling factor for LoRA")
    parser.add_argument('--task_type', type=str, default="CAUSAL_LM", help="Task type for LoRA")
    parser.add_argument('--adapter_name', type=str, default=None, help="Adapter name for BatchPeft models")
    parser.add_argument('--weight_decay_B', type=float, default=None, help="Weight decay for LoRA B matrix (for ensemble only)")

    # Training arguments
    parser.add_argument('--bf16', type=bool, default=True, help="Whether to use bf16 precision")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of epochs")
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help="Training batch size per device")
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help="Evaluation batch size per device")
    parser.add_argument('--eval_steps', type=int, default=250, help="Evaluation steps")
    parser.add_argument('--save_steps', type=int, default=250, help="Save steps")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of gradient accumulation steps")
    parser.add_argument('--num_checkpoints', type=int, default=2, help="Total number of checkpoints to keep")

    return parser.parse_args()

# Dataset processing functions
def create_template(data_point, dataset):
    if dataset in ["squad", "squad_v2"]:
        instruction = "Answer the question based only on the given context. Keep the answer short. If the answer is not in the context or if you are unsure, respond with 'I don't know'."
        user_context = data_point.get('context', '')
        user_question = data_point.get('question', '')
        assistant_answer = data_point.get('answer', '')
        return {
            "messages": [
                {"role": "user", "content": f"### Instruction\n{instruction}\n### Context\n{user_context} ### Question\n{user_question}"},
                {"role": "assistant", "content": f"{assistant_answer}"}
            ]
        }
    elif dataset == "mmlu":
        instruction = "You will be given a question followed by four options: A, B, C, and D. Your response should be either A, B, C, or D."
        user_question = data_point.get('question', '')
        options = data_point.get('choices', ["", "", "", ""])
        answer = data_point.get('answer', 0)
        answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        return {
            "messages": [
                {"role": "user", "content": f"### Instruction\n{instruction}\n### Question\n{user_question}\n### Options\n A) {options[0]} B) {options[1]} C) {options[2]} D) {options[3]}"},
                {"role": "assistant", "content": f"{answer_map[answer]}"}
            ]
        }

def retrieve_dataset_splits(dataset_name, train_size, val_size):
    if dataset_name == 'squad':
        dataset = load_dataset('squad', split='train').shuffle(seed=50)
    elif dataset_name == 'squad_v2':
        dataset = load_dataset('squad_v2', split='train').shuffle(seed=50)
        unanswerable = dataset.filter(lambda x: not bool(x['answers']['text'])).map(lambda x: {"answers": {"text": ["I don't know"]}})
        answerable = dataset.filter(lambda x: bool(x['answers']['text']))
        dataset = concatenate_datasets([answerable, unanswerable])
    elif dataset_name == 'mmlu':
        dataset = load_dataset('cais/mmlu', 'all', split='auxiliary_train').shuffle(seed=42)
    
    dataset = dataset.select(range(train_size + val_size)).map(lambda x: create_template(x, dataset_name))
    train = dataset.select(range(train_size))
    val = dataset.select(range(train_size, train_size + val_size))
    
    return train, val

# Main training function
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", padding_side='right')
    tokenizer.pad_token = tokenizer.unk_token
    if args.bf16:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True).to(device)
    
    # Setup LoRA or BatchPeft configuration
    config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        task_type=args.task_type,
        bias="none",
        target_modules=['embed_tokens', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3', 'lm_head']
    )
    
    # Apply appropriate model and trainer based on model_type
    if args.model_type in ['single', 'lora_ensemble']:
        model = PeftModel(model, config, adapter_name=args.adapter_name)
        if args.checkpoint_dir is not None:
            load_sharded_checkpoint(model, args.checkpoint_dir)
        trainer_cls = SFTTrainer

    elif args.model_type in ['batch_ensemble', 'anchored_batch']:
        model = BatchPeftModel(model, config, adapter_name=args.adapter_name)
        if args.checkpoint_dir is not None:
            load_sharded_checkpoint(model, args.checkpoint_dir)
        if args.model_type == 'anchored_batch':
            model.set_mean_prior()
        trainer_cls = AnchoredSFTTrainer if args.model_type == 'anchored_batch' else BatchSFTTrainer
    
    # Load dataset splits
    train_data, eval_data = retrieve_dataset_splits(args.dataset_name, args.train_size, args.val_size)

    # Set up optimizer with weight decay only for LoRA Ensemble
    if args.model_type == 'lora_ensemble' and args.weight_decay_B is not None:
        lora_A_params, lora_B_params = [], []
        for name, param in model.named_parameters():
            if 'lora_A' in name or 'lora_embedding_A' in name:
                lora_A_params.append(param)
            elif 'lora_B' in name or 'lora_embedding_B' in name:
                lora_B_params.append(param)
        optimizer = AdamW([
            {'params': lora_A_params, 'weight_decay': 0, 'lr': args.learning_rate},
            {'params': lora_B_params, 'weight_decay': args.weight_decay_B, 'lr': args.learning_rate}
        ])
    else:
        optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Training arguments
    training_args = TrainingArguments(
        bf16=args.bf16,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate,
        logging_steps=1,
        logging_strategy="steps",
        num_train_epochs=args.num_train_epochs,
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.num_checkpoints,
        load_best_model_at_end=True,
        group_by_length=True,
    )

    # Initialize trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        tokenizer=tokenizer,
        packing=False,
        max_seq_length=512,
        optimizers=(optimizer, None)
    )
    if args.checkpoint_dir is not None:
        trainer.train(resume_from_checkpoint=args.checkpoint_dir)
    else:
        trainer.train()

if __name__ == "__main__":
    args = parse_args()
    main(args)
