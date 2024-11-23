# Hallucination Detection in LLMs
This repository contains an official PyTorch implementation for *Hallucination Detection in LLMs: Fast and Memory-Efficient Finetuned Models*.

![Overview of the method](https://github.com/Gabriel-Arteaga/LLM-Ensemble/blob/main/assets/figure_1.jpg)

<p align="center">
 Overview of the method. The paper can be found <a href="https://arxiv.org/pdf/2409.02976" target="_blank">here</a>.
</p>

## Abstract
*Uncertainty estimation is a necessary component when implementing AI in high-risk settings, such as autonomous cars, medicine, or insurances. Large Language Models (LLMs) have seen a surge in popularity in recent years, but they are subject to hallucinations, which may cause serious harm in high-risk settings. Despite their success, LLMs are expensive to train and run: they need a large amount of computations and memory, preventing the use of ensembling methods in practice. In this work, we present a novel method that allows for fast and memory-friendly training of LLM ensembles. We show that the resulting ensembles can detect hallucinations and are a viable approach in practice as only one GPU is needed for training and inference.*

## Running scripts
### Substitute the shared weight with pre-trained weights
We highly recommend initializing the shared weights in BatchEnsemble with a set of pre-trained weights, as training from scratch is an extremely resource-intensive process.

In our experiments, we used the `Mistral-7B-Instruct-v0.2` pre-trained weights. While we have not tested other options, any pre-trained weights from the Llama-family of models should be compatible with our method, as they share the same architecture.
```
python3 load_weights.py --model_type batch_ensemble --base_model_id mistralai/Mistral-7B-Instruct-v0.2 --model_name /path/to/save/model_weights --ensemble_size 4
```
### Train the BatchEnsemble and other baselines
To train the `BatchEnsemble` or any other baseline model referenced in the paper, use the following script. To train a different model, simply adjust the `model_type` parameter as needed.
```
python3 train.py --model_type batch_ensemble --model_name /path/to/model_weights --dataset_name squad --train_size 40000 --val_size 2000 --output_dir path/to/save/checkpoints_and_results --lora_rank 8 --lora_alpha 32 --bf16 True --learning_rate 2e-5 --num_train_epochs 1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --eval_step 250 --save_steps 250 --gradient_accumulation_steps 1 --num_checkpoints 2
```
### Evaluate performance
After fine-tuning the model, you can evaluate its performance and obtain uncertainty estimates using the following command:
```
python3 eval.py --model_type batch_ensemble --dataset_name squad --test_size 5000 --output_dir /path/to/save_results --checkpoint_dir /path/to/model_weights --tokenizer_id mistralai/Mistral-7B-Instruct-v0.2 --bf16 True --batch_size 32
```
## Limitations
The current implementation of BatchEnsemble includes some necessary simplifications compared to the original Mistral model architecture to accommodate the Batch Linear layers.  
As a result, it does not yet support features like SDPA, Flash Attention, KV-cache, or the computational optimizations provided by Sliding Window attention.
## Reference
```
@article{arteaga2024hallucination,
  title={Hallucination Detection in LLMs: Fast and Memory-Efficient Finetuned Models},
  author={Arteaga, Gabriel Y and Sch{\"o}n, Thomas B and Pielawski, Nicolas},
  journal={arXiv preprint arXiv:2409.02976},
  year={2024}
}
```
