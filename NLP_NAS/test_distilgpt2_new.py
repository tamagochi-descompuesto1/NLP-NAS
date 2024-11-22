import os
import time
import nltk
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from threading import Thread
from scipy.stats import wilcoxon
from datasets import load_dataset
from jtop_stats import jtop_stats
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

def cut_text_for_generation(text, cut_percentage=0.3):
    cut_point = int(len(text) * (1 - cut_percentage))
    return text[:cut_point], text[cut_point:]

def generate_text(input_ids, attention_mask, model, tokenizer):
    # Ensure input_ids and attention_mask have compatible dimensions
    if input_ids.dim() != attention_mask.dim():
        attention_mask = attention_mask.unsqueeze(0)  # Adjust dimensions if necessary

    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask=attention_mask, 
            max_new_tokens=50, 
            num_return_sequences=1, 
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_texts(dataset, model, tokenizer, device):
    generated_texts = []
    references = []

    for example in tqdm(dataset, desc="Generating Texts on GPU"):
        # Decode the original input text
        original_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        
        # Cut the original text into incomplete_text and original_continuation
        incomplete_text, original_continuation = cut_text_for_generation(original_text)
        
        if not incomplete_text.strip():
            # Ensure incomplete_text is not empty
            continue

        # Tokenize the incomplete_text
        incomplete_input = tokenizer(incomplete_text, return_tensors='pt', padding=True, truncation=True).to(device)
        
        # Make sure input_ids and attention_mask are not empty
        input_ids = incomplete_input['input_ids']
        attention_mask = incomplete_input['attention_mask']
        if input_ids.size(1) == 0 or attention_mask.size(1) == 0:  # Skip empty input cases
            continue

        # Generate text using the tokenized incomplete_text
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        generated_text = generate_text(input_ids, attention_mask, model, tokenizer)
        
        if generated_text.strip():
            generated_texts.append(generated_text)
            references.append(original_continuation)
    
    return generated_texts, references

def tokenize(examples, tokenizer):
    # Tokenize input text
    tokenized_output = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    # Shift the input_ids to create the labels
    tokenized_output['labels'] = tokenized_output['input_ids'].copy()
    return tokenized_output

def calculate_bleu_meteor_rouge(generated_texts, references):
    bleu_scores = []
    gleu_scores = []
    meteor_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    # Ensure tokenization
    tokenized_references = [word_tokenize(reference) for reference in references]
    tokenized_generated_texts = [word_tokenize(generated_text) for generated_text in generated_texts]

    smoothing = SmoothingFunction().method1
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for reference_tokens, generated_tokens in zip(tokenized_references, tokenized_generated_texts):
        # BLEU score calculation
        bleu_score = sentence_bleu([reference_tokens], generated_tokens, smoothing_function=smoothing)
        bleu_scores.append(bleu_score)

        # Sentence GLEU score using 4--grams
        gleu_score = sentence_gleu([reference_tokens], generated_tokens, max_len=4)
        gleu_scores.append(gleu_score)

        # METEOR score calculation (requires tokenized references and hypotheses)
        meteor = meteor_score([reference_tokens], generated_tokens)
        meteor_scores.append(meteor)

    # ROUGE-L score calculation
    for reference, generated_text in zip(references, generated_texts):
        rouge_scores = scorer.score(reference, generated_text)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

    return np.mean(bleu_scores), np.mean(gleu_scores), np.mean(meteor_scores), np.mean(rouge1_scores), np.mean(rouge2_scores), np.mean(rougeL_scores)

def main(num_experiments):
    # Load models and tokenizer
    print('Loading models...')
    model_paths = ['models/distilgpt2/distilgpt2_3epochs', 'models/distilgpt2/distilgpt2_5epochs', 
                   'models/distilgpt2/distilgpt2_10epochs', 'models/distilgpt2/distilgpt2_12epochs', 
                   'models/distilgpt2/distilgpt2_15epochs']
    models = [GPT2LMHeadModel.from_pretrained(path, local_files_only=True) for path in model_paths]
    model_names = ['distilgpt2_3epochs', 'distilgpt2_5epochs', 'distilgpt2_10epochs', 'distilgpt2_12epochs', 'distilgpt2_15epochs']
    
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', force_download=True, resume_download=None, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device is {device}')

    # Move models to device
    for model in models:
        model.to(device)
        model.eval()  # Set to evaluation mode

    # Load dataset
    print('Loading WikiText...')
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', verification_mode='no_checks')['test']

    print('Tokenizing dataset...')
    tokenized_test_dataset = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer), batched=True, remove_columns=['text'])
    tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Ensure the output directories exist
    for model_name in model_names:
        os.makedirs(f'results/{model_name}', exist_ok=True)
        os.makedirs(f'stat_dumps/distilgpt2/{model_name}', exist_ok=True)

    # Metrics storage
    all_metrics = {name: [] for name in model_names}
    hardware_metrics = []

    # Experiment loop
    print('Testing models...')
    for exp in range(num_experiments):
        print(f'Experiment {exp + 1}')
        for model, name in zip(models, model_names):
            with jtop_stats.JtopStats() as stats:
                # Start continuous delta calculation in a separate thread
                delta_thread = Thread(target=stats.calculate_deltas_periodically, args=(30,))
                delta_thread.start()

                # Generate texts using GPU
                print(f'Generating texts for model {name}...')
                generated_texts, references = generate_texts(tokenized_test_dataset, model, tokenizer, device)

                # Save hardware metrics
                stats.stop_thread = True
                delta_thread.join() 
                stats.dump_deltas(f'stat_dumps/distilgpt2/{name}/metrics_{name}_exp{exp}_{time.time()}.txt', f'stat_dumps/distilgpt2/{name}/metrics_{name}_exp{exp}_raw_{time.time()}.txt')
                hardware_metrics.append(stats.__get_deltas())
            
            # Calculate BLEU, METEOR, ROUGE scores
            print(f'Calculating BLEU, METEOR, ROUGE for model {name}...')
            metrics = calculate_bleu_meteor_rouge(generated_texts, references)
            all_metrics[name].append(metrics)
            
            print(f'Model {name} --- BLEU: {metrics[0]:.4f}, GLEU: {metrics[1]:.4f}, METEOR: {metrics[2]:.4f}, ROUGE-1: {metrics[3]:.4f}, ROUGE-2: {metrics[4]:.4f}, ROUGE-L: {metrics[5]:.4f}')

    # Save final metrics
    save_final_metrics(all_metrics, 'results/distilgpt2')

    # Plot metrics
    plot_metrics(all_metrics, model_names, 'results/distilgpt2/plots')

    # Perform statistical tests
    perform_statistical_tests(all_metrics)

def save_final_metrics(all_metrics, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for name, values in all_metrics.items():
        # Calculate mean and std for the current model
        mean_metrics = np.mean(values, axis=0)
        std_metrics = np.std(values, axis=0)

        # Create a subdirectory for the current model
        model_dir = os.path.join(output_dir, name)
        os.makedirs(model_dir, exist_ok=True)

        # Write metrics to a file within the model's directory
        output_file = os.path.join(model_dir, f'metrics_{name}.txt')
        with open(output_file, 'w') as f:
            f.write(f"Model: {name}\n")
            f.write("Metric\tMean\tStd\n")
            for metric_name, (mean, std) in zip(['BLEU', 'GLEU', 'METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L'], zip(mean_metrics, std_metrics)):
                f.write(f"{metric_name}\t{mean:.4f}\t{std:.4f}\n")

def plot_metrics(all_metrics, model_names, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    epochs = ['3', '5', '10', '12', '15']

    for metric_idx, metric_name in enumerate(['BLEU', 'GLEU', 'METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']):
        plt.figure(figsize=(10, 6))
        for name in model_names:
            metric_values = np.array([metrics[metric_idx] for metrics in all_metrics[name]])
            plt.plot(range(len(metric_values)), metric_values, label=name, alpha=0.5)
        
        mean_values = np.mean([np.array([metrics[metric_idx] for metrics in all_metrics[name]]) for name in model_names], axis=0)
        std_values = np.std([np.array([metrics[metric_idx] for metrics in all_metrics[name]]) for name in model_names], axis=0)

        plt.plot(epochs, mean_values, color='black', label='Mean')
        plt.fill_between(epochs, mean_values - std_values, mean_values + std_values, color='gray', alpha=0.2)
        plt.title(f'{metric_name} across Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f'{metric_name.lower()}_epochs.png'))
        plt.close()

def perform_statistical_tests(all_metrics):
    print("\nStatistical Tests")
    for metric_idx, metric_name in enumerate(['BLEU', 'GLEU', 'METEOR', 'ROUGE-1', 'ROUGE-2', 'ROUGE-L']):
        print(f"\n{metric_name} Comparison:")
        metric_values = {name: [metrics[metric_idx] for metrics in values] for name, values in all_metrics.items()}
        
        for name1 in metric_values:
            for name2 in metric_values:
                if name1 >= name2:
                    continue
                w_stat, w_pval = wilcoxon(metric_values[name1], metric_values[name2])
                print(f"{name1} vs {name2}: Wilcoxon W={w_stat:.4f}, p={w_pval:.4e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_experiments", type=int, help="Number of experiments to run")
    args = parser.parse_args()

    main(args.num_experiments)