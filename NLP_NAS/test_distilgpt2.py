import os
import time
import nltk
import torch
import argparse
import numpy as np

from tqdm import tqdm
from threading import Thread
from scipy.stats import wilcoxon
from jtop_stats import jtop_stats
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer

nltk.download('punkt')
nltk.download('wordnet')

def cut_text_for_generation(text, cut_percentage=0.3):
    cut_point = int(len(text) * (1 - cut_percentage))
    return text[:cut_point], text[cut_point:]

def generate_text(input_ids, attention_mask, model, tokenizer, temperature=1, top_k=None, top_p=1):
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
            temperature=temperature,
            top_k=top_k,
            top_p=top_p
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

def generate_texts(dataset, model, tokenizer, device, temperature=1, top_k=None, top_p=1):
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
        generated_text = generate_text(input_ids, attention_mask, model, tokenizer, temperature, top_k, top_p)
        
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

def save_generated_texts(model_name, generated_texts, references, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{model_name}.txt")

    with open(output_file, 'w') as f:
        f.write("Generated Texts and References\n")
        for generated, reference in zip(generated_texts, references):
            f.write(f"Generated: {generated}\n")
            f.write(f"Reference: {reference}\n")
            f.write("\n")

def read_hardware_metrics(file_path):
    metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            metrics[key] = float(value)
    return metrics

def save_hardware_metrics_summary(hardware_metrics, output_dir):
    for model_name, metrics_list in hardware_metrics.items():
        raw_data_file = os.path.join(output_dir, model_name, 'raws', f'{model_name}_raw_{time.time()}.txt')
        with open(raw_data_file, 'w') as file:
            # Combine metrics from all experiments for the model
            aggregated_metrics = {}
            for metrics in metrics_list:
                for key, value in metrics.items():
                    aggregated_metrics.setdefault(key, []).append(value)
            for key, values in aggregated_metrics.items():
                mean_val = np.mean(values)
                file.write(f'{key}: {mean_val:.4f}\n')

def perform_statistical_tests_for_hardware_metrics(hardware_metrics):
    print("\nHardware Metrics Statistical Tests")
    metrics_names = next(iter(hardware_metrics.values()))[0].keys()  # Get metric names from the first model's first experiment

    for metric_name in metrics_names:
        print(f"\n{metric_name} Comparison:")
        metric_values = {
            name: [metrics[metric_name] for metrics in metrics_list]
            for name, metrics_list in hardware_metrics.items()
        }

        for model1 in metric_values:
            for model2 in metric_values:
                if model1 >= model2:
                    continue
                try:
                    w_stat, w_pval = wilcoxon(metric_values[model1], metric_values[model2])
                    print(f"{model1} vs {model2}: Wilcoxon W={w_stat:.4f}, p={w_pval:.4e}")
                except ValueError:
                    print(f"Skipping {model1} vs {model2}: Not enough paired samples for Wilcoxon test.")

def main(num_experiments, use_small_dataset=False, temperature=1, top_k=None, top_p=1):
    # Load models and tokenizer
    print('Loading models...')
    model_paths = ['models/NAS_DGPT2/model_1', 'models/NAS_DGPT2/model_2', 
                   'models/NAS_DGPT2/model_3', 'models/NAS_DGPT2/model_4']
    models = [GPT2LMHeadModel.from_pretrained(path, local_files_only=True) for path in model_paths]
    model_names = ['NAS_DGPT2_model_1', 'NAS_DGPT2_model_2', 'NAS_DGPT2_model_3', 'NAS_DGPT2_model_4']
    
    print('Loading tokenizer...')
    tokenizer = GPT2Tokenizer.from_pretrained('tokenizers/distilgpt2', padding_side='left')
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
    
    if use_small_dataset:
        dataset = dataset.select(range(200))
    else:
        dataset = dataset.shuffle(seed=42).select(range(1000))

    print('Tokenizing dataset...')
    tokenized_test_dataset = dataset.map(lambda x: tokenize(x, tokenizer=tokenizer), batched=True, remove_columns=['text'])
    tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Ensure the output directories exist
    for model_name in model_names:
        os.makedirs(f'results/NAS_DGPT2/{model_name}', exist_ok=True)
        os.makedirs(f'stat_dumps/NAS_DGPT2/{model_name}', exist_ok=True)

    # Metrics storage
    hardware_metrics = {name: [] for name in model_names}

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
                generated_texts, references = generate_texts(tokenized_test_dataset, model, tokenizer, device, temperature, top_k, top_p)

                # Save hardware metrics
                stats.stop_thread = True
                delta_thread.join()
                dump_file = f'stat_dumps/NAS_DGPT2/{name}/metrics_{name}_exp{exp + 1}_{time.time()}.txt'
                raw_file = f'stat_dumps/NAS_DGPT2/{name}/metrics_{name}_exp{exp + 1}_raw_{time.time()}.txt'
                stats.dump_deltas(dump_file, raw_file)
                hardware_metrics[name].append(read_hardware_metrics(raw_file))
            
                # Save generated texts and references
                save_generated_texts(f'{name}_exp{exp + 1}_{time.time()}', generated_texts, references, f'results/NAS_DGPT2/{name}')

    # Save hardware metrics
    save_hardware_metrics_summary(hardware_metrics, 'stat_dumps/NAS_DGPT2/')

    # Perform statistical tests
    perform_statistical_tests_for_hardware_metrics(hardware_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("num_experiments", type=int, help="Number of experiments to run")
    parser.add_argument("--use_small_dataset", type=bool, default=False, help="Whether to use a reduced version of the dataset or not")
    parser.add_argument("--temp", type=float, default=1.0, help="Sets the temperature for text generation")
    parser.add_argument("--topk", type=float, default=None, help="Sets the top_k value for text generation")
    parser.add_argument("--topp", type=float, default=1.0, help="Sets the top_p value for text generation")
    args = parser.parse_args()

    main(args.num_experiments, args.use_small_dataset, args.temp, args.topk, args.topp)