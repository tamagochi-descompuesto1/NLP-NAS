import os
import nltk
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from threading import Thread
from datasets import load_dataset
from jtop_stats import jtop_stats
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

def cut_text_for_generation(text, cut_percentage=0.3):
    cut_point = int(len(text) * (1 - cut_percentage))
    return text[:cut_point], text[cut_point:]

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

def tokenize(examples):
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

def read_metrics_from_file(file_path, bleu_array, gleu_array, meteor_array, rouge1_array, rouge2_array, rougeL_array):
    if os.path.isfile(metrics_file):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            bleu_array.append(float(lines[0].strip()))
            gleu_array.append(float(lines[1].strip()))
            meteor_array.append(float(lines[2].strip()))
            rouge1_array.append(float(lines[3].strip()))
            rouge2_array.append(float(lines[4].strip()))
            rougeL_array.append(float(lines[5].strip()))

# Function to read raw data into a DataFrame and tag with model name
def read_raw_data(file_path, model_name):
    data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            key, value = line.strip().split(": ")
            data[key] = float(value)
    df = pd.DataFrame([data])
    df['model'] = model_name  # Add the model name as a column
    return df

# Load the raw data for all models
def load_data_for_all_models(model_names, base_dir):
    all_data = []
    for model_name in model_names:
        raw_data_path = os.path.join(base_dir, model_name, f'{model_name}_raw.txt')
        df = read_raw_data(raw_data_path, model_name)
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Plot with shaded confidence interval (line plot with variance)
def plot_with_confidence(df, metric, title, ylabel):
    plt.figure(figsize=(10, 6))
    for model in df['model'].unique():
        model_df = df[df['model'] == model]
        plt.plot(model_df.index, model_df[metric], label=model, marker='o')
        plt.fill_between(model_df.index, model_df[metric] - model_df[metric].std(), 
                         model_df[metric] + model_df[metric].std(), alpha=0.2)
    plt.title(title)
    plt.xlabel('Time Steps')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Box plot to show the distribution of values across models
def plot_boxplot(df, metric, title, ylabel):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='model', y=metric, data=df)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

# Heatmap for model vs metric performance
def plot_heatmap(df, title):
    metrics = ['Power_voltage_mV_delta', 'Power_current_mA_delta', 'Power_avg_power_mW_delta',
               'Memory_used_KB_delta', 'GPU_load_delta', 'CPU_user_delta', 'CPU_system_delta']
    pivot_df = df.groupby('model')[metrics].mean()
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_df, annot=True, cmap="YlGnBu", linewidths=.5)
    plt.title(title)
    plt.show()

# Correlation matrix of metrics
def plot_correlation_matrix(df):
    metrics = ['Power_voltage_mV_delta', 'Power_current_mA_delta', 'Power_avg_power_mW_delta',
               'Memory_used_KB_delta', 'GPU_load_delta', 'CPU_user_delta', 'CPU_system_delta']
    corr_matrix = df[metrics].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", linewidths=.5)
    plt.title('Correlation Matrix of Metrics')
    plt.show()

# Calculate peak, average, and total metrics
def calculate_performance_metrics(df):
    summary = df.groupby('model').agg({
        'Power_avg_power_mW_delta': ['mean', 'max'],
        'Memory_used_KB_delta': ['mean', 'max'],
        'GPU_load_delta': ['mean', 'max'],
        'CPU_user_delta': ['mean', 'max']
    })
    print("Performance Metrics Summary:")
    print(summary)

# Main function to load data and plot comparisons
def compare_models(model_names, base_dir):
    # Load data from all models
    df = load_data_for_all_models(model_names, base_dir)

    # Plot line charts with confidence intervals
    plot_with_confidence(df, 'Power_avg_power_mW_delta', 'Average Power Consumption with Variance', 'Power (mW)')
    plot_with_confidence(df, 'Memory_used_KB_delta', 'Memory Usage with Variance', 'Memory Used (KB)')
    plot_with_confidence(df, 'GPU_load_delta', 'GPU Load with Variance', 'GPU Load')
    plot_with_confidence(df, 'CPU_user_delta', 'CPU User Utilization with Variance', 'CPU User (%)')

    # Boxplots to visualize distribution across models
    plot_boxplot(df, 'Power_avg_power_mW_delta', 'Distribution of Average Power by Model', 'Power (mW)')
    plot_boxplot(df, 'Memory_used_KB_delta', 'Distribution of Memory Used by Model', 'Memory Used (KB)')
    
    # Heatmap of model vs performance metrics
    plot_heatmap(df, 'Model vs Performance Metrics')

    # Correlation matrix for metrics
    plot_correlation_matrix(df)

    # Calculate peak, average, and total performance metrics
    calculate_performance_metrics(df)

# Load models
print('Loading models...')
model_paths = [
    'models/distilgpt2/distilgpt2_3epochs',
    'models/distilgpt2/distilgpt2_5epochs',
    'models/distilgpt2/distilgpt2_10epochs',
    'models/distilgpt2/distilgpt2_12epochs',
    'models/distilgpt2/distilgpt2_15epochs'
]
models = [GPT2LMHeadModel.from_pretrained(path, local_files_only=True) for path in model_paths]
model_names = ['distilgpt2_3epochs', 'distilgpt2_5epochs', 'distilgpt2_10epochs', 'distilgpt2_12epochs', 'distilgpt2_15epochs']

# Load tokenizer
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

# Load and tokenize dataset
print('Loading WikiText...')
wikitext_2 = load_dataset('wikitext', 'wikitext-2-raw-v1', verification_mode='no_checks')
test_dataset = wikitext_2['test']

print('Tokenizing dataset...')
tokenized_test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['text'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Ensure the output directories exist
for model_name in model_names:
    os.makedirs(f'results/{model_name}', exist_ok=True)
    os.makedirs(f'stat_dumps/distilgpt2/{model_name}', exist_ok=True)

# Testing models
print('Testing models...')

for index, model in enumerate(models):
    with jtop_stats.JtopStats() as stats:
        # Start continuous delta calculation in a separate thread
        delta_thread = Thread(target=stats.calculate_deltas_periodically, args=(30,))
        delta_thread.start()

        # Generate texts using GPU
        print(f'Generating texts for model {model_names[index]}...')
        generated_texts, references = generate_texts(tokenized_test_dataset, model, tokenizer, device)

        # Dump stats
        stats.stop_thread = True
        delta_thread.join()
        stats.dump_deltas(f'stat_dumps/distilgpt2/{model_names[index]}/{model_names[index]}.txt', f'stat_dumps/distilgpt2/{model_names[index]}/{model_names[index]}_raw.txt')

    # Calculate BLEU, METEOR, ROUGE scores
    print(f'Calculating BLEU, METEOR, ROUGE for model {model_names[index]}...')
    avg_bleu, avg_gleu, avg_meteor, avg_rouge1, avg_rouge2, avg_rougeL = calculate_bleu_meteor_rouge(generated_texts, references)

    print(f'Model {model_names[index]} --- BLEU: {avg_bleu:.4f}, GLEU: {avg_gleu:.4f}, METEOR: {avg_meteor:.4f}, ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}')

    # Save metrics to file
    metrics_file_path = f'results/distilgpt2/{model_names[index]}/metrics_{model_names[index]}.txt'
    with open(metrics_file_path, 'w') as file:
        file.write(f"{avg_bleu}\n{avg_gleu}\n{avg_meteor}\n{avg_rouge1}\n{avg_rouge2}\n{avg_rougeL}\n")

print('Test finished!')

# Loading metrics for plotting
print('Loading metrics...')
BLEU_arr = []
GLEU_arr = []
METEOR_arr = []
ROUGE_1_arr = []
ROUGE_2_arr = []
ROUGE_L_arr = []
root_dir = 'results/distilgpt2'
raw_data_dir = 'stat_dumps/distilgpt2'

for folder in model_names:
    metrics_file = os.path.join(root_dir, folder, f'metrics_{folder}.txt')
    read_metrics_from_file(metrics_file, BLEU_arr, GLEU_arr, METEOR_arr, ROUGE_1_arr, ROUGE_2_arr, ROUGE_L_arr)

print('BLEU scores:', BLEU_arr)
print('GLEU scores:', GLEU_arr)
print('METEOR scores:', METEOR_arr)
print('ROUGE-1 scores:', ROUGE_1_arr)
print('ROUGE-2 scores:', ROUGE_2_arr)
print('ROUGE-L scores:', ROUGE_L_arr)

# Plotting
print('Plotting...')
epochs = ['3', '5', '10', '12', '15']
plot_dir = 'results/distilgpt2/plots'
os.makedirs(plot_dir, exist_ok=True)

# Epochs vs BLEU
plt.figure(figsize=(10, 6))
plt.plot(epochs, BLEU_arr, marker='o', linestyle='-', color='b')
plt.xlabel('Epochs')
plt.ylabel('BLEU')
plt.title('Epochs vs BLEU')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'epochs_vs_bleu.png'))
plt.show()

# Epochs vs BLEU
plt.figure(figsize=(10, 6))
plt.plot(epochs, GLEU_arr, marker='o', linestyle='-', color='b')
plt.xlabel('Epochs')
plt.ylabel('GLEU')
plt.title('Epochs vs GLEU')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'epochs_vs_gleu.png'))
plt.show()

# Epochs vs METEOR
plt.figure(figsize=(10, 6))
plt.plot(epochs, METEOR_arr, marker='o', linestyle='-', color='g')
plt.xlabel('Epochs')
plt.ylabel('METEOR')
plt.title('Epochs vs METEOR')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'epochs_vs_meteor.png'))
plt.show()

# Epochs vs ROUGE-1
plt.figure(figsize=(10, 6))
plt.plot(epochs, ROUGE_1_arr, marker='o', linestyle='-', color='r')
plt.xlabel('Epochs')
plt.ylabel('ROUGE-1')
plt.title('Epochs vs ROUGE-1')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'epochs_vs_rouge1.png'))
plt.show()

# Epochs vs ROUGE-2
plt.figure(figsize=(10, 6))
plt.plot(epochs, ROUGE_2_arr, marker='o', linestyle='-', color='r')
plt.xlabel('Epochs')
plt.ylabel('ROUGE-2')
plt.title('Epochs vs ROUGE-2')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'epochs_vs_rouge2.png'))
plt.show()

# Epochs vs ROUGE-L
plt.figure(figsize=(10, 6))
plt.plot(epochs, ROUGE_L_arr, marker='o', linestyle='-', color='r')
plt.xlabel('Epochs')
plt.ylabel('ROUGE-L')
plt.title('Epochs vs ROUGE-L')
plt.grid(True)
plt.savefig(os.path.join(plot_dir, 'epochs_vs_rougeL.png'))
plt.show()

compare_models(model_names, raw_data_dir)

print('Plotting completed!')