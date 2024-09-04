import os
import nltk
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from datasets import load_dataset
from jtop_stats import jtop_stats
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Global arrays to store metrics
BLEU_arr = []
METEOR_arr = []
ROUGE_L_arr = []

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

def generate_texts(dataset, model, tokenizer, device):
    generated_texts = []
    references = []

    for example in tqdm(dataset, desc="Generating Texts on GPU"):
        input_ids = example['input_ids'].unsqueeze(0).to(device)
        attention_mask = example['attention_mask'].unsqueeze(0).to(device)
        generated_text = generate_text(input_ids, attention_mask, model, tokenizer, device)

        if generated_text.strip():
            generated_texts.append(generated_text)
            original_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            references.append(original_text)
    
    return generated_texts, references

def generate_text(input_ids, attention_mask, model, tokenizer, device):
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

        # METEOR score calculation (requires tokenized references and hypotheses)
        meteor = meteor_score([reference_tokens], generated_tokens)
        meteor_scores.append(meteor)

    # ROUGE-L score calculation
    for reference, generated_text in zip(references, generated_texts):
        rouge_scores = scorer.score(reference, generated_text)
        rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
        rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
        rougeL_scores.append(rouge_scores['rougeL'].fmeasure)

    return np.mean(bleu_scores), np.mean(meteor_scores), np.mean(rouge1_scores), np.mean(rouge2_scores), np.mean(rougeL_scores)

def read_metrics_from_file(file_path, precision_array, recall_array, F1_array):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        precision_array.append(float(lines[0].strip()))
        recall_array.append(float(lines[1].strip()))
        F1_array.append(float(lines[2].strip()))

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
    stats = jtop_stats.JtopStats()
    stats.set_stats()

    # Generate texts using GPU
    print(f'Generating texts for model {model_names[index]}...')
    generated_texts, references = generate_texts(tokenized_test_dataset, model, tokenizer, device)

    # Dump stats
    stats.dump_deltas(f'stat_dumps/distilgpt2/{model_names[index]}/{model_names[index]}.txt')

    # Calculate BLEU, METEOR, ROUGE scores
    print(f'Calculating BLEU, METEOR, ROUGE for model {model_names[index]}...')
    avg_bleu, avg_meteor, avg_rouge1, avg_rouge2, avg_rougeL = calculate_bleu_meteor_rouge(generated_texts, references)

    print(f'Model {model_names[index]} --- BLEU: {avg_bleu:.4f}, METEOR: {avg_meteor:.4f}, ROUGE-1: {avg_rouge1:.4f}, ROUGE-2: {avg_rouge2:.4f}, ROUGE-L: {avg_rougeL:.4f}')

    # Save metrics to file
    metrics_file_path = f'results/distilgpt2/{model_names[index]}/metrics_{model_names[index]}.txt'
    with open(metrics_file_path, 'w') as file:
        file.write(f"{avg_bleu}\n{avg_meteor}\n{avg_rouge1}\n{avg_rouge2}\n{avg_rougeL}\n")

print('Test finished!')

# Loading metrics for plotting
print('Loading metrics...')
BLEU_arr = []
METEOR_arr = []
ROUGE_1_arr = []
ROUGE_2_arr = []
ROUGE_L_arr = []
root_dir = 'results/distilgpt2'

for folder in model_names:
    metrics_file = os.path.join(root_dir, folder, f'metrics_{folder}.txt')
    if os.path.isfile(metrics_file):
        with open(metrics_file, 'r') as file:
            lines = file.readlines()
            BLEU_arr.append(float(lines[0].strip()))
            METEOR_arr.append(float(lines[1].strip()))
            ROUGE_1_arr.append(float(lines[2].strip()))
            ROUGE_2_arr.append(float(lines[3].strip()))
            ROUGE_L_arr.append(float(lines[4].strip()))

print('BLEU scores:', BLEU_arr)
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

print('Plotting completed!')