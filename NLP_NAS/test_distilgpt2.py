import os
import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from bert_score import score
from datasets import load_dataset
from jtop_stats import jtop_stats
from torch.cuda.amp import autocast
from transformers import GPT2LMHeadModel, GPT2Tokenizer

global P_arr
global R_arr
global F1_arr

def generate_text(input_ids, attention_mask, model, tokenizer, device):
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
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

def read_metrics_from_file(file_path, precision_array, recall_array, F1_array):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        precision_array.append(float(lines[0].strip()))
        recall_array.append(float(lines[1].strip()))
        F1_array.append(float(lines[2].strip()))


print('Loading models...')
distilgpt2_3epochs = GPT2LMHeadModel.from_pretrained('models/distilgpt2/distilgpt2_3epochs', local_files_only=True)
distilgpt2_5epochs = GPT2LMHeadModel.from_pretrained('models/distilgpt2/distilgpt2_5epochs', local_files_only=True)
distilgpt2_10epochs = GPT2LMHeadModel.from_pretrained('models/distilgpt2/distilgpt2_10epochs', local_files_only=True)
distilgpt2_12epochs = GPT2LMHeadModel.from_pretrained('models/distilgpt2/distilgpt2_12epochs', local_files_only=True)
distilgpt2_15epochs = GPT2LMHeadModel.from_pretrained('models/distilgpt2/distilgpt2_15epochs', local_files_only=True)

models = [distilgpt2_3epochs, distilgpt2_5epochs, distilgpt2_10epochs, distilgpt2_12epochs, distilgpt2_15epochs]
model_names = ['distilgpt2_3epochs', 'distilgpt2_5epochs', 'distilgpt2_10epochs', 'distilgpt2_12epochs', 'distilgpt2_15epochs']

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', force_download=True, resume_download=None, padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device is {device}')

for model in models:
    model.to(device)

print('Loading WikiText...')
wikitext_2 = load_dataset('wikitext', 'wikitext-2-raw-v1', verification_mode='no_checks')
test_dataset = wikitext_2['test']

print('Tokenizing dataset...')
tokenized_test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=['text'])
tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

print('Testing models...')
for index, model in enumerate(models):
    stats = jtop_stats.JtopStats()
    generated_texts = []
    references = []

    stats.set_stats()
    for i, example in tqdm(enumerate(tokenized_test_dataset), total=len(tokenized_test_dataset), desc=f'Processing {model_names[index]}'):
        input_ids = example['input_ids'].unsqueeze(0)
        attention_mask = example['attention_mask'].unsqueeze(0)
        generated_text = generate_text(input_ids, attention_mask, model, tokenizer, device)
        if generated_text.strip():
            generated_texts.append(generated_text)
            original_text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            references.append(original_text)

    with autocast():
        P, R, F1 = score(generated_texts, references, lang='en', verbose=True, model_type= 'distilbert-base-uncased', device=device)
    print(f'BERTScore   --- Precision: {P.mean().item()}, Recall: {R.mean().item()}, F1: {F1.mean().item()}')

    with open(f'results/{model_names[index]}/metrics_{model_names[index]}.txt', 'w') as file:
        file.write(str(P.mean().item()) + '\n')
        file.write(str(R.mean().item()) + '\n')
        file.write(str(F1.mean().item()) + '\n')
    
    stats.dump_deltas(f'stat_dumps/distilgpt2/{model_names[index]}')

print('Loading metrics...')
P_arr = []
R_arr = []
F1_arr = []
root_dir = 'results/distilgpt2'
for folder in os.list_dir(root_dir):
    results_path = os.path.join(root_dir, folder)
    if os.path.isdir(results_path):
        metrics_file = os.path.join(results_path, f'metrics_{folder}.txt')
        if os.path.isfile(metrics_file):
            read_metrics_from_file(metrics_file, P_arr, R_arr, F1_arr)

print('Precisions:', P_arr)
print('Recalls:', R_arr)
print('F1 scores:', F1_arr)

print('Plotting...')
epochs = ['3', '5','10', '12', '15']
plot_dir = 'results/distilgpt2/plots'

# Epochs vs P
plt.figure(figsize=(10, 6))
plt.plot(epochs, P_arr, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.title('Epochs vs Precision')
plt.savefig(os.path.join(plot_dir, 'epochs_vs_precision.png'))

# Epochs vs R
plt.figure(figsize=(10, 6))
plt.plot(epochs, R_arr, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.title('Epochs vs Recall')
plt.savefig(os.path.join(plot_dir, 'epochs_vs_recall.png'))

# Epochs vs F1
plt.figure(figsize=(10, 6))
plt.plot(epochs, F1_arr, marker='o')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('Epochs vs F1 Score')
plt.savefig(os.path.join(plot_dir, 'epochs_vs_f1.png'))

print('Test finished!')