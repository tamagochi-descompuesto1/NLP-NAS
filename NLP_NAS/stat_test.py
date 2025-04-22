import os
import numpy as np
from scipy.stats import wilcoxon

def read_hardware_metrics(file_path):
    metrics = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split(':')
            metrics[key] = float(value)
    return metrics

def perform_statistical_tests_for_hardware_metrics(hardware_metrics):
    print("\nHardware Metrics Statistical Tests")

    filtered_metrics = {k: v for k, v in hardware_metrics.items() if len(v) > 0}
    if not filtered_metrics:
        print("❌ No hay métricas cargadas para ningún modelo.")
        return

    try:
        metrics_names = next(iter(filtered_metrics.values()))[0].keys()
    except (IndexError, StopIteration):
        print("❌ No se pudieron obtener nombres de métricas.")
        return

    for metric_name in metrics_names:
        print(f"\n{metric_name} Comparison:")
        metric_values = {
            name: [metrics[metric_name] for metrics in metrics_list]
            for name, metrics_list in filtered_metrics.items()
        }

        for model1 in metric_values:
            for model2 in metric_values:
                if model1 >= model2:
                    continue
                try:
                    w_stat, w_pval = wilcoxon(metric_values[model1], metric_values[model2])
                    print(f"{model1} vs {model2}: Wilcoxon W={w_stat:.4f}, p={w_pval:.4e}")
                except ValueError:
                    print(f"⚠️ Skipping {model1} vs {model2}: Not enough paired samples.")

# 👇 Corrección: buscar archivos en el subfolder `raws`
root_dir = 'stat_dumps/NAS_DGPT2'
model_names = ['NAS_DGPT2_model_1', 'NAS_DGPT2_model_2', 'NAS_DGPT2_model_3', 'NAS_DGPT2_model_4']
hardware_metrics = {}

for model in model_names:
    model_raw_dir = os.path.join(root_dir, model, 'raws')
    hardware_metrics[model] = []

    if not os.path.exists(model_raw_dir):
        print(f"❌ No se encontró el directorio {model_raw_dir}")
        continue

    for file_name in os.listdir(model_raw_dir):
        if file_name.endswith('.txt') and '_raw_' in file_name:
            file_path = os.path.join(model_raw_dir, file_name)
            print(f"📄 Cargando: {file_path}")
            try:
                metrics = read_hardware_metrics(file_path)
                hardware_metrics[model].append(metrics)
            except Exception as e:
                print(f"⚠️ Error al leer {file_path}: {e}")

# Ejecutar pruebas estadísticas
perform_statistical_tests_for_hardware_metrics(hardware_metrics)