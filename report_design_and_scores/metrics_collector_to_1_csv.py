import os
import re
import pandas as pd

# Base folder where the plots subfolders are located
plots_base_dir =  r'D:\final_drive\(final)merged_images_with_labels_order_and_folders_mask_normalized\plots'

# List to store metrics from each model
metrics_list = []

# Regular expressions to parse each metric line
regexes = {
    'model': re.compile(r"Model:\s*(.+)"),
    'avg_bleu': re.compile(r"Average BLEU Score:\s*([\d\.]+)"),
    'avg_rouge1': re.compile(r"Average ROUGE-1 F1:\s*([\d\.]+)"),
    'avg_rouge2': re.compile(r"Average ROUGE-2 F1:\s*([\d\.]+)"),
    'avg_rougeL': re.compile(r"Average ROUGE-L F1:\s*([\d\.]+)"),
    'avg_rougeLsum': re.compile(r"Average ROUGE-Lsum F1:\s*([\d\.]+)"),
    'avg_cosine': re.compile(r"Average Cosine Similarity:\s*([\d\.]+)"),
    'avg_meteor': re.compile(r"Average METEOR Score:\s*([\d\.]+)"),
    'avg_perplexity': re.compile(r"Average Perplexity:\s*([\d\.]+)"),
    'phrase_overlap_precision': re.compile(r"Phrase Overlap Precision:\s*([\d\.]+)"),
    'phrase_overlap_recall': re.compile(r"Phrase Overlap Recall:\s*([\d\.]+)"),
    'phrase_overlap_f1': re.compile(r"Phrase Overlap F1:\s*([\d\.]+)"),
    'phrase_overlap_accuracy': re.compile(r"Phrase Overlap Accuracy:\s*([\d\.]+)")
}

# Go through each subfolder inside the plots folder
if os.path.exists(plots_base_dir):
    for subfolder in os.listdir(plots_base_dir):
        subfolder_path = os.path.join(plots_base_dir, subfolder)
        if os.path.isdir(subfolder_path):
            scores_file = os.path.join(subfolder_path, 'scores.txt')
            if os.path.exists(scores_file):
                metrics = {}
                with open(scores_file, 'r') as f:
                    lines = f.read().splitlines()
                for line in lines:
                    for key, regex in regexes.items():
                        match = regex.search(line)
                        if match:
                            # For non-model fields, convert to float
                            if key == 'model':
                                metrics[key] = match.group(1).strip()
                            else:
                                try:
                                    metrics[key] = float(match.group(1))
                                except ValueError:
                                    metrics[key] = None
                # Check that at least model and one metric were found
                if 'model' in metrics:
                    metrics_list.append(metrics)
                else:
                    print(f"Warning: Could not find model name in {scores_file}")
            else:
                print(f"Warning: scores.txt not found in {subfolder_path}")
else:
    print(f"Error: {plots_base_dir} does not exist.")

# Define the desired column order
columns = [
    'model', 'avg_bleu', 'avg_rouge1', 'avg_rouge2', 'avg_rougeL', 'avg_rougeLsum',
     'avg_cosine', 'avg_meteor', 'avg_perplexity',
    'phrase_overlap_precision', 'phrase_overlap_recall', 'phrase_overlap_f1', 'phrase_overlap_accuracy'
]

if metrics_list:
    df = pd.DataFrame(metrics_list)
    df = df[columns]  # Reorder columns if necessary

    output_csv_path = os.path.join(plots_base_dir, 'models_metrics_consolidated.csv')
    df.to_csv(output_csv_path, index=False)
    print(f"Consolidated metrics CSV saved at: {output_csv_path}")
else:
    print("No metrics data found to consolidate.")