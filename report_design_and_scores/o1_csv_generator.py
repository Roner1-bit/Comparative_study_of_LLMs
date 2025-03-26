import os
import re
import csv
import docx

cases_dir = r'D:\final_drive_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250218T095105Z-001\(final)merged_images_with_labels_order_and_folders_mask_normalized'
plots_dir = r'D:\final_drive_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250218T095105Z-001\plots'

model_encodings = {
    "deepseek-vl-small.txt": "Model A",
    "deepseek-vl-tiny.txt": "Model B",
    "deepseek-vl2-api.txt": "Model C",
    "llama-11B.txt": "Model E",
    "llava-med-response11.txt": "Model F",
    "nvlm-72b-response.txt": "Model H",
    "paligemma2-10b-report.txt": "Model I",
    "paligemma2-28b-report.txt": "Model J",
    "paligemma2-3b-report.txt": "Model K",
    "phi-3.5-vision-instruct-response.txt": "Model L",
    "qwen-vl-2b-response.txt": "Model M",
    "qwen-vl-72b-response.txt": "Model N",
    "qwen-vl-7b-response.txt": "Model O",
    "meta-vision-90b.txt": "Model Z"
}

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_scores(response_text):
    scores = {
        "Correctness": "--",
        "Conciseness": "--",
        "Completeness": "--",
        "Medical-Images-Description": "--",
        "Overall Score": "--"
    }
    for key in scores.keys():
        match = re.search(rf"{key}: (\d+)", response_text)
        if match:
            scores[key] = int(match.group(1))
    
    # Extract Overall Score separately if not in the same format
    overall_match = re.search(r"Overall Score: (\d+)%", response_text)
    if overall_match:
        scores["Overall Score"] = int(overall_match.group(1))
    else:
        overall_match = re.search(r"accuracy score of (\d+\.?\d*)%", response_text)
        if overall_match:
            scores["Overall Score"] = float(overall_match.group(1))
        else:
            overall_match = re.search(r"\((\d+\.?\d*)%\)", response_text)
            if overall_match:
                scores["Overall Score"] = float(overall_match.group(1))
    
    return scores

# Initialize a dictionary to store scores for each model
model_scores = {model: {"Correctness": [], "Conciseness": [], "Completeness": [], "Medical-Images-Description": [], "Overall Score": []} for model in model_encodings.values()}

# Initialize a list to store individual patient scores
patient_scores = []

for case in os.listdir(cases_dir):
    case_dir = os.path.join(cases_dir, case)

    for model_file, model_name in model_encodings.items():
        plot_file_path = os.path.join(plots_dir, f'{model_file.split(".")[0]}', f'{case}', f'{case}_{model_file.split(".")[0]}_o1-score.txt')
        if os.path.exists(plot_file_path):
            with open(plot_file_path, 'r', encoding='utf-8') as f:
                o1_response = f.read()

            # Extract scores and add to model_scores
            scores = extract_scores(o1_response)
            for key in scores.keys():
                if scores[key] != "--":
                    model_scores[model_name][key].append(scores[key])
            
            # Add individual patient scores to the list
            patient_scores.append({
                "Case": case,
                "Model": model_name,
                **scores
            })

# Calculate average scores
average_scores = []
for model, scores in model_scores.items():
    avg_scores = {key: (sum(values) / len(values) if values else "--") for key, values in scores.items()}
    avg_scores['Model'] = model
    average_scores.append(avg_scores)

# Write individual patient scores and average scores to a single CSV
csv_file_path = os.path.join(plots_dir, 'combined_scores.csv')
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Case', 'Model', 'Correctness', 'Conciseness', 'Completeness', 'Medical-Images-Description', 'Overall Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for score in patient_scores:
        writer.writerow(score)
    
    # Write a separator row
    writer.writerow({})

    # Write average scores
    for avg_score in average_scores:
        writer.writerow(avg_score)

print("Scores extracted, averages calculated, and results written to combined_scores.csv.")