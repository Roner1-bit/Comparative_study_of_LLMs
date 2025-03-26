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

def flexible_pattern(key):
    # Replace any group of hyphen(s) or space(s) with the flexible pattern "[-\s]*"
    return re.sub(r'[-\s]+', lambda m: "[-\\s]*", key)

def extract_scores(response_text):
    # Initialize with default values
    keys = ["Correctness", "Conciseness", "Completeness", "Medical-Images-Description"]
    scores = {key: "--" for key in keys}
    scores["Overall Score"] = "--"
    
    # Split response text into lines and clean markdown symbols.
    lines = response_text.splitlines()
    for line in lines:
        # Remove markdown symbols and extra dashes.
        clean_line = re.sub(r"[\*\-]+", "", line).strip()
        for key in keys:
            pattern_key = flexible_pattern(key)
            # Attempt plain number extraction.
            # This pattern matches an optional '[' before the number and an optional ']' after the number,
            # and avoids matching if immediately followed by "/" (i.e. a fraction) or a closing bracket.
            plain_regex = re.compile(rf"{pattern_key}:\s*\[?([\d.]+)(?!\s*(?:/|\]))", re.IGNORECASE)
            m_plain = plain_regex.search(clean_line)
            if m_plain:
                candidate = m_plain.group(1)
                scores[key] = float(candidate) if "." in candidate else int(candidate)
                continue
            # Fallback: try fraction extraction (e.g., "[3/10]" or "3/10")
            frac_regex = re.compile(rf"{pattern_key}:\s*\[?(\d+)\s*/\s*\d+\]?", re.IGNORECASE)
            m_frac = frac_regex.search(clean_line)
            if m_frac:
                scores[key] = int(m_frac.group(1))
                
    # Process Overall Score using percentage extraction on each cleaned line.
    for line in lines:
        clean_line = re.sub(r"[\*\-]+", "", line).strip()
        if "Overall Score" in clean_line:
            overall_regex = re.compile(r"Overall Score[:\s]*([\d.]+)%", re.IGNORECASE)
            m_overall = overall_regex.search(clean_line)
            if m_overall:
                candidate = m_overall.group(1)
                scores["Overall Score"] = float(candidate) if "." in candidate else int(candidate)
                break
        elif "accuracy score" in clean_line:
            acc_regex = re.compile(r"accuracy score of\s*([\d.]+)%", re.IGNORECASE)
            m_acc = acc_regex.search(clean_line)
            if m_acc:
                candidate = m_acc.group(1)
                scores["Overall Score"] = float(candidate) if "." in candidate else int(candidate)
                break
    return scores

# Initialize a dictionary to store scores for each model.
model_scores = {model: {"Correctness": [], "Conciseness": [], "Completeness": [], "Medical-Images-Description": [], "Overall Score": []} for model in model_encodings.values()}

# Initialize a list to store individual patient scores.
patient_scores = []

for case in os.listdir(cases_dir):
    case_dir = os.path.join(cases_dir, case)
    for model_file, model_name in model_encodings.items():
        plot_file_path = os.path.join(plots_dir, f'{model_file.split(".")[0]}', f'{case}', f'{case}_{model_file.split(".")[0]}_r1-score.txt')
        if os.path.exists(plot_file_path):
            with open(plot_file_path, 'r', encoding='utf-8') as f:
                o1_response = f.read()
            # Extract scores using the updated extraction method.
            scores = extract_scores(o1_response)
            for key in scores.keys():
                if scores[key] != "--":
                    model_scores[model_name][key].append(scores[key])
            # Record individual patient scores.
            patient_scores.append({
                "Case": case,
                "Model": model_name,
                **scores
            })

# Calculate average scores per model.
average_scores = []
for model, score_dict in model_scores.items():
    avg_scores = {key: (sum(values) / len(values) if values else "--") for key, values in score_dict.items()}
    avg_scores['Model'] = model
    average_scores.append(avg_scores)

# Write both individual patient scores and average scores to a CSV.
csv_file_path = os.path.join(plots_dir, 'r1_combined_scores.csv')
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['Case', 'Model', 'Correctness', 'Conciseness', 'Completeness', 'Medical-Images-Description', 'Overall Score']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for score in patient_scores:
        writer.writerow(score)
    writer.writerow({})  # Separator row.
    for avg_score in average_scores:
        writer.writerow(avg_score)

print("Scores extracted, averages calculated, and results written to combined_scores.csv.")