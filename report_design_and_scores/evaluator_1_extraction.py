import os
import re
import csv
import docx
from pathlib import Path

cases_dir = r'E:\final_drive_2\evaluators\Dr. Mousa\Dr. Mohammed Mousa-20250226T082135Z-001\Dr. Mohammed Mousa'
plots_dir = r'E:\final_drive_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250218T095105Z-001\plots\evaluator1'

# Define an accessible output directory
output_dir = r'E:\final_drive_2\evaluators\Dr. Mousa'  # Current script directory
# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Simplified model encodings - using just the base letter/name
model_encodings = {
    "a": "Model A",
    "b": "Model B",
    "c": "Model C",
    "e": "Model E",
    "f": "Model F",
    "h": "Model H",
    "i": "Model I",
    "j": "Model J",
    "k": "Model K",
    "l": "Model L",
    "m": "Model M",
    "n": "Model N",
    "o": "Model O",
    "z": "Model Z"
}

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_scores(response_text):
    scores = {
        "correctness": "--",
        "conciseness": "--",
        "completeness": "--", 
        "Overall Score": "--"
    }
    
    # Split by separator line (consecutive hyphens)
    separator_pattern = r"-{5,}"  # At least 5 consecutive hyphens
    sections = re.split(separator_pattern, response_text)
    
    # Use the section after the separator (if exists)
    if len(sections) > 1:
        feedback_section = sections[-1]  # Take the last section after separator
    else:
        feedback_section = response_text  # Use whole text if no separator found
    
    # Extract scores using more specific patterns that match the actual format
    conciseness_match = re.search(r"conciseness score out of 10:?\s*(\d+(?:\.\d+)?)", 
                                 feedback_section, re.IGNORECASE | re.DOTALL)
    if conciseness_match:
        scores["conciseness"] = float(conciseness_match.group(1))
    
    correctness_match = re.search(r"correctness score out of 10:?\s*(\d+(?:\.\d+)?)", 
                                 feedback_section, re.IGNORECASE | re.DOTALL)
    if correctness_match:
        scores["correctness"] = float(correctness_match.group(1))
    
    completeness_match = re.search(r"completeness score out of 10:?\s*(\d+(?:\.\d+)?)", 
                                  feedback_section, re.IGNORECASE | re.DOTALL)
    if completeness_match:
        scores["completeness"] = float(completeness_match.group(1))
    
    overall_match = re.search(r"Overall score out of 10:?\s*(\d+(?:\.\d+)?)", 
                             feedback_section, re.IGNORECASE | re.DOTALL)
    if overall_match:
        scores["Overall Score"] = float(overall_match.group(1))
    
    return scores

# Initialize a list to store individual file (patient) scores
patient_scores = []

# Check if cases directory exists
if not os.path.exists(cases_dir):
    print(f"Error: Cases directory does not exist: {cases_dir}")
else:
    for case in os.listdir(cases_dir):
        case_dir = os.path.join(cases_dir, case)
        
        # Skip non-directory entries
        if not os.path.isdir(case_dir):
            continue

        for model_id, model_name in model_encodings.items():
            # Look for files named simply with model_id (like "o.docx")
            docx_file_path = os.path.join(cases_dir, case, f"{model_id}.docx")
            if os.path.exists(docx_file_path):
                try:
                    # Use read_docx to extract text from the document
                    response_text = read_docx(docx_file_path)
                    # Extract scores and add to the list
                    scores = extract_scores(response_text)
                    patient_scores.append({
                        "Case": case,
                        "Model": model_name,
                        **scores
                    })
                    print(f"Successfully processed: {docx_file_path}")
                except Exception as e:
                    print(f"Error processing {docx_file_path}: {e}")

# Write individual file scores to CSV in the accessible output directory
csv_file_path = os.path.join(output_dir, 'Dr_mOhamed.csv')
try:
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Case', 'Model', 'correctness', 'conciseness', 'completeness', 'Overall Score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for score in patient_scores:
            writer.writerow(score)
    
    print(f"Scores successfully written to {csv_file_path}")
except Exception as e:
    print(f"Error writing to CSV file: {e}")

print(patient_scores)