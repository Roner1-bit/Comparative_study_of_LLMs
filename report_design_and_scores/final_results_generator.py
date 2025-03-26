import os
import csv

model_encodings = {
    "DeepSeek-VL2-small-16B": "Model A",
    "DeepSeek-VL2-tiny-3B": "Model B",
    "DeepSeek-VL2 27B": "Model C",
    "Meta-Vision3.2-11B": "Model E",
    "Llava-Med 1.5 7B": "Model F",
    "NVLM-D-72B": "Model H",
    "paligemma2-10B-pt-896": "Model I",
    "paligemma2-28B-pt-896": "Model J",
    "paligemma2-3B-pt-896": "Model K",
    "phi-3.5-vision-instruct-4B": "Model L",
    "Qwen2-VL-2B": "Model M",
    "Qwen2-VL-72B": "Model N",
    "Qwen2-VL-7B": "Model O",
    "Meta-Vision 3.2-90B": "Model Z"
}

# Invert the dictionary so that keys become values and vice-versa.
reversed_model_encodings = {v: k for k, v in model_encodings.items()}

# Define the file paths.
csv_file_path = os.path.join(
    r'D:\final_drive_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250218T095105Z-001\plots',
    'combined_scores_o1.csv'
)
output_file_path = os.path.join(
    r'D:\final_drive_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250218T095105Z-001\plots',
    'evaluator_1.txt'
)

# The score categories that contribute to a total of 40.
score_keys = ["Correctness", "Conciseness", "Completeness", "Medical-Images-Description"]

# Dictionaries to group totals and row data by descriptive model.
model_data = {}
model_rows = {}

with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Skip rows without a valid "Case" (which includes the final averages).
        if not row.get("Case") or row["Case"].strip() == "":
            print("Skipping row without a valid case:", row)
            continue
        # Map the CSV model value to its descriptive name using the model_encodings.
        model_key = row["Model"]
        human_model = model_encodings.get(model_key, model_key)
        try:
            # Convert score values to float.
            scores = [float(row[key]) for key in score_keys]
        except ValueError:
            # Skip rows with non-numeric values.
            print("Skipping row with non-numeric values:", row)
            continue

        # Calculate the total score.
        total = sum(scores)
        if human_model not in model_data:
            model_data[human_model] = []
        model_data[human_model].append(total)

        # Save the row data for later individual CSV generation.
        if human_model not in model_rows:
            model_rows[human_model] = []
        # Save only the desired columns.
        model_rows[human_model].append({
            "Case": row["Case"],
            "Correctness": row["Correctness"],
            "Conciseness": row["Conciseness"],
            "Completeness": row["Completeness"],
            "Medical-Images-Description": row["Medical-Images-Description"],
            "Total Score": total
        })

# Calculate average and accuracy for each model.
results = {}
for model, totals in model_data.items():
    if totals:
        avg_total = sum(totals) / len(totals)
        accuracy = (avg_total / 40) * 100  # Convert average to an accuracy percentage.
        results[model] = accuracy
    else:
        results[model] = None

# Write the aggregated model accuracies to a text file.
with open(output_file_path, 'w', encoding='utf-8') as f:
    for model, accuracy in results.items():
        if accuracy is not None:
            f.write(f"{model}: {accuracy:.2f}%\n")
        else:
            f.write(f"{model}: No valid data\n")

print("Model accuracies written to", output_file_path)

# Define an output directory for the separate CSV files.
output_dir = os.path.dirname(output_file_path)

# Save a separate CSV for every model using the inverted mapping for file naming.
for human_model, rows in model_rows.items():
    # Use the inverted mapping to get the original file name if available.
    original_model_id = reversed_model_encodings.get(human_model, human_model)
    model_csv_filename = f"{original_model_id.replace('.txt', '')}_results_o1.csv"
    model_csv_path = os.path.join(output_dir, model_csv_filename)
    
    # Write the rows for this model.
    with open(model_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ["Case", "Correctness", "Conciseness", "Completeness", "Medical-Images-Description", "Total Score"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in rows:
            writer.writerow(data)
    
    print("Results for", human_model, "written to", model_csv_path)