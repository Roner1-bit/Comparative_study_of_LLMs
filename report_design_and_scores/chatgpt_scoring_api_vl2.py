#%%
import os
from openai import OpenAI
import docx

client = OpenAI(
    api_key="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",
)

# Directory containing the images
cases_dir = r'D:\final_drive_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250218T095105Z-001\(final)merged_images_with_labels_order_and_folders_mask_normalized'

#%%
modified_folders = []

# Define model encodings
model_encodings = {
    # "deepseek-vl-small.txt": "Model A",
    # "deepseek-vl-tiny.txt": "Model B",
    "deepseek-vl2-api.txt": "Model C",
    # "diagnostic_prompt.txt": "Model D",
    # "llama-11B.txt": "Model E",
    # "llava-med-response11.txt": "Model F",
    # "llava_answer.txt": "Model G", typo
    # "nvlm-72b-response.txt": "Model H",
    # "paligemma2-10b-report.txt": "Model I",
    # "paligemma2-28b-report.txt": "Model J",
    # "paligemma2-3b-report.txt": "Model K",
    # "phi-3.5-vision-instruct-response.txt": "Model L",
    # "qwen-vl-2b-response.txt": "Model M",
    # "qwen-vl-72b-response.txt": "Model N",
    # "qwen-vl-7b-response.txt": "Model O",
    #"RHUH-0001.docx": "Model P",  # Example
    # "meta-vision-90b.txt":"Model_Z"
}

plots_dir = r'D:\final_drive_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250218T095105Z-001\plots'

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

for case in os.listdir(cases_dir):
    case_dir = os.path.join(cases_dir, case)

    reference_response = read_docx(os.path.join(case_dir, f'{case}.docx'))

    

    for model_file, model_name in model_encodings.items():
        model_response_path = os.path.join(case_dir, model_file)
        if os.path.exists(model_response_path):
            model_response = open(model_response_path).read()
            plot_file_path = os.path.join(plots_dir, f'{model_file.split(".")[0]}')
            plot_file_path = os.path.join(plot_file_path, f'{case}')
            
            # Create the directory if it does not exist
            os.makedirs(plot_file_path, exist_ok=True)
            
            plot_file_path = os.path.join(plot_file_path, f'{case}_{model_file.split(".")[0]}_o1-score.txt')

            # Sanity check: Skip if the file already exists
            if os.path.exists(plot_file_path):
                print(f"Skipping {case} - {model_name}, file already exists.")
                continue

            user_prompt = f"""**Reference Report:**
{reference_response}

**AI-Generated Report:**
{model_response}

**Evaluation and Scores:**
- Correctness: [Score], Reasoning: [Explanation]
- Conciseness: [Score], Reasoning: [Explanation]
- Completeness: [Score], Reasoning: [Explanation]
- Medical-Images-Description: [Score], Reasoning: [Explanation]

**Overall Score**: [Overall feedback paragraph]"""

            system_prompt = """You are a judge evaluating an AI-generated radiology report for brain MRI against a reference professional radiologist's report. Please score the AI-generated report on the following criteria out of 10, and calculate a final accuracy score out of 100%. Provide a single paragraph of feedback summarizing your evaluation. Also answer exactly as the given template. Please note that some models can sometimes provide empty responses.

Criteria to evaluate:
1. Correctness: How accurate is the AI-generated report in terms of details, diagnosis, and imaging findings? (Score 0-10)
2. Conciseness: How well does the AI-generated report balance providing enough detail without being overly verbose? (Score 0-10)
3. Completeness: Does the AI-generated report cover all necessary aspects, including tumor characteristics, surgical plan, imaging findings, and recommendations? (Score 0-10)
4. Medical-Images-Description: How well does the AI-generated report describe the medical images and is it really seeing it or making things up? (Score 0-10)"""
            response = client.chat.completions.create(
                model='o1-preview',
                messages=[
      
                    {
                        "role": "user",
                        "content": system_prompt + user_prompt
                    }
                ],
        
          
            )

            # Print the response
            print(response)
            print(response.model_dump_json(indent=2))
            o1_response = response.choices[0].message.content
            print(o1_response)
            print('----------------------------------')

            # Save the response in the plots folder
            with open(plot_file_path, 'w', encoding='utf-8') as f:
                f.write(o1_response)

            modified_folders.append(case)
            print(f"Generated O1 response for {case} - {model_name}")

print("Folders modified:")
for folder in modified_folders:
    print(folder)

