import os
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch



torch.cuda.set_per_process_memory_fraction(0.8)
# Load the LLaMA model and processor
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir='/media/elboardy/RLAB-Disk01/Large-Language-Models-Weights',
    offload_folder='/media/elboardy/RLAB-Disk01/Large-Language-Models-Weights',
)
processor = AutoProcessor.from_pretrained(model_id)

def load_image(image_path):
    return Image.open(image_path)

# Directory containing the images
cases_dir = '/media/elboardy/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'

for case in os.listdir(cases_dir):
    case_dir = os.path.join(cases_dir, case)
    image_files = [
        f for f in os.listdir(case_dir)
        if f.lower().endswith('.png')
    ]

    clinical_information_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
    if not os.path.exists(clinical_information_path):
        print(f"Missing clinical information file for case: {case}")
        continue

    clinical_information = open(clinical_information_path).read()

    system_prompt = """Consider that you are a professional radiologist with several years of experience and you are now treating a patient. Write a fully detailed diagnosis report for this case, avoiding any potential hallucination and paying close attention to all of the batch images attached to this message.

Use the following structure for the report:

## Radiologist's Report

### Patient Information:
- *Age:* 65
- *Sex:* Male
- *Days from earliest imaging to surgery:* 1
- *Histopathological Subtype:* Glioblastoma
- *WHO Grade:* 4
- *IDH Status:* Mutant
- *Preoperative KPS:* 80
- *Preoperative Contrast-Enhancing Tumor Volume (cm³):* 103.21
- *Preoperative T2/FLAIR Abnormality (cm³):* 36.29
- *Extent of Resection (EOR %):* 100.0
- *EOR Type:* Gross Total Resection (GTR)
- *Adjuvant Therapy:* Radiotherapy (RT) + Temozolomide (TMZ)
- *Progression-Free Survival (PFS) Days:* 649
- *Overall Survival (OS) Days:* 736

#### Tumor Characteristics:

#### Segmentation Analysis:

#### Surgical Considerations:

### Clinical Summary:

### Recommendations:

### Prognostic Considerations:

### Follow-Up Plan:

### Additional Notes*(if any)*:

Ensure all findings from all of the images and clinical data provided. Please mention at the end of the report how many images were reviewed."""

    user_prompt = f"""You will be given batches of images, which are different sequences of MRI scans. 
    The images are for patients who are likely to have a brain tumor. Each image will contain up to 10 slices for 5 different sequences and the segmentation masks for the tumor at the bottom row of the image. 
    Additional clinical data about the patient is: 
    {clinical_information}"""

    # Process all images
    images = []
    for image_file in image_files:
        image_path = os.path.join(case_dir, image_file)
        img = load_image(image_path)
        images.append(img)

    # Prepare the messages for LLaMA Vision
    messages = [
         {"role": "system_admin", "content": [
            {"type": "text", "text": system_prompt},
        ]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": user_prompt}
        ]}
    ]

    # Apply the chat template
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process the input and images
    inputs = processor(
        images=img,
        text=input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)

    # Generate the response
    output = model.generate(**inputs, max_new_tokens=4096)
    response_text = processor.decode(output[0], skip_special_tokens=True)

    # Save the response
    response_path = os.path.join(case_dir, 'llama-11B.txt')
    with open(response_path, 'w', encoding='utf-8') as f:
        f.write(response_text)

    print(f"Response saved for case {case}.")
