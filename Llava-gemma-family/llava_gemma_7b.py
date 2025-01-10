import requests
from PIL import Image
from transformers import (
  LlavaForConditionalGeneration,
  AutoTokenizer,
  CLIPImageProcessor
)
from processing_llavagemma import LlavaGemmaProcessor
import torch
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.8)



# Set model ID and directories
model_id = "Intel/llava-gemma-7b"
cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized (Copy)/'


model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    cache_dir='/media/RLAB-Disk01/Large-Language-Models-Weights',
    offload_folder='/media/RLAB-Disk01/Large-Language-Models-Weights',
)


processor = LlavaGemmaProcessor(
    tokenizer=AutoTokenizer.from_pretrained(model_id),
    image_processor=CLIPImageProcessor.from_pretrained(model_id)
)

cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized (Copy)/'

# Function to load an image
def load_image(image_path):
    return Image.open(image_path)

# Iterate through cases in the provided directory
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

    # Define prompts
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



    images = []
    for image_file in image_files:
        image_path = os.path.join(case_dir, image_file)
        Img = load_image(image_path)
        images.append(Img)

    user_prompt = system_prompt + user_prompt 

    # Prepare the messages for Llava-Gemma
    messages = [
        {"role": "user", "content": user_prompt}
    ]

    # Apply the chat template
    input_text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Prepare inputs
    inputs = processor(
        images=Img,
        text=input_text,
        return_tensors="pt"
    ).to(model.device)

    # Generate the response
    output = model.generate(**inputs, max_new_tokens=4096)
    response_text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # Save the response
    response_path = os.path.join(case_dir, 'llava-gemma-response-7b.txt')
    with open(response_path, 'w', encoding='utf-8') as f:
        f.write(response_text)

    print(f"Response saved for case {case}.")