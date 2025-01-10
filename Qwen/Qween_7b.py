import os
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gc

# Define paths and model configuration
model_id = "Qwen/Qwen2-VL-7B-Instruct"
cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized (Copy)/'
cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
torch.cuda.set_per_process_memory_fraction(0.8)

# Load the model and processor
print("Loading model and processor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=cache_dir,
    offload_folder=offload_folder,
)
processor = AutoProcessor.from_pretrained(model_id)
print("Model and processor loaded successfully!\n")

# Function to load images
def load_image(image_path):
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# Initialize list to track failed cases
failed_cases = []

# Iterate through cases in the directory
for case in os.listdir(cases_dir):
    case_dir = os.path.join(cases_dir, case)
    if not os.path.isdir(case_dir):
        continue  # Skip non-directory files

    image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')]

    clinical_information_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
    if not os.path.exists(clinical_information_path):
        print(f"Missing clinical information file for case: {case}")
        continue

    clinical_information = open(clinical_information_path, 'r', encoding='utf-8').read()

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

    # Collect the last image
    last_image = None
    for image_file in image_files:
        image_path = os.path.join(case_dir, image_file)
        img = load_image(image_path)
        if img is not None:
            last_image = {"type": "image", "image": img}

    if last_image is None:
        print(f"No valid images found for case: {case}")
        continue

    # Prepare messages for Qwen2-VL
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [last_image, {"type": "text", "text": user_prompt}]},
    ]

    try:
        # Prepare inputs for Qwen2-VL
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        # Generate response with gradient calculations disabled
        with torch.no_grad():
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.7, top_p=0.9)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

        # Save the response
        response_path = os.path.join(case_dir, 'qwen-vl-7b-response.txt')
        with open(response_path, 'w', encoding='utf-8') as f:
            f.write(response_text)

        # Memory management
        del inputs, generated_ids, response_text
        gc.collect()
        torch.cuda.empty_cache()

        print(f"Response saved for case {case}.")

    except torch.cuda.OutOfMemoryError:
        print(f"CUDA out of memory error for case {case}. Skipping this case.")
        failed_cases.append(case)
        continue
    except KeyboardInterrupt:
        print("Interrupted by user. Proceeding to the next case.")
        continue
    except Exception as e:
        print(f"Error processing case {case}: {e}")
        failed_cases.append(case)
        continue

# After processing all cases, save failed cases
failed_cases_path = os.path.join(cases_dir, 'failed_cases.txt')
with open(failed_cases_path, 'w', encoding='utf-8') as f:
    for failed_case in failed_cases:
        f.write(f"{failed_case}\n")
print(f"Failed cases logged in {failed_cases_path}")