#%%
import os
import torch
import gc
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining
import traceback
import shutil # Import shutil for file operations

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


#%%
import os

# Make sure cases_dir is properly defined
# cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
# deleted_count = 0
# for case in os.listdir(cases_dir):
#     case_dir = os.path.join(cases_dir, case)
#     if os.path.isdir(case_dir):
#         response_path = os.path.join(case_dir, 'llava-med-response.txt') # Changed response file name
#         if os.path.exists(response_path):
#             try:
#                 os.remove(response_path)
#                 deleted_count += 1
#                 print(f"Deleted: {response_path}")
#             except Exception as e:
#                 print(f"Error deleting {response_path}: {str(e)}")

# print(f"\nDeleted {deleted_count} response files.")
# print(f"Remaining cases without response file: {len(os.listdir(cases_dir)) - deleted_count}")

#%%

# Configuration
MODEL_PATH = "BUAADreamer/Chinese-LLaVA-Med-7B" # Changed model path
CASES_DIR = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
CACHE_DIR = '/media/RLAB-Disk01/Large-Language-Models-Weights'
IMAGE_SIZE = 224 # Fixed image size for LLaVA-Med
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Collect cases needing processing
cases_to_process = []
for case in os.listdir(CASES_DIR):
    case_dir = os.path.join(CASES_DIR, case)
    if os.path.isdir(case_dir):
        response_path = os.path.join(case_dir, 'llava-med-response11.txt') # Changed response file name
        if not os.path.exists(response_path):
            cases_to_process.append(case)
print(f"Found {len(cases_to_process)} cases to process.")


# Full image processing pipeline
# Simplified image processing: Fixed size resize only
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        # T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC), # Fixed Resize
        T.ToTensor(),
        # T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_image_from_case(case_dir):
    image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')]
    if not image_files:
        return None
    image_path = os.path.join(case_dir, image_files[-1])
    try:
        image = Image.open(image_path).convert('RGB')
        transform = build_transform(IMAGE_SIZE)
        pixels = transform # Add batch dimension and dtype
        return pixels, image # Return both pixel values and original image
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None, None


# Model initialization
print("Initializing BUAADreamer/Chinese-LLaVA-Med-7B model...")

model = AutoModelForPreTraining.from_pretrained(
    MODEL_PATH,
    use_safetensors=True,
    trust_remote_code=True,
    device_map="auto",
    cache_dir=CACHE_DIR,
    torch_dtype=torch.bfloat16,
 
).eval()

processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=CACHE_DIR)

generation_config = dict(
                          max_new_tokens=6012,
                          do_sample=False,
                          temperature=0.7,
                          )
print("Model initialized successfully.\n")

#%%

# Processing loop
failed_cases = []

for case_id in cases_to_process:
    case_dir = os.path.join(CASES_DIR, case_id)
    print(f"Processing case {case_dir}...")
    try:
        # Load clinical data
        clinical_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
        with open(clinical_path, 'r', encoding='utf-8') as f:
            clinical_info = f.read().strip()

        # Process image in case directory
        pixel_values, original_image = load_image_from_case(case_dir)
        if pixel_values is None:
            raise ValueError("No valid images found in case directory")

        # Prepare prompts - Adapting to LLaVA-Med format, using provided prompt.
        system_prompt = """<image>\n Consider that you are a professional radiologist with several years of experience and you are now treating a patient. Write a fully detailed diagnosis report for this case, avoiding any potential hallucination and paying close attention to all of the batch images attached to this message.

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
    {clinical_info}"""

        prompt = system_prompt + user_prompt


        #log print user prompt
        print(f"User prompt size: {len(prompt)}")

        # Generate response
        with torch.no_grad():
            inputs = processor(text=prompt, images=original_image , return_tensors="pt").to("cuda", torch.bfloat16) # Process prompt and image
            generate_ids = model.generate(**inputs, **generation_config)
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        #log print response size
        print(f"Response size: {len(response)}")
        print(response)


        # Save results
        output_path = os.path.join(case_dir, 'llava-med-response11.txt') # Changed output file name
        write_success = False
        error_messages = []
        temp_dir = '/tmp'

        # Method 1: Standard write
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response)
                f.flush()
                os.fsync(f.fileno())
            write_success = True
        except Exception as e:
            error_messages.append(f"Standard write failed: {str(e)}")

        # Method 2: Binary mode write
        if not write_success:
            try:
                with open(output_path, 'wb') as f:
                    f.write(response.encode('utf-8'))
                    os.fsync(f.fileno())
                write_success = True
            except Exception as e:
                error_messages.append(f"Binary write failed: {str(e)}")

        # Method 3: Temporary file + atomic rename
        if not write_success:
            temp_path = output_path + '.tmp'
            try:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                os.rename(temp_path, output_path)
                write_success = True
            except Exception as e:
                error_messages.append(f"Temp file rename failed: {str(e)}")
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

        # Method 4: Different encoding (utf-16)
        if not write_success:
            try:
                with open(output_path, 'w', encoding='utf-16') as f:
                    f.write(response)
                write_success = True
            except Exception as e:
                error_messages.append(f"UTF-16 write failed: {str(e)}")

        # Method 5: Write to alternate directory and copy
        if not write_success:
            try:
                temp_path = os.path.join(temp_dir, os.path.basename(output_path))
                with open(temp_path, 'w', encoding='utf-8') as f:
                    f.write(response)
                shutil.copy(temp_path, output_path)
                write_success = True
            except Exception as e:
                error_messages.append(f"Alternate dir write failed: {str(e)}")
                if os.path.exists(temp_path):
                    try: os.remove(temp_path)
                    except: pass

        # Method 6: Append mode (only if file doesn't exist)
        if not write_success and not os.path.exists(output_path):
            try:
                with open(output_path, 'a', encoding='utf-8') as f:
                    f.write(response)
                write_success = True
            except Exception as e:
                error_messages.append(f"Append mode failed: {str(e)}")

        # Final fallback: Force write with lowered privileges
        if not write_success:
            try:
                with open(output_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(response)
                write_success = True
            except Exception as e:
                error_messages.append(f"Force write failed: {str(e)}")

        if not write_success:
            raise IOError(f"All write methods failed:\n" + "\n".join(error_messages))


        print(f"Response saved for case {case_id}.")

        # Cleanup
        del pixel_values, original_image, inputs, generate_ids, response
        gc.collect()
        torch.cuda.empty_cache()


    except Exception as e:
        error_traceback = traceback.format_exc()
        error_message = f"Failed to process case {case_id}:\n{error_traceback}"
        print(error_message)
        failed_cases.append((case_id, error_traceback))
        continue

# Save failure log
if failed_cases:
    failed_log_path = os.path.join(CASES_DIR, 'llava_med_failed_cases11.txt') # Changed log file name
    with open(failed_log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join([f"{case_id}: {error}" for case_id, error in failed_cases]))
    print(f"Logged {len(failed_cases)} failed cases to {failed_log_path}")

print("\nProcessing completed. Summary:")
print(f"Total cases: {len(cases_to_process)}")
print(f"Successfully processed: {len(cases_to_process)-len(failed_cases)}")
print(f"Failed cases: {len(failed_cases)}")