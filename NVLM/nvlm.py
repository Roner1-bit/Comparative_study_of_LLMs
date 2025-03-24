#%%
import os
import torch
import math
import gc
from PIL import Image
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import traceback

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
#torch.cuda.set_per_process_memory_fraction(0.8)


#%%
import os

# Make sure cases_dir is properly defined
# cases_dir = "your/cases/directory/path"
cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
deleted_count = 0
for case in os.listdir(cases_dir):
    case_dir = os.path.join(cases_dir, case)
    if os.path.isdir(case_dir):
        response_path = os.path.join(case_dir, 'nvlm-72b-response.txt')
        if os.path.exists(response_path):
            try:
                os.remove(response_path)
                deleted_count += 1
                print(f"Deleted: {response_path}")
            except Exception as e:
                print(f"Error deleting {response_path}: {str(e)}")

print(f"\nDeleted {deleted_count} response files.")
print(f"Remaining cases without response file: {len(os.listdir(cases_dir)) - deleted_count}")

#%%



# Configuration
MODEL_PATH = "nvidia/NVLM-D-72B"
CASES_DIR = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
CACHE_DIR = '/media/RLAB-Disk01/Large-Language-Models-Weights'
IMAGE_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Collect cases needing processing
cases_to_process = []
for case in os.listdir(CASES_DIR):
    case_dir = os.path.join(CASES_DIR, case)
    if os.path.isdir(case_dir):
        response_path = os.path.join(case_dir, 'nvlm-72b-response.txt')
        if not os.path.exists(response_path):
            cases_to_process.append(case)
print(f"Found {len(cases_to_process)} cases to process.")

# Device mapping initialization


# Full image processing pipeline
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio



def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((int(target_width), int(target_height)))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def load_images_from_case(case_dir, max_num=6):
    image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')]
    all_pixels = []
    image_files= [image_files[-1]]
    #join the last image path with case_dir
    
    image = Image.open(os.path.join(case_dir, image_files[-1])).convert('RGB')
    transform = build_transform(IMAGE_SIZE)
    print("--------------",image_files)
    for img_file in image_files:
        img_path = os.path.join(case_dir, img_file)
        try:
            image = Image.open(img_path).convert('RGB')
            processed = dynamic_preprocess(image, image_size=IMAGE_SIZE, use_thumbnail=True, max_num=max_num)
            pixels = torch.stack([transform(img) for img in processed]).to(torch.bfloat16)
            all_pixels.append(pixels)
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
    
    return torch.cat(all_pixels) if all_pixels else None

# Model initialization
print("Initializing NVLM-72B model...")

model = AutoModel.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir=CACHE_DIR,
    offload_folder=CACHE_DIR,
    attn_implementation="flash_attention_2",
    use_safetensors=True,

).eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
generation_config = dict(
                          max_new_tokens=4096,
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

        # Process all images in case directory
        pixel_values = load_images_from_case(case_dir)
        if pixel_values is None:
            raise ValueError("No valid images found in case directory")
        
        # Prepare prompts
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

Ensure all findings from all of the images and clinical data provided. Please mention at the end of the report how many images were reviewed.\n"""

        user_prompt = f"""<image> You will be given batches of images, which are different sequences of MRI scans. 
    The images are for patients who are likely to have a brain tumor. Each image will contain up to 10 slices for 5 different sequences and the segmentation masks for the tumor at the bottom row of the image. 
    Additional clinical data about the patient is: 
    {clinical_info}"""



        user_prompt = system_prompt + user_prompt


        #log print user prompt
        print(f"User prompt size: {len(user_prompt)}")

        # Generate response
        with torch.no_grad():
            response = model.chat(
                tokenizer,
                pixel_values,
                user_prompt,
                generation_config,
                history=None,
                return_history=True
            )

        #log print response size
        print(f"Response size: {len(response)}")

        print(response)

        response = response[0]
        # Save results
        output_path = os.path.join(case_dir, 'nvlm-72b-response.txt')
        write_success = False
        error_messages = []
        temp_dir = '/tmp'  # For fallback writing methods

        # Method 1: Standard write
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(response)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure file metadata is written
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
                os.rename(temp_path, output_path)  # Atomic on Unix systems
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

        print(f"Successfully processed case {case_id}")
        
        # Cleanyupc
        del pixel_values, response
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        error_traceback = traceback.format_exc()  # Capture full error details
        error_message = f"Failed to process case {case_id}:\n{error_traceback}"
        print(error_message)  # Print full error to console
        failed_cases.append((case_id, error_traceback))  # Store traceback for logging
        continue

# Save failure log
if failed_cases:
    failed_log_path = os.path.join(CASES_DIR, 'nvlm_failed_cases.txt')
    with open(failed_log_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(failed_cases))
    print(f"Logged {len(failed_cases)} failed cases to {failed_log_path}")

print("\nProcessing completed. Summary:")
print(f"Total cases: {len(cases_to_process)}")
print(f"Successfully processed: {len(cases_to_process)-len(failed_cases)}")
print(f"Failed cases: {len(failed_cases)}")
# %%
