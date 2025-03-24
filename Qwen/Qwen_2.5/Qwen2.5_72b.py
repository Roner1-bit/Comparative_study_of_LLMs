import os
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import gc

if __name__ == '__main__':
    
    # Define paths and model configuration
    model_id = "Qwen/Qwen2.5-VL-72B-Instruct"
    cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
    cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    
    #---
    
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    # torch.cuda.set_per_process_memory_fraction(0.8)
    
    #---
    
    # Collect list of cases that do not have 'qwen-vl-72b-response.txt'
    cases_to_process = []
    for case in os.listdir(cases_dir):
        case_dir = os.path.join(cases_dir, case)
        if os.path.isdir(case_dir):
            response_path = os.path.join(case_dir, 'qwen2.5-vl-72b-response.txt')
            if not os.path.exists(response_path):
                cases_to_process.append(case)
    print(f"Found {len(cases_to_process)} cases to process.")
    
    #---
    
    # Load the model and processor
    print("Loading model and processor...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        offload_folder=offload_folder,
        attn_implementation="flash_attention_2"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    print("Model and processor loaded successfully!\n")
    
    #---
    
    def load_image(image_path):
        try:
            img = Image.open(image_path)
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    #---
    
    print("Processing cases...--->")
    print(cases_to_process)
    
    #---
    
    # Initialize list to track failed cases
    failed_cases = []
    
    # Iterate through cases in the directory
    for case in cases_to_process:
        case_dir = os.path.join(cases_dir, case)
        if not os.path.isdir(case_dir):
            continue  # Skip non-directory files
    
        image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')]
        print(f"Found {len(image_files)} image files for case {case}")
    
        clinical_information_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
        if not os.path.exists(clinical_information_path):
            print(f"Missing clinical information file for case: {case}")
            continue
    
    
        print(f"Reading clinical information for case {case}")    
        clinical_information = open(clinical_information_path, 'r', encoding='utf-8').read()
    
        system_prompt = "Consider that you are a professional radiologist with several years of experience and you are now treating a patient. Write a fully detailed diagnosis report for this case, avoiding any potential hallucination and paying close attention to all of the batch images attached to this message.\n" +\
    "\n" +\
    "Use the following structure for the report:\n" +\
    "\n" +\
    "## Radiologist's Report\n" +\
    "\n" +\
    "### Patient Information:\n" +\
    "- *Age:* 65\n" +\
    "- *Sex:* Male\n" +\
    "- *Days from earliest imaging to surgery:* 1\n" +\
    "- *Histopathological Subtype:* Glioblastoma\n" +\
    "- *WHO Grade:* 4\n" +\
    "- *IDH Status:* Mutant\n" +\
    "- *Preoperative KPS:* 80\n" +\
    "- *Preoperative Contrast-Enhancing Tumor Volume (cm³):* 103.21\n" +\
    "- *Preoperative T2/FLAIR Abnormality (cm³):* 36.29\n" +\
    "- *Extent of Resection (EOR %):* 100.0\n" +\
    "- *EOR Type:* Gross Total Resection (GTR)\n" +\
    "- *Adjuvant Therapy:* Radiotherapy (RT) + Temozolomide (TMZ)\n" +\
    "- *Progression-Free Survival (PFS) Days:* 649\n" +\
    "- *Overall Survival (OS) Days:* 736\n" +\
    "\n" +\
    "#### Tumor Characteristics:\n" +\
    "\n" +\
    "#### Segmentation Analysis:\n" +\
    "\n" +\
    "#### Surgical Considerations:\n" +\
    "\n" +\
    "### Clinical Summary:\n" +\
    "\n" +\
    "### Recommendations:\n" +\
    "\n" +\
    "### Prognostic Considerations:\n" +\
    "\n" +\
    "### Follow-Up Plan:\n" +\
    "\n" +\
    "### Additional Notes*(if any)*:\n" +\
    "\n" +\
    "Ensure all findings from all of the images and clinical data provided. Please mention at the end of the report how many images were reviewed."
    
        user_prompt = f"You will be given batches of images, which are different sequences of MRI scans. \n" +\
    f"    The images are for patients who are likely to have a brain tumor. Each image will contain up to 10 slices for 5 different sequences and the segmentation masks for the tumor at the bottom row of the image. \n" +\
    f"    Additional clinical data about the patient is: \n" +\
    f"    {clinical_information}"
    
        # Collect the last image
        last_image = None
        for image_file in image_files:
            image_path = os.path.join(case_dir, image_file)
            img = load_image(image_path)
            if img is not None:
                last_image = {"type": "image", "image": img}
                print(f"Loaded image: {image_file} for case {case}")
        if last_image is None:
            print(f"No valid images found for case: {case}")
            continue
    
        # Prepare messages for Qwen2-VL
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [last_image, {"type": "text", "text": user_prompt}]},
        ]
        print(f"Messages prepared for case {case}")
    
        try:
            # Prepare inputs for Qwen2-VL
            print(f"Processing inputs for case {case}")
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
                ).to(model.device)
    
                generated_ids = model.generate(**inputs, max_new_tokens=4096, temperature=0.7, top_p=0.9)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response_text = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
    
    
            print(f"Response generated for case {case}, length: {len(response_text)} characters")
            # Save the response
            response_path = os.path.join(case_dir, 'qwen-vl-72b-response.txt')
            with open(response_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            print(f"Response saved for case {case}.")
            # Memory management
            del inputs, generated_ids, response_text
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Memory cleared for case {case}")
            
    
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
    failed_cases_path = os.path.join(cases_dir, 'failed_cases_2.5_72.txt')
    with open(failed_cases_path, 'w', encoding='utf-8') as f:
        for failed_case in failed_cases:
            f.write(f"{failed_case}\n")
    print(f"Failed cases logged in {failed_cases_path}")
    
    #---
    
    torch.cuda.empty_cache()
    failed_cases_path = os.path.join(cases_dir, 'failed_cases.txt')
    if not os.path.exists(failed_cases_path):
        print("No failed cases to reprocess.")
        exit()
    
    with open(failed_cases_path, 'r', encoding='utf-8') as f:
        failed_cases = [line.strip() for line in f.readlines()]
    
    # Initialize a new list for cases that fail again
    new_failed_cases = []
    
    # Set a maximum number of retries
    max_retries = 2
    
    for case in failed_cases:
        retry_count = 0
        while retry_count < max_retries:
            try:
                case_dir = os.path.join(cases_dir, case)
                response_path = os.path.join(case_dir, 'qwen2.5-vl-72b-response.txt')
                if os.path.exists(response_path):
                    print(f"Response already exists for case {case}. Skipping.")
                    break  # Skip if response already exists
    
                # Load image and clinical information (same as before)
                image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')]
                clinical_information_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
                clinical_information = open(clinical_information_path, 'r', encoding='utf-8').read()
    
                # Prepare messages (same as before)
                system_prompt = "Consider that you are a professional radiologist with several years of experience and you are now treating a patient. Write a fully detailed diagnosis report for this case, avoiding any potential hallucination and paying close attention to all of the batch images attached to this message.\n" +\
    "\n" +\
    "Use the following structure for the report:\n" +\
    "\n" +\
    "## Radiologist's Report\n" +\
    "\n" +\
    "### Patient Information:\n" +\
    "- *Age:* 65\n" +\
    "- *Sex:* Male\n" +\
    "- *Days from earliest imaging to surgery:* 1\n" +\
    "- *Histopathological Subtype:* Glioblastoma\n" +\
    "- *WHO Grade:* 4\n" +\
    "- *IDH Status:* Mutant\n" +\
    "- *Preoperative KPS:* 80\n" +\
    "- *Preoperative Contrast-Enhancing Tumor Volume (cm³):* 103.21\n" +\
    "- *Preoperative T2/FLAIR Abnormality (cm³):* 36.29\n" +\
    "- *Extent of Resection (EOR %):* 100.0\n" +\
    "- *EOR Type:* Gross Total Resection (GTR)\n" +\
    "- *Adjuvant Therapy:* Radiotherapy (RT) + Temozolomide (TMZ)\n" +\
    "- *Progression-Free Survival (PFS) Days:* 649\n" +\
    "- *Overall Survival (OS) Days:* 736\n" +\
    "\n" +\
    "#### Tumor Characteristics:\n" +\
    "\n" +\
    "#### Segmentation Analysis:\n" +\
    "\n" +\
    "#### Surgical Considerations:\n" +\
    "\n" +\
    "### Clinical Summary:\n" +\
    "\n" +\
    "### Recommendations:\n" +\
    "\n" +\
    "### Prognostic Considerations:\n" +\
    "\n" +\
    "### Follow-Up Plan:\n" +\
    " \n" +\
    "### Additional Notes*(if any)*:\n" +\
    "\n" +\
    "Ensure all findings from all of the images and clinical data provided. Please mention at the end of the report how many images were reviewed."
    
                user_prompt = f"You will be given batches of images, which are different sequences of MRI scans. \n" +\
    f"    The images are for patients who are likely to have a brain tumor. Each image will contain up to 10 slices for 5 different sequences and the segmentation masks for the tumor at the bottom row of the image. \n" +\
    f"    Additional clinical data about the patient is: \n" +\
    f"    {clinical_information}"
    
                # Collect the last image (same as before)
                last_image = None
                for image_file in image_files:
                    image_path = os.path.join(case_dir, image_file)
                    img = load_image(image_path)
                    if img is not None:
                        last_image = {"type": "image", "image": img}
    
                if last_image is None:
                    print(f"No valid images found for case: {case}")
                    new_failed_cases.append(case)
                    break  # Skip this case
    
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [last_image, {"type": "text", "text": user_prompt}]},
                ]
    
                # Prepare inputs and generate response (same as before)
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
    
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
                with open(response_path, 'w', encoding='utf-8') as f:
                    f.write(response_text)
    
                # Memory management
                del inputs, generated_ids, response_text
                gc.collect()
                torch.cuda.empty_cache()
    
                print(f"Response saved for case {case} after retry.")
                break  # Success, no need to retry
    
            except torch.cuda.OutOfMemoryError:
                print(f"CUDA out of memory error for case {case} on retry {retry_count + 1}. Retrying...")
                retry_count += 1
                continue
            except Exception as e:
                print(f"Error processing case {case} on retry {retry_count + 1}: {e}")
                retry_count += 1
                continue
    
        if retry_count == max_retries:
            print(f"Failed to process case {case} after {max_retries} retries. Adding to new failed cases.")
            new_failed_cases.append(case)
    
    # Save new failed cases
    new_failed_cases_path = os.path.join(cases_dir, '72_new_failed_cases.txt')
    with open(new_failed_cases_path, 'w', encoding='utf-8') as f:
        for failed_case in new_failed_cases:
            f.write(f"{failed_case}\n")
    print(f"New failed cases logged in {new_failed_cases_path}")


