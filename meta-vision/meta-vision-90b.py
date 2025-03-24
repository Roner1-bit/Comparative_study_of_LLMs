import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import torch
import gc

if __name__ == '__main__':
    model_id = "meta-llama/Llama-3.2-90B-Vision-Instruct"
    cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
    cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    
    #---
    
    #torch.cuda.set_per_process_memory_fraction(0.8)
    
    #---
    
    cases_to_process = []
    for case in os.listdir(cases_dir):
        case_dir = os.path.join(cases_dir, case)
        if os.path.isdir(case_dir):
            response_path = os.path.join(case_dir, 'llama-90B.txt')
            if not os.path.exists(response_path):
                cases_to_process.append(case)
    print(f"Found {len(cases_to_process)} cases to process.")
    
    #---
    
    
    model_id = model_id
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        offload_folder=offload_folder,
        attn_implementation="eager",
        use_safetensors=True,
        offload_state_dict=True, 
    )
    processor = AutoProcessor.from_pretrained(model_id,use_fast=True)
    
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
    
    
    failed_cases = []
    
    for case in cases_to_process:
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
    
        # Process all images
        for image_file in image_files:
            image_path = os.path.join(case_dir, image_file)
            img = load_image(image_path)
       
    
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
    
        print(f"Messages prepared for case {case}")
    
        try:
    
    
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
            
            print(f"Response generated for case {case}, length: {len(response_text)} characters")
    
        # Save the response
            response_path = os.path.join(case_dir, 'llama-90B.txt')
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
    failed_cases_path = os.path.join(cases_dir, 'failed_cases_llama_90b.txt')
    with open(failed_cases_path, 'w', encoding='utf-8') as f:
        for failed_case in failed_cases:
            f.write(f"{failed_case}\n")
    print(f"Failed cases logged in {failed_cases_path}")


##########################################################################
# This file was converted using nb2py: https://github.com/BardiaKh/nb2py #
##########################################################################
