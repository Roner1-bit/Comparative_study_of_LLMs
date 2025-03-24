import os
from PIL import Image
import requests
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import gc

if __name__ == '__main__':
    model_id = "microsoft/Phi-3.5-vision-instruct"
    cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized'
    cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    
    #---
    
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    #torch.cuda.set_per_process_memory_fraction(0.8)
    
    #---
    
    # import os
    
    # # Make sure cases_dir is properly defined
    # # cases_dir = "your/cases/directory/path"
    
    # deleted_count = 0
    # for case in os.listdir(cases_dir):
    #     case_dir = os.path.join(cases_dir, case)
    #     if os.path.isdir(case_dir):
    #         response_path = os.path.join(case_dir, 'phi-3.5-vision-instruct-response.txt')
    #         if os.path.exists(response_path):
    #             try:
    #                 os.remove(response_path)
    #                 deleted_count += 1
    #                 print(f"Deleted: {response_path}")
    #             except Exception as e:
    #                 print(f"Error deleting {response_path}: {str(e)}")
    
    # print(f"\nDeleted {deleted_count} response files.")
    # print(f"Remaining cases without response file: {len(os.listdir(cases_dir)) - deleted_count}")
    
    #---
    
    cases_to_process = []
    for case in os.listdir(cases_dir):
        case_dir = os.path.join(cases_dir, case)
        if os.path.isdir(case_dir):
            response_path = os.path.join(case_dir, 'phi-3.5-vision-instruct-response.txt')
            if not os.path.exists(response_path):
                cases_to_process.append(case)
    print(f"Found {len(cases_to_process)} cases to process.")
    
    #---
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        _attn_implementation='flash_attention_2',
        cache_dir=cache_dir,
        offload_folder=offload_folder
    )
    
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        num_crops=32
    )
    
    #---
    
    def load_image(image_path):
        try:
            img = Image.open(image_path)
            return img
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    #---
    
    failed_cases = []
    
    for case in cases_to_process:
        case_dir = os.path.join(cases_dir, case)
        print(f"\nProcessing case: {case}")
        image_files = [
            f for f in os.listdir(case_dir)
            if f.lower().endswith('.png')
        ]
    
        
        try:
            # Load clinical information
            clinical_info_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
            if not os.path.exists(clinical_info_path):
                raise FileNotFoundError(f"Missing clinical info: {clinical_info_path}")
                
            with open(clinical_info_path, 'r', encoding='utf-8') as f:
                clinical_info = f.read()
    
            # Prepare prompts
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
    f"        The images are for patients who are likely to have a brain tumor. Each image will contain up to 10 slices for 5 different sequences and the segmentation masks for the tumor at the bottom row of the image. \n" +\
    f"        Additional clinical data about the patient is: \n" +\
    f"        {clinical_info}."
            # Create image placeholders
    
            for image_file in image_files:
                image_path = os.path.join(case_dir, image_file)
                img = load_image(image_path)
       
            images=[]
            images.append(img)
            
            image_placeholders = "\n".join([f"<|image_{i+1}|>" for i in range(len(images))])
    
            
    
    
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{image_placeholders}\n{user_prompt}"},
            ]
    
            # messages = [
            #     {"role": "user", "content": f"{image_placeholders}\n{system_prompt}\n{user_prompt}"},
            # ]
    
            # Process inputs
            prompt = processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = processor(
                text=prompt,
                images=images,
                return_tensors="pt"
            ).to(model.device)
    
            # Generate response
            generation_args = {
                "max_new_tokens": 4096,
                "temperature": 0.7,
                "do_sample": False,
            }
            
            generate_ids = model.generate(
                **inputs,
                eos_token_id=processor.tokenizer.eos_token_id,
                **generation_args
            )
    
            # Decode response
            response = processor.batch_decode(
                generate_ids[:, inputs['input_ids'].shape[1]:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
    
            # Save results
            with open(os.path.join(case_dir, 'phi-3.5-vision-instruct-response.txt'), 
                     'w', encoding='utf-8') as f:
                f.write(response)
    
            # Cleanup
            del inputs, generate_ids, response
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Completed case {case}")
    
        except Exception as e:
            print(f"Error processing {case}: {str(e)}")
            failed_cases.append(case)
            continue
    
    # Save failed cases log
    if failed_cases:
        with open(os.path.join(cases_dir, 'phi3_failed_cases.txt'), 'w') as f:
            f.write("\n".join(failed_cases))
        print(f"Logged {len(failed_cases)} failed cases")
    
    print("Processing complete.")


##########################################################################
# This file was converted using nb2py: https://github.com/BardiaKh/nb2py #
##########################################################################
