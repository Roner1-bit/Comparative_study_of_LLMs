import os
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from transformers.image_utils import load_image
import torch
import gc
import traceback

if __name__ == '__main__':
    cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
    cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    model_id = "google/paligemma2-10b-pt-896"
    offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'
        
    
    #---
    
    processor = PaliGemmaProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
            offload_folder=offload_folder,
        ).eval()
    
    #---
    
    cases_to_process = []
    for case in os.listdir(cases_dir):
        case_dir = os.path.join(cases_dir, case)
        if os.path.isdir(case_dir):
            response_path = os.path.join(case_dir, 'paligemma2-10b-report.txt')
            if not os.path.exists(response_path):
                cases_to_process.append(case)
    print(f"Found {len(cases_to_process)} cases to process.")
    
    #---
    
    failed_cases = []
        
    for case in cases_to_process:
            case_dir = os.path.join(cases_dir, case)
            try:
                # Load clinical info
                clinical_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
                if not os.path.exists(clinical_path):
                    print(f"Missing clinical info for {case}")
                    failed_cases.append(case)
                    continue
                    
                clinical_info = open(clinical_path).read()
    
                # Build prompts
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
    f"    {clinical_info}"
    
    
                full_prompt = system_prompt + "\n\n" + user_prompt
    
                print(f"Processing {case} with prompt: {full_prompt}")
                # Load image
                image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')]
                if not image_files:
                    print(f"No images found for {case}")
                    failed_cases.append(case)
                    continue
                    
                image_path = os.path.join(case_dir, image_files[-1])  # Using last image
                image = Image.open(image_path)
                print(f"Loaded image: {image_path}")
            
                # Process inputs
                model_inputs = processor(
                    text=full_prompt,
                    images=image,
                    return_tensors="pt"
                ).to(model.device)
    
                input_len = model_inputs["input_ids"].shape[-1]
    
                print("Generated inputs")
                # Generate response
                with torch.inference_mode():
                    generation = model.generate(
                        **model_inputs,
                        max_new_tokens=4096,
                        do_sample=False,
                        temperature=0.7,
                        top_p=0.9,
                    )
                    
                generation = generation[0][input_len:]
                response = processor.decode(generation, skip_special_tokens=True)
                print(f"Generated response: {response}")
                # Save response
                with open(os.path.join(case_dir, 'paligemma2-10b-report.txt'), 'w', encoding='utf-8') as f:
                    f.write(response.strip())
    
                # Cleanup
                del model_inputs, generation, response
                gc.collect()
                torch.cuda.empty_cache()
    
            except Exception as e:
                error_msg = f"\nError processing {case}:\n{traceback.format_exc()}"
                print(error_msg)
                failed_cases.append(case)
    
        # Save failed cases
    if failed_cases:
            with open(os.path.join(cases_dir, 'failed_paligemma2-10b.txt'), 'w') as f:
                f.write("\n".join(failed_cases))
                

