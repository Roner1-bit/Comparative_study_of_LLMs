import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from PIL import Image
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import torch
import gc
import traceback
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torch.cuda.amp import autocast
import pickle
import os
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
import torch
import gc
import traceback
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from torch.cuda.amp import autocast
import pickle

if __name__ == '__main__':
    
    
    if __name__ == '__main__':
        cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
        cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
        model_id = "deepseek-ai/deepseek-vl2"
        offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'
        
        cases_to_process = []
        for case in os.listdir(cases_dir):
            case_dir = os.path.join(cases_dir, case)
            if os.path.isdir(case_dir):
                response_path = os.path.join(case_dir, 'deepseek-vl2.txt')
                if not os.path.exists(response_path):
                    cases_to_process.append(case)
        print(f"Found {len(cases_to_process)} cases to process.")
    
        torch.set_default_tensor_type(torch.FloatTensor)
        processor = DeepseekVLV2Processor.from_pretrained(model_id, torch_dtype=torch.float16,
                                                         device_map="cpu", trust_remote_code=True, 
                                                         cache_dir=cache_dir, 
                                                         offload_folder=offload_folder)
    
        tokenizer = processor.tokenizer
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['<image>','<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>', '<|grounding|>','<|User|>', '<|Assistant|>']})
        
        def build_conversation(case_dir, user_prompt,system_prompt):
            image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')]
            image_paths = [os.path.join(case_dir, f) for f in image_files]
            
            image_files= [image_files[-1]]
            image_paths = [image_paths[-1]]
            # Create image placeholders
            print(image_files)
            print(image_paths)
            image_tags = " ".join(["<image>"] * len(image_files))
            print(image_tags)
            
            return [
        
                 {
                    "role": "<|User|>",
                    "content": f"{image_tags}\n<|ref|>{system_prompt}\n{user_prompt}<|/ref|>",
                    "images": image_paths
                },
        
                {"role": "<|Assistant|>", "content": ""}
            ]
        
        failed_cases = []
        
        for case in cases_to_process:
            case_dir = os.path.join(cases_dir, case)
        
            clinical_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
            if not os.path.exists(clinical_path):
                    print(f"Missing clinical info for {case}")
                    failed_cases.append(case)
                    continue
                    
            clinical_info = open(clinical_path).read()
                
        
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
            # Prepare conversation
            conversation = build_conversation(case_dir, user_prompt,system_prompt)
            pil_images = load_pil_images(conversation)
            
            print(conversation)
    
            # Process inputs on CPU
            try:
                inputs = processor(
                    conversations=conversation,
                    images=pil_images,
                    force_batchify=True,
                    system_prompt=system_prompt
                )
    
                # Convert inputs to pickle file
                input_file_path = os.path.join(case_dir, 'inputs_deepseek.pkl')
                with open(input_file_path, 'wb') as f:
                    pickle.dump(inputs, f, protocol=pickle.HIGHEST_PROTOCOL) #Added Higher protocol to pickle
                
                print(f"Inputs converted to pickle file on case {case}")
    
            except Exception as e:
                 print(f"Error during input processing for case {case}: {e}")
                 traceback.print_exc()
                 failed_cases.append(case)
                
            
            del inputs, pil_images, conversation
            gc.collect()
            torch.cuda.empty_cache()
        
        # Save failed cases
        if failed_cases:
            with open(os.path.join(cases_dir, 'failed_deepseek-vl2.txt'), 'w') as f:
                f.write("\n".join(failed_cases))
    
    #---
    
    
    if __name__ == '__main__':
        cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
        cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
        model_id = "deepseek-ai/deepseek-vl2"
        offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'
        
        cases_to_process = []
        for case in os.listdir(cases_dir):
            case_dir = os.path.join(cases_dir, case)
            if os.path.isdir(case_dir):
                response_path = os.path.join(case_dir, 'deepseek-vl2.txt')
                if not os.path.exists(response_path):
                    cases_to_process.append(case)
        print(f"Found {len(cases_to_process)} cases to process.")
    
        torch.set_default_tensor_type(torch.FloatTensor)
                # Loading the Model to GPU
    
    
    
        with init_empty_weights():
            model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=cache_dir,
                offload_folder=offload_folder,
            )
            
        # Load the model to the correct device
        model = load_checkpoint_and_dispatch(
            model,
            cache_dir+"/models--deepseek-ai--deepseek-vl2/snapshots/f363772d1c47f4239dd844015b4bd53beb87951b",
            device_map="auto",
            offload_folder=offload_folder,
            dtype=torch.float16,
        )
            
        processor = DeepseekVLV2Processor.from_pretrained(model_id, torch_dtype=torch.float16,
                                                         device_map="auto", trust_remote_code=True, 
                                                         cache_dir=cache_dir, 
                                                         offload_folder=offload_folder)
        
    
    #---
    
        tokenizer = processor.tokenizer
        # tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        # tokenizer.add_special_tokens({'additional_special_tokens': ['<image>','<|ref|>', '<|/ref|>', '<|det|>', '<|/det|>', '<|grounding|>','<|User|>', '<|Assistant|>']})    
        model = model.eval()
        failed_cases = []
        
        for case in cases_to_process:
            case_dir = os.path.join(cases_dir, case)
            input_file_path = os.path.join(case_dir, 'inputs_deepseek.pkl')
            
            if not os.path.exists(input_file_path):
                print(f"Missing input pickle for {case}")
                failed_cases.append(case)
                continue
    
            try:
                with open(input_file_path, 'rb') as f:
                    loaded_inputs = pickle.load(f)
    
                with torch.no_grad():
                    # Prepare the necessary inputs for prepare_inputs_embeds
                    prepared_inputs = {
                        "input_ids": loaded_inputs.input_ids.to(torch.long),  # Keep input_ids as Long
                        "attention_mask": loaded_inputs.attention_mask.to("cpu"),
                         "pixel_values": loaded_inputs.images.to(torch.float16) if hasattr(loaded_inputs,'images') else None,
                         "images_seq_mask": loaded_inputs.images_seq_mask.to("cpu") if hasattr(loaded_inputs,'images_seq_mask') else None
    
                       }
                    inputs_embeds = model.prepare_inputs_embeds(**prepared_inputs)
    
    
                    print(f"Messages prepared for case {case}")
                    # Generate response
                    outputs = model.language.generate(
                            inputs_embeds=inputs_embeds,
                            attention_mask=prepared_inputs["attention_mask"],
                            pad_token_id=tokenizer.pad_token_id,
                            bos_token_id=tokenizer.bos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            max_new_tokens=4096,
                            temperature=0.7,
                            top_p=0.9,
                            do_sample=False,
                            use_cache=True,
                        )
                
                        
                    # Decode and save response
                print(outputs.cpu().tolist())
                response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                
                print(f"Response generated for case {case}, length: {len(response)} characters")
                print(response)
                
                with open(os.path.join(case_dir, 'deepseek-vl2.txt'), 'w', encoding='utf-8') as f:
                    f.write(response.strip())
    
                print("Response saved.",case)
            except Exception as e:
                print(f"Error during model prediction for case {case}: {e}")
                traceback.print_exc()
                failed_cases.append(case)
    
            del inputs_embeds, loaded_inputs,prepared_inputs
        #del model, processor
        gc.collect()
        torch.cuda.empty_cache()
        
        # Save failed cases
        if failed_cases:
            with open(os.path.join(cases_dir, 'failed_deepseek-vl2.txt'), 'w') as f:
                f.write("\n".join(failed_cases))


##########################################################################
# This file was converted using nb2py: https://github.com/BardiaKh/nb2py #
##########################################################################
