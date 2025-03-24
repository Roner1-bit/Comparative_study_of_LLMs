#%%
import os
from PIL import Image
from transformers import AutoModelForCausalLM
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import torch
import gc
import traceback
import deepspeed
from deepspeed import get_accelerator
#%%
if __name__ == '__main__':
    cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
    cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    model_id = "deepseek-ai/deepseek-vl2-small"
    offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    
    #---
    
    #torch.cuda.set_per_process_memory_fraction(0.9)
    #os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    #key errors are because of inference and init difference in the deepspeed library
    ds_config = {
        "mp_size": 1,
        "dtype": torch.bfloat16,

        "zero": {
            "stage": 3,  # Use Zero Stage 3 for maximum memory savings
            "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True
        },

        "replace_with_kernel_inject": True  # Enable kernel injection
    }

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

#%%    
    #---
    
    # import os
    
    # # Make sure cases_dir is properly defined
    # # cases_dir = "your/cases/directory/path"
    
    # deleted_count = 0
    # for case in os.listdir(cases_dir):
    #     case_dir = os.path.join(cases_dir, case)
    #     if os.path.isdir(case_dir):
    #         response_path = os.path.join(case_dir, 'deepseek-vl-small.txt')
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
#%%    
    cases_to_process = []
    for case in os.listdir(cases_dir):
        case_dir = os.path.join(cases_dir, case)
        if os.path.isdir(case_dir):
            response_path = os.path.join(case_dir, 'deepseek-vl-small.txt')
            if not os.path.exists(response_path):
                cases_to_process.append(case)
    print(f"Found {len(cases_to_process)} cases to process.")
    
    #---
    
    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        use_safetensors=True,
        cache_dir=cache_dir,
        offload_folder=offload_folder,
        # offload_state_dict=True, 
        attn_implementation="eager",
        
    )
    
    
    model = deepspeed.init_inference(
        model,
        config=ds_config,
        replace_with_kernel_inject=True,
        mp_size=1,  # Number of GPUs
        dtype=torch.bfloat16
    )

    # model = model.eval()
    
    #---
    
    def build_conversation(case_dir, user_prompt,system_prompt):
        image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')]
        image_paths = [os.path.join(case_dir, f) for f in image_files]
        
        image_files= [image_files[-1]]
        image_paths = [image_paths[-1]]
        # Create image placeholders
        image_tags = " ".join(["<image>"] * len(image_files))
        
        return [
    
             {
                "role": "<|User|>",
                "content": f"{system_prompt}\n{user_prompt}\n{image_tags}",
                "images": image_paths
            },
    
            {"role": "<|Assistant|>", "content": ""}
        ]
    
    #---


#%%
    def generate_text(inputs):
        with torch.no_grad():
            outputs = model.generate(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=4096,
                temperature=0.7,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        return outputs




#%%    
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
            
            # Process inputs
            inputs = processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                system_prompt=system_prompt
            ).to(get_accelerator().current_device_name())
    
            # Generate embeddings
            inputs_embeds = model.prepare_inputs_embeds(**inputs)
    
            print(f"Messages prepared for case {case}")
            # Generate response
    
            generation_config = {
        "max_new_tokens": 4096,  # Maximum number of tokens to generate
        "do_sample": False,  # Use greedy decoding (set to True for sampling)
        "temperature": 0.7,  # Controls randomness (lower = more deterministic)
        "top_p": 0.9,  # Nucleus sampling (cumulative probability threshold)
        "repetition_penalty": 1.2,  # Penalizes repeated tokens
        "early_stopping": True,  # Stop generation if EOS token is generated
        "pad_token_id": tokenizer.eos_token_id,  # Padding token ID
        "bos_token_id": tokenizer.bos_token_id,  # Beginning of sentence token ID
        "eos_token_id": tokenizer.eos_token_id,  # End of sentence token ID
    }
    
    
            # outputs = model.language.generate(
            #     inputs_embeds=inputs_embeds,
            #     attention_mask=inputs.attention_mask,
            #     pad_token_id=tokenizer.eos_token_id,
            #     bos_token_id=tokenizer.bos_token_id,
            #     eos_token_id=tokenizer.eos_token_id,
            #     max_new_tokens=4096,
            #     temperature=0.7,
            #     generation_config=generation_config,
            #     do_sample=False,
            #     use_cache=True
            # )

            outputs = generate_text(inputs)
            
            # Decode and save response
            response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    
            print(f"Response generated for case {case}, length: {len(response)} characters")
    
    
            with open(os.path.join(case_dir, 'deepseek-vl-small.txt'), 'w', encoding='utf-8') as f:
                f.write(response.split("[/INST]")[-1].strip())
                
            # Cleanup
            del inputs, outputs, response
            gc.collect()
            torch.cuda.empty_cache()
    
            model._clear_cuda_cache()
            
        except Exception as e:
            error_msg = f"\n\nError processing {case}:\n{traceback.format_exc()}"
            print(error_msg)
            failed_cases.append(case)
    
    # Save failed cases
    if failed_cases:
        with open(os.path.join(cases_dir, 'failed_deepseek-vl-small.txt'), 'w') as f:
            f.write("\n".join(failed_cases))



