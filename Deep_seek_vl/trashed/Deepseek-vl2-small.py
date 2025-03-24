import os
from PIL import Image
from transformers import AutoModelForCausalLM, AutoConfig
from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images
import torch
import gc
import traceback
from accelerate import init_empty_weights, load_checkpoint_and_dispatch




if __name__ == '__main__':
    cases_dir = '/media/RLAB-Disk01/(final)merged_images_with_labels_order_and_folders_mask_normalized/'
    cache_dir = '/media/RLAB-Disk01/Large-Language-Models-Weights'
    model_id = "deepseek-ai/deepseek-vl2-small"
    offload_folder = '/media/RLAB-Disk01/Large-Language-Models-Weights'

    # Memory optimization settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"

    # Initialize model with memory-efficient loading
    processor = DeepseekVLV2Processor.from_pretrained(model_id)
    tokenizer = processor.tokenizer

    # Load model with Accelerate's memory optimization
    with init_empty_weights():
        config = AutoConfig.from_pretrained(model_id, 
                                            trust_remote_code=True,  
                                            use_safetensors=True,
                                            cache_dir=cache_dir,
                                            attn_implementation='eager',
                                            offload_folder=offload_folder,
                                            offload_state_dict=True, 
                                            device_map="auto",
                                            )
        model = DeepseekVLV2ForCausalLM(config)

    model = load_checkpoint_and_dispatch(
        model,
        "/media/RLAB-Disk01/Large-Language-Models-Weights/models--deepseek-ai--deepseek-vl2-small/snapshots/6033e16432a1d771cf9fe4a6f894ff5e5e1459af",
        device_map="auto",
        no_split_module_classes=["DeepseekVisionModel"],
        dtype=torch.bfloat16,
        offload_folder=offload_folder,

      
    )

    # Get cases to process
    cases_to_process = []
    for case in os.listdir(cases_dir):
        case_dir = os.path.join(cases_dir, case)
        if os.path.isdir(case_dir):
            response_path = os.path.join(case_dir, 'deepseek-vl-small.txt')
            if not os.path.exists(response_path):
                cases_to_process.append(case)
    print(f"Found {len(cases_to_process)} cases to process.")

    def build_conversation(case_dir, user_prompt, system_prompt):
        image_files = [f for f in os.listdir(case_dir) if f.lower().endswith('.png')][-1:]
        return [
            {
                "role": "<|User|>",
                "content": f"{system_prompt}\n{user_prompt}\n<image>",
                "images": [os.path.join(case_dir, image_files[0])]
            },
            {"role": "<|Assistant|>", "content": ""}
        ]

    failed_cases = []
    system_prompt_template = """Consider that you are a professional radiologist with several years of experience and you are now treating a patient. Write a fully detailed diagnosis report for this case, avoiding any potential hallucination and paying close attention to all of the batch images attached to this message.

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


    for idx, case in enumerate(cases_to_process):
        case_dir = os.path.join(cases_dir, case)
        try:
            # Load clinical info
            clinical_path = os.path.join(case_dir, 'diagnostic_prompt.txt')
            if not os.path.exists(clinical_path):
                print(f"Missing clinical info for {case}")
                failed_cases.append(case)
                continue
                
            clinical_info = open(clinical_path).read()
            user_prompt = f"""You will be given batches of images, which are different sequences of MRI scans. 
    The images are for patients who are likely to have a brain tumor. Each image will contain up to 10 slices for 5 different sequences and the segmentation masks for the tumor at the bottom row of the image. 
    Additional clinical data about the patient is: 
    {clinical_info}"""
            # Prepare conversation and images
            conversation = build_conversation(case_dir, user_prompt, system_prompt_template)
            pil_images = load_pil_images(conversation)

            # Process inputs with memory optimization
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                
                inputs = processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True,
                 ).to(model.device)
                
                inputs_embeds = model.prepare_inputs_embeds(**inputs)

                print(f"Messages prepared for case {case}")
                #Generate response

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

                
                # Move inputs to model device

                outputs = model.language.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=4096,
                    temperature=0.7,
                    generation_config=generation_config,
                    do_sample=False,
                    use_cache=True
        )
            # Process and save response
            response = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)

            print(f"Response generated for case {case}, length: {len(response)} characters")

            with open(os.path.join(case_dir, 'deepseek-vl-small.txt'), 'w', encoding='utf-8') as f:
                f.write(response.split("[/INST]")[-1].strip())

            # Memory cleanup
            del inputs, outputs, response, inputs_embeds
            if idx % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            print(f"Processed case {idx+1}/{len(cases_to_process)}: {case}")

        except Exception as e:
            error_msg = f"\nError processing {case}:\n{traceback.format_exc()}"
            print(error_msg)
            failed_cases.append(case)
            if idx % 5 == 0:
                torch.cuda.empty_cache()

    # Save failed cases
    if failed_cases:
        with open(os.path.join(cases_dir, 'failed_deepseek-vl-small.txt'), 'w') as f:
            f.write("\n".join(failed_cases))