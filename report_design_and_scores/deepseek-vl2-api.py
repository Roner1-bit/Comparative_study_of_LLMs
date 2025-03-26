import os
from PIL import Image
import replicate

os.environ['REPLICATE_API_TOKEN'] = 'r8_294ZyXjpBrZjEgfdlOMinARuwRh3LOa4btswO'
def load_image(image_path):
    return open(image_path, "rb")

# Directory containing the images
cases_dir = r'D:\final_drive_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250218T095105Z-001\(final)merged_images_with_labels_order_and_folders_mask_normalized'

for case in os.listdir(cases_dir):
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

    system_prompt = """<image>\nConsider that you are a professional radiologist with several years of experience and you are now treating a patient. Write a fully detailed diagnosis report for this case, avoiding any potential hallucination and paying close attention to all of the batch images attached to this message.

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

    # Process all images
    images = []
    for image_file in image_files:
        image_path = os.path.join(case_dir, image_file)
        img = load_image(image_path)
        images.append(img)

    

    # Check if the response file already exists
    response_path = os.path.join(case_dir, 'deepseek-vl2-api.txt')
    if os.path.exists(response_path):
        print(f"Response file already exists for case {case}. Skipping API call.")
        continue
    

    print(f"starting api for case {case}...")
    # Generate the response
    output = replicate.run(
    "deepseek-ai/deepseek-vl2:e5caf557dd9e5dcee46442e1315291ef1867f027991ede8ff95e304d4f734200",
    input={
        "image":img,
        "top_p": 0.9,
        "prompt": system_prompt + user_prompt,
        "temperature": 0.7,
        "max_length_tokens": 4096,
        "repetition_penalty": 1.1
    }
)
    response_text = output
    print(response_text)

    # Save the response
    with open(response_path, 'w', encoding='utf-8') as f:
        f.write(response_text)

    print(f"Response saved for case {case}.")