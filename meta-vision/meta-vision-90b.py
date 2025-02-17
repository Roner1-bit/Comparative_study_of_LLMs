#%%
import os
from openai import OpenAI
import base64


def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# client = OpenAI(
# api_key = "LA-abb308a9fc364cdbb43a335e611157c80614c6c329d34213be26041d8f252449",
# base_url = "https://api.llama-api.com"
# )

client = OpenAI(
api_key = "token",
base_url = "https://openrouter.ai/api/v1"
)

# Directory containing the images
cases_dir = r'C:\Users\Elboardy\Desktop\final_dataset_17_2\(final)merged_images_with_labels_order_and_folders_mask_normalized-20250214T104437Z-001\(final)merged_images_with_labels_order_and_folders_mask_normalized'

valid_cases = [
    case
    for case in os.listdir(cases_dir)
    if not os.path.exists(os.path.join(cases_dir, case, 'meta-vision-90b.txt'))
]

print(f"Number of folders without meta-vision-90b.txt: {len(valid_cases)}")
print("Folders without meta-vision-90b.txt:")
for case in valid_cases:
    print(case)

#%%
modified_folders = []

for case in valid_cases:
    case_dir = os.path.join(cases_dir, case)
    image_files = [
        f for f in os.listdir(case_dir)
        if f.lower().endswith(('.png'))
    ]

    clinical_information = open(os.path.join(case_dir, 'diagnostic_prompt.txt')).read()
    

    system_prompt="""You are a professional radiologist with several years of experience and you are now helping me treat a patient. Write a fully detailed radiology diagnosis report for this case, avoiding any potential hallucination and paying close attention to all of the batch images attached to this message.

Use the following structure for the report:


## Radiologist's Report

### Patient Information:
- **Age:** 65
- **Sex:** Male
- **Days from earliest imaging to surgery:** 1
- **Histopathological Subtype:** Glioblastoma
- **WHO Grade:** 4
- **IDH Status:** Mutant
- **Preoperative KPS:** 80
- **Preoperative Contrast-Enhancing Tumor Volume (cm³):** 103.21
- **Preoperative T2/FLAIR Abnormality (cm³):** 36.29
- **Extent of Resection (EOR %):** 100.0
- **EOR Type:** Gross Total Resection (GTR)
- **Adjuvant Therapy:** Radiotherapy (RT) + Temozolomide (TMZ)
- **Progression-Free Survival (PFS) Days:** 649
- **Overall Survival (OS) Days:** 736

#### Tumor Characteristics:

#### Segmentation Analysis:

#### Surgical Considerations:

### Clinical Summary:

### Recommendations:

### Prognostic Considerations:

### Follow-Up Plan:

### Additional Notes*(if any)*:

Ensure all findings from all of the images and clinical data provided. Please mention at the end of the report how many images were reviewed """


    user_prompt = f"""You will be given batches of images, which are different sequences of MRI scans. 
    The images are for patients who are likely to have a brain tumor. Each image will contain up to 10 slices for 5 different sequences and the segmentation masks for the tumor at the bottom row of the image. 
    Additional clinical data about the patient is: 
    {clinical_information}."""

    content_list = [ {
          "type": "text",
          "text": user_prompt,
        }]
    
  
    print('---------------------------------------------------------------------------------')

    image_files = [image_files[-1]]

    for image_file in image_files:
        image_path = os.path.join(case_dir, image_file)
        
        image_data = encode_image(image_path)
        
        content_list.append({
          "type": "image_url",
          "image_url": {
            "url":  f"data:image/png;base64,{image_data}"
          },
        })




    response = client.chat.completions.create(
        model='meta-llama/llama-3.2-90b-vision-instruct',  
        messages=[
           {
                "role": "system_admin",
                "content": system_prompt
           },
            {
                "role": "user",
                "content": content_list
            }
        ],
        temperature=0.7,
        max_tokens=4096,
        top_p=0.9,
        )

        # Print the response
    print(f"Response for {image_file}:")
    print(response)
    print(response.model_dump_json(indent=2))
    meta_response = response.choices[0].message.content
    print(meta_response)
    print('----------------------------------')
    with open(os.path.join(case_dir, 'meta-vision-90b.txt'), 'w', encoding='utf-8') as f:
        f.write(meta_response)

    modified_folders.append(case)

print("Folders modified:")
for folder in modified_folders:
    print(folder)


