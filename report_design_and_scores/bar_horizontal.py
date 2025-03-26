import pandas as pd
import matplotlib.pyplot as plt

# Sample CSV data as a dictionary
data = {
    "Model": [
        "Meta-Vision 3.2-90B", "Qwen2-VL-72B", "NVLM-D-72B", "Meta-Vision3.2-11B", 
        "DeepSeek-VL2 27B", "DeepSeek-VL2-small-16B", "Qwen2-VL-7B", "Llava-Med 1.5 7B", 
        "Qwen2-VL-2B", "phi-3.5-vision-instruct-4B", "DeepSeek-VL2-tiny-3B", 
        "paligemma2-28B-pt-896", "paligemma2-10B-pt-896", "paligemma2-3B-pt-896"
    ],
    "Accuracy": [
        70.19, 68.81, 63.19, 57.56, 53.44, 51.88, 41.31, 26.42, 
        23.88, 23.12, 15.81, 3.19, 1.62, 0.87
    ]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Sort by Accuracy in descending order
df = df.sort_values(by="Accuracy", ascending=False)

# Plotting
plt.figure(figsize=(12, 6))
bars = plt.barh(df["Model"], df["Accuracy"], color="#B983FF", edgecolor="black", linewidth=1.2)

# Make the bars rounded
for bar in bars:
    bar.set_linewidth(1.5)  # Slight border for better visibility
    bar.set_capstyle('round')  # Rounds the end of bars
    bar.set_edgecolor("#8A68C6")  # Slightly darker purple for the edges

plt.xlabel("Accuracy (%)", fontsize=12)
plt.ylabel("Model", fontsize=12)
plt.title("Model Accuracy Comparison", fontsize=14)
plt.xlim(0, 100)  # Set x-axis to 100%

# Improve visibility of model names
plt.gca().invert_yaxis()  # Invert y-axis to match the given chart order
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Display percentage labels on bars
for index, value in enumerate(df["Accuracy"]):
    plt.text(value + 1, index, f"{value:.2f}%", va='center', fontsize=10, color='black')

# Add a legend with only text (no color or markers)
plt.legend(["O1 Model's Judgement"], loc="lower right", frameon=False, fontsize=12)

plt.tight_layout()  # Adjust layout for better visibility
plt.show()
