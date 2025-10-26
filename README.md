Image Caption Generator using BLIP (Transformers)

This project demonstrates automatic image caption generation using the BLIP (Bootstrapped Language-Image Pretraining) model from Hugging Face Transformers. It allows users to upload an image and generate diverse, descriptive captions using deep learning and natural language processing.

🚀 Features

Generate five unique captions for any uploaded image

Uses Salesforce BLIP Image Captioning Base Model

Provides a visual comparison of caption lengths

Works efficiently on Google Colab using GPU acceleration

Easy to use and extend for custom applications

🧠 Model Overview

BLIP (Bootstrapped Language-Image Pretraining) is a vision-language model trained to understand both images and text. It generates meaningful and natural captions by aligning visual and linguistic features.

Model used: Salesforce/blip-image-captioning-base
Framework: Hugging Face Transformers

⚙️ Workflow Summary

Install required Python libraries

Load BLIP processor and model

Upload and display an image

Generate multiple captions with sampling for diversity

Visualize caption lengths using a bar chart

📊 Output Example

Input: Uploaded image (e.g., scenery, animal, or object)

Output:

5 diverse natural language captions

Caption length visualization chart

Example captions might look like:

“A group of people walking through a forest”

“Hikers exploring a scenic trail”

“A family enjoying nature in the woods”

🧩 Technologies Used

Python

PyTorch

Transformers (Hugging Face)

Matplotlib

Scikit-learn

Google Colab

💡 Future Improvements

Integrate semantic similarity scoring to measure caption diversity

Upgrade to BLIP-2 for more accurate descriptions

Deploy as a web app using Streamlit or Flask

Add evaluation metrics like BLEU or ROUGE

