# SAB_measure
# Causal Entity Annotation and Extraction in Financial Texts
# 📘 Overview
This project is based on the paper by Chen (2025):
🔗 SSRN Paper - "[Measuring Self-serving Attribution Bias Using Text Mining](https://download.ssrn.com/2025/2/10/5130571.pdf?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEAgaCXVzLWVhc3QtMSJHMEUCIQC0pci69wHl7%2BzNWDc9VI7OJxxsttsAnr5zUEp%2BAoH79QIgcH186XgRdD6zLViuBKAOPTyemVFWjlSuWrRJkpGz%2FwkqxgUI4f%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARAEGgwzMDg0NzUzMDEyNTciDFKrQBxHBwC9D3ZxviqaBVyYAJmG2E2oFB8cI8c7H%2BNUetYx098q068Bt8gt0cyOdgLhObVLHloEB3oHW52WeymagJXTc5iFNoq1pE7g64wCHw94OSTmyffayGcmgXgBlYFrVZvqSQRVD2%2Beh629aQ6BPWUsuKYeKpWzy3MKo5yquej38zpVDJ7vnPWSu1ZveoGcvsnFF%2BzuAN0N6taSP0k8NWK52FTQc5UJO0hhrV823gRH4irCyBhUZIib%2FN2kZOkw6q6SIPT1SgIT8l4AYaMaYfnvYunTnToxY6q48vy08OOudYjVJZDfP8E9ciF%2BhnKAc55oknseS7jYjo9VpUJbjaYStnivAuMxb7pbIyy5qYuhjyERv9yQuSoBgsMdBmmT%2B4IQZxDnWSxUfyZwM90hRrffPhmGsFM9jViSdj0roTvf7OWHad0jW8aqjJMkgekdour2LeyVF36O95cRAy0dLMGnMb80NdnNHTq2i1%2Bvl6tes8NYoN5ldFnnyDT8TZbDaRiMbj9kFxl%2BHa9kePHRZdcU0qItpBzf1He1M1tKhOgBtTQaw9nkIJE2i0NH64c2er8RtEDIPFA0R6BHNUhIC6joa21jCm4G4u5w%2B0fPZzWPzoyIzpTSHiPrmarl0Pjd67s5bbi2zrTV3ljblM6ddKvIw0LKLKdeerlc6voez5HxrGQee6Xg8bTtp79QK%2BrfMsHTzupfXEHdk50Sc3Nus%2BVDx1U4wMA6%2BXI3f0cU4IKvoZ6UFDIdzusgOgyk%2BiVobIbyq8T8DWFdco1TkHuF11vcZJ67cH0snvRGstXJyHtXQiVE6K1uVDWAI%2FdpLpPXdyNrxZAfg40JLDEaKyxF8u%2BkHnwFr4fpe8AJBaTWqe6Ol4zNgnwL5mhMdCWZocjaatcPA7KPdTCZrKjCBjqxAZ14YtGTkLcKnS7pzaoiBX9ZXzJk1MLIyck4HBLtaz2COBKabJ3XKImm2B8JxKl%2BSHcBf4ahhuiWxvZ2aqXqshpA7W1mvSiYhpWn0R%2F77xbsvCAeh0CDDkrOGreSpSoU9ONNh1sr1Q3dEUrsv2J24EJ8swd5MD457yLL7ykM4Rwe8euksvyPxPO6WTLNYTLAlX88MFrxCfKWgKp7JpsYHIVADdskTmRhifVi6bHfvlgV3Q%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20250612T004435Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAUPUUPRWEQPMPJXX3%2F20250612%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=c4866b0ca06d3dd49013e10d5326a36ae8ce147b1487a5c3fc39e122cb342f42&abstractId=5130571)"

We provide a fine-grained, expert-annotated dataset that captures causal entities in Management Discussion & Analysis (MD&A) sections of financial reports. The annotations follow definitions proposed in Chen (2025):

- Internal Cause (IC): Factors within management’s control

- External Cause (EC): Environmental or market factors

- Positive Result (PR): Favorable outcomes

- Negative Result (NR): Unfavorable outcomes

We fine-tune the best-performing BERT model identified in Chen (2025).


# 🏷 Labeling Method
We adopt the BIO sequence labeling format to identify both the boundaries and types of causal entities:

B- (Begin): Marks the start of an entity

I- (Inside): Indicates continuation within the same entity

O: Marks tokens outside any entity

An example of labeled data is shown in Figure 1 in the full documentation.
![entity_sample](https://github.com/user-attachments/assets/400d2254-c448-47e1-be14-f534b0fd6b65)

# 📊 Model Performance
Model training was conducted on:

- 600 training samples

- 200 validation samples

- 200 test samples

Our key evaluation metric is recall, as BERT tends to classify more fine-grained spans than human experts and may omit adjectives, adverbs, or verbs. For example, an expert annotation of [10,30] may be split by the model into [10,13] and [15,30].

- Our model achieved recall > 86%

- In 20 repeated experiments, recall remained stable between 85% and 90%

- These results indicate sufficient reliability for large-scale analysis

Figure 2 shows the training results in detail.
![evaluation](https://github.com/user-attachments/assets/1da991d4-87f5-4cc1-86ce-22e437aa4a70)

# ⚙️ Fine-tuning Parameters
We use HuggingFace Transformers for model training.

from transformers import AdamW, get_linear_schedule_with_warmup

learning_rate = 2e-5
num_epochs = 4
batch_size = 16  # or 8 for smaller GPUs
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
total_steps = len(dataloaders_dict["train"]) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
We also apply:

…… Gradient clipping: clip_grad_norm_ with max norm = 1.0
…… Automatic GPU detection
…… Train/validation split processing

# 🧪 Usage
We provide a standardized format for inputting MD&A text samples.
If users provide data in the same structure as shown in Figure 3,
![input](https://github.com/user-attachments/assets/4a461c6c-23c2-4426-9ad7-6cdbea7d7a7e)
the model will return outputs as illustrated in Figure 4.
![output](https://github.com/user-attachments/assets/40fffd60-7bb3-4e18-8788-9d139603b900)
The output includes:

- Predicted labels for all entities

- Classification of each entity into one of the four causal types

This structure is designed to support easy inspection and downstream analysis.
![prediction_sample](https://github.com/user-attachments/assets/09bc6505-9907-4a78-af0c-88b0a37bf6d9)

# 📁 File Structure (Optional)

├── data/

│   ├── train.json

│   ├── val.json

│   └── test.json

├── model/

│   └── fine_tuned_bert.bin

├── scripts/

│   └── train_model.py

├── README.md


# 👩‍💼 Contact
For questions or collaboration, feel free to contact us via GitHub Issues or email.





