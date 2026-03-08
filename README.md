# Cost-Aware-Prompt-Router

This repository contains machine learning pipelines designed to evaluate, predict, and route user prompts to the most appropriate Large Language Model (LLM). By predicting model performance or directly classifying the optimal endpoint, this system helps balance response quality with inference costs.

## 📁 Repository Structure

* **`mainLatest1_clean.ipynb` / `mainLatest1.ipynb`**: 
  Contains the pipeline for a **Soft-Label LLM Router**. It fine-tunes a `roberta-large` sequence classification model on a soft-label dataset to dynamically route prompts to the best open-source model via the Together AI API.
* **`model.ipynb`**: 
  Contains the **LLM Performance Predictor**. It trains regression models (`FastText` and `DistilBERT`) to predict the expected performance score of various LLMs (e.g., GPT-4, Claude-v2, LLaMA, Mistral) for a given prompt, including an ensembling method to pick the best model.
* **`README.md`**: Project documentation.

## ✨ Key Features

1. **Smart Prompt Routing**: Analyzes the input prompt and confidently routes it to the most suitable expert model:
   - *Academic/Complex tasks* ➔ `Mixtral-8x7B`
   - *Coding/Math tasks* ➔ `Qwen2.5-Coder-32B`
   - *General explanations* ➔ `Mistral-7B`
2. **Cost Savings Analyzer**: Calculates the estimated cost savings of using the smart router versus sending all requests to a heavy baseline model.
3. **Multi-Model Performance Regression**: Trains `FastText` and `DistilBERT` models to output a predicted performance score (between 0.0 and 1.0) for an LLM on a specific prompt.
4. **Ensemble Selection**: Combines the predictions from DistilBERT and FastText using a weighted average to robustly determine the highest-performing LLM for any unseen prompt.

## 🛠️ Dependencies

To run the notebooks, you will need the following libraries. Most can be installed directly within Google Colab:


pip install transformers datasets evaluate fasttext torch requests pandas numpy scikit-learn tiktoken

🚀 Usage Guide
1. Training and Using the Router (mainLatest1_clean.ipynb)
This notebook handles soft-label classification.

Train: It loads Soft_Label_Balanced_Dataset.csv, maps soft targets, and fine-tunes roberta-large using the Hugging Face Trainer API.

Inference: Uses the fine-tuned model to predict the probabilities of a prompt belonging to a specific model class.

API Integration: Once routed, it automatically queries the selected model using the Together API.

Cost Analysis: At the end of the notebook, run the cost summary cell to calculate how much money the router saved compared to a static baseline.

2. Training the Performance Predictors (model.ipynb)
This notebook trains individual regressors for multiple target LLMs.

Data Prep: Place your dataset inside a processed_data/ directory, organized by model name (e.g., processed_data/gpt-4-1106-preview/train.csv).

Train FastText/DistilBERT: Call main(model_type="fasttext") or main(model_type="distilbert") to iterate through the folders and train regressors based on prompt strings and their corresponding scores.

Ensemble Testing: Use the test_new_prompt("Your prompt here") function. The script will load the available FastText and DistilBERT checkpoints, calculate a weighted ensemble score, and output the recommended LLM.

📈 Example Output
Router Output:

Plaintext
📝 Prompt: Why am I getting a KeyError in Python when using a dictionary?

📢 Routed to: Qwen/Qwen2.5-Coder-32B-Instruct
🔢 Confidence: 0.433
🔍 Probabilities:
   mistralai/mixtral-8x7b-chat            : 0.3368
   Qwen/Qwen2.5-Coder-32B-Instruct        : 0.4330
   mistralai/mistral-7b-chat              : 0.2301
Cost Summary:

Plaintext
--- Cost Summary for 3 Prompts ---
🧱 Baseline (all to Mixtral-8x7B): $0.0053
🧠 Smart routing total           : $0.0039
💰 Estimated savings             : 25.1%
⚠️ Notes
Together API Key: You must set your TOGETHER_API_KEY environment variable in the routing notebook to execute API calls.

Hardware: It is highly recommended to run these notebooks on a GPU (e.g., T4 on Google Colab) to handle roberta-large and DistilBERT training times.
