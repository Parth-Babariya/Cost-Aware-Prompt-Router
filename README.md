# 🧠 Cost-Aware Prompt Router

This repository contains machine learning pipelines designed to evaluate, predict, and route user prompts to the most appropriate Large Language Model (LLM). By predicting model performance or directly classifying the optimal endpoint, this system helps balance response quality with inference costs.

## 📁 Repository Structure

* **`mainLatest1_clean.ipynb` / `mainLatest1.ipynb`**: 
  Contains the pipeline for the **Soft-Label LLM Router**. It fine-tunes a `roberta-large` sequence classification model on a soft-label dataset to dynamically route prompts to the best open-source model via the Together AI API.
* **`model.ipynb`**: 
  Contains the **LLM Performance Predictor**. It trains regression models (`FastText` and `DistilBERT`) to predict the expected performance score of various LLMs (e.g., GPT-4, Claude-v2, LLaMA, Mistral) for a given prompt, including an ensembling method to pick the best model.

## ✨ Key Features

1. **Smart Prompt Routing**: Analyzes the input prompt and confidently routes it to the most suitable expert model:
   - *Academic/Complex tasks* ➔ `Mixtral-8x7B`
   - *Coding/Math tasks* ➔ `Qwen2.5-Coder-32B`
   - *General explanations* ➔ `Mistral-7B`
2. **Cost Savings Analyzer**: Calculates the estimated cost savings of using the smart router versus sending all requests to a heavy, expensive baseline model.
3. **Multi-Model Performance Regression**: Trains `FastText` and `DistilBERT` models to output a predicted performance score (between 0.0 and 1.0) for an LLM on a specific prompt.
4. **Ensemble Selection**: Combines the predictions from DistilBERT and FastText using a weighted average to robustly determine the highest-performing LLM for any unseen prompt.

---

## 📖 User Guide

This guide will walk you through setting up the environment, preparing your data, training the models, and running the prompt router.

### 1. Prerequisites & Installation

**Environment:**
We highly recommend running these notebooks in a Jupyter environment with GPU acceleration (such as Google Colab with a T4 GPU) to handle the training of `roberta-large` and `DistilBERT` efficiently.

**API Keys:**
To use the live routing features, you will need a [Together AI](https://www.together.ai/) API key to query the open-source LLMs.

**Installation:**
Install the required Python packages by running the following command in your terminal or notebook cell:


pip install transformers datasets evaluate fasttext torch requests pandas numpy scikit-learn tiktoken

### 2. Data Preparation
Before training, ensure your datasets are placed in the correct directories:

1. **For the Router (mainLatest1_clean.ipynb):** Ensure Soft_Label_Balanced_Dataset.csv is uploaded to your working directory. This file should contain the prompts and their target model labels.

2. **For the Predictor (model.ipynb):** Place your dataset inside a processed_data/ directory. The structure should be organized by the target model name.

### 3. Training the Performance Predictors (model.ipynb)
This step trains the regression models that predict how well a specific LLM will perform on a scale of 0.0 to 1.0.

1. Open `model.ipynb`.

2. To train the **FastText** regressors, ensure the `main()` function is set to  
   `model_type="fasttext"` and run the notebook.

3. To train the **DistilBERT** regressors, change the parameter to  
   `model_type="distilbert"` and execute.

4. **Testing an unseen prompt:** Use the built-in ensemble function to test a new prompt.  
   The script will automatically weigh the FastText and DistilBERT outputs to recommend the best LLM:

```Python
test_new_prompt("Explain relativity simply", weight_distilbert=0.5)

```
### 4. Training the Smart Router (mainLatest1_clean.ipynb)
This notebook trains a RoBERTa sequence classifier to dynamically route prompts based on soft labels, bypassing the need to run regression on every single model during inference.

1. Open `mainLatest1_clean.ipynb`.

2. Execute the setup cells to load `roberta-large` and initialize the Hugging Face `Trainer`.

3. Train the model on your `Soft_Label_Balanced_Dataset.csv`.

4. **Set your API Key:** Before running the inference cells, set your Together API key as an environment variable:

```Python
import os
os.environ["TOGETHER_API_KEY"] = "your_api_key_here"
```
### 5. Running Inference & Cost Analysis
Once the router is trained, you can pass live prompts to it. The router will calculate the confidence scores for each model class (e.g., Mixtral, Qwen, Mistral) and automatically forward the prompt to the winning model's API endpoint.

Run the routing cell:

```Python
prompt = "Why am I getting a KeyError in Python when using a dictionary, and how can I fix it?"
response, endpoint = route_prompt(prompt)
```

## 📈 Example Output

**Router Output:**

**📝 Prompt:** Why am I getting a KeyError in Python when using a dictionary?  
**📢 Routed to:** Qwen/Qwen2.5-Coder-32B-Instruct  
**🔢 Confidence:** 0.433  
**🔍 Probabilities:**  
&nbsp;&nbsp;&nbsp;&nbsp;mistralai/mixtral-8x7b-chat : 0.3368  
&nbsp;&nbsp;&nbsp;&nbsp;Qwen/Qwen2.5-Coder-32B-Instruct : 0.4330  
&nbsp;&nbsp;&nbsp;&nbsp;mistralai/mistral-7b-chat : 0.2301 
  

## Cost Summary:

**Cost Summary for 3 Prompts**  

🧱 Baseline (all to Mixtral-8x7B): $0.0053  
🧠 Smart routing total           : $0.0039  
💰 Estimated savings             : 25.1%  
