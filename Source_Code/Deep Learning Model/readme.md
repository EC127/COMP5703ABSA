# README

## Project Overview

This project is built on the ABSA (Aspect-Based Sentiment Analysis) datasets from SemEval-2014 and SemEval-2015, using deep-learning models (RNN, GRU, LSTM, BiLSTM) to carry out an end-to-end, target-oriented sentiment analysis experiment, specifically including:

**Data Loading & Preprocessing**
Read CSV: `pd.read_csv(file_path)`, normalize column names to lowercase
Text-column detection: automatically use `"text"` or `"sentence"`
Label filtering & mapping: keep `positive`/`neutral`/`negative`, map to 2/1/0
Text cleaning: `clean_text()` + regex (lowercase, remove punctuation, collapse whitespace)
Tokenization: `TreebankWordTokenizer()`
Vocabulary building & filtering: count token frequency, include only if `freq >= 2`
Tokens → IDs: `tokens_to_ids()`, map unknown tokens to `<UNK>`
Train-test split: `train_test_split(..., stratify=df["label"])`
Dataset & DataLoader: custom `ABSADataset` + `collate_fn` for padding

**Model training and hyperparameter search**
Custom model: ABSAModel(nn.Module), supports RNN/LSTM/BiLSTM/GRU
Loss and optimizer: nn.CrossEntropyLoss() + optim.Adam(model.parameters(), lr)
Grid search: traverse the combination of embedding_dim, hidden_dim, and lr
Training loop: record train_losses, calculate accuracy and f1_score on the validation set every epoch
Best model: select the optimal parameters of each model according to Macro-F1, and track the Overall best

**Performance evaluation and visualization**
Indicator calculation: accuracy_score, f1_score, confusion_matrix
Confusion matrix visualization: ConfusionMatrixDisplay + Matplotlib
Curve drawing: Epoch vs Accuracy/F1/Loss; bar chart shows the final Accuracy/F1 / Loss
Result summary: Aggregate all grid search results into DataFrame and print the best parameters for each model

**XAI Explanation (LIME)**
Initialize the interpreter: LimeTextExplainer(class_names=[…])
Encapsulate the prediction function: model_predict_proba(texts) returns Softmax probability
Generate explanation: explainer.explain_instance(text, model_predict_proba, labels=[0,1,2], num_features=10)
Visualization: .as_pyplot_figure(label=…) and set the title (model type + hyperparameters)

The main organization is as follows:

```
Final_code_each_model.ipynb        # Main Notebook
Dataset_equal_distribution.ipynb   # This code just shows how to evenly distribute the large original data into a new data set. It does not need to be actually run.
requirements.txt                   # Project packages requirements
README.md                          # Readme file
```
The initial code is to debug the GPU. My model training is run through the GPU, but it can also run without the GPU.

## Instructions

1. Open `Final_code_each_model.ipynb` and make sure to install the following dependencies:

    ```bash
    pip install -r requirements.txt
    ```
2. Make sure that the downloaded code and dataset are in the same folder. (The datasets used in Deep Learning are all in the uploaded dataset) Below is a description of each dataset.
    ```
    Project/
    (Laptop_Train_v2.csv) is the SemEval 2014 Laptop ducument
    (Restaurants_Train_v2.csv) is the SemEval 2014 Restaurants ducument
    (Restaurants_Train_equal_distribution.csv) is a processed data set because the SemEval 2015 Restaurants dataset is too large
    (Laptops_Train_equal_distribution.csv) is a processed data set because the SemEval 2015 Laptop dataset is too large
    ```
3. Run code.

How to Run: To change the dataset, just change it in **file_path** and re-run all cells. For the final visualization, change different models in **selected_key**, where overall refers to the best model with the highest f1 value among the four models. Then run the last two cells. The changed names are described in the code. And you can change the label value in fig = exp.as_pyplot_figure(label=2) in the second-to-last cell to observe how the word is determined for different categories. (2 is positive, 1 is neutral, and 0 is negative)