# README

## Project Overview

This project is based on the ABSA (Aspect‑Based Sentiment Analysis) datasets of SemEval-2014 and SemEval-2015, and uses two pre-trained Transformer models, BERT and RoBERTa, to complete end-to-end experiments on target-oriented sentiment analysis, including:

* **Data loading and preprocessing**: unified CSV format, label mapping, stratified/balanced splitting;
* **Model fine-tuning**: BERT/RoBERTa fine-tuning, AdamW optimization, cross-entropy loss;
* **Performance evaluation**: Accuracy, Macro‑F1, Precision/Recall, confusion matrix, AUC‑ROC/PR curves;
* **XAI explanation**: Captum Integrated Gradients, visualization of Top‑K important words and coverage curves.

The main organization is as follows:

```
bert_roberta_main_new.ipynb   # Main Notebook: including data segmentation, model training, evaluation and XAI implementation
requirements.txt              # Project packages requirements
README.md                     # Readme file
```

## Instructions

1. Open `bert_roberta_main_new.ipynb` and make sure to install the following dependencies:

    ```bash
    pip install -r requirements.txt
    ```
2. Make sure your project format complies with the following format.
    ```
    Project/
    │
    └── data/
        ├── SemEval2014/
        │   ├── Laptops_Test.csv
        │   ├── Laptops_Train.csv
        │   ├── Restaurants_Test.csv
        │   └── Restaurants_Train.csv
        └── SemEval2015/
            ├── Laptops_Test.csv
            ├── Laptops_Train.csv
            ├── Restaurants_Test.csv
            └── Restaurants_Train.csv

    ```
3. Tune hyperparameters (in the Config Variables module).
4. Run code.

## Reference

* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in neural information processing systems, 30*.
* Devlin et al. (2019). *BERT: Pre-training...* NAACL.
* Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019, June). Bert: Pre-training of deep bidirectional transformers for language understanding. *In Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers)* (pp. 4171-4186).
* Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., ... & Stoyanov, V. (2019). Roberta: A robustly optimized bert pretraining approach. *arXiv preprint arXiv:1907.11692*.
