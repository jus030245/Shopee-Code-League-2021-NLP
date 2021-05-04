# Shopee-Code-League-2021-NLP
This work is done within 6 days from starting learning NLP to hand in the prediction. This work is mean to record the original work presented without discussion and time pressure. The final work scores 57% in accuracy and ranked 14% out of 1000+ teams across Asia.
Possible measure for better prediction will be discussed later.

Problem Definition:

Data:
Training Set 300,000 lines of Indonesian Address. POE/street for each Address (Answer can be ''.)
Testing Set 50,000 lines of Indonesian Address.
Data will not be provided in this work.

Challenge:
1. The data are all written in Indonesian which makes it hard for EDA and feature engineering.
2. The task was given without previous experience and knowledge in NLP.

Workflow:
1. EDA
2. Define Task - NER
3. Grouping typos/abbreviations with desired words
4. Data Preprocessing - Cleaning and Labeling (Turning sequences into tokens with associated labels)
5. Model Building - BERT-indonesian()
6. Make Prediction
7. Join the tokens
8. Adjust punctuation signs and Correct the typos/abbreviations from the dictionary
9. Submit 80% accuracy token-wise, 56% accuracy seqence-wise, Ranked top 14%
