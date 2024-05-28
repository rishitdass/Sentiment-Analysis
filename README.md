# Amazon Review Sentiment Analysis

## Overview

The Amazon Review Sentiment Analysis project aims to analyze sentiment from Amazon reviews using various machine learning models inclusing SVMs,random forest,Decision trees,GaussianNB and MultiNomialNB. Through experimentation, it has been found that Logistic Regression and LSTM deep learning models provide the best performance, achieving approximately 85% and 87% accuracy respectively. Since The dataset was huge i have utilied 100,000 rows of the Dataset due to resource restraints and to optimize training time.

## Initial Development Phase

During the initial development phase, various machine learning models were evaluated for sentiment analysis on Amazon reviews. The following table shows the performance metrics of different models:

![Sentiment Analysis](https://github.com/rishitdass/Sentiment-Analysis/blob/main/image.png)


Based on these results, it was observed that Logistic Regression outperformed other models in terms of accuracy. Consequently, further efforts were focused on tuning the Logistic Regression model for optimal performance.

Furthermore i Decided to use LSTM model for better results.
The model summary is as follows:
![LSTM model](https://github.com/rishitdass/Sentiment-Analysis/blob/main/image2.png)
## Features

- **Data Collection:** Scrapes Amazon reviews for analysis.
- **Preprocessing:** Cleans and preprocesses the raw text data.
- **Model Training:** Trains machine learning models including Logistic Regression and LSTM.
- **Evaluation:** Evaluates model performance using accuracy metrics.
- **Prediction:** Allows users to input new reviews for sentiment analysis.

## Additional Features

In addition to the above features, the project utilizes the following:

- **GridSearchCV:** Utilizes GridSearchCV for hyperparameter tuning to optimize model performance.

- **TF-IDF** used TF-IDF vectorizer for feature extraction.

- **LSTM**  i have used Long Short term memory networks which are a type of RNN.
- **Text Summarizer**
 In addition to sentiment analysis, this project incorporates a text summarization feature. The text summarizer extracts the most important information from lengthy reviews, generating concise summaries without losing key details.

### Features

- **Automatic Summarization**: The text summarizer intelligently selects essential sentences from the input text, condensing it into a shorter form while retaining the original meaning.

- **Enhanced Readability**: Summaries are crafted to be easily understandable, making them ideal for quick insights or decision-making.

- **Flexibility**: Users can adjust parameters to customize the length and style of the generated summaries, tailoring them to their preferences.

## Reposiory Content

The following files are included in the repository:
1. LSTM_sentiment_model.h5 : The LSTM trained model file.
2. Logistic_sentiment_model.pkl : The train logistic regression model pcikle file.
3. tokenizer_config.json : The tokenizer text map that was generated for LSTM.
4. sentiment-analysis.ipynb : Jupytor Notebook when preprocessing and model training has been done

## Dataset
[https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews/data)
