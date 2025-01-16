Political News Classification: Exploring Text and Sentiment Features
Project Overview

This study focuses on:
	1.	Analyzing the sentiment polarity (the degree of positivity or negativity) across political news categories.
	2.	Evaluating whether the inclusion of sentiment features as complementary variables enhances the accuracy of a text-only classification model for political news headlines.

Key Features
	•	Data Preprocessing: Includes data cleaning, feature extraction (TF-IDF and sentiment analysis), and dataset splitting for training and testing.
	•	Exploratory Data Analysis (EDA): Explores trends in news categories and sentiment score distributions.
	•	Classification Models:
	•	Text-only logistic regression model.
	•	Combined text and sentiment-based logistic regression model.
	•	Model Evaluation: Accuracy, precision, recall, specificity, and F1-score are used as performance metrics.

--------------------------------------------------------------------------------------------------------------------------------
 How to Use

1. Prerequisites
	•	R version 4.0 or higher.
	•	R packages: glmnet, tm, tidyverse, caret, textdata, syuzhet, wordcloud.

Install the required packages using:
install.packages(c("glmnet", "tm", "tidyverse", "caret", "textdata", "syuzhet", "wordcloud"))

2. Run the Code
	1.	Clone this repository:
 git clone https://github.com/<your-github-username>/<repository-name>.git
cd <repository-name>
	2.	Load your dataset into the data/ folder.
	3.	Run the R scripts in the following order:
	•	data_preprocessing.R for cleaning and feature extraction.
	•	eda.R for exploratory analysis.
	•	models.R for model training and evaluation.
	•	visualization.R for generating plots and visualizations.
--------------------------------------------------------------------------------------------------------------------------------
Results
	•	Model Comparison:
	•	Text-only model achieved an accuracy of 85.41%.
	•	Combined text and sentiment model achieved an accuracy of 85.38%.
	•	Key Insights:
	•	Sentiment features did not significantly improve model performance but provided additional interpretability in certain scenarios.
	•	The dataset showed a significant imbalance after 2018, influencing classification accuracy.

 Contact

For questions or feedback, feel free to contact:
	•	Name: Shixiong Xu
	•	Email: robinxuuu@gmail.com
	•	GitHub: https://github.com/Robinxuuu/Robinxuuu/tree/main
 
