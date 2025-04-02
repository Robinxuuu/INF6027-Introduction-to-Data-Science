# INF6027 Intro to Data Science Report
library(tidyverse) # map_df()
library(jsonlite) # fromJSON()
library(tm) # corpus()
library(textstem) # lemmatize_strings
library(syuzhet) # get_sentiment()
library(glmnet) # glmnet()
library(caret) # confusionmatrix

# Data input
# (I) READ the 'News_Category_Dataset_v3'
# because each row is a JSON object, we have to read the JSON doc line by line
# and parse each row as an independent JSON object individually into a list/df in R
news_file_path <- 'News_Category_Dataset_v3.json'
news_lines <- readLines(news_file_path)
news_json_data_df <- map_df(news_lines, ~ fromJSON(.x))

# (II) Explanatory Data Analysis (EDA)
str(news_json_data_df) # no missing: 209527 x 6 tibble
any(is.na(news_json_data_df)) # False = no NaN
# mutate YYYY-MM, YYYY data
news_json_data_df$date <- as.Date(news_json_data_df$date, '%Y-%m-%d') # convert 'date' to class 'Date'
news_json_data_df$year <- format(news_json_data_df$date, '%Y') # extract the year as a new column

# Look for the consistency of news category by frequency linechart
yearly_distribution <- news_json_data_df %>%
  group_by(year, category) %>%
  summarise(count = n(), .groups = 'drop') %>%
  mutate(percentage = count / sum(count) * 100, year = as.Date(paste0(year, "-01-01")))

# The frequency bar chart of all category in descending
news_data_wide <- yearly_distribution %>%
  select(year, category, count) %>% # 只保留需要的列
  pivot_wider(names_from = year, values_from = count, values_fill = 0) %>%
  mutate(Total = rowSums(across(where(is.numeric)))) %>%# add a total column
  arrange(desc(Total)) %>%
  mutate(frequency_category = case_when(
    Total > 5000 ~ 'High',
    Total > 2500 ~ 'Medium', 
    Total <= 2500 ~ 'Low'
  ))
ggplot(news_data_wide, aes(x = reorder(category, Total), y = Total)) + 
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = Total), hjust = -0.2, size = 3) +
  coord_flip() +
  theme(panel.background = element_blank()) +
  labs(title = "Total News Counts by Category", x = "Category", y = "Total Count", caption = 'Kaggle: Rishabh Misra (2022)')

# create a frequency label d.f. for checking category consistency
frequency_category_df <- news_data_wide %>%
  select(category, frequency_category)
yearly_distribution <- yearly_distribution %>%
  left_join(frequency_category_df, by = "category")
# Follow 'frequency_category' to draw the faceted plot
ggplot(yearly_distribution, aes(year, count)) +
  geom_line(aes(color = category)) +
  facet_grid(frequency_category ~ ., scales = "free_y") +
  labs(title = "News Categories Trends by Frequency", x = "Year", y = "Count", caption = 'Kaggle: Rishabh Misra (2022)') +
  theme(panel.background =element_blank()) +
  geom_vline(xintercept = as.Date('2018-01-01'), linetype = "dashed", color = "red", size = 0.8) + # 添加红色虚线
  ggplot2::annotate("text", x = as.Date('2018-01-01'), y = Inf, label = "Year = 2018", color = "red", vjust = 5, size = 3.5)

# (III) Sentiment analysis
# Step 1: Create a sentiment analysis dataframe
sentiment_news_data <- news_json_data_df %>% 
  select(headline, category, date, year)
# Step 2: Preprocessing the headlines
headlines_corpus <- Corpus(VectorSource(sentiment_news_data$headline))
headlines_corpus_cleaned <- tm_map(headlines_corpus, content_transformer(tolower))    # Convert to lowercase
headlines_corpus_cleaned <- tm_map(headlines_corpus_cleaned, removePunctuation)              # Remove punctuation
headlines_corpus_cleaned <- tm_map(headlines_corpus_cleaned, removeNumbers)                  # Remove numbers
headlines_corpus_cleaned <- tm_map(headlines_corpus_cleaned, removeWords, stopwords("en"))   # Remove common stopwords
headlines_corpus_cleaned <- tm_map(headlines_corpus_cleaned, stripWhitespace)                # Remove extra whitespace
headlines_corpus_cleaned <- tm_map(headlines_corpus_cleaned, lemmatize_strings) # Lemmatization
# Convert the cleaned corpus back into a character vector
sentiment_news_data$cleaned_headlines <- sapply(headlines_corpus_cleaned, as.character)
# Step 3: sentiment analysis and answer RQ1
# Calculate sentiment scores for each cleaned headline
sentiment_news_data$afinn_sentiment_score <- get_sentiment(sentiment_news_data$cleaned_headlines, method = "afinn")
# Boxplot of sentiment scores
sentiment_news_data %>% 
  ggplot(aes(x=category, y=afinn_sentiment_score)) + 
  geom_boxplot(notch=FALSE, varwidth = TRUE, fill='plum') + 
  geom_hline(yintercept = 0, linetype = "dashed", color = "red", size = 0.3) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  labs(
    title = 'Sentiment Score Distribution Across All Categories (AFINN)', 
    subtitle = 'The box width represents the count of news headlines',
    x='News Category', 
    y='Sentiment Score (AFINN Lexicon)', 
    caption='Source: Kaggle Dataset (Misra, 2022)') 
# t test for political news' sentiment scores
ks_test <- ks.test(sentiment_news_data$afinn_sentiment_score, "pnorm",
                   mean = mean(sentiment_news_data$afinn_sentiment_score),
                   sd = sd(sentiment_news_data$afinn_sentiment_score))
print(ks_test) # not normal distributed
wilcoxon_test <- wilcox.test(sentiment_news_data$afinn_sentiment_score, mu = 0)
print(wilcoxon_test) # significant different from 0
# Step 4: Create target variable (assume classification is political vs non-political)
sentiment_news_data$target <- ifelse(sentiment_news_data$category == "POLITICS", 1, 0)
# Step 5: Build logistic regression models and make predictions
# (1) Text feature only
# (i) Extraction of text features (TF-IDF)
tfidf_matrix <- DocumentTermMatrix(
  headlines_corpus_cleaned, 
  control = list(
    weighting = weightTfIdf
  )
)
tfidf_matrix <- removeSparseTerms(tfidf_matrix, 0.99)
tfidf_matrix <- as.matrix(tfidf_matrix)
# (ii) Define the training set and the test set
set.seed(123)
train_index <- sample(1:nrow(tfidf_matrix), 0.8 * nrow(tfidf_matrix))
X_train_tfidf <- tfidf_matrix[train_index, ]
X_test_tfidf <- tfidf_matrix[-train_index, ]
y_train_tfidf <- sentiment_news_data$target[train_index]
y_test_tfidf <- sentiment_news_data$target[-train_index]
# (iii) Train the logistic regression model
# Use cross-validation to find the best lambda
classification_model_tfidf <- cv.glmnet(X_train_tfidf, y_train_tfidf, family = "binomial")
best_lambda_tfidf <- classification_model_tfidf$lambda.min
# (iv) Predict on the test data
predictions_tfidf <- predict(classification_model_tfidf, X_test_tfidf, s = best_lambda_tfidf, type = "response")
# (v) Evaluation of the model
# Convert probabilities to class predictions
predicted_classes_tfidf <- ifelse(predictions_tfidf > 0.5, 1, 0)
# Calculate error and generate confusion matrix
confusion_tfidf <- confusionMatrix(as.factor(predicted_classes_tfidf), as.factor(y_test_tfidf))
print(confusion_tfidf)
# (2) Combined features
# (i) Combine features and define the training set and the test set
combined_features <- cbind(tfidf_matrix, sentiment_news_data$afinn_sentiment_score)
set.seed(123)
train_index <- sample(1:nrow(combined_features), 0.8 * nrow(combined_features))
X_train <- combined_features[train_index, ]
X_test <- combined_features[-train_index, ]
y_train <- sentiment_news_data$target[train_index]
y_test <- sentiment_news_data$target[-train_index]
# (ii) Train the logistic regression model
# Use cross-validation to find the best lambda
cv_model <- cv.glmnet(X_train, y_train, family = "binomial")
best_lambda <- cv_model$lambda.min
# (iii) Predict on the test data
predictions <- predict(classification_model, X_test, s = best_lambda, type = "response")
# (iv) Evaluation of the model
# Convert probabilities to class predictions
predicted_classes <- ifelse(predictions > 0.5, 1, 0)
# Calculate error and generate confusion matrix
confusion <- confusionMatrix(as.factor(predicted_classes), as.factor(y_test))
print(confusion)




