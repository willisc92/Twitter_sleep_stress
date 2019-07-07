# ENSF 619-3/4 Project
# Name: Willis Cheung, Julian Mulia, Justin WOods

#ls()
remove(list = ls())

install.packages("ggplot2")
install.packages("RColorBrew")
install.packages("corrplot")
install.packages("dplyr")
install.packages("Hmisc")
install.packages("devtools")
install.packages("psych")
install.packages("stargazer")
install.packages("ggpubr")
install.packages("BBmisc")
install.packages("tidyverse")
install.packages("caret")
install.packages("nnet")
install.packages("tm")
install.packages("wordcloud")
install.packages("e1071")
install.packages("gmodel")
install.packages("RWeka")
install.packages("xtable")
install.packages("tidytext")
install.packages("stringr")


setRepositories()

library("ggplot2")
library("RColorBrew")
library("corrplot")
library("dplyr")
library("Hmisc")
library("devtools")
library("psych")
library("stargazer")
library("ggpubr")
library("BBmisc")
library("tidyverse")
library("caret")
library("nnet")
library("tm")
library("wordcloud")
library("e1071")
library("gmodel")
library("RWeka")
library("xtable")
library("tidytext")
library("stringr")
library("caret")

cleaned_data_df <- read.csv("G:/Users/Willis/Documents/MENG_Software/ENSF 619-04 Machine Learning/ensf-619-3-4-project/Data To Label/ITER_2_labeled_combined.csv", header=T, stringsAsFactors=FALSE)
str(cleaned_data_df) #Check data types

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))

cleaned_data_df <- cleaned_data_df[complete.cases(cleaned_data_df),] #Remove NAN and empty label rows.
cleaned_data_df <- cleaned_data_df[!(is.na(cleaned_data_df$label) | cleaned_data_df$label==""),]
cleaned_data_df <- cleaned_data_df[, -1] #Remove index column

#cols to remove first iteration
#cols_to_remove <- c("orig_text", "country", "user_ID","user_creation_day","user_creation_month","user_creation_year", "tweet_year")

cols_to_remove <- c("orig_text","country","user_id")
cleaned_data_df <- select(cleaned_data_df, -cols_to_remove) #Remove columns that are unnecessary.
cols_to_factor <- c("label","city", "province","tweet_day","tweet_hour","tweet_weekday","user_verified") #Change these to factor type.
cleaned_data_df[cols_to_factor] <- lapply(cleaned_data_df[cols_to_factor], factor) 
normalized <- normalize(cleaned_data_df, method = "range", range = c(0,1), margin = 2, on.constant = "quiet") #Normalize numericalcolumns
sd(cleaned_data_df$account_life_days)

install.packages("Hmisc")
library("Hmisc")
describe(cleaned_data_df)

normalized$filtered_text_non_neg <- gsub("NEG_", "", normalized$filtered_text) # Remove NEG_ tags for naive bayes classification.
normalized$filtered_text_no_dup <- sapply(normalized$filtered_text_non_neg, function(x) paste(unique(unlist(strsplit(x," "))), collapse = " "))
normalized$sleep_label <- sapply(normalized$label, function(x) if (x == "B" | x == "Z") "Z" else "N")
normalized$stress_label <- sapply(normalized$label, function(x) if (x == "B" | x == "P") "P" else "N")
normalized$both <- sapply(normalized$label, function(x) if (x =="B") "B" else "N")
normalized$sleep_label <- as.factor(normalized$sleep_label)
normalized$stress_label <- as.factor(normalized$stress_label)
normalized$both <- as.factor(normalized$both)

#Write to csv for unsupervised learning
#write.csv(normalized$filtered_text_non_neg,"D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\ML Scripts\\clean_df_for_unsupervised_v2.csv", row.names = FALSE)


count_table <- data.frame(table(normalized$label))
colnames(count_table) <- c('Label', 'Freq')
count_table$Perc <- count_table$Freq / sum(count_table$Freq) * 100
count_table <- count_table[with(count_table, order(-Freq)),]
count_table
xtable(count_table)

province_count_table <- data.frame(table(normalized$province))
colnames(province_count_table) <- c('Province', 'Freq')
province_count_table$Perc <- province_count_table$Freq / sum(province_count_table$Freq) * 100
province_count_table <- province_count_table[with(province_count_table, order(-Freq)),]
province_count_table

?ggplot
ggplot(data = province_count_table, aes(x = reorder(Province, -Perc), y= Perc, fill=-Perc)) + geom_bar(stat="identity", colour="black") + theme(plot.title = element_text(hjust = 0.5), axis.text.x=element_text(angle=90, hjust=1), legend.position = "none") + 
  labs(title = "Percent of labelled tweets by province", x = "Province")


xtable(province_count_table)

sapply(normalized, function(x) length(unique(x)))
sapply(normalized, function(x) sum(is.na(x)))

#Binary Naive Bayes Classification based on text - Create a corpus on already filtered text.
df <- normalized[, c(1,16,17,18, 19,20)]

#For old file
#df <- normalized[, c(1,16,17,18,19)]
str(normalized)
glimpse(df)

corpus <- VCorpus(VectorSource(df$filtered_text_no_dup))
corpus <- VCorpus(VectorSource(df$filtered_text_non_neg))

corpus
inspect(corpus[1:3])

corpus.clean <- corpus %>% #Remove numbers,lower, remove stopwords, and stem
  tm_map(content_transformer(tolower)) %>%
  tm_map(removeNumbers) %>%
  tm_map(removeWords, stopwords(kind="en")) %>%
  tm_map(stemDocument) %>%
  tm_map(stripWhitespace)

dtm <- DocumentTermMatrix(corpus.clean) #Split train/test data and create a DTM, and create 80:20 test/training split.  
dtm <- DocumentTermMatrix(corpus.clean, control=list(tokenize=BigramTokenizer)) #Split train/test data and create a DTM, and create 80:20 test/training split.  
dtm <- DocumentTermMatrix(corpus.clean, control=list(tokenize=TrigramTokenizer))

split = 0.8

set.seed(69)
trainIndex <- createDataPartition(df$label, p=split, list=FALSE)
inspect(dtm[40:50, 10:15])
df.train <- df[trainIndex,]
df.test <- df[-trainIndex,]
dtm.train <- dtm[trainIndex,]
dtm.test <- dtm[-trainIndex,]

corpus.clean.train <- corpus.clean[trainIndex]
corpus.clean.test <- corpus.clean[-trainIndex]

dim(dtm.test)
dim(dtm.train)

# Feature reduction - Ignore words that appear in less than five tweets.  Restricts dtm to use only the frequent words.  
fivefreq <- findFreqTerms(dtm.train, 5)
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control = list(dictionary = fivefreq)) 
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control = list(dictionary = fivefreq))
dim(dtm.test.nb)

#Check to see if train/test samples are representative of each other.  
df.train$label%>% table %>% prop.table
df.test$label %>% table %>% prop.table

#Train the NB Model to use binary model - replaces word frequencies to yes (presence) and no (absence)
convert_counts <- function(x) {
  y <- ifelse(x > 0, 1, 0)
  y <- factor(y, levels=c(0,1), labels = c("No", "Yes"))
  y
}

trainNB <- apply(dtm.train.nb, 2, convert_counts)
testNB <- apply(dtm.test.nb, 2, convert_counts)
classifier <- naiveBayes(trainNB, df.train$label, laplace = 1)
stress_nb_classifier <- naiveBayes(trainNB, df.train$stress_label, laplace =1)
sleep_nb_classifier <- naiveBayes(trainNB, df.train$sleep_label, laplace=1)
both_nb_classifier <- naiveBayes(trainNB, df.train$both, laplace=1)

pred <- predict(classifier, newdata=testNB) #Use the NB Classifier to make predictions on test set.  
stressPred <- predict(stress_nb_classifier, newdata=testNB)
sleepPred <- predict(sleep_nb_classifier, newdata=testNB)
bothPred <- predict(both_nb_classifier, newdata=testNB)

table("Predictions" = pred, "Actual" = df.test$label)

conf.mat <- confusionMatrix(pred, df.test$label)
ConfMat <- as.data.frame.matrix(conf.mat$table)
xtable(ConfMat)
conf.mat
conf.mat$byClass
conf.mat$overall
conf.mat$overall['Accuracy']

stress.conf.mat <- confusionMatrix(stressPred, df.test$stress_label, positive="P")
stress.ConfMat <- as.data.frame.matrix(stress.conf.mat$table)
stress.conf.mat
ls(stress.conf.mat)
stress_precision = stress.ConfMat[2,2]/sum(stress.ConfMat[2,1:2])
stress_recall = stress.ConfMat[2,2]/sum(stress.ConfMat[1:2,2])
stress_f_score = 2*stress_precision*stress_recall/(stress_precision+stress_recall)
stress_precision
stress_recall
stress_f_score

sleep.conf.mat <- confusionMatrix(sleepPred, df.test$sleep_label, positive="Z")
sleep.ConfMat <- as.data.frame.matrix(sleep.conf.mat$table)
sleep.conf.mat
sleep_precision = sleep.ConfMat[2,2]/sum(sleep.ConfMat[2,1:2])
sleep_recall = sleep.ConfMat[2,2]/sum(sleep.ConfMat[1:2,2])
sleep_f_score = 2*sleep_precision*sleep_recall/(sleep_precision+sleep_recall)
sleep_precision
sleep_recall
sleep_f_score


both.conf.mat <- confusionMatrix(bothPred, df.test$both, positive="B")
both.ConfMat <- as.data.frame.matrix(both.conf.mat)
both.conf.mat 

#LM Modeling 
LM_df <- normalized
AFINN_lexicon <- get_sentiments("afinn")

unnested <- LM_df %>%
            unnest_tokens(word, filtered_text)
unnested$neg <- ifelse((grepl("neg_", unnested$word)), -1, 1)
unnested$word <- gsub("neg_", "",unnested$word) 
unnested <- unnested %>% left_join(AFINN_lexicon) %>%
          left_join(unnested)
unnested[is.na(unnested)] <- 0
unnested$score <- unnested$score * unnested$neg 


unnested <- unnested %>%
    group_by(filtered_text_non_neg) %>%
    dplyr::summarize(total_score = sum(score))

unnested <- unnested %>% inner_join(LM_df)


cols_to_drop <- c("filtered_text_non_neg", "filtered_text","filtered_text_no_dup", "label")
clean_df_sent <- select(unnested, -cols_to_drop) #Remove columns that are unnecessary.
clean_df_sent_norm <- normalize(clean_df_sent, method = "range", range = c(0,1), margin = 2, on.constant = "quiet") #Normalize numericalcolumns


ctrl <- trainControl(method = "repeatedcv", number = 10, savePredictions = TRUE)


library(plyr)

k = 10
sleep_fpr <- NULL
sleep_fnr <- NULL
sleep_acc <- NULL

stress_fpr <- NULL
stress_fnr <- NULL
stress_acc <- NULL
set.seed(123)

pbar <- create_progress_bar('text')
pbar$init(k)

remove_missing_levels <- function(fit, test_data) {
  library(magrittr)
  
  # https://stackoverflow.com/a/39495480/4185785
  
  # drop empty factor levels in test data
  test_data %>%
    droplevels() %>%
    as.data.frame() -> test_data
  
  # 'fit' object structure of 'lm' and 'glmmPQL' is different so we need to
  # account for it
  if (any(class(fit) == "glmmPQL")) {
    # Obtain factor predictors in the model and their levels
    factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                     names(unlist(fit$contrasts))))
    # do nothing if no factors are present
    if (length(factors) == 0) {
      return(test_data)
    }
    
    map(fit$contrasts, function(x) names(unmatrix(x))) %>%
      unlist() -> factor_levels
    factor_levels %>% str_split(":", simplify = TRUE) %>%
      extract(, 1) -> factor_levels
    
    model_factors <- as.data.frame(cbind(factors, factor_levels))
  } else {
    # Obtain factor predictors in the model and their levels
    factors <- (gsub("[-^0-9]|as.factor|\\(|\\)", "",
                     names(unlist(fit$xlevels))))
    # do nothing if no factors are present
    if (length(factors) == 0) {
      return(test_data)
    }
    
    factor_levels <- unname(unlist(fit$xlevels))
    model_factors <- as.data.frame(cbind(factors, factor_levels))
  }
  
  # Select column names in test data that are factor predictors in
  # trained model
  
  predictors <- names(test_data[names(test_data) %in% factors])
  
  # For each factor predictor in your data, if the level is not in the model,
  # set the value to NA
  
  for (i in 1:length(predictors)) {
    found <- test_data[, predictors[i]] %in% model_factors[
      model_factors$factors == predictors[i], ]$factor_levels
    if (any(!found)) {
      # track which variable
      var <- predictors[i]
      # set to NA
      test_data[!found, predictors[i]] <- NA
      # drop empty factor levels in test data
      test_data %>%
        droplevels() -> test_data
      # issue warning to console
      message(sprintf(paste0("Setting missing levels in '%s', only present",
                             " in test data but missing in train data,",
                             " to 'NA'."),
                      var))
    }
  }
  return(test_data)
}

for(i in 1:k)
{
  smp_size <- floor(0.6 *nrow(clean_df_sent_norm))
  index <- sample(seq_len(nrow(clean_df_sent_norm)), size=smp_size)  
  train <- clean_df_sent_norm[index,]
  test <- clean_df_sent_norm[-index,]
  
  sleeplm.fit <- glm(sleep_label ~ total_score + account_life_days + province + latitude + longitude + tweet_day + tweet_hour + tweet_weekday + 
                        user_followers_count + user_friends_count + user_listed_count + user_statuses_count + user_verified,
                      data = train, family ="binomial")
  sleep_results_prob <- predict(sleeplm.fit, remove_missing_levels(fit=sleeplm.fit, test_data=test), type='response')
  sleep_results <- as.factor(ifelse(sleep_results_prob > 0.5, "Z", "N"))

  sleep_label_actual_test <- test$sleep_label
  sleep_misClasificError <- mean(sleep_label_actual_test != sleep_results)
  
  sleep_score = 0
  for (j in 1:length(sleep_label_actual_test)){
    if (is.na(sleep_label_actual_test[j]) | is.na(sleep_results[j])) {
        next 
    }
    
    if (sleep_label_actual_test[j] == sleep_results[j]) {
      sleep_score = sleep_score + 1 
    }
  }
  
  mean_sleep_score = sum(sleep_score)/length(sleep_label_actual_test)
  
  sleep_acc[i] <- mean_sleep_score
  sleep_cm <- confusionMatrix(data = sleep_results, reference=sleep_label_actual_test, positive = "Z")
  print(sleep_cm)
  sleep_fpr[i] <- sleep_cm$table[2]/(nrow(clean_df_sent_norm)-smp_size)
  sleep_fnr[i] <- sleep_cm$table[3]/(nrow(clean_df_sent_norm)-smp_size)
  
  
  stresslm.fit <- glm(stress_label ~ total_score + account_life_days + province + latitude + longitude + tweet_day + tweet_hour + tweet_weekday + 
                       user_followers_count + user_friends_count + user_listed_count + user_statuses_count + user_verified,
                     data = train, family ="binomial")
  stress_results_prob <- predict(stresslm.fit, remove_missing_levels(fit=stresslm.fit, test_data=test), type='response')
  stress_results <- as.factor(ifelse(stress_results_prob > 0.5, "P", "N"))
  stress_label_actual_test <- test$stress_label

  stress_score = 0
  for (l in 1:length(stress_label_actual_test)){
    if (is.na(stress_label_actual_test[l]) | is.na(stress_results[l])) {
      next 
    }
    
    if (stress_label_actual_test[l] == stress_results[l]) {
      stress_score = stress_score + 1 
    }
  }
  
  mean_stress_score = sum(stress_score)/length(stress_label_actual_test)
  
  stress_acc[i] <- mean_stress_score
  
  stress_cm <- confusionMatrix(data = stress_results, reference=stress_label_actual_test, positive = "P")
  print(stress_cm)
  stress_fpr[i] <- stress_cm$table[2]/(nrow(clean_df_sent_norm)-smp_size)
  stress_fnr[i] <- stress_cm$table[3]/(nrow(clean_df_sent_norm)-smp_size)
  pbar$step()
}

sleep_acc_non_na <- mean(sleep_acc, na.rm=TRUE)
sleep_acc_non_na
par(mfcol=(c(1,2)), mar = c(4,4,4,4))
hist(sleep_acc, xlab="Accuracy", ylab= 'Freq', col='cyan', border='blue', density=30, main = 'Sleep Logistic Regression Accuracy')
mean(sleep_fpr)
mean(sleep_fnr)
hist(sleep_fpr, xlab = 'fpr', ylab = 'Freq', main='Sleep Logistic Regression FPR', col='cyan', border='blue', density=30)
hist(sleep_fnr, xlab = 'fnr', ylab = 'Freq', main='Sleep Logistic Regression FNR', col='cyan', border='blue', density=30)

mean(stress_acc, na.rm=TRUE)
hist(stress_acc, xlab="Accuracy", ylab= 'Freq', col='cyan', border='blue', density=30, main = 'Stress Logistic Regrssion Accuracy')
mean(stress_fpr)
mean(stress_fnr)
hist(stress_fpr, xlab = 'fpr', ylab = 'Freq', main='Stress Logistic Regression FPR', col='cyan', border='blue', density=30)
hist(stress_fnr, xlab = 'fnr', ylab = 'Freq', main='Stress Logistic Regression FNR', col='cyan', border='blue', density=30)



sleeplm.fit <- train(sleep_label ~  total_score + account_life_days + province + latitude + longitude + tweet_day + tweet_hour + tweet_weekday + 
                           user_followers_count + user_friends_count + user_listed_count + user_statuses_count + user_verified,
                         data = clean_df_sent_norm, method = "glm", family = "binomial", trControl = ctrl, tuneLength = 5)
sleep_pred = predict(sleeplm.fit, newdata = clean_df_sent_norm[-trainIndex,])
sleep_label_actual_test = clean_df_sent_norm[-trainIndex,]$sleep_label
confusionMatrix(data = sleep_pred, sleep_label_actual_test, positive = "Z")
summary(sleeplm.fit)

glm1 <- glm(sleep_label ~ total_score + account_life_days + province + latitude + longitude +tweet_day + tweet_hour + tweet_weekday + 
              user_followers_count + user_friends_count + user_listed_count + user_statuses_count + user_verified,
            data = clean_df_sent_norm, family ="binomial")

summary(glm1)
sleeplm.fit$finalModel$call <- glm1$call
stargazer(sleeplm.fit$finalModel)


stresslm.fit <- train(stress_label ~  total_score + account_life_days +  province + latitude + longitude + tweet_day + tweet_hour + tweet_weekday + 
                           user_followers_count + user_friends_count + user_listed_count + user_statuses_count + user_verified,
                         data = clean_df_sent_norm, method = "glm", family = "binomial", trControl = ctrl, tuneLength = 5)
stress_pred = predict(stresslm.fit, newdata = clean_df_sent_norm[-trainIndex,])
summary(stresslm.fit)
stress_label_actual_test = clean_df_sent_norm[-trainIndex,]$stress_label
confusionMatrix(data = stress_pred, stress_label_actual_test, positive="P")

glm2 <- glm(stress_label ~ total_score + account_life_days +  province + latitude + longitude + tweet_day + tweet_hour + tweet_weekday + 
              user_followers_count + user_friends_count + user_listed_count + user_statuses_count + user_verified,
            data = clean_df_sent_norm, family ="binomial")

summary(glm2)
stresslm.fit$finalModel$call <- glm2$call
stargazer(stresslm.fit$finalModel)





#Visualizing the most common words in the corpus, as well as the stressed, sleep, and both labels.  
?wordcloud
wc_colors = c("chartreuse", "cornflowerblue", "darkorange")
wordcloud(corpus.clean, random.order = FALSE, min.freq = 25, colors = wc_colors )
par(mfcol = c(1,1))
sleep <- df  %>% subset(label == "Z" | label == 'B')
sleep_and_stressed <- df %>% subset(label == "B")
stressed <- df %>% subset(label == "P" | label == "B") 
sleep_cloud <- wordcloud(sleep$filtered_text_no_dup, max.words = 40, scale = c(3,0.5),  colors = wc_colors)
stressed_cloud <- wordcloud(stressed$filtered_text_no_dup, max.words = 40, scale = c(3,0.5),  colors = wc_colors)
sleep_and_stressed_cloud <- wordcloud(sleep_and_stressed$filtered_text_no_dup, max.words = 40, 
                                      scale = c(3,0.5), colors = wc_colors)

# Neither unigrams/bigrams
neither <- df %>% subset(label == "N")
neither.Corpus<-VCorpus(VectorSource(neither$filtered_text_no_dup))
# Convert the text to lower case
# Remove numbers
neither.Clean <- tm_map(neither.Corpus, removeNumbers)
# Remove english common stopwords
neither.Clean <- tm_map(neither.Clean,removeWords,stopwords("english"))
neither.Clean <- tm_map(neither.Clean,removeWords,stopwords("SMART"))
# Remove whitespace 
neither.Clean <- tm_map(neither.Clean, stripWhitespace)

dtm_neither <- TermDocumentMatrix(neither.Clean)
m <- as.matrix(dtm_neither)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 100)



tdm.neither.bigram <- TermDocumentMatrix(neither.Clean, control = list(tokenize = BigramTokenizer))
tdm.neither.bigram



memory.limit(2010241024*1024) 
neither_bigram.freq <- sort(rowSums(as.matrix(tdm.neither.bigram)), decreasing = TRUE)
neither_bigram.freq.df = data.frame(word= names(neither_bigram.freq), freq=neither_bigram.freq)
head(neither_bigram.freq.df, 50)

neither_cloud <- wordcloud(neither_bigram.freq.df$word, freq=neither_bigram.freq.df$freq, max.words = 40, min.freq = 5,
                           scale = c(3,0.5), colors = wc_colors, rot.per=0)
neither_cloud

# Stress unigrams/bigrams 
stress.Corpus<-VCorpus(VectorSource(stressed$filtered_text_no_dup))
# Convert the text to lower case
# Remove numbers
stress.Clean <- tm_map(stress.Corpus, removeNumbers)
# Remove english common stopwords
stress.Clean <- tm_map(stress.Clean,removeWords,stopwords("english"))
stress.Clean <- tm_map(stress.Clean,removeWords,stopwords("SMART"))
# Remove whitespace 
stress.Clean <- tm_map(stress.Clean, stripWhitespace)

dtm_stress <- TermDocumentMatrix(stress.Clean)
m <- as.matrix(dtm_stress)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 100)

tdm.stress.bigram <- TermDocumentMatrix(stress.Clean, control = list(tokenize = BigramTokenizer))
tdm.stress.bigram

memory.limit(2010241024*1024) 
stress_bigram.freq <- sort(rowSums(as.matrix(tdm.stress.bigram)), decreasing = TRUE)
stress_bigram.freq.df = data.frame(word= names(stress_bigram.freq), freq=stress_bigram.freq)
head(stress_bigram.freq.df, 50)

wordcloud(words = stress_bigram.freq.df$word, freq=stress_bigram.freq.df$freq, min.freq = 0,
          max.words=100, random.order=FALSE, rot.per=0, 
          colors=brewer.pal(8, "Dark2"))

# VISUALIZATIONS by province during sleeping hours.  
sleeping.hours = c(0,1,2,3,4,5,6)
normalized$during_sleep_hours = ifelse(normalized$tweet_hour %in% sleeping.hours, "Midnight - 7am", "Regular Hours")
normalized$during_sleep_hours <- as.factor(normalized$during_sleep_hours)
sleep_from_normalized <- normalized  %>% subset(label == "Z" | label == 'B') # Filtered by label Z
sleep_from_normalized_agg_by_province <- count(sleep_from_normalized, province, during_sleep_hours) # Aggregated by province

#Aggregate DFson all labels
agg_df <- count(normalized, label, during_sleep_hours)
agg_ord <- mutate(agg_df, during_sleep_hours = reorder(during_sleep_hours, -n, sum), label = reorder(label, -n, sum))
p5b <- ggplot(agg_ord) + 
  geom_col(aes(x = label, y = n, fill = during_sleep_hours)) + coord_flip() +
  labs(title = "Aggregate counts for all labels, separated by reg/sleeping hours", x = "Label")+ guides(fill=guide_legend(title="Hours"))
p5b

# Plot for all labels (normalized to proportion of label)
agg_df2 <- agg_df  %>%  # Add proportions to data frame by label 
  arrange(label, desc(during_sleep_hours)) %>% # Rearranging in stacking order      
  group_by(label) %>% 
  mutate(Freq2 = sum(n), # Calculating position of stacked Freq
         prop = 100*n/sum(n)%>% round(2)) # Calculating proportion of Freq
agg_df2

p5 <- ggplot(data = agg_df2, aes(x = label, y = prop, fill = during_sleep_hours)) +
  geom_bar(stat = "identity") +
  geom_text(aes(y = prop, label = sprintf('%.0f%%', prop)), angle=0, position = position_stack(vjust = 0.5)) +
  labs(title = "Percentage filled counts for all labels, separated by reg/sleeping hours", x = "Label", y = "Percent") +
    guides(fill=guide_legend(title="Hours"))

p5

# Sleep Label Aggregates
str(sleep_from_normalized_agg_by_province)
sleep_agg_ord_by_province <- mutate(sleep_from_normalized_agg_by_province, province = reorder(province, -n, sum), 
                                    during_sleep_hours = reorder(during_sleep_hours, -n, sum))

# Non-filled plot for sleep aggregates.
p1 <- ggplot(sleep_agg_ord_by_province) + geom_col(aes(x = province, y = n, fill = during_sleep_hours)) +
  coord_flip() + labs(title = "Sleep counts per province, by reg/sleeping hours", x = "Province") +
  guides(fill=guide_legend(title="Hours"))
p1

# Filled plot for sleep aggregates.
sleep_agg_ord_by_province2 <- sleep_agg_ord_by_province  %>% 
  arrange(province, desc(during_sleep_hours)) %>%
  group_by(province) %>%
  mutate(Freq2 = sum(n), # Calculating position of stacked Freq
         prop = 100*n/sum(n) %>% round(2)) # Calculating proportion of Freq
sleep_agg_ord_by_province2

p1b <- ggplot(data = sleep_agg_ord_by_province2,aes(x = province, y = prop, fill = during_sleep_hours))+
  geom_bar(stat = "identity") +
  labs(title = "Percentage filed counts per province, by reg/sleeping hours", x = "Province", y = "Percent") +
  theme(axis.text.x=element_text(angle = -90, hjust = 0)) + 
  guides(fill=guide_legend(title="Hours")) + coord_flip()
p1b

# Non-filled plot for stress aggregates.  
stress_from_normalized <- normalized  %>% subset(label == "P" | label == "B")
stress_from_normalized_agg_by_province <- count(stress_from_normalized, province, during_sleep_hours)
str(stress_from_normalized_agg_by_province)
stress_agg_ord_by_province <- mutate(stress_from_normalized_agg_by_province, province = reorder(province, -n, sum),
                                     during_sleep_hours = reorder(during_sleep_hours, -n, sum))
p2 <- ggplot(stress_agg_ord_by_province) + geom_col(aes(x = province, y = n, fill = during_sleep_hours)) + 
  coord_flip() + labs(title = "Stress counts per province, by reg/sleeping hours", x = "Province")+ 
  guides(fill=guide_legend(title="Hours"))
p2

# Filled plot for stress aggregates.
stress_agg_ord_by_province2 <- stress_agg_ord_by_province  %>% 
  arrange(province, desc(during_sleep_hours)) %>%
  group_by(province) %>%
  mutate(Freq2 = cumsum(n), # Calculating position of stacked Freq
         prop = 100*n/sum(n) %>% round(2)) # Calculating proportion of Freq

p2b <- ggplot(data = stress_agg_ord_by_province2,aes(x = province, y = prop, fill = during_sleep_hours))+
  geom_bar(stat = "identity") +
  labs(title = "Percentage filled stress counts per province, by reg/sleeping hours", x = "Province", y = "Percent") +
  theme(axis.text.x=element_text(angle = -90, hjust = 0)) + 
  coord_flip() + guides(fill=guide_legend(title="Hours"))
p2b


# Non-filled plot for neither sleep/stressed aggregates
neither_from_normalized <-  normalized  %>% subset(label == "N")
neither_from_normalized_agg_by_province <- count(neither_from_normalized, province, during_sleep_hours)
neither_agg_ord_by_province <- mutate(neither_from_normalized_agg_by_province, province = reorder(province, -n, sum),
                                      during_sleep_hours = reorder(during_sleep_hours, -n, sum))
p4 <- ggplot(neither_agg_ord_by_province) + geom_col(aes(x = province, y = n, fill = during_sleep_hours)) +coord_flip()+
  labs(title = "Neither stress/sleep counts per province, by reg/sleeping hours", x = "Province") +
  guides(fill=guide_legend(title="Hours"))
p4

# Filled plot for neither sleep and stressed aggregates.
neither_agg_ord_by_province2 <- neither_agg_ord_by_province  %>% 
  arrange(province, desc(during_sleep_hours)) %>%
  group_by(province) %>%
  mutate(Freq2 = cumsum(n), # Calculating position of stacked Freq
         prop = 100*n/sum(n) %>% round(2)) # Calculating proportion of Freq

p4b <- ggplot(data = neither_agg_ord_by_province2,aes(x = province, y = prop, fill = during_sleep_hours))+
  geom_bar(stat = "identity") +
  labs(title = "Neither stress/sleep percentage filled counts per province, by reg/sleeping hours") +
  theme(axis.text.x=element_text(angle = -90, hjust = 0))+coord_flip() + guides(fill=guide_legend(title="Hours"))
p4b



