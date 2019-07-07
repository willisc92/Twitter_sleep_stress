install.packages("BTM")
install.packages("udpipe")
install.packages("SnowballC")
install.packages("hunspell")
install.packages("dplyr")
install.packages("tidytext")
install.packages("stringr")
install.packages("xtable")

library(BTM)
library(udpipe)
library(SnowballC) #for text Stemmign
library(hunspell) # for spell check and spelling
setRepositories()

#Changing the working directory
setwd("D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\ML Scripts")

PA<- read.csv("clean_df_for_unsupervised.csv", header=TRUE, sep=",", fileEncoding = "UTF-8") #importing the data as a single 
PA <- PA[-c(1)]

#In order to turn it into a tidy text dataset, we first need to put it into a data frame.
library(dplyr)

class(PA$text)
PA$text <- lapply(PA$text, as.character) #the format was 'factor' /check class(PA$text)
PA$text<-as.character(unlist(PA$text)) #to unlist the list to character format
PA$text <- iconv(PA$text, "UTF-8", "ASCII", sub="") #To remove all the special characters from Tweets
#PA$text <- hunspell_stem(PA$text)
#PA_df<- data_frame(line=1:6511, text=PA$text) 
PA_df <- PA[c(1:6511),]
PA_df <- data.frame(PA_df)


class(PA_df$PA_df)
PA_df$text <- lapply(PA_df$PA_df, as.character) #the format was 'factor' /check class(PA$text)
#---------------------------- Tokenization -----------------------------------
library(tidytext)
library(dplyr)
library(stringr)
data(stop_words)

myStopwords <- c("can", "say","one","way","use", "test", 
                 "also","however","tell","will", "think",
                 "much","need","take","tend","even", "like", "particular", "rather", "said", "must", " s " , " t ", "agile", "test", "part", "write", 
                 "get", "well", "make", "ask", "come", "end", "first", "two", "may", "might", "see", "code", "project", "test", "done",
                 "something", "thing", "point", "post", "look", "'ve", "'re", "parentid" , "id", "title", "postlink", "body", "thing","think", "just"
                 , "thing","case", "cases", "really", "non", "etc", "using", "set", "level", "going", "lot", "back", "since", "let", "don","functional",
                 "nonfunctional", "thus","else", "software development projects","software development project", "aren", "others", "things","requirement",
                 "tags", "score", "answercount", "commentcount", "favoritecount", "ownerdisplayname", "ownerid", "viewcount", "http", "name", "osm", "create",
                 "var", "points", "line", "find", "used", "now", "trying", "looking", "many", "thanks", "another", "following", "per", "within", "avail", 
                 "try", "names", "along", "please", "without", "tried", "still", "either", "anyone", "usually", "looks", "seem","owneruserid", "line", "lines",
                 "dont","just","one", "ive", "know", "cant","got")

#break the text into individual tokens and stem
PA_df<- PA_df %>% 
  unnest_tokens(word, text) %>% 
  filter(!str_detect(word, "^[0-9]*$")) %>%
  filter(!str_detect(word, myStopwords)) %>%
  anti_join(stop_words)  

PA_df%>% count(word, sort=TRUE)

#---------------------------- Buiding the Model -----------------------------------
## Building the model
set.seed(321)
model  <- BTM(PA_df, k = 3, beta = 0.01, iter = 1000, trace = 100, background = TRUE)

## Inspect the model - topic frequency + conditional term probabilities

model$theta

topicterms <- terms(model, top_n = 5)
topicterms

joined <- bind_cols(topicterms)
joined <- joined %>% select(-contains("probability"))
xtable(joined)
joined


