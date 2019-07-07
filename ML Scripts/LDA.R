#load text mining library
install.packages("tm")
install.packages("mlbench")
install.packages("class")
install.packages("SparseM")
install.packages("RTextTools")
install.packages("limma")

install.packages(c("dismo"))
install.packages(c("caTools"))
install.packages(c("randomForest"))
install.packages(c("tree"))
install.packages(c("e1071"))
install.packages(c("ipred"))
install.packages(c("tau"))
install.packages(c("glmnet"))
install.packages("xtable")

library(tm)
library(mlbench)
library(class)
library(SparseM)
library(limma)
library(RTextTools)

library(dismo)
library(caTools)
library(randomForest)
library(tree)
library(e1071)
library(ipred)
library(tau)
library(glmnet)
library(topicmodels)
library(xtable)

#set working directory (modify path as needed)
setwd("D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\ML Scripts\\split_text_files_v2");


#load files into corpus
#get listing of .txt files in directory
filenames <- list.files(getwd(),pattern="*")
names <- filenames

#read files into a character vector
files <- lapply(filenames,readLines)

#create corpus from vector
docs <- Corpus(VectorSource(files))

docs <-tm_map(docs,content_transformer(tolower))

#Create the toSpace content transformer
toSpace<- content_transformer(function(x, pattern){return (gsub(pattern, " ", x))});
docs<- tm_map(docs, toSpace, "<.*?>");
docs <- tm_map(docs, toSpace, ":");
docs<- tm_map(docs, toSpace, ",");
docs<- tm_map(docs, toSpace, "_");
docs<-tm_map(docs, toSpace, "-");
docs<-tm_map(docs, toSpace, "'");
#docs<-tm_map(docs, toSpace, ".");
docs<-tm_map(docs, toSpace, "`");

docs<- tm_map(docs, removePunctuation);
docs<- tm_map(docs, removeNumbers);
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, stripWhitespace);
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
#remove custom stopwords
docs <- tm_map(docs, removeWords, myStopwords)

#sample manual transformation
docs <- tm_map(docs, content_transformer(gsub), pattern = "give", replacement = "")
docs <- tm_map(docs, content_transformer(gsub), pattern = "restore", replacement = "storage")
docs <- tm_map(docs, removeWords, myStopwords)

memory.limit()

#Create document-term matrix
dtm <- DocumentTermMatrix(docs)
#convert rownames to filenames
rownames(dtm) <- names
#collapse matrix by summing over columns
memory.limit(2010241024*1024) 

freq <- colSums(as.matrix(dtm))
#length should be total number of terms
length(freq)
#create sort order (descending)
ord <- order(freq,decreasing=TRUE)
#List all terms in decreasing order of freq and write to disk
freq[ord]
write.csv(freq[ord],"word_freq.csv")

#load topic models library
library(topicmodels)
#Set parameters for Gibbs sampling
k <-4
burnin <- 4000
iter <- 3000
thin <- 500
keep <-50
seed <-list(2003,5,63,100001,765)
nstart <- 5
best <- TRUE



#Number of topics
#

rowTotal <- apply(dtm, 1, sum)
dtm <- dtm[rowTotal >0,]

#Run LDA using Gibbs sampling
ldaOut <-LDA(dtm,k, method="Gibbs", control=list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))

#write out results
#docs to topics
ldaOut.topics <- as.matrix(topics(ldaOut))
write.csv(ldaOut.topics,file=paste("LDAGibbs",k,"DocsToTopics.csv"))
ldaOut.topics
#top 6 terms in each topic
ldaOut.terms <- as.matrix(terms(ldaOut,5))
xtable(ldaOut.terms)

write.csv(ldaOut.terms,file=paste("LDAGibbs",k,"TopicsToTerms.csv"))

#probabilities associated with each topic assignment
topicProbabilities <- as.data.frame(ldaOut@gamma)
write.csv(topicProbabilities,file=paste("LDAGibbs",k,"TopicProbabilities.csv"))

theta <- posterior(ldaOut)$topics

#Find relative importance of top 2 topics
topic1ToTopic2 <- lapply(1:nrow(dtm),function(x)
  sort(topicProbabilities[x,])[k]/sort(topicProbabilities[x,])[k-1])


#Find relative importance of second and third most important topics
topic2ToTopic3 <- lapply(1:nrow(dtm),function(x)
  sort(topicProbabilities[x,])[k-1]/sort(topicProbabilities[x,])[k-2])


#write to file
write.csv(topic1ToTopic2,file=paste("LDAGibbs",k,"Topic1ToTopic2.csv"))
write.csv(topic2ToTopic3,file=paste("LDAGibbs",k,"Topic2ToTopic3.csv"))
