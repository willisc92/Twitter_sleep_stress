# Install
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes

# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")

docs<- Corpus(DirSource(directory = "D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\Stress_keyword_texts"))  

text <- readLines(file.choose())
docs <- Corpus(VectorSource(text))

toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))


docs <- tm_map(docs, toSpace, "[^a-zA-Z0-9]")
# Convert the text to lower case
docs <- tm_map(docs, content_transformer(tolower))
# Remove numbers
docs <- tm_map(docs, removeNumbers)
# Remove english common stopwords
docs <- tm_map(docs, removeWords, stopwords("english"))
docs <- tm_map(docs, removeWords, stopwords("SMART"))

# Remove punctuations
docs <- tm_map(docs, removePunctuation)
# Remove custom stopwords.  
docs <- tm_map(docs, removeWords, c("can", "one", "people","just","like","blood","time","will","response","levels","don","research","zebras","make",
                                    "system", "victor", "things", "chapter", "back", "glucocorticoids","immune","glucocorticoid","human","humans",
                                    "data", "sleep", "person", "page", "process", "hormone", "individual", "good", "brain", "cortisol", "measuring",
                                    "day")) 
#docs <- tm_map(docs, stemDocument)

dtm <- TermDocumentMatrix(docs)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 100)
set.seed(1234)
wordcloud(words = d$word, freq = d$freq, min.freq = 224,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Set1"))

sleep_keywords_df <- read.csv("D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\sleep_keywords.csv", header = T)
wordcloud(words = sleep_keywords_df$word, freq = sleep_keywords_df$frequency, min.freq = 1,
          max.words=200, random.order=FALSE, rot.per=0.35,
          colors=brewer.pal(8, "Set1"))
dev.new(width = 1000, height = 1000, unit = "px")
