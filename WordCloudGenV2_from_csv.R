#ls()
remove(list = ls()) #Clear environment


# Install
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("wordcloud") # word-cloud generator 
install.packages("RColorBrewer") # color palettes
install.packages("RWeka")
install.packages("corpus")
# Load
library("tm")
library("SnowballC")
library("wordcloud")
library("RColorBrewer")
library("RWeka")
library("corpus")
Sys.setenv(JaVA_HOME='C:\\Program Files\\Java\\jdk-10.0.2')

tweetsDS <- read.csv("D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\Stress Keyword Scraping From Reddit\\compiled_subreddit_titles_and_bodies.csv",
                     header = FALSE, stringsAsFactors = FALSE)
tweetsDS <- data.frame(tweetsDS)
tweetsDS.Corpus<-VCorpus(VectorSource(tweetsDS$V2))

remove_char <- content_transformer(function (x , pattern ) gsub(pattern, "", x))
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))

tweetsDS.Clean <- tm_map(tweetsDS.Corpus, remove_char, "\'")

tweetsDS.Clean <- tm_map(tweetsDS.Corpus, toSpace, "[^a-zA-Z0-9]")
# Convert the text to lower case
tweetsDS.Clean <- tm_map(tweetsDS.Clean, content_transformer(tolower))
# Remove numbers
tweetsDS.Clean <- tm_map(tweetsDS.Clean, removeNumbers)
# Remove english common stopwords
tweetsDS.Clean <- tm_map(tweetsDS.Clean,removeWords,stopwords("english"))
tweetsDS.Clean <- tm_map(tweetsDS.Clean,removeWords,stopwords("SMART"))
# Remove punctuations
tweetsDS.Clean <- tm_map(tweetsDS.Clean, removePunctuation)
# Remove whitespace 
tweetsDS.Clean <- tm_map(tweetsDS.Clean, stripWhitespace)
# Remove custom stopwords.  
#tweetsDS.Clean <- tm_map(tweetsDS.Clean, removeWords, c("don", "didn" )) 

dtm <- TermDocumentMatrix(tweetsDS.Clean)
m <- as.matrix(dtm)
v <- sort(rowSums(m),decreasing=TRUE)
d <- data.frame(word = names(v),freq=v)
head(d, 100)

wordcloud(words = tweetsDS.Clean, min.freq = 50,
          max.words=100, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram <- TermDocumentMatrix(tweetsDS.Clean, control = list(tokenize = BigramTokenizer))
tdm.bigram

memory.limit(2010241024*1024) 
freq <- sort(rowSums(as.matrix(tdm.bigram)), decreasing = TRUE)
freq.df = data.frame(word= names(freq), freq=freq)
head(freq.df, 50)

wordcloud(words = freq.df$word, freq=freq.df$freq, min.freq = 50,
         max.words=100, random.order=FALSE, rot.per=0, 
         colors=brewer.pal(8, "Dark2"))

TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.trigram <- TermDocumentMatrix(tweetsDS.Clean, control = list(tokenize = TrigramTokenizer))
tdm.trigram
freq_trigram <- sort(rowSums(as.matrix(tdm.trigram)), decreasing = TRUE)
freq_trigram.df = data.frame(word= names(freq_trigram), freq=freq_trigram)
head(freq_trigram.df, 50)

wordcloud(words = freq_trigram.df$word, freq=freq_trigram.df$freq, min.freq = 10,
          max.words=100, random.order=FALSE, rot.per=0, 
          colors=brewer.pal(8, "Dark2"))
