library(plyr);
setwd("D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\ML Scripts");
getwd()
Stack = read.csv("clean_df_for_unsupervised.csv")
setwd("D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\ML Scripts\\split_text_files");
d_ply(Stack, .(id), function(sdf) write.csv(sdf, file=paste(sdf$id,".txt",sep="")))


setwd("D:\\Documents\\MEng Software\\ENSF 619-4 Machine Learning\\ensf-619-3-4-project\\ML Scripts\\split_text_files_v2");
for(i in 1:nrow(Stack)){
  write.table(Stack[i,2],paste0("line",i,".txt"),row.names=FALSE,col.names = FALSE)
}
