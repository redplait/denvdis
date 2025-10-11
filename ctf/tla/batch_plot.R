library(ggplot2)
args = commandArgs(trailingOnly=TRUE)
# R has start array index 1
for(aidx in 1:length(args) )
{
  fname <- args[aidx]
  data <- read.csv(file = fname)
  # try to plot - uncomment below line to produce points plot
  # p <- ggplot(data = data, aes(x = scale, y = ms)) + scale_x_continuous(trans = "log2") + geom_point() +
  # to produce bar plot
  # always good idea to add axes descriptions
  p <- ggplot(data = data, aes(x = scale, y = ms)) + geom_bar(stat = "identity") + scale_x_continuous(trans = "log2") +
   xlab("scale log") + ylab("time, ms")
  ggsave(sub("csv", "png", fname), plot = p)
}
