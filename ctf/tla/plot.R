library(ggplot2)
args = commandArgs(trailingOnly=TRUE)
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else {
  # R has start array index 1
  data <- read.csv(file = args[1])
  # try to plot
  # p <- ggplot(data = data, aes(x = scale, y = ms)) + scale_x_continuous(trans = "log2") + geom_point()
  p <- ggplot(data = data, aes(x = scale, y = ms)) + geom_bar(stat = "identity") + scale_x_continuous(trans = "log2")
  ggsave("my_plot.png", plot = p)
}
