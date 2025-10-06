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
  png_fname <- ifelse(length(args)>1, args[2], sub("csv", "png", args[1]) )
  ggsave(png_fname, plot = p)
}
