library("hesim")
library("data.table")


sequence <- c("a", "b", "a", "a", "a", "a", "b", "a", "b", "a", "b", "a", "a", 
              "b", "b", "b", "a")        
sequenceMatr <- createSequenceMatrix(sequence, sanitize = FALSE)
mcFitMLE <- markovchainFit(data = sequence)
mcFitMLE
mcFitBSP <- markovchainFit(data = sequence, method = "bootstrap", nboot = 5, name = "Bootstrap Mc")

na.sequence <- c("a", NA, "a", "b")
# There will be only a (a,b) transition        
na.sequenceMatr <- createSequenceMatrix(na.sequence, sanitize = FALSE)
mcFitMLE <- markovchainFit(data = na.sequence)

# data can be a list of character vectors
sequences <- list(x = c("a", "b", "a"), y = c("b", "a", "b", "a", "c"))
mcFitMap <- markovchainFit(sequences, method = "map")
mcFitMle <- markovchainFit(sequences, method = "mle")


sequence <- list(c("a", "b", "a", "a", "a", "a", "b", "a", "b", "a", "b", "a", "a", 
              "b", "b", "b", "a") 
              
sequences <- list(x = c("a", "b", "a","a"), y = c("b", "a", "b","a"),z = c("a","a","b","a"))              
sequenceMatr <- createSequenceMatrix(sequences, sanitize = FALSE)
mcFitMLE <- markovchainFit(data = sequences)
mcFitMLE


x <- matrix(1:9, nrow = 3, ncol = 4)
x[c(1),c(1)] <- "a"
x[c(1),c(2)] <- "b"
x[c(1),c(3)] <- "a"
x[c(1),c(4)] <- "a"
x[c(2),c(1)] <- "b"
x[c(2),c(2)] <- "a"
x[c(2),c(3)] <- "b"
x[c(2),c(4)] <- "a"
x[c(3),c(1)] <- "a"
x[c(3),c(2)] <- "a"
x[c(3),c(3)] <- "b"
x[c(3),c(4)] <- "a"
x

sequenceMatr <- createSequenceMatrix(x, sanitize = FALSE)
mcFitMLE <- markovchainFit(data = x)
mcFitMLE

library("markovchain")


dat<-matrix(c('a','b','c','a','a','a','b','b','a','c','a','a','c','c','a'),nrow = 2)


mylistMc<-markovchainFit(data=dat[1,])

mc <- apply(t(dat),2,function(x) markovchainFit(x))
trans_mat <- list(mc[[1]][[1]],mc[[2]][[1]])

trans_mat <- list(mc[[1]][[1]],mc[[2]][[1]],mc[[3]][[1]])
