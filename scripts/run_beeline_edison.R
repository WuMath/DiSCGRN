args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2L)
x <- as.matrix(read.csv(args[[1]], row.names = 1, check.names = FALSE))
stopifnot(is.numeric(x), all(is.finite(x)), nrow(x) >= 12L)
set.seed(20261007L)
fit <- EDISON::EDISON.run(t(x), information.sharing = "poisson", num.iter = 5000L)
prob <- EDISON::calculateEdgeProbabilitiesTimePoints(fit, c(1L, nrow(x) + 1L), ncol(x))
score <- Reduce(`+`, prob) / length(prob)
# EDISON stores parent x target; common score files store target x parent.
score <- t(score)
dimnames(score) <- list(colnames(x), colnames(x))
diag(score) <- 0
write.csv(score, args[[2]])
cat("EDISON version", as.character(packageVersion("EDISON")), "\n")
