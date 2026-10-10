seed <- as.integer(Sys.getenv("SCODE_SEED", "1"))
source_file <- Sys.getenv("SCODE_SOURCE", "")
if (!nzchar(source_file) || !file.exists(source_file)) {
  stop("SCODE_SOURCE must point to the official SCODE.R source")
}
set.seed(seed)
source(source_file, chdir = FALSE)
