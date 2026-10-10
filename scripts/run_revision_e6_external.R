#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
split_name <- if (length(args) >= 1) args[[1]] else "calibration"
method_name <- tolower(if (length(args) >= 2) args[[2]] else "edison")
only_system <- if (length(args) >= 3) args[[3]] else NA_character_

expected_versions <- switch(
  method_name,
  edison = c(EDISON = "1.1.2"),
  vardetect = c(VARDetect = "0.1.8", sparsevar = "1.0.0"),
  stop(
    sprintf("Unknown method '%s'. Choose 'edison' or 'vardetect'.", method_name),
    call. = FALSE
  )
)

missing_packages <- names(expected_versions)[
  !vapply(names(expected_versions), requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0) {
  package_vector <- paste(sprintf('"%s"', missing_packages), collapse = ", ")
  stop(
    paste0(
      "Missing required R package(s) for method '", method_name, "': ",
      paste(missing_packages, collapse = ", "), ".\n",
      "R executable: ", file.path(R.home("bin"), "R"), "\n",
      "Libraries searched: ", paste(.libPaths(), collapse = "; "), "\n",
      "Install them in this active R environment with:\n",
      "  Rscript -e 'install.packages(c(", package_vector,
      "), repos=\"https://cloud.r-project.org\", dependencies=TRUE)'\n",
      "Then rerun this command."
    ),
    call. = FALSE
  )
}

for (package_name in names(expected_versions)) {
  actual_version <- as.character(utils::packageVersion(package_name))
  expected_version <- unname(expected_versions[[package_name]])
  if (!identical(actual_version, expected_version)) {
    warning(
      sprintf(
        "Reproducibility warning: %s %s is installed, but the paper used %s.",
        package_name, actual_version, expected_version
      ),
      call. = FALSE
    )
  }
  message(
    sprintf(
      "Using %s %s from %s",
      package_name, actual_version, find.package(package_name)
    )
  )
}

results_root <- Sys.getenv("CTCPGRN_RESULTS_ROOT", unset = "results")
if (!nzchar(trimws(results_root))) stop("CTCPGRN_RESULTS_ROOT must not be empty")
input_dir <- file.path(results_root, "revision_e6", "input")
output_dir <- file.path(results_root, "revision_e6")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

manifest_path <- file.path(input_dir, "manifest.tsv")
if (!file.exists(manifest_path)) {
  stop(
    paste0(
      "Missing ", manifest_path, ". Run the following command from the project root first:\n",
      "  python scripts/prepare_revision_e6.py"
    ),
    call. = FALSE
  )
}
manifest <- read.delim(manifest_path, check.names = FALSE)
manifest <- manifest[manifest$split == split_name, , drop = FALSE]
if (!is.na(only_system)) manifest <- manifest[manifest$system_id == only_system, , drop = FALSE]
if (nrow(manifest) == 0) {
  stop(
    sprintf(
      "No systems matched split='%s'%s in %s.",
      split_name,
      if (is.na(only_system)) "" else paste0(" and system_id='", only_system, "'"),
      manifest_path
    ),
    call. = FALSE
  )
}

edge_rows <- list()
summary_rows <- list()
edge_cursor <- 1
summary_cursor <- 1

append_edges <- function(system_id, method, config, matrices) {
  for (regime in seq_along(matrices)) {
    mat <- as.matrix(matrices[[regime]])
    idx <- expand.grid(target = seq_len(nrow(mat)), regulator = seq_len(ncol(mat)))
    idx$score <- as.vector(t(mat))
    idx$regime <- regime
    idx$config <- config
    idx$method <- method
    idx$system_id <- system_id
    edge_rows[[edge_cursor]] <<- idx[, c("system_id", "method", "config", "regime", "target", "regulator", "score")]
    edge_cursor <<- edge_cursor + 1
  }
}

safe_stationary_var <- function(x) {
  p <- ncol(x)
  fit <- try(sparsevar::fit_var(x, p = 1, penalty = "ENET", method = "cv"), silent = TRUE)
  if (inherits(fit, "try-error")) return(matrix(0, p, p))
  as.matrix(fit$A[[1]])
}

run_edison <- function(x, system_id, config) {
  p <- ncol(x)
  n <- nrow(x)
  set.seed(sum(utf8ToInt(system_id)) + ifelse(config == "poisson", 11, 29))
  elapsed <- system.time({
    log_text <- capture.output({
      fit <- EDISON::EDISON.run(t(x), information.sharing = config, num.iter = 5000)
    })
  })[["elapsed"]]
  cp_post <- EDISON::calculateCPProbabilities(fit)
  cp_score <- colMeans(cp_post$node.cps)
  eligible <- seq.int(6, n - 4)
  best_position <- eligible[which.max(cp_score[eligible])]
  best_score <- cp_score[best_position]
  # Keep a continuous changepoint score; the decision threshold is calibrated
  # only on the calibration systems in the downstream scorer.
  time_prob <- EDISON::calculateEdgeProbabilitiesTimePoints(fit, c(1, n + 1), p)
  before <- Reduce(`+`, time_prob[seq_len(best_position - 1)]) / (best_position - 1)
  after_ids <- seq.int(best_position, length(time_prob))
  after <- Reduce(`+`, time_prob[after_ids]) / length(after_ids)
  # EDISON stores parent x target; transpose to target x regulator.
  append_edges(system_id, "EDISON", config, list(t(before), t(after)))
  summary_rows[[summary_cursor]] <<- data.frame(
    system_id = system_id, method = "EDISON", config = config,
    cp_positions = as.character(best_position),
    primary_cp_index = best_position - 1,
    cp_score = best_score, runtime_seconds = elapsed,
    status = "ok", stringsAsFactors = FALSE
  )
  summary_cursor <<- summary_cursor + 1
}

run_vardetect <- function(x, system_id, config) {
  p <- ncol(x)
  n <- nrow(x)
  block_size <- if (n <= 25) 2 else 3
  if (config == "wide") {
    l1 <- c(1e-4, 1e-3, 1e-2, 1e-1)
    l2 <- c(1e-3, 1e-2, 1e-1, 1)
  } else {
    l1 <- c(1e-3, 1e-2, 1e-1)
    l2 <- c(1e-2, 1e-1, 1)
  }
  fit <- NULL
  warning_text <- character()
  elapsed <- system.time({
    fit <- tryCatch(
      withCallingHandlers(
        VARDetect::tbss(x, method = "sparse", q = 1, block.size = block_size,
                       lambda.1.cv = l1, lambda.2.cv = l2, refit = FALSE,
                       use.BIC = TRUE, verbose = FALSE),
        warning = function(w) {
          warning_text <<- c(warning_text, conditionMessage(w))
          invokeRestart("muffleWarning")
        }
      ),
      error = function(e) e
    )
  })[["elapsed"]]
  if (inherits(fit, "error")) {
    cps <- numeric()
    status <- paste0("error: ", conditionMessage(fit))
    mats <- list(safe_stationary_var(x), safe_stationary_var(x))
  } else {
    cps <- as.numeric(fit$cp)
    status <- if (length(warning_text)) paste0("ok_with_warning: ", paste(unique(warning_text), collapse = " | ")) else "ok"
    mats <- fit$est_phi
    if (length(mats) == 0) {
      stationary <- safe_stationary_var(x)
      mats <- list(stationary, stationary)
    } else if (length(mats) == 1) {
      mats <- list(mats[[1]], mats[[1]])
    } else {
      # A single-change truth is used in E6.  Preserve the first and last
      # estimated regimes when VARDetect proposes more than one boundary.
      mats <- list(mats[[1]], mats[[length(mats)]])
    }
  }
  primary <- if (length(cps)) cps[[1]] - 1 else NA_real_
  append_edges(system_id, "VARDetect", config, lapply(mats, abs))
  summary_rows[[summary_cursor]] <<- data.frame(
    system_id = system_id, method = "VARDetect", config = config,
    cp_positions = if (length(cps)) paste(cps, collapse = ";") else "",
    primary_cp_index = primary, cp_score = as.numeric(length(cps) > 0),
    runtime_seconds = elapsed, status = status, stringsAsFactors = FALSE
  )
  summary_cursor <<- summary_cursor + 1
}

for (i in seq_len(nrow(manifest))) {
  row <- manifest[i, ]
  dat <- read.delim(file.path(input_dir, paste0(row$system_id, ".tsv")), check.names = FALSE)
  x <- scale(as.matrix(dat[, -1, drop = FALSE]))
  x[!is.finite(x)] <- 0
  if (method_name == "edison") {
    # Compare both priors only during calibration.  The frozen formal-test
    # configuration is Poisson; running bino_hard for e4test only duplicates
    # MCMC work because the scorer intentionally discards that configuration.
    configs <- if (split_name == "calibration") c("poisson", "bino_hard") else c("poisson")
    for (config in configs) {
      if (split_name == "frozen_short") {
        problem <- tryCatch({run_edison(x, row$system_id, config); NULL}, error = function(e) e)
        if (inherits(problem, "error")) {
          summary_rows[[summary_cursor]] <- data.frame(
            system_id = row$system_id, method = "EDISON", config = config,
            cp_positions = NA_character_, primary_cp_index = NA_real_,
            cp_score = NA_real_, runtime_seconds = NA_real_,
            status = paste0("error: ", conditionMessage(problem)),
            stringsAsFactors = FALSE
          )
          summary_cursor <- summary_cursor + 1
        }
      } else {
        run_edison(x, row$system_id, config)
      }
    }
  }
  if (method_name == "vardetect") {
    configs <- if (split_name == "calibration") c("wide", "compact") else c("compact")
    for (config in configs) run_vardetect(x, row$system_id, config)
  }
  cat(sprintf("completed %s (%d/%d)\n", row$system_id, i, nrow(manifest)))
}

suffix <- if (is.na(only_system)) paste(split_name, method_name, sep = "_") else paste(split_name, method_name, only_system, sep = "_")
write.table(do.call(rbind, summary_rows),
            file.path(output_dir, paste0("external_", suffix, "_summary.tsv")),
            sep = "\t", row.names = FALSE, quote = FALSE)
write.table(do.call(rbind, edge_rows),
            file.path(output_dir, paste0("external_", suffix, "_edges.tsv")),
            sep = "\t", row.names = FALSE, quote = FALSE)
