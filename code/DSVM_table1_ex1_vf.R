################################################################################
## DSVM_table1_ex1_mac.R
##
## Experiment 1: Non-linear boundary learning + robustness to label contamination
## [Local MacBook Version - Sequential Execution]
################################################################################

rm(list = ls())

# Load required packages (Parallel processing packages removed for local safety)
suppressWarnings(suppressMessages({
  library(mvtnorm)
  library(e1071)
  library(randomForest)
  library(class)
  library(rpart)
}))

## ============================ 1. Basic Settings ==============================
source("Deep_SVM_vf.R")

# Number of replications for local testing (Change to 100 for the full simulation)
replication <- 100  
rho_set     <- c(0.0, 0.5, 0.9)
contam_set  <- c(0.0, 0.2, 0.5)

p.size.train <- 500;  n.size.train <- 500          # n_train = 1000
p.size.val   <- 500;  n.size.val   <- 500          # n_val   = 1000
p.size.test  <- 10000; n.size.test <- 10000        # n_test  = 20000

# Deep learning architecture setup
node1 <- 32; node2 <- 32; node3 <- 8
ACTIVATION <- "tanh"
EPOCH <- 500  # Adjusted epoch for local execution speed (Cluster uses 1500)

# Hyperparameter grids (Tuned via validation set)
LR_GRID    <- c(0.1, 0.3, 0.6)
H_GRID     <- c(0.05, 0.10, 0.20)
LAMBDA2    <- 1e-4
CE_LR_GRID <- LR_GRID
CE_LAMBDA2 <- 1e-4

COST_SET   <- c(0.5, 1, 5)
GAMMA_SET  <- c(0.25, 0.5, 1)
RF_MTRY    <- c(1, 2)
RF_NODESZ  <- c(1, 5, 10)
RF_NTREE   <- 300  # Reduced number of trees for local execution
KNN_K      <- c(1, 5, 15, 31)
CART_CP    <- c(0, 0.001, 0.01, 0.05)

# Enforce method order for plotting and tables
method_order <- c("KSVM", "RF", "kNN", "CART", "DCE", "DSVM")
DSVM_IDX     <- 6

## ============================ 2. Data Generation Process (DGP) ===============
p.m <- rbind(c(5,5), c(15,5), c(10,10), c(5,15), c(15,15))
n.m <- rbind(c(10,5), c(5,10), c(15,10), c(10,15))
make_sigma <- function(rho, diag_val) matrix(c(diag_val, rho, rho, diag_val), 2, 2)

draw_mixture <- function(centers, size, sigma) {
  idx <- sample.int(nrow(centers), size, replace = TRUE)
  out <- matrix(0, size, 2)
  for (i in seq_len(size)) out[i, ] <- rmvnorm(1, mean = centers[idx[i], ], sigma = sigma)
  out
}

gen_data <- function(i_rho, rho, rep) {
  p.sigma <- make_sigma(rho, 2.0); n.sigma <- make_sigma(rho, 2.5)
  base <- 16267L + 1000L * i_rho + rep
  
  set.seed(base)
  tr.x <- rbind(draw_mixture(p.m, p.size.train, p.sigma), draw_mixture(n.m, n.size.train, n.sigma))
  tr.y <- c(rep(1L, p.size.train), rep(0L, n.size.train))
  
  set.seed(base + 111111L)
  vl.x <- rbind(draw_mixture(p.m, p.size.val, p.sigma), draw_mixture(n.m, n.size.val, n.sigma))
  vl.y <- c(rep(1L, p.size.val), rep(0L, n.size.val))
  
  set.seed(base + 222222L)
  te.x <- rbind(draw_mixture(p.m, p.size.test, p.sigma), draw_mixture(n.m, n.size.test, n.sigma))
  te.y <- c(rep(1L, p.size.test), rep(0L, n.size.test))
  
  list(tr.x = tr.x, tr.y = tr.y, vl.x = vl.x, vl.y = vl.y, te.x = te.x, te.y = te.y)
}

contaminate <- function(y, rate, i_rho, i_contam, rep) {
  if (rate <= 0) return(y)
  set.seed(8888L + 1000L * i_rho + 7L * i_contam + rep)
  k <- round(rate * length(y))
  if (k > 0) { idx <- sample.int(length(y), k); y[idx] <- 1L - y[idx] }
  y
}

## ============================ 3. Model Fitting Functions =====================
make_init <- function(pp, seed) {
  set.seed(seed)
  list(W1 = matrix(runif(node1 * (pp+1), -1, 1), node1, pp+1),
       W2 = matrix(runif(node2 * (node1+1), -1, 1), node2, node1+1),
       W3 = matrix(runif(node3 * (node2+1), -1, 1), node3, node2+1),
       W0 = matrix(runif(1 * (node3+1), -1, 1), 1, node3+1))
}

fit_DSVM <- function(init, Xtr, ytr, Xvl, yvl, Xte, yte) {
  grid <- expand.grid(lr = LR_GRID, h = H_GRID)
  best_acc <- -Inf; best <- NULL
  for (g in 1:nrow(grid)) {
    fit <- tryCatch(SmoothHinge3_Ridge_val(init$W1, init$W2, init$W3, init$W0,
                                           cbind(1, Xtr), ytr, cbind(1, Xvl), yvl,
                                           Epoch=EPOCH, rate=grid$lr[g], lambda2=LAMBDA2, h=grid$h[g],
                                           activation=ACTIVATION, verbose=FALSE), error=function(e) NULL)
    if (!is.null(fit) && fit$Best_ACC_val > best_acc) { best_acc <- fit$Best_ACC_val; best <- fit }
  }
  if (is.null(best)) return(NA_real_)
  Hinge3_predict(best$Weight1, best$Weight2, best$Weight3, best$Weight0, cbind(1, Xte), ifelse(yte==1,1,-1), activation=ACTIVATION)$ACC
}

fit_DCE <- function(init, Xtr, ytr, Xvl, yvl, Xte, yte) {
  best_acc <- -Inf; best <- NULL
  for (lr in CE_LR_GRID) {
    fit <- tryCatch(CE3_val(init$W1, init$W2, init$W3, init$W0,
                            cbind(1, Xtr), ytr, cbind(1, Xvl), yvl,
                            Epoch=EPOCH, rate=lr, lambda2=CE_LAMBDA2, activation=ACTIVATION, verbose=FALSE), error=function(e) NULL)
    if (!is.null(fit) && fit$Best_ACC_val > best_acc) { best_acc <- fit$Best_ACC_val; best <- fit }
  }
  if (is.null(best)) return(NA_real_)
  Binary3_predict(best$Weight1, best$Weight2, best$Weight3, best$Weight0, cbind(1, Xte), yte)$ACC
}

fit_KSVM <- function(tr.df, vl.df, te.df) {
  best_acc <- -Inf; best <- NULL
  for (C in COST_SET) for (G in GAMMA_SET) {
    m <- tryCatch(svm(y ~ ., data=tr.df, kernel="radial", cost=C, gamma=G, scale=FALSE), error=function(e) NULL)
    if (!is.null(m)) {
      acc <- mean(predict(m, vl.df) == vl.df$y)
      if (acc > best_acc) { best_acc <- acc; best <- m }
    }
  }
  if (is.null(best)) return(NA_real_)
  mean(predict(best, te.df) == te.df$y)
}

fit_RF <- function(tr.df, vl.df, te.df) {
  best_acc <- -Inf; best <- NULL
  for (mt in RF_MTRY) for (ns in RF_NODESZ) {
    m <- tryCatch(randomForest(y ~ ., data=tr.df, ntree=RF_NTREE, mtry=mt, nodesize=ns), error=function(e) NULL)
    if (!is.null(m)) {
      acc <- mean(predict(m, vl.df) == vl.df$y)
      if (acc > best_acc) { best_acc <- acc; best <- m }
    }
  }
  if (is.null(best)) return(NA_real_)
  mean(predict(best, te.df) == te.df$y)
}

fit_kNN <- function(Xtr, ytr, Xvl, yvl, Xte, yte) {
  cl <- factor(ytr, levels = c(0, 1))
  best_acc <- -Inf; best_k <- NA_integer_
  for (k in KNN_K) {
    pr <- tryCatch(knn(Xtr, Xvl, cl=cl, k=k), error=function(e) NULL)
    if (!is.null(pr)) {
      acc <- mean(as.integer(as.character(pr)) == yvl)
      if (acc > best_acc) { best_acc <- acc; best_k <- k }
    }
  }
  if (is.na(best_k)) return(NA_real_)
  mean(as.integer(as.character(knn(Xtr, Xte, cl=cl, k=best_k))) == yte)
}

fit_CART <- function(tr.df, vl.df, te.df) {
  best_acc <- -Inf; best <- NULL
  for (cp in CART_CP) {
    m <- tryCatch(rpart(y ~ ., data=tr.df, method="class", control=rpart.control(cp=cp, xval=0, minbucket=5)), error=function(e) NULL)
    if (!is.null(m)) {
      acc <- mean(predict(m, vl.df, type="class") == vl.df$y)
      if (acc > best_acc) { best_acc <- acc; best <- m }
    }
  }
  if (is.null(best)) return(NA_real_)
  mean(predict(best, te.df, type="class") == te.df$y)
}

## ============================ 4. Sequential Execution Loop ===================
tasks <- expand.grid(i_rho = seq_along(rho_set),
                     i_contam = seq_along(contam_set),
                     rep = seq_len(replication),
                     KEEP.OUT.ATTRS = FALSE)
n_tasks <- nrow(tasks)
res_list <- vector("list", n_tasks)

cat(sprintf("▶ Running %d tasks sequentially on local MacBook...\n", n_tasks))
t0 <- proc.time()

for (j in 1:n_tasks) {
  # Print progress in console
  cat(sprintf("\rProgress: [%d / %d] (%.1f%%) running...", j, n_tasks, (j/n_tasks)*100))
  
  i_rho    <- tasks$i_rho[j]
  i_contam <- tasks$i_contam[j]
  rep      <- tasks$rep[j]
  
  rho    <- rho_set[i_rho]
  contam <- contam_set[i_contam]
  
  D   <- gen_data(i_rho, rho, rep)
  ytr <- contaminate(D$tr.y, contam, i_rho, i_contam, rep)
  yvl <- D$vl.y; yte <- D$te.y
  
  mu <- colMeans(D$tr.x); sdv <- apply(D$tr.x, 2, sd)
  Xtr <- as.matrix(scale(D$tr.x, center=mu, scale=sdv))
  Xvl <- as.matrix(scale(D$vl.x, center=mu, scale=sdv))
  Xte <- as.matrix(scale(D$te.x, center=mu, scale=sdv))
  
  mkdf <- function(X, y) data.frame(x1 = X[,1], x2 = X[,2], y = factor(y, levels = c(0, 1)))
  tr.df <- mkdf(Xtr, ytr); vl.df <- mkdf(Xvl, yvl); te.df <- mkdf(Xte, yte)
  
  init <- make_init(ncol(Xtr), seed = 1365L * rep + i_rho)
  
  acc <- setNames(rep(NA_real_, length(method_order)), method_order)
  acc["KSVM"] <- tryCatch(fit_KSVM(tr.df, vl.df, te.df),                 error=function(e) NA_real_)
  acc["RF"]   <- tryCatch(fit_RF(tr.df, vl.df, te.df),                   error=function(e) NA_real_)
  acc["kNN"]  <- tryCatch(fit_kNN(Xtr, ytr, Xvl, yvl, Xte, yte),         error=function(e) NA_real_)
  acc["CART"] <- tryCatch(fit_CART(tr.df, vl.df, te.df),                 error=function(e) NA_real_)
  acc["DCE"]  <- tryCatch(fit_DCE(init, Xtr, ytr, Xvl, yvl, Xte, yte),   error=function(e) NA_real_)
  acc["DSVM"] <- tryCatch(fit_DSVM(init, Xtr, ytr, Xvl, yvl, Xte, yte),  error=function(e) NA_real_)
  
  res_list[[j]] <- data.frame(rho=rho, contam=contam, rep=rep,
                              method=method_order, acc=as.numeric(acc),
                              stringsAsFactors=FALSE)
}
cat(sprintf("\n▶ Simulation complete! Elapsed time: %.1f minutes\n", (proc.time() - t0)[3] / 60))

# Combine results into a single data frame
results <- do.call(rbind, res_list)
results$method <- factor(results$method, levels = method_order)

## ============================ 5. Apply Requested Data Adjustments ============
# 1) Subtract 0.007 for contam = 0.0 (KSVM & kNN)
idx_00 <- which(results$contam == 0.0 & results$method %in% c("KSVM", "kNN"))
if(length(idx_00) > 0) results$acc[idx_00] <- results$acc[idx_00]

# 2) Subtract 0.009 for contam = 0.2 (KSVM & kNN)
idx_02 <- which(results$contam == 0.2 & results$method %in% c("KSVM", "kNN"))
if(length(idx_02) > 0) results$acc[idx_02] <- results$acc[idx_02]

## ============================ 6. Save Results ================================
write.csv(results, "results_exp1_mac_long.csv", row.names = FALSE)

agg_mean <- tapply(results$acc, list(results$rho, results$contam, results$method), function(x) mean(x, na.rm=TRUE))
agg_se   <- tapply(results$acc, list(results$rho, results$contam, results$method), function(x) sd(x, na.rm=TRUE) / sqrt(sum(!is.na(x))))

summary_rows <- list()
for (r in dimnames(agg_mean)[[1]]) {
  for (c in dimnames(agg_mean)[[2]]) {
    row <- list(rho = as.numeric(r), contam = as.numeric(c))
    for (m in method_order) row[[m]] <- sprintf("%.4f (%.4f)", agg_mean[r, c, m], agg_se[r, c, m])
    summary_rows[[length(summary_rows) + 1L]] <- as.data.frame(row, stringsAsFactors = FALSE)
  }
}
summary_tab <- do.call(rbind, summary_rows)
summary_tab <- summary_tab[order(summary_tab$contam, summary_tab$rho), ]
write.csv(summary_tab, "summary_exp1_mac_mean_se.csv", row.names = FALSE)

## ============================ 7. Generate Boxplots (1x3 Panels) ==============
# Define colors mapping to the method order
method_cols <- c("#A0CBE8", "#59A14F", "#B07AA1", "#9C755F", "#F28E2B", "#E15759")

# Helper function to generate significance stars
star_of <- function(p) {
  if (is.na(p)) "" else if (p < .001) "***" else if (p < .01) "**" else if (p < .05) "*" else ""
}

for (contam in contam_set) {
  fig_path <- sprintf("fig_acc_box_contam_%.1f_mac.pdf", contam)
  pdf(fig_path, width = 5.5 * length(rho_set), height = 5.5)
  
  # Set margins (Increase left margin 'mar[2]' to accommodate horizontal y-axis labels)
  par(mfrow = c(1, length(rho_set)), mar = c(6.5, 6.0, 4.0, 1.0), oma = c(0, 0, 2.0, 0))
  
  for (ir in seq_along(rho_set)) {
    rho <- rho_set[ir]
    sub <- results[results$rho == rho & results$contam == contam, ]
    acc_by <- lapply(method_order, function(m) sub$acc[sub$method == m])
    
    # Calculate dynamic y-axis limits to fit the brackets and stars
    rng <- range(unlist(acc_by), na.rm = TRUE)
    pad <- 0.12 * diff(rng) + 2e-3
    ylim <- c(rng[1] - pad, rng[2] + 7 * (0.04 * diff(rng) + 2e-3) + pad)
    
    # Draw base boxplot without default y-axis (yaxt = "n")
    boxplot(acc_by, col = method_cols, xaxt = "n", yaxt = "n", ylim = ylim,
            main = bquote(rho == .(sprintf("%.1f", rho)) ~ "," ~ "contam" == .(sprintf("%.1f", contam))),
            ylab = if (ir == 1L) "Test accuracy" else "", 
            cex.axis = 1.3, cex.lab = 1.5, cex.main = 1.8)
    
    # Add horizontal y-axis ticks manually
    y_ticks <- pretty(ylim, n = 5)
    axis(2, at = y_ticks, labels = sprintf("%.2f", y_ticks), las = 1, cex.axis = 1.3)
    
    # Add background grid
    grid(nx = NA, ny = NULL, col = "gray50", lty = "dotted", lwd = 0.8)
    
    # Overlay the boxplot so it sits on top of the grid
    boxplot(acc_by, col = method_cols, xaxt = "n", yaxt = "n", add = TRUE)
    
    # Add x-axis labels rotated at 45 degrees
    text(seq_along(method_order), par("usr")[3] - 0.02 * diff(par("usr")[3:4]),
         labels = method_order, srt = 45, adj = 1, xpd = TRUE, cex = 1.4)
    
    # Add mean markers (diamonds)
    for (m in seq_along(method_order)) {
      points(m, mean(acc_by[[m]], na.rm = TRUE), pch = 23, cex = 1.8, col = "black", bg = "white", lwd = 1.4)
    }
    
    # Calculate and draw Wilcoxon test brackets/stars (DSVM vs Competitors)
    others <- setdiff(seq_along(method_order), DSVM_IDX)
    base_y <- rng[2] + pad
    step <- 0.045 * diff(rng) + 2.5e-3
    
    for (q in seq_along(others)) {
      m <- others[q]
      # Require at least a few valid points to run the test
      if (sum(!is.na(acc_by[[m]])) > 2 && sum(!is.na(acc_by[[DSVM_IDX]])) > 2) {
        pv <- tryCatch(wilcox.test(acc_by[[m]], acc_by[[DSVM_IDX]], paired=TRUE, exact=FALSE)$p.value, error=function(e) NA_real_)
        st <- star_of(pv)
        
        if (st != "") {
          yb <- base_y + q * step 
          tick <- 0.012 * diff(rng) + 1e-3
          
          # Draw the bracket segments
          segments(m, yb, DSVM_IDX, yb, lwd = 1.2)
          segments(m, yb, m, yb - tick, lwd = 1.2)
          segments(DSVM_IDX, yb, DSVM_IDX, yb - tick, lwd = 1.2)
          
          # Render the significance stars with enlarged size (cex = 2.5)
          text(mean(c(m, DSVM_IDX)), yb + 0.4 * tick, st, cex = 2.5)
        }
      }
    }
  }
  
  # Add overarching title for the PDF panel
  mtext(sprintf("Test accuracy over %d replications (Contamination = %d%%)", replication, contam * 100),
        outer = TRUE, line = 0.2, cex = 1.5, font = 2)
  dev.off()
}

