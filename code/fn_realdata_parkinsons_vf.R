################################################################################
## parkinsons_pdsvm.R
## End-to-end (local) PDSVM pipeline for Parkinson's speech features.
################################################################################
rm(list = ls())
if (!requireNamespace("e1071", quietly = TRUE))
  install.packages("e1071", repos = "http://cran.us.r-project.org")
suppressPackageStartupMessages({ library(e1071); library(parallel) })

source("Deep_SVM_vf.R")

## ============================== CONFIG ======================================
PREP_RDS   <- "parkinsons_prep_754.rds"
RESULT_DIR <- "results_parkinsons_754"
REP_DIR    <- file.path(RESULT_DIR, "rep_results")
TUNING_RDS <- file.path(RESULT_DIR, "tuning.rds")
dir.create(REP_DIR, recursive = TRUE, showWarnings = FALSE)

ARCH_GRID         <- list(c(16, 8, 4), c(24, 12, 6), c(32, 16, 8), c(48, 24, 12))
H_GRID            <- c(0.01, 0.02, 0.05, 0.10, 0.20)
LR_GRID           <- c(0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
ACT_GRID          <- c("tanh", "relu", "sigmoid")
LAMBDA_GRID_CE    <- c(40, 20, 10, 5, 2.5, 1.25, 0.6, 0.3, 0.15, 0.07, 0.03, 0.01)
LAMBDA_GRID_HINGE <- c(80, 40, 20, 10, 5, 2.5, 1.25, 0.6, 0.3, 0.15, 0.07, 0.03, 0.01)

R_REPS         <- 50      # selection-stability subsamples
SUBSAMPLE_FRAC <- 0.8
TUNE_EPOCH     <- 3000
FINAL_EPOCH    <- 3000
CV_K           <- 5       # subject-grouped folds for lambda selection
SE_MULT        <- 1       # 1-SE rule multiplier
WARMUP_FRAC    <- 0.3
SEL_THRESH     <- 1e-4
LAMBDA2_FIXED  <- 1e-4
GAMMA          <- 1       
NCORES         <- 1       
SEED           <- 20260529

l1.names <- c("SPINN", "PDSVM-GL", "PDSVM-AGL")
methods  <- c("KSVM", "D-CE", "D-Ridge", l1.names)
## ============================================================================

## ------------------------------ helpers -------------------------------------
to01  <- function(v) ifelse(v == 1, 1L, 0L)
to_pm <- function(v) ifelse(v == 1, 1, -1)
se    <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))
bal_acc <- function(pred, ytrue) {
  pr <- to01(pred); yt <- to01(ytrue)
  mean(c(mean(pr[yt == 1] == 1), mean(pr[yt == 0] == 0)))
}
standardize_by <- function(Xref, ...) {
  mu <- colMeans(Xref); sdv <- apply(Xref, 2, sd); sdv[sdv == 0] <- 1
  lapply(list(...), function(M) scale(M, mu, sdv))
}
make_init <- function(pp, n1, n2, n3, seed) {
  set.seed(seed); g <- function(a, b) sqrt(6 / (a + b))
  c1 <- g(pp + 1, n1); c2 <- g(n1 + 1, n2); c3 <- g(n2 + 1, n3); c0 <- g(n3 + 1, 1)
  list(W1 = matrix(runif(n1 * (pp + 1), -c1, c1), n1, pp + 1),
       W2 = matrix(runif(n2 * (n1 + 1), -c2, c2), n2, n1 + 1),
       W3 = matrix(runif(n3 * (n2 + 1), -c3, c3), n3, n2 + 1),
       W0 = matrix(runif(1  * (n3 + 1), -c0, c0), 1,  n3 + 1))
}
calc_nogueira <- function(pj, d_avg, p_total, M) {
  if (d_avg == 0 || d_avg == p_total) return(NA_real_)
  1 - (M / (M - 1)) * (mean(pj * (1 - pj)) / ((d_avg / p_total) * (1 - d_avg / p_total)))
}

SPINN3_GLasso_val <- function(W1, W2, W3, W0, X, y, X_val, y_val, Epoch = 1000, rate = 0.05,
                              lambda1 = 0.05, lambda2 = 1e-4, warmup_frac = 0,
                              penalty_type = "prox", activation = "tanh") {
  n <- nrow(X); eta <- rate; w_hat <- rep(1, ncol(X) - 1)
  warm_E <- max(1L, floor(warmup_frac * Epoch))
  best_acc <- -Inf; W1b <- W1; W2b <- W2; W3b <- W3; W0b <- W0
  for (t in seq_len(Epoch)) {
    fp <- forward_pass_3(W1, W2, W3, W0, X, activation)
    p0 <- Activation(as.vector(fp$z0), "sigmoid"); d0 <- matrix((p0 - y) / n, nrow = 1)
    grad_W0 <- d0 %*% t(fp$y3a); back0 <- t(W0) %*% d0
    delta3 <- back0[-1, , drop = FALSE] * Activation_prime(fp$y3, activation); grad_W3 <- delta3 %*% t(fp$y2a); back3 <- t(W3) %*% delta3
    delta2 <- back3[-1, , drop = FALSE] * Activation_prime(fp$y2, activation); grad_W2 <- delta2 %*% t(fp$y1a); back2 <- t(W2) %*% delta2
    delta1 <- back2[-1, , drop = FALSE] * Activation_prime(fp$y1, activation); grad_W1 <- delta1 %*% X
    if (lambda2 > 0) { grad_W0 <- grad_W0 + 2 * lambda2 * W0; grad_W3 <- grad_W3 + 2 * lambda2 * W3; grad_W2 <- grad_W2 + 2 * lambda2 * W2 }
    W0 <- W0 - eta * grad_W0; W3 <- W3 - eta * grad_W3; W2 <- W2 - eta * grad_W2
    lam_t <- lambda1 * min(1, t / warm_E)
    if (lam_t > 0 && penalty_type == "prox") {
      W1h <- W1 - eta * grad_W1; W1f <- group_soft_threshold(W1h[, -1, drop = FALSE], eta * lam_t * w_hat)
      W1 <- cbind(W1h[, 1, drop = FALSE], W1f)
    } else if (lam_t > 0) {
      pen <- glasso_penalty_grad(W1[, -1, drop = FALSE], w_hat); g1 <- grad_W1; g1[, -1] <- grad_W1[, -1] + lam_t * pen; W1 <- W1 - eta * g1
    } else W1 <- W1 - eta * grad_W1
    pvl <- Activation(as.vector(forward_pass_3(W1, W2, W3, W0, X_val, activation)$z0), "sigmoid")
    a <- bal_acc(pvl > 0.5, y_val); if (a > best_acc) { best_acc <- a; W1b <- W1; W2b <- W2; W3b <- W3; W0b <- W0 }
  }
  list(Weight1 = W1b, Weight2 = W2b, Weight3 = W3b, Weight0 = W0b, Best_ACC_val = best_acc)
}

## --------------------------- load prepped data ------------------------------
if (!file.exists(PREP_RDS)) stop(paste("Cannot find", PREP_RDS))
D <- readRDS(PREP_RDS)
p          <- ncol(D$Xtr)
feat_names <- D$feature_names
base_mask  <- as.logical(D$base_mask)
has_mask   <- length(base_mask) > 0
set.seed(SEED)

## ============================================================================
## ============================================================================
calibrate_baseline <- function() {
  std <- standardize_by(D$Xtr, D$Xtr, D$Xvl); Xtr_s <- std[[1]]; Xvl_s <- std[[2]]
  ytr_pm <- to_pm(D$ytr); yvl_pm <- to_pm(D$yvl); ytr_01 <- to01(D$ytr); yvl_01 <- to01(D$yvl)
  
  ## hinge baseline: arch x h x lr x act
  gh <- expand.grid(arch = seq_along(ARCH_GRID), h = H_GRID, lr = LR_GRID, act = ACT_GRID, stringsAsFactors = FALSE)
  gh$val <- NA_real_
  for (i in seq_len(nrow(gh))) {
    a <- ARCH_GRID[[gh$arch[i]]]; ini <- make_init(p, a[1], a[2], a[3], seed = 1001)
    f <- PDSVM3_val(ini$W1, ini$W2, ini$W3, ini$W0, cbind(1, Xtr_s), ytr_pm, cbind(1, Xvl_s), yvl_pm,
                    Epoch = TUNE_EPOCH, rate = gh$lr[i], lambda1 = 0, lambda2 = LAMBDA2_FIXED,
                    h = gh$h[i], activation = gh$act[i], verbose = FALSE)
    pr <- Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1, Xvl_s), yvl_pm, activation = gh$act[i])$Pred
    gh$val[i] <- bal_acc(pr, D$yvl)
  }
  bH <- gh[which.max(gh$val), ]; ARCH_STAR <- ARCH_GRID[[bH$arch]]
  NODE1 <- ARCH_STAR[1]; NODE2 <- ARCH_STAR[2]; NODE3 <- ARCH_STAR[3]
  H_STAR_HINGE <- bH$h; LR_STAR_HINGE <- bH$lr; ACT_STAR_HINGE <- bH$act
  init0 <- make_init(p, NODE1, NODE2, NODE3, seed = 1001)
  cat(sprintf("[Hinge Baseline*] arch=(%d,%d,%d) h=%.2f lr=%.4f act=%s val=%.4f\n",
              NODE1, NODE2, NODE3, H_STAR_HINGE, LR_STAR_HINGE, ACT_STAR_HINGE, bH$val))
  
  ## CE baseline: lr x act at ARCH_STAR
  gc <- expand.grid(lr = LR_GRID, act = ACT_GRID, stringsAsFactors = FALSE); gc$val <- NA_real_
  for (i in seq_len(nrow(gc))) {
    f <- CE3_val(init0$W1, init0$W2, init0$W3, init0$W0, cbind(1, Xtr_s), ytr_01, cbind(1, Xvl_s), yvl_01,
                 Epoch = TUNE_EPOCH, rate = gc$lr[i], lambda2 = LAMBDA2_FIXED, activation = gc$act[i], verbose = FALSE)
    pr <- Binary3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1, Xvl_s), D$yvl, activation = gc$act[i])$Pred
    gc$val[i] <- bal_acc(pr, D$yvl)
  }
  bC <- gc[which.max(gc$val), ]; LR_STAR_CE <- bC$lr; ACT_STAR_CE <- bC$act
  cat(sprintf("[CE Baseline*]    lr=%.4f act=%s val=%.4f\n", LR_STAR_CE, ACT_STAR_CE, bC$val))
  
  list(NODE1 = NODE1, NODE2 = NODE2, NODE3 = NODE3,
       H_STAR_HINGE = H_STAR_HINGE, LR_STAR_HINGE = LR_STAR_HINGE, ACT_STAR_HINGE = ACT_STAR_HINGE,
       LR_STAR_CE = LR_STAR_CE, ACT_STAR_CE = ACT_STAR_CE)
}

cat("=== CALIB BASELINE ===\n")
tune <- if (file.exists(TUNING_RDS)) readRDS(TUNING_RDS) else { t0 <- calibrate_baseline(); saveRDS(t0, TUNING_RDS); t0 }

## ============================================================================
## ITERATION
## ============================================================================
run_rep <- function(r, tune) {
  NODE1 <- tune$NODE1; NODE2 <- tune$NODE2; NODE3 <- tune$NODE3
  H_STAR_HINGE <- tune$H_STAR_HINGE; LR_STAR_HINGE <- tune$LR_STAR_HINGE; ACT_STAR_HINGE <- tune$ACT_STAR_HINGE
  LR_STAR_CE <- tune$LR_STAR_CE; ACT_STAR_CE <- tune$ACT_STAR_CE
  
  yte_pm <- to_pm(D$yte); yte_01 <- to01(D$yte)
  
  # 1. data Subsampling
  subj_tr <- unique(D$id_tr); cls_s <- D$ytr[match(subj_tr, D$id_tr)]
  pos_s <- subj_tr[cls_s == 1]; neg_s <- subj_tr[cls_s == 0]
  set.seed(7000 + r)
  keep_s <- c(sample(pos_s, floor(SUBSAMPLE_FRAC * length(pos_s))),
              sample(neg_s, floor(SUBSAMPLE_FRAC * length(neg_s))))
  sub <- which(D$id_tr %in% keep_s)
  Xtr_r <- D$Xtr[sub, , drop = FALSE]; ytr_r <- D$ytr[sub]
  
  st <- standardize_by(Xtr_r, Xtr_r, D$Xvl, D$Xte); Xtr_s <- st[[1]]; Xvl_s <- st[[2]]; Xte_s <- st[[3]]
  ytr_pm <- to_pm(ytr_r); yvl_pm <- to_pm(D$yvl); ytr_01 <- to01(ytr_r); yvl_01 <- to01(D$yvl)
  init <- make_init(p, NODE1, NODE2, NODE3, seed = 1365 * r)
  
  acc  <- setNames(numeric(length(methods)), methods)
  nsel <- setNames(numeric(length(l1.names)), l1.names)
  selfreq <- lapply(l1.names, function(.) numeric(p)); names(selfreq) <- l1.names
  
  # 2. Baseline Methods (KSVM, D-CE, D-Ridge)
  tk  <- tune.svm(Xtr_s, as.factor(ytr_r), kernel = "radial", gamma = 10^(-4:-1), cost = 10^(-1:2),
                  scale = FALSE, tunecontrol = tune.control(cross = 3))
  acc["KSVM"] <- bal_acc(as.numeric(as.character(predict(tk$best.model, Xte_s))), D$yte)
  
  fce <- CE3_val(init$W1, init$W2, init$W3, init$W0, cbind(1, Xtr_s), ytr_01, cbind(1, Xvl_s), yvl_01,
                 Epoch = FINAL_EPOCH, rate = LR_STAR_CE, lambda2 = LAMBDA2_FIXED, activation = ACT_STAR_CE, verbose = FALSE)
  acc["D-CE"] <- bal_acc(Binary3_predict(fce$Weight1, fce$Weight2, fce$Weight3, fce$Weight0, cbind(1, Xte_s), yte_01, activation = ACT_STAR_CE)$Pred, D$yte)
  
  frd <- PDSVM3_val(init$W1, init$W2, init$W3, init$W0, cbind(1, Xtr_s), ytr_pm, cbind(1, Xvl_s), yvl_pm,
                    Epoch = FINAL_EPOCH, rate = LR_STAR_HINGE, lambda1 = 0, lambda2 = LAMBDA2_FIXED, h = H_STAR_HINGE, activation = ACT_STAR_HINGE, verbose = FALSE)
  acc["D-Ridge"] <- bal_acc(Hinge3_predict(frd$Weight1, frd$Weight2, frd$Weight3, frd$Weight0, cbind(1, Xte_s), yte_pm, activation = ACT_STAR_HINGE)$Pred, D$yte)
  
  # Adaptive Weights
  w_hat_r <- compute_adaptive_weights(frd$Weight1[, -1, drop = FALSE], gamma = GAMMA, eps = 1e-8)
  
  # 3. CV Lambda Tuning
  Xcv_r <- rbind(Xtr_r, D$Xvl); ycv_r <- c(ytr_r, D$yvl); idcv_r <- c(D$id_tr[sub], D$id_vl)
  uid <- unique(idcv_r); ucl <- ycv_r[match(uid, idcv_r)]; fs <- integer(length(uid))
  for (cl in unique(ucl)) { w <- sample(which(ucl == cl)); fs[w] <- (seq_along(w) - 1) %% CV_K + 1L }
  rowf <- setNames(fs, as.character(uid))[as.character(idcv_r)]
  
  # 4. main iteration
  for (m in l1.names) {
    is_ce <- (m == "SPINN"); lam_grid <- if (is_ce) LAMBDA_GRID_CE else LAMBDA_GRID_HINGE
    A <- matrix(NA_real_, CV_K, length(lam_grid))
    
    for (k in seq_len(CV_K)) {
      tri <- which(rowf != k); vai <- which(rowf == k)
      sc <- standardize_by(Xcv_r[tri, , drop = FALSE], Xcv_r[tri, , drop = FALSE], Xcv_r[vai, , drop = FALSE])
      Xtk <- sc[[1]]; Xvk <- sc[[2]]
      ytk_pm <- to_pm(ycv_r[tri]); yvk_pm <- to_pm(ycv_r[vai]); ytk_01 <- to01(ycv_r[tri]); yvk_01 <- to01(ycv_r[vai])
      ik <- make_init(p, NODE1, NODE2, NODE3, seed = 1001 + r + k)
      
      if (m == "PDSVM-AGL") {
        pk <- PDSVM3_val(ik$W1, ik$W2, ik$W3, ik$W0, cbind(1, Xtk), ytk_pm, cbind(1, Xvk), yvk_pm,
                         Epoch = TUNE_EPOCH, rate = LR_STAR_HINGE, lambda1 = 0, lambda2 = LAMBDA2_FIXED,
                         h = H_STAR_HINGE, activation = ACT_STAR_HINGE, verbose = FALSE)
        wk <- compute_adaptive_weights(pk$Weight1[, -1, drop = FALSE], gamma = GAMMA, eps = 1e-8)
      } else { wk <- NULL }
      
      for (j in seq_along(lam_grid)) {
        if (is_ce) {
          f <- SPINN3_GLasso_val(ik$W1, ik$W2, ik$W3, ik$W0, cbind(1, Xtk), ytk_01, cbind(1, Xvk), yvk_01,
                                 Epoch = TUNE_EPOCH, rate = LR_STAR_CE, lambda1 = lam_grid[j], lambda2 = LAMBDA2_FIXED,
                                 penalty_type = "prox", warmup_frac = WARMUP_FRAC, activation = ACT_STAR_CE)
          pr <- Binary3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1, Xvk), yvk_01, activation = ACT_STAR_CE)$Pred
        } else if (m == "PDSVM-GL") {
          f <- PDSVM3_val(ik$W1, ik$W2, ik$W3, ik$W0, cbind(1, Xtk), ytk_pm, cbind(1, Xvk), yvk_pm,
                          Epoch = TUNE_EPOCH, rate = LR_STAR_HINGE, lambda1 = lam_grid[j], lambda2 = LAMBDA2_FIXED,
                          h = H_STAR_HINGE, penalty_type = "prox", warmup_frac = WARMUP_FRAC, activation = ACT_STAR_HINGE, verbose = FALSE)
          pr <- Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1, Xvk), yvk_pm, activation = ACT_STAR_HINGE)$Pred
        } else {
          f <- PDSVM3_val(ik$W1, ik$W2, ik$W3, ik$W0, cbind(1, Xtk), ytk_pm, cbind(1, Xvk), yvk_pm,
                          Epoch = TUNE_EPOCH, rate = LR_STAR_HINGE, lambda1 = lam_grid[j], lambda2 = LAMBDA2_FIXED,
                          h = H_STAR_HINGE, adaptive_weights = wk, penalty_type = "prox", warmup_frac = WARMUP_FRAC, activation = ACT_STAR_HINGE, verbose = FALSE)
          pr <- Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1, Xvk), yvk_pm, activation = ACT_STAR_HINGE)$Pred
        }
        A[k, j] <- bal_acc(pr, ycv_r[vai])
      }
    }
    
    mean_acc <- colMeans(A, na.rm = TRUE); se_acc <- apply(A, 2, sd) / sqrt(CV_K)
    jbest <- which.max(mean_acc)
    thr <- mean_acc[jbest] - SE_MULT * se_acc[jbest]
    lambda_star_m <- lam_grid[which(mean_acc >= thr)[which.max(lam_grid[which(mean_acc >= thr)])]] #sparsest 1-SE lambda
    

    if (is_ce) {
      f <- SPINN3_GLasso_val(init$W1, init$W2, init$W3, init$W0, cbind(1, Xtr_s), ytr_01, cbind(1, Xvl_s), yvl_01,
                             Epoch = FINAL_EPOCH, rate = LR_STAR_CE, lambda1 = lambda_star_m, lambda2 = LAMBDA2_FIXED,
                             penalty_type = "prox", warmup_frac = WARMUP_FRAC, activation = ACT_STAR_CE)
      acc[m] <- bal_acc(Binary3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1, Xte_s), yte_01, activation = ACT_STAR_CE)$Pred, D$yte)
    } else if (m == "PDSVM-GL") {
      f <- PDSVM3_val(init$W1, init$W2, init$W3, init$W0, cbind(1, Xtr_s), ytr_pm, cbind(1, Xvl_s), yvl_pm,
                      Epoch = FINAL_EPOCH, rate = LR_STAR_HINGE, lambda1 = lambda_star_m, lambda2 = LAMBDA2_FIXED,
                      h = H_STAR_HINGE, penalty_type = "prox", warmup_frac = WARMUP_FRAC, activation = ACT_STAR_HINGE, verbose = FALSE)
      acc[m] <- bal_acc(Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1, Xte_s), yte_pm, activation = ACT_STAR_HINGE)$Pred, D$yte)
    } else {
      f <- PDSVM3_val(init$W1, init$W2, init$W3, init$W0, cbind(1, Xtr_s), ytr_pm, cbind(1, Xvl_s), yvl_pm,
                      Epoch = FINAL_EPOCH, rate = LR_STAR_HINGE, lambda1 = lambda_star_m, lambda2 = LAMBDA2_FIXED,
                      h = H_STAR_HINGE, adaptive_weights = w_hat_r, penalty_type = "prox", warmup_frac = WARMUP_FRAC, activation = ACT_STAR_HINGE, verbose = FALSE)
      acc[m] <- bal_acc(Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1, Xte_s), yte_pm, activation = ACT_STAR_HINGE)$Pred, D$yte)
    }
    sel <- pdsvm_selected_features(f, SEL_THRESH)$selected
    selfreq[[m]][sel] <- 1; nsel[m] <- length(sel)
  }
  
  cat(sprintf("rep %d | %s\n", r, paste(sprintf("%s=%.3f", methods, acc[methods]), collapse = " ")))
  out <- list(acc = acc, nsel = nsel, selfreq = selfreq)
  saveRDS(out, file.path(REP_DIR, sprintf("rep_%03d.rds", r)))
  out
}

cat(sprintf("=== SIM: %d reps (NCORES=%d) ===\n", R_REPS, NCORES))
reps <- if (NCORES > 1)
  mclapply(seq_len(R_REPS), run_rep, tune = tune, mc.cores = NCORES) else
    lapply(seq_len(R_REPS), run_rep, tune = tune)

## ============================================================================
## COMBINE
## ============================================================================
cat("=== COMBINE ===\n")
M    <- length(reps)
ACC  <- do.call(rbind, lapply(reps, function(z) z$acc[methods]));  dimnames(ACC)  <- list(NULL, methods)
NSEL <- do.call(rbind, lapply(reps, function(z) z$nsel[l1.names])); dimnames(NSEL) <- list(NULL, l1.names)
SELfreq <- setNames(lapply(l1.names, function(m)
  Reduce(`+`, lapply(reps, function(z) z$selfreq[[m]])) / M), l1.names)

summ <- data.frame(method = methods,
                   bal_acc = round(colMeans(ACC, na.rm = TRUE), 4),
                   bal_se  = round(apply(ACC, 2, se), 4),
                   nsel = NA_real_, base_hit_ratio = NA_real_, base_prec_ratio = NA_real_, stability = NA_real_,
                   row.names = methods)
base_rate <- if (has_mask) mean(base_mask) else NA_real_
for (m in l1.names) {
  pi_m <- SELfreq[[m]]; avg_d <- mean(NSEL[, m]); summ[m, "nsel"] <- round(avg_d, 2)
  if (M > 1) summ[m, "stability"] <- round(calc_nogueira(pi_m, avg_d, p, M), 4)
  if (has_mask) {
    summ[m, "base_hit_ratio"] <- ifelse(sum(pi_m) > 0, round((sum(pi_m[base_mask]) / sum(pi_m)) / base_rate, 4), NA)
    sel_set <- which(pi_m > 0.5)
    summ[m, "base_prec_ratio"] <- ifelse(length(sel_set) > 0, round(mean(base_mask[sel_set]) / base_rate, 4), NA)
  }
}
write.csv(summ, file.path(RESULT_DIR, "metrics.csv"), row.names = FALSE)
cat("\n--- metrics ---\n"); print(summ)

sf <- data.frame(feature = feat_names, base_clinical = base_mask, check.names = FALSE)
for (m in l1.names) sf[[m]] <- round(SELfreq[[m]], 4)
sf <- sf[order(-sf[["PDSVM-AGL"]], -sf[["PDSVM-GL"]]), ]
write.csv(sf, file.path(RESULT_DIR, "selection_frequency_full.csv"), row.names = FALSE)

TOPK <- 20
png(file.path(RESULT_DIR, "top_features_comparison.png"), width = 1500, height = 650, res = 120)
par(mfrow = c(1, 3), mar = c(9, 4, 3, 1))
for (m in l1.names) {
  pi_m <- SELfreq[[m]]; ord <- order(pi_m, decreasing = TRUE)[seq_len(min(TOPK, p))]
  cols <- ifelse(base_mask[ord], "firebrick", "gray70")
  barplot(pi_m[ord], names.arg = feat_names[ord], las = 2, col = cols, border = NA,
          main = sprintf("%s (Top %d)", m, TOPK), ylab = "Selection frequency", ylim = c(0, 1), cex.names = 0.7)
}
legend("topright", legend = c("clinical baseline", "derived"), fill = c("firebrick", "gray70"), bty = "n", cex = 0.8)
dev.off()

save(ACC, SELfreq, NSEL, summ, file = file.path(RESULT_DIR, "parkinsons_results_final.RData"))
cat(sprintf("\n[OK] outputs written to %s/\n", RESULT_DIR))