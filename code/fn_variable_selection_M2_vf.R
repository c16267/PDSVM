rm(list = ls())

suppressPackageStartupMessages({
  library(e1071); library(ggplot2); library(dplyr); library(tidyr)
})

RUN_CALIB   <- TRUE   
RUN_SIM     <- TRUE  
RUN_COMBINE <- TRUE  

REPS        <- 1    # number of production replications
ACTIVATION  <- "tanh" # activation ("tanh", "relu", "sigmoid")

TAG_MODEL  <- "model4_easy_local"
CODE_DIR   <- getwd()
RESULT_DIR <- file.path(CODE_DIR, paste0("results_", TAG_MODEL))
dir.create(RESULT_DIR, showWarnings = FALSE, recursive = TRUE)

# v6 source (full-batch trainers, multi-activation)
source("Deep_SVM_v6.R")

## =============================================================================
## Custom SPINN trainer (CE + group lasso), full batch, using v6 activations.
## =============================================================================
SPINN3_GLasso_val <- function(W1, W2, W3, W0, X, y, X_val, y_val,
                              Epoch = 1000, rate = 0.05,
                              lambda1 = 0.05, lambda2 = 1e-4,
                              warmup_frac = 0, penalty_type = "prox",
                              activation = "tanh", verbose = FALSE) {
  n <- nrow(X); eta <- rate; w_hat <- rep(1, ncol(X) - 1)
  warm_E <- max(1L, floor(warmup_frac * Epoch))
  best_acc <- -Inf; W1_best <- W1; W2_best <- W2; W3_best <- W3; W0_best <- W0
  
  for (t in seq_len(Epoch)) {
    fp <- forward_pass_3(W1, W2, W3, W0, X, activation)
    p0 <- Activation(as.vector(fp$z0), "sigmoid")
    d0 <- matrix((p0 - y) / n, nrow = 1)
    
    grad_W0 <- d0 %*% t(fp$y3a); back0 <- t(W0) %*% d0
    delta3  <- back0[-1, , drop = FALSE] * Activation_prime(fp$y3, activation)
    grad_W3 <- delta3 %*% t(fp$y2a); back3 <- t(W3) %*% delta3
    delta2  <- back3[-1, , drop = FALSE] * Activation_prime(fp$y2, activation)
    grad_W2 <- delta2 %*% t(fp$y1a); back2 <- t(W2) %*% delta2
    delta1  <- back2[-1, , drop = FALSE] * Activation_prime(fp$y1, activation)
    grad_W1 <- delta1 %*% X
    
    if (lambda2 > 0) { grad_W0 <- grad_W0+2*lambda2*W0; grad_W3 <- grad_W3+2*lambda2*W3; grad_W2 <- grad_W2+2*lambda2*W2 }
    W0 <- W0 - eta*grad_W0; W3 <- W3 - eta*grad_W3; W2 <- W2 - eta*grad_W2
    
    lam_t <- lambda1 * min(1, t / warm_E)
    if (lam_t > 0 && penalty_type == "grad") {
      pen <- glasso_penalty_grad(W1[,-1,drop=FALSE], w_hat)
      g1 <- grad_W1; g1[,-1] <- grad_W1[,-1] + lam_t*pen
      W1 <- W1 - eta*g1
    } else if (lam_t > 0 && penalty_type == "prox") {
      W1_half <- W1 - eta*grad_W1
      W1f <- group_soft_threshold(W1_half[,-1,drop=FALSE], eta*lam_t*w_hat)
      W1 <- cbind(W1_half[, 1, drop=FALSE], W1f)
    } else {
      W1 <- W1 - eta*grad_W1
    }
    
    fp_vl <- forward_pass_3(W1, W2, W3, W0, X_val, activation); pvl <- Activation(as.vector(fp_vl$z0), "sigmoid")
    acc_val <- mean((pvl > 0.5) == (y_val == 1))
    if (acc_val > best_acc) { best_acc <- acc_val; W1_best <- W1; W2_best <- W2; W3_best <- W3; W0_best <- W0 }
  }
  list(Weight1 = W1_best, Weight2 = W2_best, Weight3 = W3_best, Weight0 = W0_best, Best_ACC_val = best_acc)
}

## ---- 1. Settings -------------------------------------------------------------
P_SIGNAL <- 5
P_TOTAL  <- 20
P_NOISE  <- P_TOTAL - P_SIGNAL
NOISE_SD <- 0.5
n_per_class_train <- 1000; n_per_class_val <- 1000; n_per_class_test <- 5000
node1 <- 64; node2 <- 32; node3 <- 8

FINAL_EPOCH <- 3000; TUNE_EPOCH <- 1000
LR_FIXED <- 0.07; H_FIXED <- 0.05; LAMBDA2_FIXED <- 1e-4

PENALTY_TYPE <- "prox"   # proximal group lasso (matches Algorithm 1)
WARMUP_FRAC  <- 0.5      # lambda1 annealed 0 -> full over the first 50% of epochs
JITTER       <- 0.05     # per-rep lambda* multiplier ~ Uniform(1-JITTER, 1+JITTER)
SEL_THRESH   <- 1e-5     # absolute selection threshold on ||w^(1)_{.,j}||_2^2

STANDARDIZE_INPUT <- TRUE
S_true <- 1:P_SIGNAL     # used ONLY for evaluation, never for tuning
method.names <- c("KSVM","Oracle","D-CE","D-Ridge","SPINN","D-GLasso","D-AGLasso")
l1.names     <- c("SPINN","D-GLasso","D-AGLasso")

## ---- 2. DGP (easy additive, independent features) ----------------------------
h_fun <- function(X) X[,1]^2 + X[,2]^2 + X[,3]*X[,4] + sin(X[,5])
set.seed(20260425)
.lx <- matrix(rnorm(2e5 * P_TOTAL), 2e5, P_TOTAL)
POP_MEDIAN <- median(h_fun(.lx) + rnorm(2e5, 0, NOISE_SD)); rm(.lx)

generate_balanced_data <- function(n_per_class, seed) {
  set.seed(seed); X_out <- matrix(0, 0, P_TOTAL); y_out <- numeric(0)
  while (sum(y_out == 1) < n_per_class || sum(y_out == 0) < n_per_class) {
    Xt <- matrix(rnorm(n_per_class * 2 * P_TOTAL), n_per_class * 2, P_TOTAL)
    yt <- ifelse(h_fun(Xt) + rnorm(nrow(Xt), 0, NOISE_SD) > POP_MEDIAN, 1, 0)
    X_out <- rbind(X_out, Xt); y_out <- c(y_out, yt)
  }
  i1 <- which(y_out == 1)[1:n_per_class]; i0 <- which(y_out == 0)[1:n_per_class]
  Xf <- X_out[c(i1, i0), ]; yf <- y_out[c(i1, i0)]; sh <- sample(length(yf))
  list(x = Xf[sh, ], y = yf[sh])
}

make_init <- function(pp, node1, node2, node3, seed) {
  set.seed(seed); glorot <- function(a,b) sqrt(6/(a+b))
  c1<-glorot(pp+1,node1);c2<-glorot(node1+1,node2);c3<-glorot(node2+1,node3);c0<-glorot(node3+1,1)
  list(W1=matrix(runif(node1*(pp+1),-c1,c1),node1,pp+1),
       W2=matrix(runif(node2*(node1+1),-c2,c2),node2,node1+1),
       W3=matrix(runif(node3*(node2+1),-c3,c3),node3,node2+1),
       W0=matrix(runif(1*(node3+1),-c0,c0),1,node3+1))
}

## =============================================================================
## MODE 1: CALIBRATION  (select lambda1 by validation accuracy)
## =============================================================================
if (RUN_CALIB) {
  cat(sprintf("\n========== STEP 0: CALIBRATION (prox mode | %s) ==========\n", ACTIVATION))
  
  K_CALIB <- 10
  COARSE_GRID_BY_METHOD <- list(
    "SPINN"     = c(0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.002),
    "D-GLasso"  = c(0.05, 0.03, 0.02, 0.015, 0.01, 0.008, 0.005, 0.002),
    "D-AGLasso" = c(0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005)
  )
  
  calib_one_rep <- function(k, grids) {
    tr <- generate_balanced_data(n_per_class_train, 500000L + k)
    vl <- generate_balanced_data(n_per_class_val,   600000L + k)
    if (STANDARDIZE_INPUT) {
      mu<-colMeans(tr$x); sdv<-apply(tr$x,2,sd); sdv[sdv==0]<-1
      Xtr<-scale(tr$x,mu,sdv); Xvl<-scale(vl$x,mu,sdv)
    } else { Xtr<-tr$x; Xvl<-vl$x }
    train.y <- tr$y; val.y <- vl$y
    train.y_h <- ifelse(tr$y==1,1,-1); val.y_h <- ifelse(vl$y==1,1,-1)
    
    init <- make_init(P_TOTAL,node1,node2,node3, seed = 700000L + k)
    
    fit_ridge <- PDSVM3_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y_h, cbind(1,Xvl),val.y_h,
                            Epoch=TUNE_EPOCH, rate=LR_FIXED, lambda1=0, lambda2=LAMBDA2_FIXED,
                            h=H_FIXED, activation=ACTIVATION, verbose=FALSE)
    w_hat <- compute_adaptive_weights(fit_ridge$Weight1[,-1,drop=FALSE], gamma=1, eps=1e-8)
    
    fitters <- list(
      "SPINN"     = function(l1) SPINN3_GLasso_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y, cbind(1,Xvl),val.y,
                                                   Epoch=TUNE_EPOCH, rate=LR_FIXED, lambda1=l1, lambda2=LAMBDA2_FIXED,
                                                   penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACTIVATION, verbose=FALSE),
      "D-GLasso"  = function(l1) PDSVM3_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y_h, cbind(1,Xvl),val.y_h,
                                            Epoch=TUNE_EPOCH, rate=LR_FIXED, lambda1=l1, lambda2=LAMBDA2_FIXED, h=H_FIXED,
                                            penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACTIVATION, verbose=FALSE),
      "D-AGLasso" = function(l1) PDSVM3_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y_h, cbind(1,Xvl),val.y_h,
                                            Epoch=TUNE_EPOCH, rate=LR_FIXED, lambda1=l1, lambda2=LAMBDA2_FIXED, h=H_FIXED,
                                            adaptive_weights=w_hat, penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACTIVATION, verbose=FALSE))
    
    rows <- list()
    for (m in names(grids)) for (lam in grids[[m]]) {
      fit <- fitters[[m]](lam)
      sel <- pdsvm_selected_features(fit, SEL_THRESH)$selected
      rows[[length(rows)+1]] <- data.frame(method=m, k=k, lambda=lam,
                                           val=fit$Best_ACC_val, nsel=length(sel))
    }
    do.call(rbind, rows)
  }
  
  all_rows <- do.call(rbind, lapply(1:K_CALIB, function(k) { cat(sprintf("  Calib draw %d/%d...\n", k, K_CALIB)); calib_one_rep(k, COARSE_GRID_BY_METHOD) }))
  ag <- aggregate(cbind(val, nsel) ~ method + lambda, data = all_rows, FUN = mean)
  
  best_lam <- list()
  for (m in unique(ag$method)) {
    a <- ag[ag$method == m, ]
    # Select lambda1 by validation accuracy; tie-break toward a sparser solution.
    a <- a[order(-a$val, -a$lambda), ]
    best_lam[[m]] <- a$lambda[1]
  }
  
  lambda_star <- setNames(c(best_lam[["SPINN"]], best_lam[["D-GLasso"]], best_lam[["D-AGLasso"]]), l1.names)
  saveRDS(list(lambda_star = lambda_star, activation = ACTIVATION), file.path(RESULT_DIR, sprintf("%s_lambda_calib.rds", TAG_MODEL)))
  cat("\n[OK] Calibration saved!\n"); print(lambda_star)
}

## =============================================================================
## MODE 2: SIMULATION  (production loop with per-rep lambda jitter + SVM tuning)
## =============================================================================
if (RUN_SIM) {
  cat(sprintf("\n========== STEP 1: SIMULATION (Reps: 1 to %d) ==========\n", REPS))
  
  calib_file <- file.path(RESULT_DIR, sprintf("%s_lambda_calib.rds", TAG_MODEL))
  if (!file.exists(calib_file)) stop("Calibration file not found! Set RUN_CALIB <- TRUE first.")
  lambda_star <- readRDS(calib_file)$lambda_star
  
  for (rep_id in 1:REPS) {
    cat(sprintf("\n--- Running Rep %d / %d ---\n", rep_id, REPS))
    
    set.seed(900000L + rep_id)
    fac <- setNames(runif(length(lambda_star), 1 - JITTER, 1 + JITTER), names(lambda_star))
    lambda_rep <- lambda_star * fac
    
    tr <- generate_balanced_data(n_per_class_train, 16267 + rep_id)
    vl <- generate_balanced_data(n_per_class_val,   16267*2 + rep_id)
    te <- generate_balanced_data(n_per_class_test,  16267*3 + rep_id)
    
    train.y <- tr$y; val.y <- vl$y; test.y <- te$y
    train.y_h <- ifelse(tr$y==1,1,-1); val.y_h <- ifelse(vl$y==1,1,-1); test.y_h <- ifelse(te$y==1,1,-1)
    train.df <- data.frame(tr$x, y=as.factor(train.y)); colnames(train.df)<-c(paste0("x",1:P_TOTAL),"y")
    
    if (STANDARDIZE_INPUT) {
      mu<-colMeans(tr$x); sdv<-apply(tr$x,2,sd); sdv[sdv==0]<-1
      Xtr<-scale(tr$x,mu,sdv); Xvl<-scale(vl$x,mu,sdv); Xte<-scale(te$x,mu,sdv)
    } else { Xtr<-tr$x; Xvl<-vl$x; Xte<-te$x }
    
    init <- make_init(P_TOTAL,node1,node2,node3, seed=1365*rep_id)
    acc <- setNames(rep(NA_real_,7), method.names); lam <- setNames(rep(NA_real_,3), l1.names)
    ex <- setNames(rep(NA,3), l1.names); sel <- list()
    
    # ---------------------------------------------------------
    # 1. KSVM & Oracle (tuned by e1071::tune.svm, 3-fold CV)
    # ---------------------------------------------------------
    cat("  > Tuning KSVM (p=20)...\n")
    tune_ksvm <- tune.svm(y ~ ., data=train.df,
                          gamma = 10^(-3:1), cost = 10^(-1:2),
                          tunecontrol = tune.control(cross = 3))
    ks <- tune_ksvm$best.model
    acc["KSVM"] <- sum(diag(table(predict(ks,te$x),test.y)))/length(test.y)
    
    cat("  > Tuning Oracle (p=5)...\n")
    od <- train.df[,c(paste0("x",1:P_SIGNAL),"y")]
    tune_oracle <- tune.svm(y ~ ., data=od,
                            gamma = 10^(-2:1), cost = 10^(-1:2),
                            tunecontrol = tune.control(cross = 3))
    oc <- tune_oracle$best.model
    acc["Oracle"] <- sum(diag(table(predict(oc,te$x[,1:P_SIGNAL]),test.y)))/length(test.y)
    
    # ---------------------------------------------------------
    # 2. Base deep models
    # ---------------------------------------------------------
    cat("  > Training Deep Baselines...\n")
    fit_ce <- CE3_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y, cbind(1,Xvl),val.y,
                      Epoch=FINAL_EPOCH, rate=LR_FIXED, lambda2=LAMBDA2_FIXED, activation=ACTIVATION, verbose=FALSE)
    acc["D-CE"] <- Binary3_predict(fit_ce$Weight1,fit_ce$Weight2,fit_ce$Weight3,fit_ce$Weight0, cbind(1,Xte),test.y, activation=ACTIVATION)$ACC
    
    fit_ridge <- PDSVM3_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y_h, cbind(1,Xvl),val.y_h,
                            Epoch=FINAL_EPOCH, rate=LR_FIXED, lambda1=0, lambda2=LAMBDA2_FIXED, h=H_FIXED, activation=ACTIVATION, verbose=FALSE)
    acc["D-Ridge"] <- Hinge3_predict(fit_ridge$Weight1,fit_ridge$Weight2,fit_ridge$Weight3,fit_ridge$Weight0, cbind(1,Xte),test.y_h, activation=ACTIVATION)$ACC
    w_hat <- compute_adaptive_weights(fit_ridge$Weight1[,-1,drop=FALSE], gamma=1, eps=1e-8)
    
    # ---------------------------------------------------------
    # 3. Penalized deep models (prox group lasso)
    # ---------------------------------------------------------
    cat("  > Training Penalized Models...\n")
    for (m in l1.names) {
      lam_m <- as.numeric(lambda_rep[[m]])
      if (m=="SPINN") {
        f <- SPINN3_GLasso_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y, cbind(1,Xvl),val.y,
                               Epoch=FINAL_EPOCH, rate=LR_FIXED, lambda1=lam_m, lambda2=LAMBDA2_FIXED,
                               penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACTIVATION, verbose=FALSE)
        acc[m] <- Binary3_predict(f$Weight1,f$Weight2,f$Weight3,f$Weight0, cbind(1,Xte),test.y, activation=ACTIVATION)$ACC
      } else if (m=="D-GLasso") {
        f <- PDSVM3_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y_h, cbind(1,Xvl),val.y_h,
                        Epoch=FINAL_EPOCH, rate=LR_FIXED, lambda1=lam_m, lambda2=LAMBDA2_FIXED, h=H_FIXED,
                        penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACTIVATION, verbose=FALSE)
        acc[m] <- Hinge3_predict(f$Weight1,f$Weight2,f$Weight3,f$Weight0, cbind(1,Xte),test.y_h, activation=ACTIVATION)$ACC
      } else {
        f <- PDSVM3_val(init$W1,init$W2,init$W3,init$W0, cbind(1,Xtr),train.y_h, cbind(1,Xvl),val.y_h,
                        Epoch=FINAL_EPOCH, rate=LR_FIXED, lambda1=lam_m, lambda2=LAMBDA2_FIXED, h=H_FIXED,
                        adaptive_weights=w_hat, penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACTIVATION, verbose=FALSE)
        acc[m] <- Hinge3_predict(f$Weight1,f$Weight2,f$Weight3,f$Weight0, cbind(1,Xte),test.y_h, activation=ACTIVATION)$ACC
      }
      lam[m] <- lam_m
      s <- pdsvm_selected_features(f, SEL_THRESH)$selected; sel[[m]] <- s
      ex[m] <- (length(intersect(s,S_true))==P_SIGNAL && length(setdiff(s,S_true))==0)
    }
    
    out <- list(rep_id=rep_id, acc=acc, lam=lam, exact=ex, sel=sel, activation=ACTIVATION)
    saveRDS(out, file.path(RESULT_DIR, sprintf("%s_rep%04d.rds", TAG_MODEL, rep_id)))
  }
}

## =============================================================================
## MODE 3: COMBINE
## =============================================================================
if (RUN_COMBINE) {
  cat("\n========== STEP 2: AGGREGATING RESULTS ==========\n")
  res_files <- list.files(RESULT_DIR, pattern = sprintf("^%s_rep[0-9]+\\.rds$", TAG_MODEL), full.names = TRUE)
  if (length(res_files) == 0) stop("No result files found! Please run SIM mode first.")
  
  all_acc <- list(); all_tpr <- list(); all_fpr <- list(); all_nsel <- list(); all_exact <- list()
  for (f in res_files) {
    dat <- readRDS(f); rep_id <- dat$rep_id
    df_acc <- as.data.frame(t(dat$acc)); df_acc$rep <- rep_id
    all_acc[[length(all_acc) + 1]] <- df_acc
    
    for (m in l1.names) {
      sel <- dat$sel[[m]]
      tp <- length(intersect(sel, S_true)); fp <- length(setdiff(sel, S_true))
      all_tpr[[length(all_tpr)+1]]   <- data.frame(method=m, rep=rep_id, val=tp/P_SIGNAL)
      all_fpr[[length(all_fpr)+1]]   <- data.frame(method=m, rep=rep_id, val=fp/P_NOISE)
      all_nsel[[length(all_nsel)+1]] <- data.frame(method=m, rep=rep_id, val=length(sel))
      all_exact[[length(all_exact)+1]] <- data.frame(method=m, rep=rep_id, val=as.numeric(dat$exact[[m]]))
    }
  }
  
  df_acc_final <- bind_rows(all_acc) %>% pivot_longer(cols = all_of(method.names), names_to="Method", values_to="Accuracy")
  df_tpr_final <- bind_rows(all_tpr); df_fpr_final <- bind_rows(all_fpr)
  df_nsel_final <- bind_rows(all_nsel); df_exact_final <- bind_rows(all_exact)
  
  write.csv(df_acc_final, file.path(RESULT_DIR, "Final_Accuracy_Detailed.csv"), row.names=FALSE)
  summary_acc <- df_acc_final %>% group_by(Method) %>% summarize(Mean_ACC=mean(Accuracy), SE_ACC=sd(Accuracy)/sqrt(n()))
  write.csv(summary_acc, file.path(RESULT_DIR, "Final_Summary_Accuracy.csv"), row.names=FALSE)
  
  summary_sel <- data.frame(
    Method = l1.names,
    TPR  = sapply(l1.names, function(m) mean(df_tpr_final$val[df_tpr_final$method==m])),
    FPR  = sapply(l1.names, function(m) mean(df_fpr_final$val[df_fpr_final$method==m])),
    nsel = sapply(l1.names, function(m) mean(df_nsel_final$val[df_nsel_final$method==m])),
    Exact= sapply(l1.names, function(m) mean(df_exact_final$val[df_exact_final$method==m]))
  )
  write.csv(summary_sel, file.path(RESULT_DIR, "Final_Summary_Selection.csv"), row.names=FALSE)
  cat("\n--- Selection summary (TPR / FPR / #sel / exact) ---\n"); print(summary_sel)
  
  p1 <- ggplot(df_acc_final, aes(x=reorder(Method, Accuracy, FUN=median), y=Accuracy, fill=Method)) +
    geom_boxplot(alpha=0.8) + theme_bw() + coord_flip() +
    labs(title="Test Accuracy (Model 4, easy additive)", x="", y="Accuracy") + theme(legend.position="none")
  ggsave(file.path(RESULT_DIR, "Plot_Accuracy.png"), p1, width=8, height=5, bg="white")
  
  p2 <- ggplot(df_tpr_final, aes(x=method, y=val, fill=method)) +
    geom_boxplot(alpha=0.8) + theme_bw() + labs(title="Signal Recovery (TPR)", y="True Positive Rate") + theme(legend.position="none")
  ggsave(file.path(RESULT_DIR, "Plot_TPR.png"), p2, width=6, height=4, bg="white")
  
}