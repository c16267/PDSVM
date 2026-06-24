rm(list = ls())
suppressPackageStartupMessages({ 
  library(e1071)
  library(parallel) 
})


setwd("/Users/shin.991/Library/CloudStorage/Dropbox/postdoc/research/side_project/Dr. Bang/Penalized Deep SVM/code")  
source("Deep_SVM_v6.R")

## ============================== CONTROL PANEL ===============================
PREP_RDS   <- "celeba_eyeglasses_prep_64.rds"
RESULT_DIR <- "results_celeba_eyeglasses_local"
dir.create(RESULT_DIR, showWarnings = FALSE)


NODE1 <- 32; NODE2 <- 16; NODE3 <- 8    
PENALTY_TYPE <- "prox"                   
WARMUP_FRAC  <- 0.3
SEL_THRESH   <- 1e-4                     
LAMBDA2_FIXED <- 1e-4


REPS           <- 50
NUM_CORES      <- max(1, detectCores() - 1)
SUBSAMPLE_FRAC <- 0.8        

TUNE_EPOCH     <- 5000
FINAL_EPOCH    <- 5000
ACC_TOL        <- 0.005      

H_GRID    <- c(0.01, 0.05, 0.10)
LR_GRID   <- c(0.001, 0.005, 0.01, 0.05, 0.1) 
ACT_GRID  <- c("tanh", "relu")

LAMBDA_GRID_CE    <- c(5.0, 3.0, 1.5, 0.8, 0.4, 0.2, 0.1, 0.05, 0.01)
LAMBDA_GRID_HINGE <- c(8.0, 5.0, 3.0, 1.5, 0.8, 0.4, 0.2, 0.1, 0.05, 0.01)
SEED <- 20260529
## ============================================================================

set.seed(SEED)
D <- readRDS(PREP_RDS)
S <- D$img_size; p <- S * S
eye_mask <- D$eye_mask
has_eye  <- !is.null(eye_mask)

standardize_by <- function(Xref, ...) {
  mu <- colMeans(Xref); sdv <- apply(Xref, 2, sd); sdv[sdv == 0] <- 1
  lapply(list(...), function(M) scale(M, mu, sdv))
}
to01  <- function(v) ifelse(v == 1, 1L, 0L)
to_pm <- function(v) ifelse(v == 1, 1, -1)
bal_acc <- function(pred, ytrue) {
  pr <- to01(pred); yt <- to01(ytrue)
  s1 <- mean(pr[yt == 1] == 1); s0 <- mean(pr[yt == 0] == 0)
  mean(c(s1, s0))
}

SPINN3_GLasso_val <- function(W1, W2, W3, W0, X, y, X_val, y_val, Epoch=1000, rate=0.05, lambda1=0.05, lambda2=1e-4, warmup_frac=0, penalty_type="prox", activation="tanh") {
  n <- nrow(X); eta <- rate; w_hat <- rep(1, ncol(X) - 1)
  warm_E <- max(1L, floor(warmup_frac * Epoch))
  best_acc <- -Inf; W1b<-W1; W2b<-W2; W3b<-W3; W0b<-W0
  for (t in seq_len(Epoch)) {
    fp <- forward_pass_3(W1, W2, W3, W0, X, activation)
    p0 <- Activation(as.vector(fp$z0), "sigmoid"); d0 <- matrix((p0 - y)/n, nrow = 1)
    grad_W0 <- d0 %*% t(fp$y3a); back0 <- t(W0) %*% d0
    delta3 <- back0[-1,,drop=FALSE]*Activation_prime(fp$y3,activation); grad_W3<-delta3%*%t(fp$y2a); back3<-t(W3)%*%delta3
    delta2 <- back3[-1,,drop=FALSE]*Activation_prime(fp$y2,activation); grad_W2<-delta2%*%t(fp$y1a); back2<-t(W2)%*%delta2
    delta1 <- back2[-1,,drop=FALSE]*Activation_prime(fp$y1,activation); grad_W1<-delta1%*%X
    if (lambda2>0){grad_W0<-grad_W0+2*lambda2*W0;grad_W3<-grad_W3+2*lambda2*W3;grad_W2<-grad_W2+2*lambda2*W2}
    W0<-W0-eta*grad_W0; W3<-W3-eta*grad_W3; W2<-W2-eta*grad_W2
    lam_t <- lambda1 * min(1, t/warm_E)
    if (lam_t>0 && penalty_type=="prox") {
      W1h <- W1-eta*grad_W1; W1f <- group_soft_threshold(W1h[,-1,drop=FALSE], eta*lam_t*w_hat)
      W1 <- cbind(W1h[,1,drop=FALSE], W1f)
    } else if (lam_t>0) {
      pen <- glasso_penalty_grad(W1[,-1,drop=FALSE], w_hat); g1<-grad_W1; g1[,-1]<-grad_W1[,-1]+lam_t*pen; W1<-W1-eta*g1
    } else W1 <- W1-eta*grad_W1
    pvl <- Activation(as.vector(forward_pass_3(W1,W2,W3,W0,X_val,activation)$z0), "sigmoid")
    a <- bal_acc(pvl>0.5, y_val); if (a>best_acc){best_acc<-a;W1b<-W1;W2b<-W2;W3b<-W3;W0b<-W0}
  }
  list(Weight1=W1b,Weight2=W2b,Weight3=W3b,Weight0=W0b,Best_ACC_val=best_acc)
}

make_init <- function(pp, n1, n2, n3, seed) {
  set.seed(seed); g <- function(a,b) sqrt(6/(a+b))
  c1<-g(pp+1,n1);c2<-g(n1+1,n2);c3<-g(n2+1,n3);c0<-g(n3+1,1)
  list(W1=matrix(runif(n1*(pp+1),-c1,c1),n1,pp+1),
       W2=matrix(runif(n2*(n1+1),-c2,c2),n2,n1+1),
       W3=matrix(runif(n3*(n2+1),-c3,c3),n3,n2+1),
       W0=matrix(runif(1*(n3+1),-c0,c0),1,n3+1))
}

l1.names <- c("SPINN", "PDSVM-GL", "PDSVM-AGL")
methods <- c("KSVM", "D-CE", "D-Ridge", l1.names)

cat("\n========== STAGE A & B: Hyperparameter & Lambda Tuning ==========\n")
flush.console()
std <- standardize_by(D$Xtr, D$Xtr, D$Xvl)
Xtr_s <- std[[1]]; Xvl_s <- std[[2]]
ytr_pm <- to_pm(D$ytr); yvl_pm <- to_pm(D$yvl)
ytr_01 <- to01(D$ytr); yvl_01 <- to01(D$yvl)
init0 <- make_init(p, NODE1, NODE2, NODE3, seed = 1001)


grid_hinge <- expand.grid(h = H_GRID, lr = LR_GRID, act = ACT_GRID, stringsAsFactors = FALSE)
grid_hinge$val <- NA_real_
for (i in seq_len(nrow(grid_hinge))) {
  f <- PDSVM3_val(init0$W1, init0$W2, init0$W3, init0$W0, cbind(1, Xtr_s), ytr_pm, cbind(1, Xvl_s), yvl_pm,
                  Epoch = TUNE_EPOCH, rate = grid_hinge$lr[i], lambda1 = 0, lambda2 = LAMBDA2_FIXED, 
                  h = grid_hinge$h[i], activation = grid_hinge$act[i], verbose = FALSE)
  pr <- Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1,Xvl_s), yvl_pm, activation=grid_hinge$act[i])$Pred
  grid_hinge$val[i] <- bal_acc(pr, D$yvl)
}
bestH <- grid_hinge[which.max(grid_hinge$val), ]
H_STAR_HINGE <- bestH$h; LR_STAR_HINGE <- bestH$lr; ACT_STAR_HINGE <- bestH$act
cat(sprintf("[Hinge*] h=%.2f lr=%.3f act=%s (val.bal=%.4f)\n", H_STAR_HINGE, LR_STAR_HINGE, ACT_STAR_HINGE, bestH$val))
flush.console()


grid_ce <- expand.grid(lr = LR_GRID, act = ACT_GRID, stringsAsFactors = FALSE)
grid_ce$val <- NA_real_
for (i in seq_len(nrow(grid_ce))) {
  f <- CE3_val(init0$W1, init0$W2, init0$W3, init0$W0, cbind(1, Xtr_s), ytr_01, cbind(1, Xvl_s), yvl_01,
               Epoch = TUNE_EPOCH, rate = grid_ce$lr[i], lambda2 = LAMBDA2_FIXED, 
               activation = grid_ce$act[i], verbose = FALSE)
  pr <- Binary3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1,Xvl_s), D$yvl, activation=grid_ce$act[i])$Pred
  grid_ce$val[i] <- bal_acc(pr, D$yvl)
}
bestCE <- grid_ce[which.max(grid_ce$val), ]
LR_STAR_CE <- bestCE$lr; ACT_STAR_CE <- bestCE$act
cat(sprintf("[CE*]    lr=%.3f act=%s (val.bal=%.4f)\n", LR_STAR_CE, ACT_STAR_CE, bestCE$val))
flush.console()


pilot <- PDSVM3_val(init0$W1, init0$W2, init0$W3, init0$W0, cbind(1,Xtr_s), ytr_pm, cbind(1,Xvl_s), yvl_pm,
                    Epoch=TUNE_EPOCH, rate=LR_STAR_HINGE, lambda1=0, lambda2=LAMBDA2_FIXED, h=H_STAR_HINGE, activation=ACT_STAR_HINGE, verbose=FALSE)
w_hat0 <- compute_adaptive_weights(pilot$Weight1[,-1,drop=FALSE], gamma=1, eps=1e-8)

fit_method <- function(method, lam, ep, Wi = init0, w_hat = w_hat0) {
  if (method == "SPINN") {
    SPINN3_GLasso_val(Wi$W1, Wi$W2, Wi$W3, Wi$W0, cbind(1,Xtr_s), ytr_01, cbind(1,Xvl_s), yvl_01,
                      Epoch=ep, rate=LR_STAR_CE, lambda1=lam, lambda2=LAMBDA2_FIXED, penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACT_STAR_CE)
  } else if (method == "PDSVM-GL") {
    PDSVM3_val(Wi$W1, Wi$W2, Wi$W3, Wi$W0, cbind(1,Xtr_s), ytr_pm, cbind(1,Xvl_s), yvl_pm,
               Epoch=ep, rate=LR_STAR_HINGE, lambda1=lam, lambda2=LAMBDA2_FIXED, h=H_STAR_HINGE, penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACT_STAR_HINGE, verbose=FALSE)
  } else {
    PDSVM3_val(Wi$W1, Wi$W2, Wi$W3, Wi$W0, cbind(1,Xtr_s), ytr_pm, cbind(1,Xvl_s), yvl_pm,
               Epoch=ep, rate=LR_STAR_HINGE, lambda1=lam, lambda2=LAMBDA2_FIXED, h=H_STAR_HINGE, adaptive_weights=w_hat, penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACT_STAR_HINGE, verbose=FALSE)
  }
}

lambda_star <- setNames(numeric(3), l1.names)
for (m in l1.names) {
  is_ce <- (m == "SPINN")
  lam_grid <- if (is_ce) LAMBDA_GRID_CE else LAMBDA_GRID_HINGE
  act_used <- if (is_ce) ACT_STAR_CE else ACT_STAR_HINGE
  tab <- data.frame(lambda = lam_grid, val = NA_real_, nsel = NA_real_)
  
  for (i in seq_along(lam_grid)) {
    f <- fit_method(m, lam_grid[i], TUNE_EPOCH)
    if (is_ce) { pr <- Binary3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1,Xvl_s), yvl_01, activation=act_used)$Pred
    } else {     pr <- Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1,Xvl_s), yvl_pm, activation=act_used)$Pred }
    tab$val[i]  <- bal_acc(pr, D$yvl)
    tab$nsel[i] <- length(pdsvm_selected_features(f, SEL_THRESH)$selected)
  }
  vmax <- max(tab$val)
  cand <- tab[tab$val >= vmax - ACC_TOL, ]
  lambda_star[m] <- cand$lambda[which.max(cand$lambda)]
  cat(sprintf("  [%s] lambda*=%.4f (val.bal=%.4f, nsel=%.0f)\n", m, lambda_star[m], tab$val[which(tab$lambda==lambda_star[m])], tab$nsel[which(tab$lambda==lambda_star[m])]))
  flush.console()
}

cat(sprintf("\n========== STAGE C: Parallel Training (%d Cores, %d Reps) ==========\n", NUM_CORES, REPS))
flush.console()

yte_pm <- to_pm(D$yte); yte_01 <- to01(D$yte)
pos_idx_tr <- which(D$ytr == 1); neg_idx_tr <- which(D$ytr == 0)
n_sub_pos <- floor(SUBSAMPLE_FRAC * length(pos_idx_tr))
n_sub_neg <- floor(SUBSAMPLE_FRAC * length(neg_idx_tr))


results_list <- mclapply(1:REPS, function(r) {
  
  set.seed(7000 + r)
  
  acc_row <- setNames(numeric(length(methods)), methods)
  nsel_row <- setNames(numeric(length(l1.names)), l1.names)
  selfreq_list <- lapply(l1.names, function(.) numeric(p)); names(selfreq_list) <- l1.names

  sub <- sample(c(sample(pos_idx_tr, n_sub_pos), sample(neg_idx_tr, n_sub_neg)))
  Xtr_r <- D$Xtr[sub, , drop = FALSE]; ytr_r <- D$ytr[sub]
  
  st <- standardize_by(Xtr_r, Xtr_r, D$Xvl, D$Xte)
  Xtr_s <- st[[1]]; Xvl_s <- st[[2]]; Xte_s <- st[[3]]
  
  ytr_pm <- to_pm(ytr_r); yvl_pm <- to_pm(D$yvl)
  ytr_01 <- to01(ytr_r);  yvl_01 <- to01(D$yvl)
  
  init <- make_init(p, NODE1, NODE2, NODE3, seed = 1365 * r)
  
  ## 1. KSVM
  df_tr <- data.frame(Xtr_s, y = as.factor(ytr_r)); colnames(df_tr) <- c(paste0("v", 1:p), "y")
  tk <- tune.svm(y ~ ., data = df_tr, kernel = "radial", gamma = 10^(-4:-1), cost = 10^(-1:2), tunecontrol = tune.control(cross = 3))
  prk <- predict(tk$best.model, Xte_s)
  acc_row["KSVM"] <- bal_acc(as.numeric(as.character(prk)), D$yte)
  
  ## 2. D-CE
  fce <- CE3_val(init$W1, init$W2, init$W3, init$W0, cbind(1,Xtr_s), ytr_01, cbind(1,Xvl_s), yvl_01, Epoch=FINAL_EPOCH, rate=LR_STAR_CE, lambda2=LAMBDA2_FIXED, activation=ACT_STAR_CE, verbose=FALSE)
  acc_row["D-CE"] <- bal_acc(Binary3_predict(fce$Weight1, fce$Weight2, fce$Weight3, fce$Weight0, cbind(1,Xte_s), yte_01, activation=ACT_STAR_CE)$Pred, D$yte)
  
  frd <- PDSVM3_val(init$W1, init$W2, init$W3, init$W0, cbind(1,Xtr_s), ytr_pm, cbind(1,Xvl_s), yvl_pm, Epoch=FINAL_EPOCH, rate=LR_STAR_HINGE, lambda1=0, lambda2=LAMBDA2_FIXED, h=H_STAR_HINGE, activation=ACT_STAR_HINGE, verbose=FALSE)
  acc_row["D-Ridge"] <- bal_acc(Hinge3_predict(frd$Weight1, frd$Weight2, frd$Weight3, frd$Weight0, cbind(1,Xte_s), yte_pm, activation=ACT_STAR_HINGE)$Pred, D$yte)
  w_hat <- compute_adaptive_weights(frd$Weight1[,-1,drop=FALSE], gamma=1, eps=1e-8)
  
  ## 4. Penalty Methods
  for (m in l1.names) {
    if (m == "SPINN") {
      f <- SPINN3_GLasso_val(init$W1, init$W2, init$W3, init$W0, cbind(1,Xtr_s), ytr_01, cbind(1,Xvl_s), yvl_01, Epoch=FINAL_EPOCH, rate=LR_STAR_CE, lambda1=lambda_star[m], lambda2=LAMBDA2_FIXED, penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACT_STAR_CE)
      acc_row[m] <- bal_acc(Binary3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1,Xte_s), yte_01, activation=ACT_STAR_CE)$Pred, D$yte)
    } else if (m == "PDSVM-GL") {
      f <- PDSVM3_val(init$W1, init$W2, init$W3, init$W0, cbind(1,Xtr_s), ytr_pm, cbind(1,Xvl_s), yvl_pm, Epoch=FINAL_EPOCH, rate=LR_STAR_HINGE, lambda1=lambda_star[m], lambda2=LAMBDA2_FIXED, h=H_STAR_HINGE, penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACT_STAR_HINGE, verbose=FALSE)
      acc_row[m] <- bal_acc(Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1,Xte_s), yte_pm, activation=ACT_STAR_HINGE)$Pred, D$yte)
    } else {
      f <- PDSVM3_val(init$W1, init$W2, init$W3, init$W0, cbind(1,Xtr_s), ytr_pm, cbind(1,Xvl_s), yvl_pm, Epoch=FINAL_EPOCH, rate=LR_STAR_HINGE, lambda1=lambda_star[m], lambda2=LAMBDA2_FIXED, h=H_STAR_HINGE, adaptive_weights=w_hat, penalty_type=PENALTY_TYPE, warmup_frac=WARMUP_FRAC, activation=ACT_STAR_HINGE, verbose=FALSE)
      acc_row[m] <- bal_acc(Hinge3_predict(f$Weight1, f$Weight2, f$Weight3, f$Weight0, cbind(1,Xte_s), yte_pm, activation=ACT_STAR_HINGE)$Pred, D$yte)
    }
    sel <- pdsvm_selected_features(f, SEL_THRESH)$selected
    selfreq_list[[m]][sel] <- 1  
    nsel_row[m] <- length(sel)
  }
  
  list(acc = acc_row, nsel = nsel_row, selfreq = selfreq_list)
  
}, mc.cores = NUM_CORES)

flush.console()

## ============================================================================
## ============================================================================
cat("\n========== STAGE D: Metrics & Figures ==========\n")

ACC <- matrix(NA_real_, REPS, length(methods), dimnames = list(NULL, methods))
NSEL <- matrix(NA_real_, REPS, length(l1.names), dimnames = list(NULL, l1.names))
SELfreq <- lapply(l1.names, function(.) numeric(p)); names(SELfreq) <- l1.names

for (r in seq_len(REPS)) {
  ACC[r, ] <- results_list[[r]]$acc[methods]
  NSEL[r, ] <- results_list[[r]]$nsel[l1.names]
  for (m in l1.names) SELfreq[[m]] <- SELfreq[[m]] + results_list[[r]]$selfreq[[m]]
}
for (m in l1.names) SELfreq[[m]] <- SELfreq[[m]] / REPS

se <- function(x) sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))

calc_nogueira <- function(p_j_vec, d_avg, p_total, M) {
  if (d_avg == 0 || d_avg == p_total) return(NA_real_)
  num <- mean(p_j_vec * (1 - p_j_vec))
  denom <- (d_avg / p_total) * (1 - d_avg / p_total)
  stab <- 1 - (M / (M - 1)) * (num / denom)
  return(stab)
}

summ <- data.frame(
  method  = methods,
  bal_acc = round(colMeans(ACC, na.rm = TRUE), 4),
  bal_se  = round(apply(ACC, 2, se), 4),
  nsel    = NA_real_, eye_hit_ratio = NA_real_, eye_prec_ratio = NA_real_, stability = NA_real_
)
rownames(summ) <- methods
base_rate <- mean(eye_mask) 

for (m in l1.names) {
  avg_d <- mean(NSEL[, m])
  summ[m, "nsel"] <- round(avg_d, 2)
  
  if (REPS > 1) {
    summ[m, "stability"] <- round(calc_nogueira(SELfreq[[m]], avg_d, p, REPS), 4)
  }
  
  if (has_eye) {
    pi_m <- SELfreq[[m]]
    summ[m, "eye_hit_ratio"] <- ifelse(sum(pi_m) > 0, round((sum(pi_m[eye_mask]) / sum(pi_m)) / base_rate, 4), NA)
    sel_set <- which(pi_m > 0.5) 
    summ[m, "eye_prec_ratio"] <- ifelse(length(sel_set) > 0, round(mean(eye_mask[sel_set]) / base_rate, 4), NA)
  }
}

write.csv(summ, file.path(RESULT_DIR, "metrics.csv"), row.names = FALSE)
cat("\n--- Final Metrics ---\n"); print(summ)

flip <- function(v) { m <- matrix(v, S, S); t(m)[, S:1] }

png(file.path(RESULT_DIR, "selection_frequency.png"), width = 1200, height = 340, res = 130)
par(mfrow = c(1, 4), mar = c(1, 1, 2.5, 1))
image(flip(D$mean_face), col = gray.colors(64), axes = FALSE, main = "Mean face")
for (m in l1.names) {
  image(flip(SELfreq[[m]]), col = hcl.colors(64, "YlOrRd", rev = TRUE), zlim = c(0, 1), axes = FALSE, main = m)
}
dev.off()

masked <- function(v_freq, thr = 0.5) D$mean_face * as.numeric(v_freq > thr)
png(file.path(RESULT_DIR, "masked_faces.png"), width = 700, height = 360, res = 130)
par(mfrow = c(1, 2), mar = c(1, 1, 2.5, 1))
image(flip(masked(SELfreq[["SPINN"]])),     col = gray.colors(64), axes = FALSE, main = "SPINN")
image(flip(masked(SELfreq[["PDSVM-AGL"]])), col = gray.colors(64), axes = FALSE, main = "PDSVM-AGL")
dev.off()

save(ACC, SELfreq, NSEL, summ, file = file.path(RESULT_DIR, "celeba_results_local_final.RData"))


# ################################################################################
################################################################################
################################################################################
rm(list = ls())
library(ggplot2)
library(dplyr)

setwd("/Users/shin.991/Library/CloudStorage/Dropbox/postdoc/research/side_project/Dr. Bang/Penalized Deep SVM/code")  

PREP_RDS <- '/Users/shin.991/Library/CloudStorage/Dropbox/postdoc/research/side_project/Dr. Bang/Penalized Deep SVM/code/results_celeba_eyeglasses/64x64/celeba_eyeglasses_prep_64.rds'
RESULT_RDATA <- '/Users/shin.991/Library/CloudStorage/Dropbox/postdoc/research/side_project/Dr. Bang/Penalized Deep SVM/code/results_celeba_eyeglasses/64x64/celeba_results_final.RData'

D <- readRDS(PREP_RDS)
load(RESULT_RDATA)

S <- D$img_size
p <- S * S

flip <- function(v) { m <- matrix(v, S, S); t(m)[, S:1] }

masked <- function(img_vec, freq_vec, thr = 0.5) {
  img_vec * as.numeric(freq_vec > thr)
}

vec_to_df <- function(vec, S) {
  m <- matrix(vec, nrow = S, ncol = S)
  m_flip <- t(m)[, S:1] 
  
  df <- expand.grid(x = 1:S, y = 1:S)
  df$value <- as.vector(m_flip)
  return(df)
}

set.seed(4733)
#test_pos_idx <- which(D$yte == 1)
sample_idx <- sample(1:nrow(D$Xte), 3)
X_sample <- D$Xte[sample_idx, ]
plot_data <- data.frame()

for (i in 1:length(sample_idx)) {
  img_orig <- X_sample[i, ]
  
  img_spinn <- masked(img_orig, SELfreq[["SPINN"]])
  img_gl    <- masked(img_orig, SELfreq[["PDSVM-GL"]])   
  img_agl   <- masked(img_orig, SELfreq[["PDSVM-AGL"]])
  
  df_orig  <- vec_to_df(img_orig, S)  %>% mutate(Person = paste("Person", i), Method = "1.Original")
  df_dsvm  <- vec_to_df(img_orig, S)  %>% mutate(Person = paste("Person", i), Method = "2.DSVM")
  df_spinn <- vec_to_df(img_spinn, S) %>% mutate(Person = paste("Person", i), Method = "3.SPINN")
  df_gl    <- vec_to_df(img_gl, S)    %>% mutate(Person = paste("Person", i), Method = "4.PDSVM-GL")
  df_agl   <- vec_to_df(img_agl, S)   %>% mutate(Person = paste("Person", i), Method = "5.PDSVM-AGL")
  
  plot_data <- bind_rows(plot_data, df_orig, df_dsvm, df_spinn, df_gl, df_agl)
}

g <- ggplot(plot_data, aes(x = x, y = y, fill = value)) +
  geom_raster() + 
  facet_grid(Person ~ Method) + 
  scale_fill_gradient(low = "black", high = "white", guide = "none") + 
  coord_fixed() + 
  theme_void() +  
  theme(
    strip.text.x = element_text(size = 10, face = "bold", margin = margin(b = 10, t = 10)),
    strip.text.y = element_text(size = 10, face = "bold", margin = margin(l = 10, r = 10), angle = -90),
    panel.spacing = unit(1.0, "lines")
  )
g

output_pdf <- "results_celeba_eyeglasses/individual_masked_faces_ggplot_5cols.pdf"
ggsave(output_pdf, plot = g, width = 10, height = 6)

################################################################################
################################################################################
rm(list = ls())
suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
})

setwd("/Users/shin.991/Library/CloudStorage/Dropbox/postdoc/research/side_project/Dr. Bang/Penalized Deep SVM/code")  


cat("Loading data and results...\n")
PREP_RDS <- '/Users/shin.991/Library/CloudStorage/Dropbox/postdoc/research/side_project/Dr. Bang/Penalized Deep SVM/code/results_celeba_eyeglasses/64x64/celeba_eyeglasses_prep_64.rds'
RESULT_RDATA <- '/Users/shin.991/Library/CloudStorage/Dropbox/postdoc/research/side_project/Dr. Bang/Penalized Deep SVM/code/results_celeba_eyeglasses/64x64/celeba_results_final.RData'

D <- readRDS(PREP_RDS)
load(RESULT_RDATA) 

S <- D$img_size
p <- S * S

vec_to_df <- function(vec, S) {
  m <- matrix(vec, nrow = S, ncol = S)
  m_flip <- t(m)[, S:1] 
  
  df <- expand.grid(x = 1:S, y = 1:S)
  df$value <- as.vector(m_flip)
  return(df)
}

output_pdf <- "results_celeba_eyeglasses_64/selection_frequency_with_dsvm_new_scale.pdf"


df_mean <- vec_to_df(D$mean_face, S) %>% mutate(Method = "1. Mean face")

p_mean <- ggplot(df_mean, aes(x = x, y = y, fill = value)) +
  geom_raster() +
  facet_wrap(~ Method) +
  scale_fill_gradient(low = "black", high = "white", guide = "none") +
  coord_fixed() + 
  theme_void() +
  theme(
    strip.text = element_text(size = 11, face = "bold", margin = margin(b = 10, t = 10)),
    # Top=5, Right=0 (여백 제거), Bottom=5, Left=5
    plot.margin = margin(5, 0, 5, 5) 
  )

df_dsvm  <- vec_to_df(rep(1.0, p), S) %>% mutate(Method = "2. DSVM")
df_spinn <- vec_to_df(SELfreq[["SPINN"]], S) %>% mutate(Method = "3. SPINN")
df_gl    <- vec_to_df(SELfreq[["PDSVM-GL"]], S) %>% mutate(Method = "4. PDSVM-GL")
df_agl   <- vec_to_df(SELfreq[["PDSVM-AGL"]], S) %>% mutate(Method = "5. PDSVM-AGL")

df_freq <- bind_rows(df_dsvm, df_spinn, df_gl, df_agl)

p_freq <- ggplot(df_freq, aes(x = x, y = y, fill = value)) +
  geom_raster() +
  facet_wrap(~ Method, nrow = 1) +
  scale_fill_gradientn(
    colors = c("blue", "white", "red"),
    limits = c(0, 1),
    name = "Selection\nFrequency"
  ) +
  coord_fixed() + 
  theme_void() +
  theme(
    strip.text = element_text(size = 11, face = "bold", margin = margin(b = 10, t = 10)),
    plot.margin = margin(5, 5, 5, 0), 
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold", margin = margin(b = 10)),
    legend.text = element_text(size = 10),
    legend.key.height = unit(0.8, "cm"),
    legend.key.width = unit(0.6, "cm")
  )

final_plot <- p_mean + p_freq + 
  plot_layout(widths = c(1, 4)) & 
  theme(panel.spacing = unit(0.1, "lines")) 
output_pdf <- "results_celeba_eyeglasses_64/selection_frequency_with_dsvm_new_scale.pdf"

ggsave(output_pdf, plot = final_plot, width = 15, height = 3.0)
