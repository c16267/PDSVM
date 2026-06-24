
rm(list = ls())

OUT_DIR <- file.path(getwd(), "theorems")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

suppressPackageStartupMessages({
  library(mvtnorm)
  library(ggplot2)
  library(patchwork)
})

## ============================================================================
## 0. Inline model functions (tanh activation, 3 hidden layers)
## ============================================================================

## --- activation and its derivative (in terms of the post-activation value) ---
Activation       <- function(x) tanh(x)
Activation_prime <- function(y) 1 - y^2          # d/dz tanh(z) = 1 - tanh(z)^2

## --- convolution-smoothed hinge loss and its derivative (closed form) -------
phi_Hh <- function(u, h) {
  stopifnot(h > 0)
  out  <- numeric(length(u))
  left <- u <= (1 - h)
  mid  <- (u > (1 - h)) & (u < (1 + h))
  out[left] <- 1 - u[left]
  out[mid]  <- (1 - u[mid] + h)^2 / (4 * h)
  out
}
phi_Hh_prime <- function(u, h) {
  stopifnot(h > 0)
  out  <- numeric(length(u))
  left <- u <= (1 - h)
  mid  <- (u > (1 - h)) & (u < (1 + h))
  out[left] <- -1
  out[mid]  <- -(1 - u[mid] + h) / (2 * h)
  out
}

## --- group-soft-thresholding (proximal operator of the group L2 norm) --------
group_soft_threshold <- function(W1_cols, tau_vec) {
  col_norms <- sqrt(colSums(W1_cols^2))
  shrink    <- pmax(0, 1 - tau_vec / pmax(col_norms, .Machine$double.eps))
  sweep(W1_cols, 2, shrink, `*`)
}

## --- forward pass (X is n x (p+1) with a leading 1-column for the bias) ------
forward_pass_3 <- function(W1, W2, W3, W0, X) {
  z1 <- W1 %*% t(X); y1 <- Activation(z1); y1a <- rbind(1, y1)
  z2 <- W2 %*% y1a;  y2 <- Activation(z2); y2a <- rbind(1, y2)
  z3 <- W3 %*% y2a;  y3 <- Activation(z3); y3a <- rbind(1, y3)
  z0 <- W0 %*% y3a
  list(z1=z1, y1=y1, y1a=y1a, z2=z2, y2=y2, y2a=y2a, z3=z3, y3=y3, y3a=y3a, z0=z0)
}

## --- full-batch proximal gradient training (returns best-val weights) -------
## lambda1 = 0 here recovers smoothed-hinge + ridge (the DSVM baseline).
PDSVM3_val <- function(W1, W2, W3, W0, X, y, X_val, y_val,
                       Epoch = 1000, rate = 0.05,
                       lambda1 = 0, lambda2 = 0, h = 0.1,
                       adaptive_weights = NULL, verbose = FALSE) {
  W1 <- as.matrix(W1); W2 <- as.matrix(W2); W3 <- as.matrix(W3); W0 <- as.matrix(W0)
  X  <- as.matrix(X);  y  <- ifelse(as.numeric(y) == 1, 1, -1)
  X_val <- as.matrix(X_val); y_val <- ifelse(as.numeric(y_val) == 1, 1, -1)
  n <- nrow(X); p <- ncol(X) - 1; eta <- rate
  w_hat <- if (is.null(adaptive_weights)) rep(1, p) else as.numeric(adaptive_weights)
  
  best_acc <- -Inf; best_epoch <- 1L
  W1_best <- W1; W2_best <- W2; W3_best <- W3; W0_best <- W0
  
  for (t in seq_len(Epoch)) {
    fp <- forward_pass_3(W1, W2, W3, W0, X); y0 <- as.vector(fp$z0)
    d0 <- matrix(y * phi_Hh_prime(y * y0, h) / n, nrow = 1)
    grad_W0 <- d0 %*% t(fp$y3a); back0 <- t(W0) %*% d0
    delta3  <- back0[-1, , drop = FALSE] * Activation_prime(fp$y3); grad_W3 <- delta3 %*% t(fp$y2a)
    back3   <- t(W3) %*% delta3
    delta2  <- back3[-1, , drop = FALSE] * Activation_prime(fp$y2); grad_W2 <- delta2 %*% t(fp$y1a)
    back2   <- t(W2) %*% delta2
    delta1  <- back2[-1, , drop = FALSE] * Activation_prime(fp$y1); grad_W1 <- delta1 %*% X
    
    if (lambda2 > 0) { grad_W0 <- grad_W0+2*lambda2*W0; grad_W3 <- grad_W3+2*lambda2*W3; grad_W2 <- grad_W2+2*lambda2*W2 }
    
    W0 <- W0 - eta*grad_W0; W3 <- W3 - eta*grad_W3; W2 <- W2 - eta*grad_W2
    W1_half <- W1 - eta*grad_W1
    if (lambda1 > 0) {
      W1f <- group_soft_threshold(W1_half[, -1, drop = FALSE], eta*lambda1*w_hat)
      W1  <- cbind(W1_half[, 1, drop = FALSE], W1f)
    } else {
      W1 <- W1_half
    }
    
    z0_vl <- as.vector(forward_pass_3(W1, W2, W3, W0, X_val)$z0)
    acc_val <- mean(ifelse(z0_vl >= 0, 1, -1) == y_val)
    if (acc_val > best_acc) { best_acc <- acc_val; best_epoch <- t; W1_best<-W1; W2_best<-W2; W3_best<-W3; W0_best<-W0 }
  }
  list(Weight1=W1_best, Weight2=W2_best, Weight3=W3_best, Weight0=W0_best,
       Best_epoch=best_epoch, Best_ACC_val=best_acc, h=h, lambda1=lambda1, lambda2=lambda2)
}

## convenience wrapper: smoothed-hinge + ridge (DSVM), lambda1 = 0
SmoothHinge3_Ridge_val <- function(W1, W2, W3, W0, X, y, X_val, y_val,
                                   Epoch = 1000, rate = 0.05, lambda2 = 1e-4, h = 0.1,
                                   verbose = FALSE) {
  PDSVM3_val(W1, W2, W3, W0, X, y, X_val, y_val,
             Epoch = Epoch, rate = rate, lambda1 = 0, lambda2 = lambda2, h = h, verbose = verbose)
}

## --- full-batch PGD trajectory (records L_n and the prox-gradient mapping) ---
## Returns L_n_traj and G_eta_norm_sq = ||W^t - W^{t+1}||^2 / eta^2.
PDSVM3_fullbatch <- function(W1, W2, W3, W0, X, y,
                             T_iter = 500, rate = 0.05,
                             lambda1 = 0, lambda2 = 0, h = 0.1,
                             adaptive_weights = NULL, verbose = FALSE, tag = "PDSVM-FB") {
  W1 <- as.matrix(W1); W2 <- as.matrix(W2); W3 <- as.matrix(W3); W0 <- as.matrix(W0)
  X  <- as.matrix(X);  y  <- ifelse(as.numeric(y) == 1, 1, -1)
  n <- nrow(X); p <- ncol(X) - 1; eta <- rate
  w_hat <- if (is.null(adaptive_weights)) rep(1, p) else as.numeric(adaptive_weights)
  
  L_n_traj <- numeric(T_iter + 1); step_sq <- numeric(T_iter)
  
  compute_Ln <- function(W1, W2, W3, W0) {
    fp <- forward_pass_3(W1, W2, W3, W0, X)
    Rh <- mean(phi_Hh(y * as.vector(fp$z0), h))
    P1 <- if (lambda1 > 0) sum(w_hat * sqrt(colSums(W1[, -1, drop = FALSE]^2))) else 0
    P2 <- if (lambda2 > 0) sum(W2^2) + sum(W3^2) + sum(W0^2) else 0
    Rh + lambda1 * P1 + lambda2 * P2
  }
  
  L_n_traj[1] <- compute_Ln(W1, W2, W3, W0)
  
  for (t in seq_len(T_iter)) {
    fp <- forward_pass_3(W1, W2, W3, W0, X); y0 <- as.vector(fp$z0)
    d0 <- matrix(y * phi_Hh_prime(y * y0, h) / n, nrow = 1)
    grad_W0 <- d0 %*% t(fp$y3a); back0 <- t(W0) %*% d0
    delta3  <- back0[-1, , drop = FALSE] * Activation_prime(fp$y3); grad_W3 <- delta3 %*% t(fp$y2a)
    back3   <- t(W3) %*% delta3
    delta2  <- back3[-1, , drop = FALSE] * Activation_prime(fp$y2); grad_W2 <- delta2 %*% t(fp$y1a)
    back2   <- t(W2) %*% delta2
    delta1  <- back2[-1, , drop = FALSE] * Activation_prime(fp$y1); grad_W1 <- delta1 %*% X
    
    if (lambda2 > 0) { grad_W0 <- grad_W0+2*lambda2*W0; grad_W3 <- grad_W3+2*lambda2*W3; grad_W2 <- grad_W2+2*lambda2*W2 }
    
    W0_new <- W0 - eta*grad_W0; W3_new <- W3 - eta*grad_W3; W2_new <- W2 - eta*grad_W2
    W1_half <- W1 - eta*grad_W1
    if (lambda1 > 0) {
      W1f <- group_soft_threshold(W1_half[, -1, drop = FALSE], eta*lambda1*w_hat)
      W1_new <- cbind(W1_half[, 1, drop = FALSE], W1f)
    } else {
      W1_new <- W1_half
    }
    
    step_sq[t] <- sum((c(W1_new, W2_new, W3_new, W0_new) - c(W1, W2, W3, W0))^2)
    W1 <- W1_new; W2 <- W2_new; W3 <- W3_new; W0 <- W0_new
    L_n_traj[t + 1] <- compute_Ln(W1, W2, W3, W0)
    
    if (verbose && (t %% 100 == 0))
      cat(sprintf("[%s] iter %d | L_n=%.6f | ||dW||^2=%.2e\n", tag, t, L_n_traj[t+1], step_sq[t]))
  }
  
  list(L_n_traj = L_n_traj, step_sq = step_sq,
       G_eta_norm_sq = step_sq / (eta^2),
       rate = eta, h = h, lambda1 = lambda1, lambda2 = lambda2)
}

## ============================================================================
## 1. Settings
## ============================================================================

n_per_class_grid <- round(10^seq(log10(80), log10(720), length.out = 10))
reps_for <- function(npc) if (npc <= 150) 50 else 100

n_floor    <- 5000
floor_reps <- 50

n_per_class_traj <- 500
T_iter_traj      <- 1000     # full-batch iterations for the trajectory panel
rate_fb          <- 0.05

P_SIGNAL <- 2
P_NOISE  <- 0
P_TOTAL  <- P_SIGNAL + P_NOISE

n_eval_per_class <- 5000

node1 <- 32; node2 <- 32; node3 <- 8
nepoch        <- 3000
lr            <- 0.08
LAMBDA2_FIXED <- 1e-5
## Panel (b)/(c) exercise the proximal step, so the trajectory uses lambda1 > 0.
LAMBDA1_TRAJ  <- 0.02
STANDARDIZE_INPUT <- TRUE

h_anchor   <- 0.10
n_anchor_h <- 600
h_min      <- 0.03
h_max      <- 0.25
h_schedule <- function(n_total) pmax(h_min, pmin(h_max, h_anchor * sqrt(n_anchor_h / n_total)))
h_fixed    <- h_anchor

## ============================================================================
## 2. DGP (chessboard, signal-only)
## ============================================================================

p.m <- rbind(c(5,5), c(15,5), c(10,10), c(5,15), c(15,15))
n.m <- rbind(c(10,5), c(5,10), c(15,10), c(10,15))
p.sigma <- matrix(c(2,   0, 0, 2  ), 2, 2)
n.sigma <- matrix(c(2.5, 0, 0, 2.5), 2, 2)

sample_data <- function(n_pos, n_neg, seed) {
  set.seed(seed)
  p.idx <- sample.int(nrow(p.m), n_pos, replace = TRUE)
  n.idx <- sample.int(nrow(n.m), n_neg, replace = TRUE)
  p.sig <- t(vapply(p.idx, function(i) mvtnorm::rmvnorm(1, mean = p.m[i, ], sigma = p.sigma), numeric(2)))
  n.sig <- t(vapply(n.idx, function(i) mvtnorm::rmvnorm(1, mean = n.m[i, ], sigma = n.sigma), numeric(2)))
  list(x = rbind(p.sig, n.sig), y = c(rep(1, n_pos), rep(0, n_neg)))
}

eval_data    <- sample_data(n_eval_per_class, n_eval_per_class, seed = 99999)
eval_y_hinge <- ifelse(eval_data$y == 1, 1, -1)
val_fixed    <- sample_data(2500, 2500, seed = 99998)

bayes_eta <- function(x2d, prior_pos = 0.5) {
  d_pos <- numeric(nrow(x2d)); d_neg <- numeric(nrow(x2d))
  for (k in 1:nrow(p.m)) d_pos <- d_pos + mvtnorm::dmvnorm(x2d, mean = p.m[k, ], sigma = p.sigma)
  for (k in 1:nrow(n.m)) d_neg <- d_neg + mvtnorm::dmvnorm(x2d, mean = n.m[k, ], sigma = n.sigma)
  d_pos <- d_pos / nrow(p.m); d_neg <- d_neg / nrow(n.m)
  num <- prior_pos * d_pos
  num / (num + (1 - prior_pos) * d_neg)
}

eta_eval   <- bayes_eta(eval_data$x[, 1:P_SIGNAL, drop = FALSE], prior_pos = 0.5)
bayes_risk <- mean(pmin(eta_eval, 1 - eta_eval))


## ============================================================================
## 3. Helpers
## ============================================================================

make_init <- function(pp, node1, node2, node3, seed) {
  set.seed(seed)
  list(
    W1 = matrix(runif(node1 * (pp    + 1), -1, 1), node1, pp    + 1),
    W2 = matrix(runif(node2 * (node1 + 1), -1, 1), node2, node1 + 1),
    W3 = matrix(runif(node3 * (node2 + 1), -1, 1), node3, node2 + 1),
    W0 = matrix(runif(1     * (node3 + 1), -1, 1),     1, node3 + 1)
  )
}

apply_standardization <- function(train_x, other_x_list) {
  mu  <- colMeans(train_x); sdx <- apply(train_x, 2, sd); sdx[sdx == 0] <- 1
  out <- list(train = scale(train_x, center = mu, scale = sdx))
  for (nm in names(other_x_list)) out[[nm]] <- scale(other_x_list[[nm]], center = mu, scale = sdx)
  out
}

eval_risks <- function(fit, eval_x, eval_y_hinge, h) {
  fp   <- forward_pass_3(fit$Weight1, fit$Weight2, fit$Weight3, fit$Weight0, cbind(1, eval_x))
  y0   <- as.vector(fp$z0)
  Rh   <- mean(phi_Hh(eval_y_hinge * y0, h))
  pred <- ifelse(y0 >= 0, 1, -1)
  list(Rh = Rh, R = mean(pred != eval_y_hinge))
}

fit_and_evaluate <- function(npc, seed_data, seed_init, val_data = val_fixed,
                             eval_x_raw = eval_data$x, nepoch_use = nepoch, h_override = NULL) {
  n_total <- 2 * npc
  h_use   <- if (is.null(h_override)) h_schedule(n_total) else h_override
  train <- sample_data(npc, npc, seed = seed_data)
  if (STANDARDIZE_INPUT) {
    std <- apply_standardization(train$x, list(val = val_data$x, eval = eval_x_raw))
    train_x <- std$train; val_x <- std$val; eval_x <- std$eval
  } else {
    train_x <- train$x; val_x <- val_data$x; eval_x <- eval_x_raw
  }
  init <- make_init(P_TOTAL, node1, node2, node3, seed = seed_init)
  fit  <- SmoothHinge3_Ridge_val(
    W1 = init$W1, W2 = init$W2, W3 = init$W3, W0 = init$W0,
    X = cbind(1, train_x), y = train$y,
    X_val = cbind(1, val_x), y_val = val_data$y,
    Epoch = nepoch_use, rate = lr, lambda2 = LAMBDA2_FIXED, h = h_use, verbose = FALSE
  )
  risks <- eval_risks(fit, eval_x, eval_y_hinge, h_use)
  list(Rh = risks$Rh, R = risks$R, h_used = h_use, Best_epoch = fit$Best_epoch, Best_ACC_val = fit$Best_ACC_val)
}

## ============================================================================
## 4. PANEL (a): excess classification risk vs n
## ============================================================================

cat("\n==== PANEL (a): excess classification risk vs n ====\n")

cat(sprintf("[Floor] estimating R_h floor at n_per_class = %d (%d reps)\n", n_floor, floor_reps))
floor_Rh <- numeric(floor_reps); floor_R <- numeric(floor_reps)
for (k in 1:floor_reps) {
  res <- fit_and_evaluate(npc = n_floor, seed_data = 88880 + k, seed_init = 88990 + k, nepoch_use = nepoch)
  floor_Rh[k] <- res$Rh; floor_R[k] <- res$R
}
Rh_floor <- mean(floor_Rh)
cat(sprintf("R_h floor estimate: %.5f\n", Rh_floor))

n_npc     <- length(n_per_class_grid)
reps_grid <- vapply(n_per_class_grid, reps_for, numeric(1))
max_reps  <- max(reps_grid)
R_mat  <- matrix(NA_real_, n_npc, max_reps)
Rh_mat <- matrix(NA_real_, n_npc, max_reps)

cat(sprintf("[Grid] %d cells; reps = %s; total runs = %d\n", n_npc, paste(reps_grid, collapse = ", "), sum(reps_grid)))
pb <- txtProgressBar(min = 0, max = sum(reps_grid), style = 3); done <- 0L
for (i in seq_along(n_per_class_grid)) {
  npc <- n_per_class_grid[i]; reps <- reps_grid[i]
  for (r in 1:reps) {
    res <- fit_and_evaluate(npc = npc, seed_data = 1000 * i + r, seed_init = 3000 * i + r)
    Rh_mat[i, r] <- res$Rh; R_mat[i, r] <- res$R
    done <- done + 1L; setTxtProgressBar(pb, done)
  }
}
close(pb)

## ============================================================================
## 5. PANEL (b)/(c): full-batch proximal-gradient trajectory (lambda1 > 0)
## ============================================================================

cat("\n==== PANEL (b)/(c): full-batch proximal-gradient trajectory ====\n")
cat(sprintf("n_per_class = %d, T_iter = %d, lambda1 = %.3f\n", n_per_class_traj, T_iter_traj, LAMBDA1_TRAJ))

train <- sample_data(n_per_class_traj, n_per_class_traj, seed = 7777)
train_x <- if (STANDARDIZE_INPUT) apply_standardization(train$x, list())$train else train$x
init <- make_init(P_TOTAL, node1, node2, node3, seed = 7778)

fb_result <- PDSVM3_fullbatch(
  W1 = init$W1, W2 = init$W2, W3 = init$W3, W0 = init$W0,
  X = cbind(1, train_x), y = train$y,
  T_iter = T_iter_traj, rate = rate_fb,
  lambda1 = LAMBDA1_TRAJ, lambda2 = LAMBDA2_FIXED, h = h_fixed,
  adaptive_weights = NULL, verbose = TRUE, tag = "PDSVM-FB"
)

## ============================================================================
## 6. Data prep for plotting
## ============================================================================

# --- Panel (a) ---
total_n       <- 2 * n_per_class_grid
excess_R      <- pmax(R_mat - bayes_risk, 0)
excess_R_mean <- rowMeans(excess_R, na.rm = TRUE)
excess_R_se   <- apply(excess_R, 1, function(v) sd(v, na.rm = TRUE) / sqrt(sum(!is.na(v))))

nls_data <- data.frame(em = excess_R_mean, total_n = total_n)
nls_fit  <- nls(em ~ C / sqrt(total_n) + floor, data = nls_data,
                start = list(C = excess_R_mean[1] * sqrt(total_n[1]), floor = min(excess_R_mean) / 2),
                control = list(maxiter = 200))
C_hat     <- as.numeric(coef(nls_fit)["C"])
floor_hat <- as.numeric(coef(nls_fit)["floor"])

plot_data_a <- data.frame(
  n = total_n,
  pure_excess = pmax(excess_R_mean - floor_hat, 1e-6),  # after subtracting the FITTED floor
  se = excess_R_se,
  theory_rate = C_hat / sqrt(total_n)
)

# --- Panel (b)/(c) ---
t_axis <- seq_len(T_iter_traj)
running_min_G2 <- cummin(fb_result$G_eta_norm_sq)

ok_traj  <- t_axis > 3 & t_axis < 50 & running_min_G2 > 0
fit_traj <- lm(log(running_min_G2[ok_traj]) ~ log(t_axis[ok_traj]))
slope_traj     <- coef(fit_traj)[2]
intercept_traj <- coef(fit_traj)[1]

t_floor_onset <- which(abs(diff(running_min_G2)) / pmax(running_min_G2[-length(running_min_G2)], 1e-30) < 1e-2)[1]
if (is.na(t_floor_onset)) t_floor_onset <- max(t_axis)

plot_data_b <- data.frame(t = t_axis, min_G2 = running_min_G2, Ln = fb_result$L_n_traj[-1])
t_anchor <- t_axis[ok_traj][1]; y_anchor <- running_min_G2[ok_traj][1]
plot_data_b$theory_T      <- ifelse(t_axis >= t_anchor & t_axis <= 50, y_anchor * (t_axis / t_anchor)^(-1.0), NA)
plot_data_b$empirical_fit <- ifelse(t_axis >= t_anchor & t_axis <= 50, exp(intercept_traj) * (t_axis)^slope_traj, NA)

## ============================================================================
## 7. Figure (1 x 3 layout)
## ============================================================================

# --- Panel (a): excess classification risk vs n ---
p_a <- ggplot(plot_data_a, aes(x = n)) +
  geom_errorbar(aes(ymin = pure_excess - se, ymax = pure_excess + se), width = 0.05, color = "gray50", na.rm = TRUE) +
  geom_point(aes(y = pure_excess, color = "Empirical (mean +/- SE)"), size = 3, na.rm = TRUE) +
  geom_line(aes(y = theory_rate, color = "Theory: C/sqrt(n)"), linetype = "dashed", linewidth = 1, na.rm = TRUE) +
  scale_x_log10(breaks = total_n, labels = total_n) +
  scale_y_log10() +
  scale_color_manual(values = c("Empirical (mean +/- SE)" = "black", "Theory: C/sqrt(n)" = "darkgreen")) +
  labs(x = "Total Training Size (n)", y = "Excess Classification Risk (floor-subtracted)", color = NULL) +
  theme_classic(base_size = 14) +
  theme(legend.position = c(0.95, 0.95), legend.justification = c("right", "top"),
        legend.background = element_rect(fill = "transparent"), panel.grid.minor = element_blank())

# --- Panel (b): stationarity of the proximal-gradient mapping ---
p_b <- ggplot(plot_data_b[plot_data_b$t <= 100, ], aes(x = t)) +
  annotate("rect", xmin = t_anchor, xmax = 50, ymin = 0, ymax = Inf, alpha = 0.15, fill = "blue") +
  geom_line(aes(y = min_G2, color = "Trajectory min"), linewidth = 0.8, na.rm = TRUE) +
  geom_line(aes(y = empirical_fit, color = "Empirical fit"), linewidth = 1.2, linetype = "dotted", na.rm = TRUE) +
  geom_line(aes(y = theory_T, color = "Reference C/T (slope -1)"), linewidth = 1.2, linetype = "dotdash", na.rm = TRUE) +
  geom_vline(xintercept = t_floor_onset, linetype = "dashed", color = "gray50") +
  scale_x_log10() + scale_y_log10() +
  scale_color_manual(
    values = c("Trajectory min" = "gray60", "Empirical fit" = "blue", "Reference C/T (slope -1)" = "darkgreen"),
    labels = c("Trajectory min", sprintf("Empirical fit (~T^{%.2f})", slope_traj), "Reference C/T (slope -1)")) +
  labs(x = "Proximal-Gradient Step (t, log scale)", y = "Stationarity Measure", color = NULL) +
  theme_classic(base_size = 14) +
  theme(legend.position = c(0.05, 0.05), legend.justification = c("left", "bottom"),
        legend.background = element_rect(fill = "transparent", color = NA),
        legend.key = element_rect(fill = "transparent", color = NA), legend.text = element_text(size = 11))

# --- Panel (c): descent of the penalized objective L_n ---
p_c <- ggplot(plot_data_b, aes(x = t, y = Ln)) +
  geom_line(color = "black", linewidth = 0.8) +
  labs(x = "Proximal-Gradient Step (t)", y = expression("Penalized Objective "*L[n]*"("*W^"(t)"*")")) +
  theme_classic(base_size = 14) +
  theme(panel.grid.minor = element_blank())

final_plot <- p_a + p_b + p_c + plot_layout(ncol = 3) +
  plot_annotation(tag_levels = 'a') &
  theme(plot.tag = element_text(size = 16, face = 'bold'))

ggsave(file.path(OUT_DIR, "fig_theorems_visualization.pdf"),
       plot = final_plot, width = 16, height = 5.5, device = cairo_pdf)
cat(sprintf("\nFigure saved to: %s\n", file.path(OUT_DIR, "fig_theorems_visualization.pdf")))

## ============================================================================
## 8. Save numerical results
## ============================================================================

save(Rh_mat, R_mat, n_per_class_grid, reps_grid, total_n,
     excess_R, excess_R_mean, excess_R_se,
     Rh_floor, floor_Rh, floor_R, n_floor, floor_reps, bayes_risk,
     fb_result, running_min_G2, t_axis, slope_traj,
     T_iter_traj, n_per_class_traj, rate_fb, LAMBDA1_TRAJ,
     file = file.path(OUT_DIR, "theorems_visualization_data.RData"))
cat(sprintf("Saved data to: %s\n", file.path(OUT_DIR, "theorems_visualization_data.RData")))