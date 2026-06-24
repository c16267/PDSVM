## ============================== 1. Activations ===============================
Activation <- function(x, type = c("tanh", "relu", "sigmoid")) {
  type <- match.arg(type)                       
  if (type == "tanh")    return(tanh(x))
  if (type == "relu")  { x[x < 0] <- 0; return(x) }
  if (type == "sigmoid") return(1 / (1 + exp(-x)))
  stop("Unknown activation type")
}

Activation_prime <- function(y, type = c("tanh", "relu", "sigmoid")) {
  # y is the post-activation value: y = f(x)
  if (type == "tanh") return(1 - y^2)
  if (type == "relu") return(ifelse(y > 0, 1, 0))
  if (type == "sigmoid") return(y * (1 - y))
  stop("Unknown activation type")
}

## ===================== 2. Smoothed hinge + derivative =======================
phi_Hh <- function(u, h) {
  stopifnot(h > 0); out <- numeric(length(u))
  left <- u <= (1 - h); mid <- (u > (1 - h)) & (u < (1 + h))
  out[left] <- 1 - u[left]; out[mid] <- (1 - u[mid] + h)^2 / (4 * h); out
}
phi_Hh_prime <- function(u, h) {
  stopifnot(h > 0); out <- numeric(length(u))
  left <- u <= (1 - h); mid <- (u > (1 - h)) & (u < (1 + h))
  out[left] <- -1; out[mid] <- -(1 - u[mid] + h) / (2 * h); out
}

## ===================== 3a. Proximal group-soft-threshold (v4) ===============
group_soft_threshold <- function(W1_cols, tau_vec) {
  col_norms <- sqrt(colSums(W1_cols^2))
  shrink <- pmax(0, 1 - tau_vec / pmax(col_norms, .Machine$double.eps))
  sweep(W1_cols, 2, shrink, `*`)
}

## ===================== 3b. Gradient of group-L2 penalty (v3) ================
glasso_penalty_grad <- function(W1_cols, w_hat, eps = .Machine$double.eps) {
  col_norms <- sqrt(colSums(W1_cols^2))
  scale <- w_hat / pmax(col_norms, eps)
  scale[col_norms <= eps] <- 0      # dead column -> no gradient contribution
  sweep(W1_cols, 2, scale, `*`)
}

## ===================== 4. Adaptive weights (gamma=1) ========================
compute_adaptive_weights <- function(W1_pilot_cols, gamma = 1, eps = 1e-8) {
  col_norms <- sqrt(colSums(W1_pilot_cols^2)); 1 / (col_norms + eps)^gamma
}

## ===================== 5. Forward pass + predictors =========================
forward_pass_3 <- function(W1, W2, W3, W0, X, activation = "tanh") {
  z1 <- W1 %*% t(X); y1 <- Activation(z1, activation); y1a <- rbind(1, y1)
  z2 <- W2 %*% y1a;  y2 <- Activation(z2, activation); y2a <- rbind(1, y2)
  z3 <- W3 %*% y2a;  y3 <- Activation(z3, activation); y3a <- rbind(1, y3)
  z0 <- W0 %*% y3a
  list(z1=z1,y1=y1,y1a=y1a, z2=z2,y2=y2,y2a=y2a, z3=z3,y3=y3,y3a=y3a, z0=z0)
}
Hinge3_predict <- function(W1, W2, W3, W0, X, y, activation = "tanh") {
  fp <- forward_pass_3(W1,W2,W3,W0,X, activation)
  y0 <- as.vector(fp$z0); pred <- ifelse(y0>=0,1,-1)
  y_use <- if (all(y %in% c(0,1))) ifelse(y==1,1,-1) else y
  list(Pred=pred, ACC=mean(y_use==pred))
}
Binary3_predict <- function(W1, W2, W3, W0, X, y, activation = "tanh") {
  fp <- forward_pass_3(W1,W2,W3,W0,X, activation)
  p0 <- Activation(as.vector(fp$z0), "sigmoid") 
  pred <- as.integer(p0 > 0.5); list(Pred=pred, ACC=mean(pred==y))
}

## =====================================================================
## 6. Main routine: full-batch; penalty_type = "grad" (v3) or "prox" (v4)
## =====================================================================
PDSVM3_val <- function(W1, W2, W3, W0, X, y, X_val, y_val,
                       Epoch=1000, Bsize=NULL, rate=0.05,
                       lambda1=0, lambda2=0, h=0.1, adaptive_weights=NULL,
                       penalty_type=c("grad","prox"), warmup_frac=0,
                       activation=c("tanh", "relu", "sigmoid"),
                       verbose=TRUE, tag="PDSVM", pb=NULL) {
  penalty_type <- match.arg(penalty_type)
  activation <- match.arg(activation)
  warm_E <- max(1L, floor(warmup_frac * Epoch))   
  W1<-as.matrix(W1);W2<-as.matrix(W2);W3<-as.matrix(W3);W0<-as.matrix(W0)
  X<-as.matrix(X); y<-as.numeric(y); y<-ifelse(y==1,1,-1)
  X_val<-as.matrix(X_val); y_val<-as.numeric(y_val); y_val<-ifelse(y_val==1,1,-1)
  n<-nrow(X); p<-ncol(X)-1; eta<-rate
  w_hat <- if (is.null(adaptive_weights)) rep(1,p) else as.numeric(adaptive_weights)
  stopifnot(length(w_hat)==p)
  acc_train<-acc_val<-loss_train<-loss_val<-step_sq<-numeric(Epoch)
  best_acc<- -Inf; best_epoch<-1L; W1_best<-W1;W2_best<-W2;W3_best<-W3;W0_best<-W0
  
  for (t in seq_len(Epoch)) {
    fp <- forward_pass_3(W1,W2,W3,W0,X, activation); y0 <- as.vector(fp$z0)
    d0 <- matrix(y * phi_Hh_prime(y*y0, h) / n, nrow=1)
    grad_W0 <- d0 %*% t(fp$y3a); back0 <- t(W0) %*% d0
    
    delta3  <- back0[-1,,drop=FALSE] * Activation_prime(fp$y3, activation)
    grad_W3 <- delta3 %*% t(fp$y2a)
    back3   <- t(W3) %*% delta3
    
    delta2  <- back3[-1,,drop=FALSE] * Activation_prime(fp$y2, activation)
    grad_W2 <- delta2 %*% t(fp$y1a)
    back2   <- t(W2) %*% delta2
    
    delta1  <- back2[-1,,drop=FALSE] * Activation_prime(fp$y1, activation)
    grad_W1 <- delta1 %*% X
    
    if (lambda2 > 0) { grad_W0<-grad_W0+2*lambda2*W0; grad_W3<-grad_W3+2*lambda2*W3; grad_W2<-grad_W2+2*lambda2*W2 }
    
    ## upper layers: plain gradient step
    W0_new<-W0-eta*grad_W0; W3_new<-W3-eta*grad_W3; W2_new<-W2-eta*grad_W2
    
    ## ---- W1 update: group lasso via gradient (v3) or proximal (v4) ----------
    lam_t <- lambda1 * min(1, t / warm_E)   
    if (lam_t > 0 && penalty_type == "grad") {
      pen <- glasso_penalty_grad(W1[,-1,drop=FALSE], w_hat)          
      grad_W1_pen <- grad_W1; grad_W1_pen[,-1] <- grad_W1[,-1] + lam_t*pen
      W1_new <- W1 - eta*grad_W1_pen
    } else if (lam_t > 0 && penalty_type == "prox") {
      W1_half <- W1 - eta*grad_W1
      W1f <- group_soft_threshold(W1_half[,-1,drop=FALSE], eta*lam_t*w_hat)
      W1_new <- cbind(W1_half[,1,drop=FALSE], W1f)
    } else {
      W1_new <- W1 - eta*grad_W1
    }
    
    step_sq[t] <- sum((c(W1_new,W2_new,W3_new,W0_new) - c(W1,W2,W3,W0))^2)
    W1<-W1_new; W2<-W2_new; W3<-W3_new; W0<-W0_new
    
    z0_tr <- as.vector(forward_pass_3(W1,W2,W3,W0,X, activation)$z0)
    z0_vl <- as.vector(forward_pass_3(W1,W2,W3,W0,X_val, activation)$z0)
    loss_train[t] <- mean(phi_Hh(y*z0_tr,h)); loss_val[t] <- mean(phi_Hh(y_val*z0_vl,h))
    acc_train[t]  <- mean(ifelse(z0_tr>=0,1,-1)==y); acc_val[t] <- mean(ifelse(z0_vl>=0,1,-1)==y_val)
    if (acc_val[t] > best_acc) { best_acc<-acc_val[t]; best_epoch<-t; W1_best<-W1;W2_best<-W2;W3_best<-W3;W0_best<-W0 }
    if (verbose) {
      cn <- sqrt(colSums(W1[,-1,drop=FALSE]^2)); nz <- sum(cn <= 1e-3)
      cat(sprintf("[%s|%s|%s] iter %d | h=%.3f lam1=%.4f lam2=%.4f | val.acc=%.4f | step^2=%.2e | small groups=%d/%d\n",
                  tag, penalty_type, activation, t, h, lambda1, lambda2, acc_val[t], step_sq[t], nz, p))
    }
    if (!is.null(pb)) setTxtProgressBar(pb, t)
  }
  list(Weight1=W1_best,Weight2=W2_best,Weight3=W3_best,Weight0=W0_best,
       ACC_train=acc_train,ACC_val=acc_val,Loss_train=loss_train,Loss_val=loss_val,
       step_sq=step_sq,Best_epoch=best_epoch,Best_ACC_val=best_acc,
       h=h,lambda1=lambda1,lambda2=lambda2,adaptive_weights=w_hat,
       penalty_type=penalty_type, activation=activation, optimizer="fullbatch")
}

## =====================================================================
## 6b. Mini-batch SGD variant with the same penalty_type switch.
## =====================================================================
PDSVM3_minibatch <- function(W1, W2, W3, W0, X, y, X_val, y_val,
                             Epoch=1000, Bsize=64, rate=0.02,
                             lambda1=0, lambda2=0, h=0.1, adaptive_weights=NULL,
                             penalty_type=c("grad","prox"), warmup_frac=0,
                             activation=c("tanh", "relu", "sigmoid"),
                             verbose=TRUE, tag="PDSVM-MB", pb=NULL) {
  penalty_type <- match.arg(penalty_type)
  activation <- match.arg(activation)
  warm_E <- max(1L, floor(warmup_frac * Epoch))   
  W1<-as.matrix(W1);W2<-as.matrix(W2);W3<-as.matrix(W3);W0<-as.matrix(W0)
  X<-as.matrix(X); y<-as.numeric(y); y<-ifelse(y==1,1,-1)
  X_val<-as.matrix(X_val); y_val<-as.numeric(y_val); y_val<-ifelse(y_val==1,1,-1)
  n<-nrow(X); p<-ncol(X)-1; eta<-rate; n_iter<-ceiling(n/Bsize)
  w_hat <- if (is.null(adaptive_weights)) rep(1,p) else as.numeric(adaptive_weights)
  acc_train<-acc_val<-loss_train<-loss_val<-numeric(Epoch)
  best_acc<- -Inf; best_epoch<-1L; W1_best<-W1;W2_best<-W2;W3_best<-W3;W0_best<-W0
  for (e in seq_len(Epoch)) {
    lam_e <- lambda1 * min(1, e / warm_E)   
    idx<-sample.int(n); Xe<-X[idx,,drop=FALSE]; ye<-y[idx]; ns<-1L
    for (tt in seq_len(n_iter)) {
      ne<-min(ns+Bsize-1L,n); Xb<-Xe[ns:ne,,drop=FALSE]; yb<-ye[ns:ne]; nb<-nrow(Xb)
      fp<-forward_pass_3(W1,W2,W3,W0,Xb, activation); y0<-as.vector(fp$z0)
      d0<-matrix(yb*phi_Hh_prime(yb*y0,h)/nb,nrow=1)
      grad_W0<-d0%*%t(fp$y3a); back0<-t(W0)%*%d0
      
      delta3<-back0[-1,,drop=FALSE] * Activation_prime(fp$y3, activation)
      grad_W3<-delta3%*%t(fp$y2a); back3<-t(W3)%*%delta3
      
      delta2<-back3[-1,,drop=FALSE] * Activation_prime(fp$y2, activation)
      grad_W2<-delta2%*%t(fp$y1a); back2<-t(W2)%*%delta2
      
      delta1<-back2[-1,,drop=FALSE] * Activation_prime(fp$y1, activation)
      grad_W1<-delta1%*%Xb
      
      if (lambda2>0){grad_W0<-grad_W0+2*lambda2*W0;grad_W3<-grad_W3+2*lambda2*W3;grad_W2<-grad_W2+2*lambda2*W2}
      W0<-W0-eta*grad_W0; W3<-W3-eta*grad_W3; W2<-W2-eta*grad_W2
      if (lam_e>0 && penalty_type=="grad") {
        pen<-glasso_penalty_grad(W1[,-1,drop=FALSE],w_hat)
        g1<-grad_W1; g1[,-1]<-grad_W1[,-1]+lam_e*pen; W1<-W1-eta*g1
      } else if (lam_e>0 && penalty_type=="prox") {
        W1h<-W1-eta*grad_W1; W1f<-group_soft_threshold(W1h[,-1,drop=FALSE],eta*lam_e*w_hat)
        W1<-cbind(W1h[,1,drop=FALSE],W1f)
      } else W1<-W1-eta*grad_W1
      ns<-ne+1L; if (ns>n) break
    }
    z0_tr<-as.vector(forward_pass_3(W1,W2,W3,W0,X, activation)$z0); z0_vl<-as.vector(forward_pass_3(W1,W2,W3,W0,X_val, activation)$z0)
    loss_train[e]<-mean(phi_Hh(y*z0_tr,h)); loss_val[e]<-mean(phi_Hh(y_val*z0_vl,h))
    acc_train[e]<-mean(ifelse(z0_tr>=0,1,-1)==y); acc_val[e]<-mean(ifelse(z0_vl>=0,1,-1)==y_val)
    if (acc_val[e]>best_acc){best_acc<-acc_val[e];best_epoch<-e;W1_best<-W1;W2_best<-W2;W3_best<-W3;W0_best<-W0}
    if (verbose) cat(sprintf("[%s|%s|%s] Epoch %d | val.acc=%.4f\n", tag, penalty_type, activation, e, acc_val[e]))
    if (!is.null(pb)) setTxtProgressBar(pb, e)
  }
  list(Weight1=W1_best,Weight2=W2_best,Weight3=W3_best,Weight0=W0_best,
       ACC_train=acc_train,ACC_val=acc_val,Loss_train=loss_train,Loss_val=loss_val,
       Best_epoch=best_epoch,Best_ACC_val=best_acc,h=h,lambda1=lambda1,lambda2=lambda2,
       adaptive_weights=w_hat,penalty_type=penalty_type,activation=activation,optimizer="minibatch")
}

## ===================== 7. Convenience wrappers ==============================
SmoothHinge3_val <- function(W1,W2,W3,W0,X,y,X_val,y_val, Epoch=1000,Bsize=NULL,rate=0.05,h=0.1,activation="tanh",verbose=TRUE,tag="DSVM",pb=NULL)
  PDSVM3_val(W1,W2,W3,W0,X,y,X_val,y_val,Epoch=Epoch,Bsize=Bsize,rate=rate,lambda1=0,lambda2=0,h=h,activation=activation,verbose=verbose,tag=tag,pb=pb)

SmoothHinge3_Ridge_val <- function(W1,W2,W3,W0,X,y,X_val,y_val, Epoch=1000,Bsize=NULL,rate=0.05,lambda2=1e-4,h=0.1,activation="tanh",verbose=TRUE,tag="DSVM+Ridge",pb=NULL)
  PDSVM3_val(W1,W2,W3,W0,X,y,X_val,y_val,Epoch=Epoch,Bsize=Bsize,rate=rate,lambda1=0,lambda2=lambda2,h=h,activation=activation,verbose=verbose,tag=tag,pb=pb)

SmoothHinge3_GLasso_val <- function(W1,W2,W3,W0,X,y,X_val,y_val, Epoch=1000,Bsize=NULL,rate=0.05,lambda1=0.05,lambda2=1e-4,h=0.1,penalty_type="grad",warmup_frac=0,activation="tanh",verbose=TRUE,tag="PDSVM-GL",pb=NULL)
  PDSVM3_val(W1,W2,W3,W0,X,y,X_val,y_val,Epoch=Epoch,Bsize=Bsize,rate=rate,lambda1=lambda1,lambda2=lambda2,h=h,penalty_type=penalty_type,warmup_frac=warmup_frac,activation=activation,verbose=verbose,tag=tag,pb=pb)

SmoothHinge3_AGLasso_val <- function(W1,W2,W3,W0,X,y,X_val,y_val, Epoch=1000,Bsize=NULL,rate=0.05,lambda1=0.05,lambda2=1e-4,h=0.1,adaptive_weights,penalty_type="grad",warmup_frac=0,activation="tanh",verbose=TRUE,tag="PDSVM-AGL",pb=NULL){
  stopifnot(!is.null(adaptive_weights))
  PDSVM3_val(W1,W2,W3,W0,X,y,X_val,y_val,Epoch=Epoch,Bsize=Bsize,rate=rate,lambda1=lambda1,lambda2=lambda2,h=h,adaptive_weights=adaptive_weights,penalty_type=penalty_type,warmup_frac=warmup_frac,activation=activation,verbose=verbose,tag=tag,pb=pb)
}

## ===================== 7b. Cross-entropy baseline (D-CE) ====================
CE3_val <- function(W1,W2,W3,W0,X,y,X_val,y_val, Epoch=1000,Bsize=NULL,rate=0.05,lambda2=0,activation=c("tanh", "relu", "sigmoid"),verbose=TRUE,tag="D-CE",pb=NULL) {
  activation <- match.arg(activation)
  W1<-as.matrix(W1);W2<-as.matrix(W2);W3<-as.matrix(W3);W0<-as.matrix(W0)
  X<-as.matrix(X); y<-as.numeric(y); X_val<-as.matrix(X_val); y_val<-as.numeric(y_val)
  stopifnot(all(y%in%c(0,1)), all(y_val%in%c(0,1)))
  n<-nrow(X); eta<-rate; EPS<-1e-12
  acc_train<-acc_val<-loss_train<-loss_val<-numeric(Epoch); best_acc<- -Inf; best_epoch<-1L
  W1_best<-W1;W2_best<-W2;W3_best<-W3;W0_best<-W0
  for (t in seq_len(Epoch)) {
    fp<-forward_pass_3(W1,W2,W3,W0,X, activation); z0<-as.vector(fp$z0); p0<-Activation(z0, "sigmoid") # Output layer for CE is always Sigmoid
    d0<-matrix((p0-y)/n,nrow=1)
    grad_W0<-d0%*%t(fp$y3a); back0<-t(W0)%*%d0
    
    delta3<-back0[-1,,drop=FALSE] * Activation_prime(fp$y3, activation)
    grad_W3<-delta3%*%t(fp$y2a); back3<-t(W3)%*%delta3
    
    delta2<-back3[-1,,drop=FALSE] * Activation_prime(fp$y2, activation)
    grad_W2<-delta2%*%t(fp$y1a); back2<-t(W2)%*%delta2
    
    delta1<-back2[-1,,drop=FALSE] * Activation_prime(fp$y1, activation)
    grad_W1<-delta1%*%X
    
    if (lambda2>0){grad_W0<-grad_W0+2*lambda2*W0;grad_W3<-grad_W3+2*lambda2*W3;grad_W2<-grad_W2+2*lambda2*W2}
    W0<-W0-eta*grad_W0;W3<-W3-eta*grad_W3;W2<-W2-eta*grad_W2;W1<-W1-eta*grad_W1
    
    p_tr<-Activation(as.vector(forward_pass_3(W1,W2,W3,W0,X, activation)$z0), "sigmoid")
    p_vl<-Activation(as.vector(forward_pass_3(W1,W2,W3,W0,X_val, activation)$z0), "sigmoid")
    p_tr_c<-pmin(pmax(p_tr,EPS),1-EPS); p_vl_c<-pmin(pmax(p_vl,EPS),1-EPS)
    loss_train[t]<- -mean(y*log(p_tr_c)+(1-y)*log(1-p_tr_c)); loss_val[t]<- -mean(y_val*log(p_vl_c)+(1-y_val)*log(1-p_vl_c))
    acc_train[t]<-mean((p_tr>0.5)==(y==1)); acc_val[t]<-mean((p_vl>0.5)==(y_val==1))
    if (acc_val[t]>best_acc){best_acc<-acc_val[t];best_epoch<-t;W1_best<-W1;W2_best<-W2;W3_best<-W3;W0_best<-W0}
    if (verbose) cat(sprintf("[%s|%s] iter %d | lam2=%.4f | val.acc=%.4f | val.CE=%.4f\n",tag,activation,t,lambda2,acc_val[t],loss_val[t]))
    if (!is.null(pb)) setTxtProgressBar(pb,t)
  }
  list(Weight1=W1_best,Weight2=W2_best,Weight3=W3_best,Weight0=W0_best,
       ACC_train=acc_train,ACC_val=acc_val,Loss_train=loss_train,Loss_val=loss_val,
       Best_epoch=best_epoch,Best_ACC_val=best_acc,lambda2=lambda2,activation=activation,optimizer="fullbatch_gd")
}

## ===================== 8. Selected-feature extractor ========================
pdsvm_selected_features <- function(fit, sel_thresh = 1e-3) {
  W1_feat <- fit$Weight1[, -1, drop = FALSE]
  col_norms <- sqrt(colSums(W1_feat^2))
  list(selected = which(col_norms > sel_thresh),
       zeroed   = which(col_norms <= sel_thresh),
       col_norms = col_norms)
}