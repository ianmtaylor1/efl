data {
    int<lower=0> N;
    int<lower=1> P;
    int<lower=0> Y[N];
    matrix[N,P] X;
    
    int<lower=0> N_new;
    matrix[N_new, P] X_new;
    
    vector[P] beta_prior_mean;
    cov_matrix[P] beta_prior_var;
}
transformed data {
    cholesky_factor_cov[P] beta_prior_var_chol;
    beta_prior_var_chol = cholesky_decompose(beta_prior_var);
}
parameters {
    vector[P] beta;
}
model {
    beta ~ multi_normal_cholesky(beta_prior_mean, beta_prior_var_chol);
    if (N > 0) {
        Y ~ poisson_log(X * beta);
    };
}
generated quantities {
    int<lower=0> Y_pred[N];
    int<lower=0> Y_new_pred[N_new];
    if (N > 0) {
        Y_pred = poisson_log_rng(X * beta);
    };
    if (N_new > 0) {
        Y_new_pred = poisson_log_rng(X_new * beta);
    };
}
