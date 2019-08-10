data {
    int<lower=0> N; 
    int<lower=1> P;
    int<lower=1,upper=3> Y[N];
    matrix[N,P] X;

    int<lower=0> N_new;
    matrix[N_new, P] X_new;
    
    vector[P] beta_prior_mean;
    cov_matrix[P] beta_prior_var;
    real theta_prior_loc;
    real<lower=0> theta_prior_scale;
}
parameters {
    vector[P] beta;
    real<lower=0> theta;
}
model {
    theta ~ logistic(theta_prior_loc, theta_prior_scale);
    beta ~ multi_normal(beta_prior_mean, beta_prior_var);
    if (N > 0) {
        Y ~ ordered_logistic(X * beta, [ -theta, theta ]');
    };
}
generated quantities {
    int<lower=1,upper=3> Y_pred[N];
    int<lower=1,upper=3> Y_new_pred[N_new];
    for (i in 1:N) {
        Y_pred[i] = ordered_logistic_rng(X[i] * beta, [ -theta, theta]');
    };
    for (i in 1:N_new) {
        Y_new_pred[i] = ordered_logistic_rng(X_new[i] * beta, [ -theta, theta]');
    };
}