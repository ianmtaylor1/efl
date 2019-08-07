data {
  int<lower=0> N; 
  int<lower=0> P;
  int<lower=1,upper=3> Y[N];
  matrix[N,P] X;
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
  Y ~ ordered_logistic(X * beta, [ -theta, theta ]');
}
