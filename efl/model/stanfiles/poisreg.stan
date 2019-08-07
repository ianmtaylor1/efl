data {
  int<lower=0> N;
  int<lower=0> P;
  int<lower=0> Y[N];
  matrix[N,P] X;
  vector[P] beta_prior_mean;
  cov_matrix[P] beta_prior_var;
}
parameters {
  vector[P] beta;
}
model {
  beta ~ multi_normal(beta_prior_mean, beta_prior_var);
  Y ~ poisson(exp(X * beta));
}
