data {
#include std_data_result.stan
    
    // Prior mean and variance for teams parameters
    vector[nTeams] teams_prior_mean;
    cov_matrix[nTeams] teams_prior_var;
    
    // Prior location and scale for win/draw threshold parameter
    real theta_prior_loc;
    real<lower=0> theta_prior_scale;

    // Prior mean and sd for homefield advantage parameter
    real home_prior_mean;
    real<lower=0> home_prior_sd;
}
transformed data {
    // Cholesky decomposition of prior teams variance
    cholesky_factor_cov[nTeams] teams_prior_var_chol;
    teams_prior_var_chol = cholesky_decompose(teams_prior_var);
}
parameters {
    vector[nTeams-1] teams_raw;
    real home;
    real<lower=0> theta;
}
transformed parameters {
    vector[nTeams] teams = append_row(teams_raw, -sum(teams_raw));
}
model {
    // Priors
    theta ~ logistic(theta_prior_loc, theta_prior_scale) T[0,];
    home ~ normal(home_prior_mean, home_prior_sd);
    teams ~ multi_normal_cholesky(teams_prior_mean, teams_prior_var_chol);
    // Model
    if (nGames > 0) {
        result ~ ordered_logistic(
                teams[hometeamidx] - teams[awayteamidx] + home, 
                [ -theta, theta ]'
                );
    };
}
generated quantities {
    int<lower=1,upper=3> result_pred[nGames];
    int<lower=1,upper=3> result_new_pred[nGames_new];
    for (i in 1:nGames) {
        result_pred[i] = ordered_logistic_rng(
                teams[hometeamidx[i]] - teams[awayteamidx[i]] + home, 
                [ -theta, theta ]'
                );
    };
    for (i in 1:nGames_new) {
        result_new_pred[i] = ordered_logistic_rng(
                teams[hometeamidx_new[i]] - teams[awayteamidx_new[i]] + home, 
                [ -theta, theta ]'
                );
    };
}
