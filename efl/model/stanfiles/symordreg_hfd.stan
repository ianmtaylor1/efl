data {
    // Number of games and number of teams
    int<lower=0> nGames;
    int<lower=1> nTeams;
    
    // Describe the known games and their outcomes
    int<lower=1,upper=nTeams> hometeamidx[nGames];
    int<lower=1,upper=nTeams> awayteamidx[nGames];
    int<lower=1,upper=3> result[nGames];
    
    // Number of new games and description of new games
    int<lower=0> nGames_new;
    int<lower=1,upper=nTeams> hometeamidx_new[nGames_new];
    int<lower=1,upper=nTeams> awayteamidx_new[nGames_new];
    
    // Prior mean and variance for teams home strength parameters
    vector[nTeams] home_prior_mean;
    cov_matrix[nTeams] home_prior_var;
    
    // Prior mean and variance for teams away strength parameters
    vector[nTeams] away_prior_mean;
    cov_matrix[nTeams] away_prior_var;
    
    // Prior location and scale for win/draw threshold parameter
    real theta_prior_loc;
    real<lower=0> theta_prior_scale;

    // Prior mean and sd for homefield advantage parameter
    real homefield_prior_mean;
    real<lower=0> homefield_prior_sd;
}
transformed data {
    // Cholesky decomposition of prior teams variance
    cholesky_factor_cov[nTeams] home_prior_var_chol;
    cholesky_factor_cov[nTeams] away_prior_var_chol;
    home_prior_var_chol = cholesky_decompose(home_prior_var);
    away_prior_var_chol = cholesky_decompose(away_prior_var);
}
parameters {
    vector[nTeams-1] home_raw;
    vector[nTeams-1] away_raw;
    real homefield;
    real<lower=0> theta;
}
transformed parameters {
    vector[nTeams] home = append_row(home_raw, -sum(home_raw));
    vector[nTeams] away = append_row(away_raw, -sum(away_raw));
}
model {
    // Priors
    theta ~ logistic(theta_prior_loc, theta_prior_scale) T[0,];
    homefield ~ normal(homefield_prior_mean, homefield_prior_sd);
    home ~ multi_normal_cholesky(home_prior_mean, home_prior_var_chol);
    away ~ multi_normal_cholesky(away_prior_mean, away_prior_var_chol);
    // Model
    if (nGames > 0) {
        result ~ ordered_logistic(
                home[hometeamidx] - away[awayteamidx] + homefield, 
                [ -theta, theta ]'
                );
    };
}
generated quantities {
    int<lower=1,upper=3> result_pred[nGames];
    int<lower=1,upper=3> result_new_pred[nGames_new];
    for (i in 1:nGames) {
        result_pred[i] = ordered_logistic_rng(
                home[hometeamidx[i]] - away[awayteamidx[i]] + homefield, 
                [ -theta, theta ]'
                );
    };
    for (i in 1:nGames_new) {
        result_new_pred[i] = ordered_logistic_rng(
                home[hometeamidx_new[i]] - away[awayteamidx_new[i]] + homefield, 
                [ -theta, theta ]'
                );
    };
}
