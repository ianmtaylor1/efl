data {
    // Number of games and number of teams
    int<lower=0> nGames;
    int<lower=1> nTeams;
    
    // Describe the known games and their outcomes
    int<lower=1,upper=nTeams> hometeamidx[nGames];
    int<lower=1,upper=nTeams> awayteamidx[nGames];
    int<lower=1,upper=3> results[nGames];
    
    // Number of new games and description of new games
    int<lower=0> nGames_new;
    int<lower=1,upper=nTeams> hometeamidx_new[nGames_new];
    int<lower=1,upper=nTeams> awayteamidx_new[nGames_new];
    
    // Prior mean and variance for teams parameters
    vector[nTeams-1] teams_prior_mean;
    cov_matrix[nTeams-1] teams_prior_var;
    
    // Prior location and scale for win/draw threshold parameter
    real theta_prior_loc;
    real<lower=0> theta_prior_scale;

    // Prior mean and sd for homefield advantage parameter
    real home_prior_mean;
    real<lower=0> home_prior_sd;
}
transformed data {
    // Cholesky decomposition of prior teams variance
    cholesky_factor_cov[nTeams-1] teams_prior_var_chol;
    // Indicator matrices for which team is in which game
    matrix[nGames,nTeams-1] X;
    matrix[nGames_new, nTeams-1] X_new;
    
    teams_prior_var_chol = cholesky_decompose(teams_prior_var);
    
    X = rep_matrix(0, nGames, nTeams-1);
    for (r in 1:nGames) {
        if (hometeamidx[r] < nTeams) {
            X[r,hometeamidx[r]] = 1;
        }
        if (awayteamidx[r] < nTeams) {
            X[r,awayteamidx[r]] = -1;
        }
    }
    
    X_new = rep_matrix(0, nGames_new, nTeams-1);
    for (r in 1:nGames_new) {
        if (hometeamidx[r] < nTeams) {
            X_new[r,hometeamidx[r]] = 1;
        }
        if (awayteamidx[r] < nTeams) {
            X_new[r,awayteamidx[r]] = -1;
        }
    }
}
parameters {
    vector[nTeams-1] teams;
    real home;
    real<lower=0> theta;
}
model {
    theta ~ logistic(theta_prior_loc, theta_prior_scale);
    teams ~ multi_normal_cholesky(teams_prior_mean, teams_prior_var_chol);
    home ~ normal(home_prior_mean, home_prior_sd);
    if (N > 0) {
        results ~ ordered_logistic(X * teams + home, [ -theta, theta ]');
    };
}
generated quantities {
    int<lower=1,upper=3> results_pred[nGames];
    int<lower=1,upper=3> results_new_pred[nGames_new];
    for (i in 1:nGames) {
        results_pred[i] = ordered_logistic_rng(X[i] * teams + home, [ -theta, theta ]');
    };
    for (i in 1:nGames_new) {
        results_new_pred[i] = ordered_logistic_rng(X_new[i] * teams + home, [ -theta, theta ]');
    };
}
