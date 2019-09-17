data {
    // Number of games and number of teams
    int<lower=0> nGames;
    int<lower=1> nTeams;
    
    // Describe the known games and their outcomes
    int<lower=1,upper=nTeams> hometeamidx[nGames];
    int<lower=1,upper=nTeams> awayteamidx[nGames];
    int<lower=0> homegoals[nGames];
    int<lower=0> awaygoals[nGames];
    
    // Number of new games and description of new games
    int<lower=0> nGames_new;
    int<lower=1,upper=nTeams> hometeamidx_new[nGames_new];
    int<lower=1,upper=nTeams> awayteamidx_new[nGames_new];
    
    // Prior parameters for home and away goals
    real          log_home_goals_prior_mean;
    real<lower=0> log_home_goals_prior_sd;
    real          log_away_goals_prior_mean;
    real<lower=0> log_away_goals_prior_sd;
    
    // Prior parameters for team modifiers
    vector[nTeams]     homeoff_prior_mean;
    cov_matrix[nTeams] homeoff_prior_var;
    vector[nTeams]     homedef_prior_mean;
    cov_matrix[nTeams] homedef_prior_var;
    vector[nTeams]     awayoff_prior_mean;
    cov_matrix[nTeams] awayoff_prior_var;
    vector[nTeams]     awaydef_prior_mean;
    cov_matrix[nTeams] awaydef_prior_var;
}
transformed data {
    cholesky_factor_cov[nTeams] homeoff_prior_var_chol;
    cholesky_factor_cov[nTeams] homedef_prior_var_chol;
    cholesky_factor_cov[nTeams] awayoff_prior_var_chol;
    cholesky_factor_cov[nTeams] awaydef_prior_var_chol;
    
    homeoff_prior_var_chol = cholesky_decompose(homeoff_prior_var);
    homedef_prior_var_chol = cholesky_decompose(homedef_prior_var);
    awayoff_prior_var_chol = cholesky_decompose(awayoff_prior_var);
    awaydef_prior_var_chol = cholesky_decompose(awaydef_prior_var);
}
parameters {
    // "raw" team modifiers (i.e. only the first nTeams-1 teams)
    vector[nTeams-1] homeoff_raw;
    vector[nTeams-1] homedef_raw;
    vector[nTeams-1] awayoff_raw;
    vector[nTeams-1] awaydef_raw;
    // Baseline goals for home and away teams
    real log_home_goals;
    real log_away_goals;
}
transformed parameters {
    // Transformed team modifiers, including nTeams'th component to add to 0
    vector[nTeams] homeoff = append_row(homeoff_raw, -sum(homeoff_raw));
    vector[nTeams] homedef = append_row(homedef_raw, -sum(homedef_raw));
    vector[nTeams] awayoff = append_row(awayoff_raw, -sum(awayoff_raw));
    vector[nTeams] awaydef = append_row(awaydef_raw, -sum(awaydef_raw));
}
model {
    // Prior contribution from home/away goals
    log_home_goals ~ normal(log_home_goals_prior_mean, log_home_goals_prior_sd);
    log_away_goals ~ normal(log_away_goals_prior_mean, log_away_goals_prior_sd);
    // Prior contribution from team modifiers
    homeoff ~ multi_normal_cholesky(homeoff_prior_mean, homeoff_prior_var_chol);
    homedef ~ multi_normal_cholesky(homedef_prior_mean, homedef_prior_var_chol);
    awayoff ~ multi_normal_cholesky(awayoff_prior_mean, awayoff_prior_var_chol);
    awaydef ~ multi_normal_cholesky(awaydef_prior_mean, awaydef_prior_var_chol);
    // Model, goals follow poisson distribution
    if (nGames > 0) {
        homegoals ~ poisson_log(homeoff[hometeamidx] - awaydef[awayteamidx] + log_home_goals);
        awaygoals ~ poisson_log(awayoff[awayteamidx] - homedef[hometeamidx] + log_away_goals);
    };
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    if (nGames > 0) {
        homegoals_pred = poisson_log_rng(homeoff[hometeamidx] - awaydef[awayteamidx] + log_home_goals);
        awaygoals_pred = poisson_log_rng(awayoff[awayteamidx] - homedef[hometeamidx] + log_away_goals);
    };
    if (nGames_new > 0) {
        homegoals_new_pred = poisson_log_rng(homeoff[hometeamidx_new] - awaydef[awayteamidx_new] + log_home_goals);
        awaygoals_new_pred = poisson_log_rng(awayoff[awayteamidx_new] - homedef[hometeamidx_new] + log_away_goals);
    };
}
