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
    
    // Prior parameters of the latent team parameters
    // The last team is a reference, its latent parameter == 0
    vector[nTeams] teams_prior_mean;
    cov_matrix[nTeams] teams_prior_var;
    
    // Prior parameters for the covariance matrix linking team offense/defense
    // sub-parameters to the latent team parameter
    real<lower=1> teamvar_prior_nu;
    cov_matrix[2] teamvar_prior_Sigma;
    
    // Prior parameters for the log baseline goals parameter
    real log_goals_prior_mean;
    real<lower=0> log_goals_prior_sd;
    
    // Prior parameters for the homefield advantage parameter
    real home_prior_mean;
    real<lower=0> home_prior_sd;
}
transformed data {
    // Cholesky decomp of the prior variance of the latent team strengths
    cholesky_factor_cov[nTeams] teams_prior_var_chol;
    
    // Precompute cholesky decomposition of the teams prior variance
    teams_prior_var_chol = cholesky_decompose(teams_prior_var);
}
parameters {
    // Latent "team strength" with a holdout
    vector[nTeams-1] teams_raw;
    
    // Baseline goals parameter, log scale
    real log_goals;
    
    // Homefield advantage, applied to home offense and home defense
    real home;
    
    // Covariance of beta (two components at a time) 
    // given teams, log_goals, home
    cov_matrix[2] teamvar;
    
    // Home/away offense/defense parameters, hierarchically defined from
    // latent team strength, log_goals, home, and teamvar
    vector[nTeams] homeoff;
    vector[nTeams] homedef;
    vector[nTeams] awayoff;
    vector[nTeams] awaydef;
}
transformed parameters {
    vector[nTeams] teams = append_row(teams_raw, -sum(teams_raw))
}
model {
    // Local Variables: Arrays of vectors for vectorizing hierarchical
    // distribution of beta
    vector[2] beta_stacked[2*nTeams];
    vector[2] beta_means[2*nTeams];
    
    // Log goals distributed by normal prior
    log_goals ~ normal(log_goals_prior_mean, log_goals_prior_sd);
    
    // Homefield advantage distributed by normal prior
    home ~ normal(home_prior_mean, home_prior_sd);
    
    // Team strengths distributed by a MVN prior
    teams ~ multi_normal_cholesky(teams_prior_mean, teams_prior_var_chol);
    
    // Hierarchical variance has Wishart prior
    teamvar ~ wishart(teamvar_prior_nu, teamvar_prior_Sigma);
    
    // Hierarchical distribution of beta
    // Build arrays of 2-vectors for vectorization
    for (t in 1:nTeams) {
        beta_means[2*t-1]   = [teams[t]+home, teams[t]]';
        beta_means[2*t]     = [teams[t]+home, teams[t]]';
        beta_stacked[2*t-1] = [homeoff[t], awayoff[t]]';
        beta_stacked[2*t]   = [homedef[t], awaydef[t]]';
    }
    // Vectorized sampling statement
    beta_stacked ~ multi_normal(beta_means, teamvar);
    
    // Distribution of goals
    if (nGames > 0) {
        homegoals ~ poisson_log(homeoff[hometeamidx] - awaydef[awayteamidx] + log_goals);
        awaygoals ~ poisson_log(awayoff[awayteamidx] - homedef[hometeamidx] + log_goals);
    }
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    
    if (nGames > 0) {
        homegoals_pred = poisson_log_rng(homeoff[hometeamidx] - awaydef[awayteamidx] + log_goals);
        awaygoals_pred = poisson_log_rng(awayoff[awayteamidx] - homedef[hometeamidx] + log_goals);
    }
    if (nGames_new > 0) {
        homegoals_new_pred = poisson_log_rng(homeoff[hometeamidx_new] - awaydef[awayteamidx_new] + log_goals);
        awaygoals_new_pred = poisson_log_rng(awayoff[awayteamidx_new] - homedef[hometeamidx_new] + log_goals);
    }
}
