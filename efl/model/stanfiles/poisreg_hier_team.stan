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
    vector[nTeams-1] teams_prior_mean;
    cov_matrix[nTeams-1] teams_prior_var;
    
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
    // X matrices have one row for each game, and have columns grouped 4 per
    // team, ordered (HomeOff, AwayOff, HomeDef, AwayDef)
    matrix[nGames, 4*nTeams] X_home;
    matrix[nGames, 4*nTeams] X_away;
    matrix[nGames_new, 4*nTeams] X_home_new;
    matrix[nGames_new, 4*nTeams] X_away_new;
    // Cholesky decomp of the prior variance of the latent team strengths
    cholesky_factor_cov[nTeams-1] teams_prior_var_chol;
    
    // Precompute cholesky decomposition of the teams prior variance
    teams_prior_var_chol = cholesky_decompose(teams_prior_var);
    
    // Create X matrices and fill with appropriate indicators
    X_home = rep_matrix(0, nGames, 4*nTeams);
    X_away = rep_matrix(0, nGames, 4*nTeams);
    for (r in 1:nGames) {
        X_home[r,4*hometeamidx[r]-3] = 1;  // Home Offense for homegoals
        X_home[r,4*awayteamidx[r]-0] = -1; // Away Defense for homegoals
        X_away[r,4*awayteamidx[r]-2] = 1;  // Away Offense for awaygoals
        X_away[r,4*hometeamidx[r]-1] = -1; // Home Defense for awaygoals
    }
    X_home_new = rep_matrix(0, nGames_new, 4*nTeams);
    X_away_new = rep_matrix(0, nGames_new, 4*nTeams);
    for (r in 1:nGames_new) {
        X_home_new[r,4*hometeamidx[r]-3] = 1;  // Home Offense for homegoals
        X_home_new[r,4*awayteamidx[r]-0] = -1; // Away Defense for homegoals
        X_away_new[r,4*awayteamidx[r]-2] = 1;  // Away Offense for awaygoals
        X_away_new[r,4*hometeamidx[r]-1] = -1; // Home Defense for awaygoals
    }
}
parameters {
    // Latent "team strength" with a holdout
    vector[nTeams-1] teams;
    
    // Baseline goals parameter, log scale
    real log_goals;
    
    // Homefield advantage, applied to home offense and home defense
    real home;
    
    // Covariance of beta (two components at a time) 
    // given teams, log_goals, home
    cov_matrix[2] teamvar;
    
    // Home/away offense/defense parameters, hierarchically defined from
    // latent team strength, log_goals, home, and teamvar
    vector[4*nTeams] beta;
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
    // First, the first nTeams-1 teams
    for (t in 1:(nTeams-1)) {
        beta_means[2*t-1]   = [teams[t]+home, teams[t]]';
        beta_means[2*t]     = [teams[t]+home, teams[t]]';
        beta_stacked[2*t-1] = beta[(4*t-3):(4*t-2)];
        beta_stacked[2*t]   = beta[(4*t-1):(4*t)];
    }
    // Reference (nTeams^th) team
    beta_means[2*nTeams-1]   = [home, 0]';
    beta_means[2*nTeams]     = [home, 0]';
    beta_stacked[2*nTeams-1] = beta[(4*nTeams-3):(4*nTeams-2)];
    beta_stacked[2*nTeams]   = beta[(4*nTeams-1):(4*nTeams)];
    // Vectorized sampling statement
    beta_stacked ~ multi_normal(beta_means, teamvar);
    
    // Distribution of goals
    if (nGames > 0) {
        homegoals ~ poisson_log(X_home * beta + log_goals);
        awaygoals ~ poisson_log(X_away * beta + log_goals);
    }
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    
    if (nGames > 0) {
        homegoals_pred = poisson_log_rng(X_home * beta + log_goals);
        awaygoals_pred = poisson_log_rng(X_away * beta + log_goals);
    }
    if (nGames_new > 0) {
        homegoals_new_pred = poisson_log_rng(X_home_new * beta + log_goals);
        awaygoals_new_pred = poisson_log_rng(X_away_new * beta + log_goals);
    }
}
