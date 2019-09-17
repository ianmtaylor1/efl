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
    
    // Prior parameters for the log baseline goals parameter
    real log_goals_prior_mean;
    real<lower=0> log_goals_prior_sd;
    
    // Prior parameters for the homefield advantage parameter
    real home_prior_mean;
    real<lower=0> home_prior_sd;
    
    // Prior parameters for the variance of the hierarchical team
    // offense/defense home/away parameters, given the latent team strength
    // and homefield advantage
    real<lower=0> sigma2_prior_alpha;
    real<lower=0> sigma2_prior_beta;
    
    // Prior parameters for the variance of homefield advantage by team
    real<lower=0> s2home_prior_alpha;
    real<lower=0> s2home_prior_beta;
    
    // Prior parameters for the correlation between off/def subparameters
    real<lower=0> rho_prior_alpha;
    real<lower=0> rho_prior_beta;
}
transformed data {
    // Cholesky decomp of the prior variance of the latent team strengths
    cholesky_factor_cov[nTeams] teams_prior_var_chol;
    
    // Precompute cholesky decomposition of the teams prior variance
    teams_prior_var_chol = cholesky_decompose(teams_prior_var);
}
parameters {
    // First n-1 components of teams vector (n-1 degrees of freedom)
    vector[nTeams-1] teams_raw;
    
    // Baseline goals parameter, log scale
    real log_goals;
    
    // Homefield advantage, applied to home offense and home defense
    real home;
    
    // Variance of how far the hierarchical team parameters can vary from the
    // latent team strength
    real<lower=0> sigma2;
    
    // Variance of the difference in homefield advantage between teams
    real<lower=0> s2home;
    
    // Correlation between home/away offense and home/away defense
    real<lower=-1,upper=1> rho;
    
    // Home/away offense/defense parameters, hierarchically defined from
    // latent team strength, log_goals, home, and teamvar
    vector[nTeams] homeoff;
    vector[nTeams] homedef;
    vector[nTeams] awayoff;
    vector[nTeams] awaydef;
}
transformed parameters {
    // Teams vector centered at zero
    vector[nTeams] teams = append_row(teams_raw, -sum(teams_raw));
}
model {
    // Local Variables: Arrays of vectors for vectorizing hierarchical
    // distribution of beta
    vector[4] beta_stacked[nTeams];
    vector[4] beta_means[nTeams];
    // sigma2 and rho assembled into a 4x4 covariance matrix
    matrix[4,4] teamvar;
    
    // Log goals distributed by normal prior
    log_goals ~ normal(log_goals_prior_mean, log_goals_prior_sd);
    
    // Homefield advantage distributed by normal prior
    home ~ normal(home_prior_mean, home_prior_sd);
    
    // Team strengths distributed by a MVN prior
    teams ~ multi_normal_cholesky(teams_prior_mean, teams_prior_var_chol);
    
    // Hierarchical variance has gamma prior
    sigma2 ~ gamma(sigma2_prior_alpha, sigma2_prior_beta);
    
    // Homefield variance has gamma prior
    s2home ~ gamma(s2home_prior_alpha, s2home_prior_beta);
    
    // Offense and defense correlation has (shifted,scaled) beta prior
    // Linear transformation, fine to do without Jacobian
    (rho + 1)/2 ~ beta(rho_prior_alpha, rho_prior_beta);
    
    // Hierarchical distribution of beta
    // Build arrays of 2-vectors for vectorization
    for (t in 1:nTeams) {
        beta_means[t]   = [teams[t]+home, teams[t], teams[t]+home, teams[t]]';
        beta_stacked[t] = [homeoff[t], awayoff[t], homedef[t], awaydef[t]]';
    }
    // Build variance matrix
    teamvar = [[sigma2+s2home , rho*sigma2 , s2home        , 0         ],
               [rho*sigma2    , sigma2     , 0             , 0         ],
               [s2home        , 0          , sigma2+s2home , rho*sigma2],
               [0             , 0          , rho*sigma2    , sigma2    ]];
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
