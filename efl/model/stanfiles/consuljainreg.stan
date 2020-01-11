functions {
    // Log PMF function for the Consul-Jain generalized Poisson distribution
    // See Consul and Jain (1973)
    real consuljain_lpmf(int[] x, vector mu, real theta) {
        // how many elements do we have?
        int n = num_elements(x);
        vector[n] xv; // vectorized x for math
        // comverted parameters
        real delta;
        vector[n] lambda;
        // convenient precomputation
        vector[n] lxd;
        // the gamut of integrity checks on parameters, etc
        if (!(min(x) >= 0)) {
            reject("consuljain_lpmf: x must be positive. ",
                   "(found x=", min(x), ")");
        }
        if (!(min(mu) > 0)) {
            reject("consuljain_lpmf: mu must be positive. ",
                   "(found mu=", min(mu), ")");
        }
        if (!(theta > 0.25)) {
            reject("consuljain_lpmf: theta must be greater than 0.25. ",
                   " (found theta=", theta, ")");
        }
        // calculate and return log pmf
        xv = to_vector(x);
        delta = 1 - 1 / sqrt(theta);
        lambda = mu / sqrt(theta);
        lxd = lambda + xv * delta;
        if (min(lxd) <= 0) {
            // Any x's such that this condition holds have probability zero
            return negative_infinity();
        } else {
            return sum(-lxd + log(lambda) - lgamma(xv + 1)) + dot_product((xv - 1), log(lxd));
        }
    }
    // Generate random numbers from the Consul-Jain generalized Poisson distribution
    int consuljain_rng(real mu, real theta) {
        real u; // Uniform random variable to be used
        real cdf = 0.0; // Keep track of total probability
        int x = 0; // Value that will eventually be returned
        real delta = 1 - 1 / sqrt(theta); // Standard second parameter
        real lambda = mu * (1 - delta); // Standard first parameter
        real m = positive_infinity(); // Maximum allowable x value
        // Draw from uniform
        u = uniform_rng(0.0,1.0);
        // Do we have a max?
        if (delta < 0) {
            m = -lambda/delta;
        }
        // Accumulate probability until we're above u, then we stop
        while ((cdf < u) && (x < m)) {
            real lprob = -lambda - delta*x + log(lambda) + (x-1) * log(lambda + delta*x) - lgamma(x+1);
            cdf += exp(lprob);
            x += 1; 
        }
        // Go back one and return
        return x - 1;
    }
}
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
    
    // Prior parameters for the index of dispersion parameter
    real dispersion_prior_mean;
    real<lower=0> dispersion_prior_sd;
    
    // Prior parameters for team modifiers
    vector[nTeams]     offense_prior_mean;
    cov_matrix[nTeams] offense_prior_var;
    vector[nTeams]     defense_prior_mean;
    cov_matrix[nTeams] defense_prior_var;
}
transformed data {
    cholesky_factor_cov[nTeams] offense_prior_var_chol;
    cholesky_factor_cov[nTeams] defense_prior_var_chol;
    
    offense_prior_var_chol = cholesky_decompose(offense_prior_var);
    defense_prior_var_chol = cholesky_decompose(defense_prior_var);
}
parameters {
    // "raw" team modifiers (i.e. only the first nTeams-1 teams)
    vector[nTeams-1] offense_raw;
    vector[nTeams-1] defense_raw;
    // Baseline goals for home and away teams
    real log_home_goals;
    real log_away_goals;
    // Index of dispersion for goals
    // Has a theoretical lower bound of 0.25 (corresponds to delta = -1)
    real<lower=0.25> dispersion;
}
transformed parameters {
    // Transformed team modifiers, including nTeams'th component to add to 0
    vector[nTeams] offense = append_row(offense_raw, -sum(offense_raw));
    vector[nTeams] defense = append_row(defense_raw, -sum(defense_raw));
}
model {
    // Prior contribution from home/away goals
    log_home_goals ~ normal(log_home_goals_prior_mean, log_home_goals_prior_sd);
    log_away_goals ~ normal(log_away_goals_prior_mean, log_away_goals_prior_sd);
    // Prior contribution from team modifiers
    offense ~ multi_normal_cholesky(offense_prior_mean, offense_prior_var_chol);
    defense ~ multi_normal_cholesky(defense_prior_mean, defense_prior_var_chol);
    // Prior index of dispersion
    dispersion ~ normal(dispersion_prior_mean, dispersion_prior_sd) T[0.25,];
    // Model, goals follow Consul-Jain generalized Poisson distribution
    if (nGames > 0) {
        // local variables to hold means
        vector[nGames] mu_home;
        vector[nGames] mu_away;
        mu_home = exp(offense[hometeamidx] - defense[awayteamidx] + log_home_goals);
        mu_away = exp(offense[awayteamidx] - defense[hometeamidx] + log_away_goals);
        homegoals ~ consuljain(mu_home, dispersion);
        awaygoals ~ consuljain(mu_away, dispersion);
    }
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    {
        vector[nGames] mu_home = exp(offense[hometeamidx] - defense[awayteamidx] + log_home_goals);
        vector[nGames] mu_away = exp(offense[awayteamidx] - defense[hometeamidx] + log_away_goals);
        vector[nGames_new] mu_home_new = exp(offense[hometeamidx_new] - defense[awayteamidx_new] + log_home_goals);
        vector[nGames_new] mu_away_new = exp(offense[awayteamidx_new] - defense[hometeamidx_new] + log_away_goals);
        // Generate home/away scores for observed games
        for (i in 1:nGames) {
            homegoals_pred[i] = consuljain_rng(mu_home[i], dispersion);
            awaygoals_pred[i] = consuljain_rng(mu_away[i], dispersion);
        }
        // Generate home/away scores for unobserved games
        for (i in 1:nGames_new) {
            homegoals_new_pred[i] = consuljain_rng(mu_home_new[i], dispersion);
            awaygoals_new_pred[i] = consuljain_rng(mu_away_new[i], dispersion);
        }
    }
    
}
