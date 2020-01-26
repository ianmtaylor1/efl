functions {
    // Log normalizing constant for the Consul-Jain generalized Poisson
    // distribution, due to truncation when delta < 0
    // See Consul and Jain (1973) and follow-up papers
    real cj_log_norm(real lambda, real delta) {
        int truncpoint = 10; // Larger than this errors are negligible
        real logc;
        real m = positive_infinity();
        if (delta < 0) {
            // If delta is negative we need to calculate m
            m = -lambda/delta;
        }
        if (m > truncpoint) {
            // If truncation is large enough, series sums to 1 as expected
            // (this catches the case when delta is positive)
            logc = 0;
        } else { // Just sum up all the terms
            int x;
            real lprob;
            real lfac = 0;
            logc = -log(lambda); // x=0 term
            x = 1;
            while (x < m) {
                lfac += log(x);
                lprob =  (x - 1) * log(lambda + x * delta) - x * delta - lfac;
                logc = log_sum_exp(logc, lprob);
                x += 1;
            }
            logc += log(lambda) - lambda; // factored out components w/ no x
        }
        return logc;
    }
    // Log PMF function for the Consul-Jain generalized Poisson distribution
    // See Consul and Jain (1973)
    real consuljain_lpmf(int x, real mu, real theta) {
        // converted parameters
        real delta;
        real lambda;
        // convenient precomputation
        real lxd;
        // the gamut of integrity checks on parameters
        if (!(x >= 0)) {
            reject("consuljain_lpmf: x must be positive. ",
                   "(found x=", x, ")");
        }
        if (!(mu > 0)) {
            reject("consuljain_lpmf: mu must be positive. ",
                   "(found mu=", mu, ")");
        }
        if (!(theta > 0.25)) {
            reject("consuljain_lpmf: theta must be greater than 0.25. ",
                   " (found theta=", theta, ")");
        }
        // Calculate and return log pmf
        delta = 1 - 1 / sqrt(theta);
        lambda = mu / sqrt(theta);
        lxd = lambda + x * delta;
        if (lxd <= 0) {
            // Any x's such that this is negative have probability zero
            return negative_infinity();
        } else {
            return -lxd + log(lambda) + (x - 1) * log(lxd) - lgamma(x + 1) - cj_log_norm(lambda, delta);
        }
    }
    // Generate random numbers from the Consul-Jain generalized Poisson distribution
    int consuljain_rng(real mu, real theta) {
        real lu; // log of Uniform random variable
        real lcdf; // Keep track of total probability
        int x = 0; // Value that will eventually be returned
        real delta = 1 - 1 / sqrt(theta); // Standard second parameter
        real lambda = mu * (1 - delta); // Standard first parameter
        real logc = cj_log_norm(lambda, delta); // Log of normalizing constant
        real m = positive_infinity(); // Maximum allowable x value
        // Draw from uniform
        lu = log(uniform_rng(0.0,1.0));
        // Do we have a max?
        if (delta < 0) {
            m = -lambda/delta;
        }
        // Term for x=0
        lcdf = -lambda;
        // Accumulate probability until we're above u, then we stop
        {
            real log_lambda = log(lambda);
            real lfac = 0.0; // log factorial tracker
            while ((lcdf - logc < lu) && (x + 1 < m)) {
                real lprob;
                x += 1; 
                lfac += log(x);
                lprob = -lambda - delta*x + log_lambda + (x-1) * log(lambda + delta*x) - lfac;
                lcdf = log_sum_exp(lcdf, lprob);
            }
        }
        // Return the value
        return x;
    }
}
data {
#include std_data_goal.stan
    
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
        for (i in 1:nGames) {
            homegoals[i] ~ consuljain(mu_home[i], dispersion);
            awaygoals[i] ~ consuljain(mu_away[i], dispersion);
        }
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
