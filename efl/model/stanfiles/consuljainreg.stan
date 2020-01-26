functions {
#include fn_consuljain.stan
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
