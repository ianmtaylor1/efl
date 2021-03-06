functions {
#include fn_consuljain2.stan
}
data {
#include std_data_goal.stan
    
    // Prior parameters for home and away goals
    real          log_home_goals_prior_mean;
    real<lower=0> log_home_goals_prior_sd;
    real          log_away_goals_prior_mean;
    real<lower=0> log_away_goals_prior_sd;
    
    // Prior parameters for the index of dispersion parameter
    real delta_prior_mean;
    real<lower=0> delta_prior_sd;
    
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
    // Dispersion parameter for goals
    real<lower=-1, upper=1> delta;
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
    // Prior dispersion
    delta ~ normal(delta_prior_mean, delta_prior_sd) T[-1, 1];
    // Model, goals follow Consul-Jain generalized Poisson distribution
    if (nGames > 0) {
        // local variables to hold means
        vector[nGames] lmu_home;
        vector[nGames] lmu_away;
        lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        for (i in 1:nGames) {
            homegoals[i] ~ consuljain(lmu_home[i], delta);
            awaygoals[i] ~ consuljain(lmu_away[i], delta);
        }
    }
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    real dispersion; // index of dispersion based on delta
    {
        vector[nGames] lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        vector[nGames] lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        vector[nGames_new] lmu_home_new = offense[hometeamidx_new] - defense[awayteamidx_new] + log_home_goals;
        vector[nGames_new] lmu_away_new = offense[awayteamidx_new] - defense[hometeamidx_new] + log_away_goals;
        // Generate home/away scores for observed games
        for (i in 1:nGames) {
            homegoals_pred[i] = consuljain_rng(lmu_home[i], delta);
            awaygoals_pred[i] = consuljain_rng(lmu_away[i], delta);
        }
        // Generate home/away scores for unobserved games
        for (i in 1:nGames_new) {
            homegoals_new_pred[i] = consuljain_rng(lmu_home_new[i], delta);
            awaygoals_new_pred[i] = consuljain_rng(lmu_away_new[i], delta);
        }
    }
    dispersion = inv_square(1 - delta);
}
