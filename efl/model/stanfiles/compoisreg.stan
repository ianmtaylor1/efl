functions {
#include fn_com_poisson.stan
}
data {
#include std_data_goal.stan
    
    // Prior parameters for home and away goals
    real          log_home_goals_prior_mean;
    real<lower=0> log_home_goals_prior_sd;
    real          log_away_goals_prior_mean;
    real<lower=0> log_away_goals_prior_sd;
    
    // Prior parameters for the dispersion/decay parameter
    real nu_prior_mean;
    real<lower=0> nu_prior_sd;
    real<lower=0> nu_lower_limit;
    
    // Prior parameters for team modifiers
    vector[nTeams]     offense_prior_mean;
    cov_matrix[nTeams] offense_prior_var;
    vector[nTeams]     defense_prior_mean;
    cov_matrix[nTeams] defense_prior_var;
    
    // Maximum truncation point for distribution
    int<lower=(max(append_array(append_array(homegoals, awaygoals), {0})) + 1) * 100> truncpoint;
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
    // Dispersion-controlling parameter for goals
    // nu=1 -> Poisson distribution. Higher = lower dispersion
    // controls the "rate of decay" in the COM-Poisson distribution
    real<lower=nu_lower_limit> nu;
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
    // Prior dispersion/decay parameter
    nu ~ normal(nu_prior_mean, nu_prior_sd) T[nu_lower_limit,];
    // Model, goals follow COM-Poisson distribution
    if (nGames > 0) {
        // local variables to hold means
        vector[nGames] lmu_home;
        vector[nGames] lmu_away;
        lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        for (i in 1:nGames) {
            homegoals[i] ~ com_poisson_log(lmu_home[i], nu, truncpoint);
            awaygoals[i] ~ com_poisson_log(lmu_away[i], nu, truncpoint);
        }
    }
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    int max_trunc; // Highest effective truncation point for this sample
    int min_trunc; // Lowest effective truncation point for this sample
    
    {
        vector[nGames] lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        vector[nGames] lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        vector[nGames_new] lmu_home_new = offense[hometeamidx_new] - defense[awayteamidx_new] + log_home_goals;
        vector[nGames_new] lmu_away_new = offense[awayteamidx_new] - defense[hometeamidx_new] + log_away_goals;
        int eff_trunc_home[nGames+nGames_new]; // Effective truncation points for home scores
        int eff_trunc_away[nGames+nGames_new]; // Same for away scores
        // Generate home/away scores for observed games
        for (i in 1:nGames) {
            homegoals_pred[i] = com_poisson_log_rng(lmu_home[i], nu, truncpoint);
            awaygoals_pred[i] = com_poisson_log_rng(lmu_away[i], nu, truncpoint);
        }
        // Generate home/away scores for unobserved games
        for (i in 1:nGames_new) {
            homegoals_new_pred[i] = com_poisson_log_rng(lmu_home_new[i], nu, truncpoint);
            awaygoals_new_pred[i] = com_poisson_log_rng(lmu_away_new[i], nu, truncpoint);
        }
        // Effective truncation point of distribution for observed games
        for (i in 1:nGames) {
            eff_trunc_home[i] = com_poisson_truncpoint(lmu_home[i], nu, truncpoint);
            eff_trunc_away[i] = com_poisson_truncpoint(lmu_away[i], nu, truncpoint);
        }
        // Effective truncation point of distribution for unobserved games
        for (i in 1:nGames_new) {
            eff_trunc_home[nGames + i] = com_poisson_truncpoint(lmu_home_new[i], nu, truncpoint);
            eff_trunc_away[nGames + i] = com_poisson_truncpoint(lmu_away_new[i], nu, truncpoint);
        }
        max_trunc = max(append_array(eff_trunc_home, eff_trunc_away));
        min_trunc = min(append_array(eff_trunc_home, eff_trunc_away));
        // Alert for max_trunc == truncpoint
        if (max_trunc >= truncpoint) {
            print("*** Info: non-negligible probability at com_poisson truncation point.")
        } 
    }
}
