functions {
#include fn_bi_efgm.stan
#include fn_consuljain.stan
    
    real consuljain_efgm_lpmf(int[] x, real[] mu, real theta, real phi) {
        // All parameter checking done by the underlying distributional
        // functions
        real u[2]; // Lower and upper cdf for x[1]
        real v[2]; // Lower and upper cdf for x[2]
        real corners[4];
        // CDF for first x
        if (x[1] == 0) {
            u[1] = 0;
            u[2] = consuljain_cdf(x[1], mu[1], theta);
        } else {
            u = consuljain_cdf_array({x[1]-1, x[1]}, mu[1], theta);
        }
        // CDF for second x
        if (x[2] == 0) {
            v[1] = 0;
            v[2] = consuljain_cdf(x[2], mu[2], theta);
        } else {
            v = consuljain_cdf_array({x[2]-1, x[2]}, mu[2], theta);
        }
        // compute CDF of EFGM copula in bound area
        corners[1] = bi_efgm_cdf({u[1], v[1]}, phi);
        corners[2] = -bi_efgm_cdf({u[1], v[2]}, phi);
        corners[3] = -bi_efgm_cdf({u[2], v[1]}, phi);
        corners[4] = bi_efgm_cdf({u[2], v[2]}, phi);
        return log(sum(corners));
    }
    
    int[] consuljain_efgm_rng(real[] mu, real theta, real phi) {
        real u[2] = bi_efgm_rng(phi);
        return { consuljain_icdf(u[1], mu[1], theta),
                 consuljain_icdf(u[2], mu[2], theta) };
    }
}
data {
#include std_data_goal.stan
    
    // Prior parameters for home and away goals
    real          log_home_goals_prior_mean;
    real<lower=0> log_home_goals_prior_sd;
    real          log_away_goals_prior_mean;
    real<lower=0> log_away_goals_prior_sd;
    
    // Prior parameters for team modifiers
    vector[nTeams]     offense_prior_mean;
    cov_matrix[nTeams] offense_prior_var;
    vector[nTeams]     defense_prior_mean;
    cov_matrix[nTeams] defense_prior_var;
    
    // Prior parameters for the index of dispersion parameter
    real dispersion_prior_mean;
    real<lower=0> dispersion_prior_sd;
    
    // Prior parameters for the inter-goals correlation
    real<lower=-1, upper=1> phi_prior_mean;
    real<lower=0> phi_prior_sd;
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
    // EFGM copula parameter
    real<lower=-1, upper=1> phi;
}
transformed parameters {
    // Transformed team modifiers, including nTeams'th component to add to 0
    vector[nTeams] offense;
    vector[nTeams] defense;
    
    offense = append_row(offense_raw, -sum(offense_raw));
    defense = append_row(defense_raw, -sum(defense_raw));
}
model {
    // Prior EFGM
    phi ~ normal(phi_prior_mean, phi_prior_sd) T[-1, 1];
    // Prior index of dispersion
    dispersion ~ normal(dispersion_prior_mean, dispersion_prior_sd) T[0.25,];
    // Prior contribution from home/away goals
    log_home_goals ~ normal(log_home_goals_prior_mean, log_home_goals_prior_sd);
    log_away_goals ~ normal(log_away_goals_prior_mean, log_away_goals_prior_sd);
    // Prior contribution from team modifiers
    offense ~ multi_normal_cholesky(offense_prior_mean, offense_prior_var_chol);
    defense ~ multi_normal_cholesky(defense_prior_mean, defense_prior_var_chol);
    // Model, goals follow poisson distribution
    {
        vector[nGames] mu_home;
        vector[nGames] mu_away;
        mu_home = exp(offense[hometeamidx] - defense[awayteamidx] + log_home_goals);
        mu_away = exp(offense[awayteamidx] - defense[hometeamidx] + log_away_goals);
        for (i in 1:nGames) {
            int g[2] = {homegoals[i], awaygoals[i]};
            g ~ consuljain_efgm({mu_home[i], mu_away[i]}, dispersion, phi);
        }
    }
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    real rho;
    
    {
        vector[nGames] mu_home = exp(offense[hometeamidx] - defense[awayteamidx] + log_home_goals);
        vector[nGames] mu_away = exp(offense[awayteamidx] - defense[hometeamidx] + log_away_goals);
        for (i in 1:nGames) {
            int gp[2] = consuljain_efgm_rng({mu_home[i], mu_away[i]}, dispersion, phi);
            homegoals_pred[i] = gp[1];
            awaygoals_pred[i] = gp[2];
        }
    }
    {
        vector[nGames_new] mu_home_new = exp(offense[hometeamidx_new] - defense[awayteamidx_new] + log_home_goals);
        vector[nGames_new] mu_away_new = exp(offense[awayteamidx_new] - defense[hometeamidx_new] + log_away_goals);
        for (i in 1:nGames_new) {
            int gp[2] = consuljain_efgm_rng({mu_home_new[i], mu_away_new[i]}, dispersion, phi);
            homegoals_new_pred[i] = gp[1];
            awaygoals_new_pred[i] = gp[2];
        }
    }
    
    rho = phi / 3;
}
