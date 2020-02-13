functions {
#include fn_bi_efgm.stan
#include fn_consuljain2.stan
    
    real consuljain_efgm_lpmf(int[] x, real[] log_mu, real delta, real phi) {
        // All parameter checking done by the underlying distributional
        // functions
        real u[2]; // Lower and upper cdf for x[1]
        real v[2]; // Lower and upper cdf for x[2]
        real corners[4];
        // CDF for first x
        if (x[1] == 0) {
            u[1] = 0;
            u[2] = consuljain_cdf(x[1], log_mu[1], delta);
        } else {
            u = consuljain_cdf_array({x[1]-1, x[1]}, log_mu[1], delta);
        }
        // CDF for second x
        if (x[2] == 0) {
            v[1] = 0;
            v[2] = consuljain_cdf(x[2], log_mu[2], delta);
        } else {
            v = consuljain_cdf_array({x[2]-1, x[2]}, log_mu[2], delta);
        }
        // compute CDF of EFGM copula in bound area
        corners[1] = bi_efgm_cdf({u[1], v[1]}, phi);
        corners[2] = -bi_efgm_cdf({u[1], v[2]}, phi);
        corners[3] = -bi_efgm_cdf({u[2], v[1]}, phi);
        corners[4] = bi_efgm_cdf({u[2], v[2]}, phi);
        return log(sum(corners));
    }
    
    int[] consuljain_efgm_rng(real[] log_mu, real delta, real phi) {
        real u[2] = bi_efgm_rng(phi);
        return { consuljain_icdf(u[1], log_mu[1], delta),
                 consuljain_icdf(u[2], log_mu[2], delta) };
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
    
    // Prior parameters for the dispersion parameter
    real delta_prior_mean;
    real<lower=0> delta_prior_sd;
    
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
    // Dispersion for goals
    real<lower=-1, upper=1> delta;
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
    delta ~ normal(delta_prior_mean, delta_prior_sd) T[-1, 1];
    // Prior contribution from home/away goals
    log_home_goals ~ normal(log_home_goals_prior_mean, log_home_goals_prior_sd);
    log_away_goals ~ normal(log_away_goals_prior_mean, log_away_goals_prior_sd);
    // Prior contribution from team modifiers
    offense ~ multi_normal_cholesky(offense_prior_mean, offense_prior_var_chol);
    defense ~ multi_normal_cholesky(defense_prior_mean, defense_prior_var_chol);
    // Model, goals follow poisson distribution
    {
        vector[nGames] lmu_home;
        vector[nGames] lmu_away;
        lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        for (i in 1:nGames) {
            int g[2] = {homegoals[i], awaygoals[i]};
            g ~ consuljain_efgm({lmu_home[i], lmu_away[i]}, delta, phi);
        }
    }
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    real rho;
    real dispersion;
    
    {
        vector[nGames] lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        vector[nGames] lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        for (i in 1:nGames) {
            int gp[2] = consuljain_efgm_rng({lmu_home[i], lmu_away[i]}, delta, phi);
            homegoals_pred[i] = gp[1];
            awaygoals_pred[i] = gp[2];
        }
    }
    {
        vector[nGames_new] lmu_home_new = offense[hometeamidx_new] - defense[awayteamidx_new] + log_home_goals;
        vector[nGames_new] lmu_away_new = offense[awayteamidx_new] - defense[hometeamidx_new] + log_away_goals;
        for (i in 1:nGames_new) {
            int gp[2] = consuljain_efgm_rng({lmu_home_new[i], lmu_away_new[i]}, delta, phi);
            homegoals_new_pred[i] = gp[1];
            awaygoals_new_pred[i] = gp[2];
        }
    }
    
    rho = phi / 3;
    dispersion = inv_square(1 - delta);
}
