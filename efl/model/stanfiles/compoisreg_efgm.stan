functions {
#include fn_bi_efgm.stan
#include fn_com_poisson.stan
    
    real com_poisson_efgm_lpmf(int[] x, real[] log_mu, real nu, int tp, real phi) {
        // All parameter checking done by the underlying distributional
        // functions
        real u[2]; // Lower and upper cdf for x[1]
        real v[2]; // Lower and upper cdf for x[2]
        real corners[4];
        // CDF for first x
        if (x[1] == 0) {
            u[1] = 0;
            u[2] = com_poisson_log_cdf(x[1], log_mu[1], nu, tp);
        } else {
            u = com_poisson_log_cdf_array({x[1]-1, x[1]}, log_mu[1], nu, tp);
        }
        // CDF for second x
        if (x[2] == 0) {
            v[1] = 0;
            v[2] = com_poisson_log_cdf(x[2], log_mu[2], nu, tp);
        } else {
            v = com_poisson_log_cdf_array({x[2]-1, x[2]}, log_mu[2], nu, tp);
        }
        // compute CDF of EFGM copula in bound area
        corners[1] = bi_efgm_cdf({u[1], v[1]}, phi);
        corners[2] = -bi_efgm_cdf({u[1], v[2]}, phi);
        corners[3] = -bi_efgm_cdf({u[2], v[1]}, phi);
        corners[4] = bi_efgm_cdf({u[2], v[2]}, phi);
        return log(sum(corners));
    }
    
    int[] com_poisson_efgm_rng(real[] log_mu, real nu, int tp, real phi) {
        real u[2] = bi_efgm_rng(phi);
        return { com_poisson_log_icdf(u[1], log_mu[1], nu, tp),
                 com_poisson_log_icdf(u[2], log_mu[2], nu, tp) };
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
    
    // Prior parameters for the concentration parameter
    real nu_prior_mean;
    real<lower=0> nu_prior_sd;
    real<lower=0> nu_lower_limit;
    
    // Maximum truncation point for distribution
    int<lower=(max(append_array(homegoals,awaygoals))+1)*100> truncpoint;
    
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
    real<lower=nu_lower_limit> nu;
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
    // Prior concentration parameter
    nu ~ normal(nu_prior_mean, nu_prior_sd) T[nu_lower_limit,];
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
            g ~ com_poisson_efgm({lmu_home[i], lmu_away[i]}, nu, truncpoint, phi);
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
    real rho;
    
    {
        vector[nGames] lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        vector[nGames] lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        vector[nGames_new] lmu_home_new = offense[hometeamidx_new] - defense[awayteamidx_new] + log_home_goals;
        vector[nGames_new] lmu_away_new = offense[awayteamidx_new] - defense[hometeamidx_new] + log_away_goals;
        int eff_trunc_home[nGames+nGames_new]; // Effective truncation points for home scores
        int eff_trunc_away[nGames+nGames_new]; // Same for away scores
        for (i in 1:nGames) {
            int gp[2] = com_poisson_efgm_rng({lmu_home[i], lmu_away[i]}, nu, truncpoint, phi);
            homegoals_pred[i] = gp[1];
            awaygoals_pred[i] = gp[2];
            // Effective truncation point of distribution
            eff_trunc_home[i] = com_poisson_truncpoint(lmu_home[i], nu, truncpoint);
            eff_trunc_away[i] = com_poisson_truncpoint(lmu_away[i], nu, truncpoint);
        }
        for (i in 1:nGames_new) {
            int gp[2] = com_poisson_efgm_rng({lmu_home_new[i], lmu_away_new[i]}, nu, truncpoint, phi);
            homegoals_new_pred[i] = gp[1];
            awaygoals_new_pred[i] = gp[2];
            // Effective truncation point of distribution
            eff_trunc_home[nGames + i] = com_poisson_truncpoint(lmu_home_new[i], nu, truncpoint);
            eff_trunc_away[nGames + i] = com_poisson_truncpoint(lmu_away_new[i], nu, truncpoint);
        }
        max_trunc = max(append_array(eff_trunc_home, eff_trunc_away));
        min_trunc = min(append_array(eff_trunc_home, eff_trunc_away));
    }
    // Alert for max_trunc == truncpoint
    if (max_trunc >= truncpoint) {
        print("Info: non-negligible probability at com_poisson truncation point.")
    }
    
    rho = phi / 3;
}
