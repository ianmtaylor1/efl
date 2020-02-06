functions {
#include fn_bi_efgm.stan
    
    real poisson_log_efgm_lpmf(int[] x, real[] log_lambda, real phi) {
        // All parameter checking done by the underlying distributional
        // functions
        real low[2];
        real high[2];
        real corners[4];
        // CDF at low endpoints
        if (x[1] == 0) {
            low[1] = 0;
        } else {
            low[1] = poisson_cdf(x[1]-1, exp(log_lambda[1]));
        }
        if (x[2] == 0) {
            low[2] = 0;
        } else {
            low[2] = poisson_cdf(x[2]-1, exp(log_lambda[2]));
        }
        // CDF at high endpoints
        high[1] = low[1] + exp(poisson_log_lpmf(x[1] | log_lambda[1]));
        high[2] = low[2] + exp(poisson_log_lpmf(x[2] | log_lambda[2]));
        // compute CDF of EFGM copula in bound area
        corners[1] = bi_efgm_cdf(low, phi);
        corners[2] = -bi_efgm_cdf({low[1], high[2]}, phi);
        corners[3] = -bi_efgm_cdf({high[1], low[2]}, phi);
        corners[4] = bi_efgm_cdf(high, phi);
        return log(sum(corners));
    }
    
    // Inverse CDF for Poisson (log parameterization)
    int poisson_log_icdf(real u, real log_lambda) {
        real lcdf;
        real log_u;
        int x;
        if (!((u <= 1) && (u >= 0))) {
            reject("poisson_icdf: u must be between 0 and 1. ",
                   "(found u=", u, ")");
        }
        lcdf = poisson_log_lpmf(0 | log_lambda);
        log_u = log(u);
        x = 0;
        while (lcdf < log_u) {
            x += 1;
            lcdf = log_sum_exp(lcdf, poisson_log_lpmf(x | log_lambda));
        }
        return x;
    }
    
    int[] poisson_log_efgm_rng(real[] log_lambda, real phi) {
        real u[2] = bi_efgm_rng(phi);
        return { poisson_log_icdf(u[1], log_lambda[1]),
                 poisson_log_icdf(u[2], log_lambda[2]) };
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
    // Prior contribution from home/away goals
    log_home_goals ~ normal(log_home_goals_prior_mean, log_home_goals_prior_sd);
    log_away_goals ~ normal(log_away_goals_prior_mean, log_away_goals_prior_sd);
    // Prior contribution from team modifiers
    offense ~ multi_normal_cholesky(offense_prior_mean, offense_prior_var_chol);
    defense ~ multi_normal_cholesky(defense_prior_mean, defense_prior_var_chol);
    // Model, goals follow poisson distribution
    {
        vector[nGames] log_lambda_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        vector[nGames] log_lambda_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        for (i in 1:nGames) {
            int g[2] = {homegoals[i], awaygoals[i]};
            g ~ poisson_log_efgm({log_lambda_home[i], log_lambda_away[i]}, phi);
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
        vector[nGames] log_lambda_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        vector[nGames] log_lambda_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        for (i in 1:nGames) {
            int gp[2] = poisson_log_efgm_rng({log_lambda_home[i], log_lambda_away[i]}, phi);
            homegoals_pred[i] = gp[1];
            awaygoals_pred[i] = gp[2];
        }
    }
    {
        vector[nGames_new] log_lambda_home = offense[hometeamidx_new] - defense[awayteamidx_new] + log_home_goals;
        vector[nGames_new] log_lambda_away = offense[awayteamidx_new] - defense[hometeamidx_new] + log_away_goals;
        for (i in 1:nGames_new) {
            int gp[2] = poisson_log_efgm_rng({log_lambda_home[i], log_lambda_away[i]}, phi);
            homegoals_new_pred[i] = gp[1];
            awaygoals_new_pred[i] = gp[2];
        }
    }
    
    rho = phi / 3;
}
