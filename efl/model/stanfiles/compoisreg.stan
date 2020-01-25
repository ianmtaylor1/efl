// The COM Poisson functions are taken from Paul-Christian Bürkner's 'brms'
// project: https://github.com/paul-buerkner/brms
// As of 2020-01-24 these functions aren't officially supported in brms, but
// are available on Github. They've been linked on Stan discussions:
// https://discourse.mc-stan.org/t/conway-maxwell-poisson-distribution/2370
// They have since been modified by me to fit my use case here.
functions {
    // log approximate normalizing constant of the COM poisson distribuion
    // approximation based on doi:10.1007/s10463-017-0629-6
    // Args: see log_Z_com_poisson()
    real log_Z_com_poisson_approx(real log_mu, real nu) {
        real nu_mu = nu * exp(log_mu); 
        real nu2 = nu^2;
        // first 4 terms of the residual series
        real log_sum_resid = log(
            1 + nu_mu^(-1) * (nu2 - 1) / 24 + 
            nu_mu^(-2) * (nu2 - 1) / 1152 * (nu2 + 23) +
            nu_mu^(-3) * (nu2 - 1) / 414720 * (5 * nu2^2 - 298 * nu2 + 11237)
            );
        return nu_mu + log_sum_resid  - 
            ((log(2 * pi()) + log_mu) * (nu - 1) / 2 + log(nu) / 2);
    }
    // log normalizing constant of the COM Poisson distribution
    // implementation inspired by code of Ben Goodrich
    // Args:
    //   log_mu: log location parameter
    //   shape: positive shape parameter
    real log_Z_com_poisson(real log_mu, real nu) {
        real log_Z;
        real lfac;
        real term;
        real k;
        int M;
        real log_thres;
        if (nu == 1) {
            return exp(log_mu);
        }
        // nu == 0 or Inf will fail in this parameterization
        if (nu <= 0) {
            reject("log_Z_com_poisson: nu must be positive. ",
                   "(found nu=", nu, ")");
        }
        if (nu == positive_infinity()) {
            reject("log_Z_com_poisson: nu must be finite")
        }
        // direct computation of the truncated series
        M = 1000;
        log_thres = log(1e-8);
        // check if the Mth term of the series is small enough
        if (log_mu > log(M/2)) {
            reject("log_Z_com_poisson: log_mu is too large. ",
                   "(found log_mu=", log_mu, ")")
        }
        if (nu * (M * log_mu - lgamma(M + 1)) > log_thres) {
            reject("log_Z_com_poisson: nu is too close to zero. ",
                   "(found nu=", nu, ")");
        }
        log_Z = log1p_exp(nu * log_mu);  // first 2 terms of the series
        lfac = 0;
        term = 0;
        k = 2;
        while ((log(k) < log_mu) || (term > log_thres)) { 
            lfac += log(k);
            term = nu * (k * log_mu - lfac);
            log_Z = log_sum_exp(log_Z, term);
            k += 1;
        }
        return log_Z;
    }
    // COM Poisson log-PMF for a single response (log parameterization)
    // Args: 
    //   y: the response value 
    //   log_mu: log location parameter
    //   nu: positive shape parameter
    real com_poisson_log_lpmf(int[] y, vector log_mu, real nu) {
        int n = num_elements(y);
        vector[n] yvec;
        vector[n] log_Z;
        if (nu == 1) return poisson_log_lpmf(y | log_mu);
        yvec = to_vector(y);
        for (i in 1:n) {
            log_Z[i] = log_Z_com_poisson(log_mu[i], nu);
        }
        return nu * dot_product(yvec, log_mu) - sum(nu * lgamma(yvec + 1) + log_Z);
    }
    // Random number generator for COM Poisson
    int com_poisson_log_rng(real log_mu, real nu) {
        real log_num;  // log numerator
        real log_Z;  // log denominator
        int x;  // number that will eventually be returned
        real lu; // log of uniform random variable between 0 and logZ
        if (nu == 1) {
            // Simplify if we just have poisson (rare)
            return poisson_log_rng(log_mu);
        }
        // Find the normalizing constant, Z
        log_Z = log_Z_com_poisson(log_mu, nu);
        // Draw a uniform r.v. to transform via inverse cdf
        lu = log(uniform_rng(0.0, 1.0));
        // Start at x=0, and set the appropriate log numerator
        x = 0;
        log_num = 0;
        {
            real lfac = 0.0; // log factorial tracker
            while (log_num - log_Z < lu) {
                // Keep incrementing the numerator until we exceed the random value
                x += 1;
                lfac += log(x);
                log_num = log_sum_exp(log_num, nu * (x * log_mu - lfac));
            }
        }
        return x;
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
    
    // Prior parameters for the dispersion/decay parameter
    real<lower=0> nu_prior_mu;
    real<lower=0> nu_prior_sigma;
    
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
    // Dispersion-controlling parameter for goals
    // nu=1 -> Poisson distribution. Higher = lower dispersion
    // controls the "rate of decay" in the COM-Poisson distribution
    real<lower=0> nu;
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
    nu ~ lognormal(nu_prior_mu, nu_prior_sigma);
    // Model, goals follow Consul-Jain generalized Poisson distribution
    if (nGames > 0) {
        // local variables to hold means
        vector[nGames] lmu_home;
        vector[nGames] lmu_away;
        lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        homegoals ~ com_poisson_log(lmu_home, nu);
        awaygoals ~ com_poisson_log(lmu_away, nu);
    }
}
generated quantities {
    int<lower=0> homegoals_pred[nGames];
    int<lower=0> awaygoals_pred[nGames];
    int<lower=0> homegoals_new_pred[nGames_new];
    int<lower=0> awaygoals_new_pred[nGames_new];
    {
        vector[nGames] lmu_home = offense[hometeamidx] - defense[awayteamidx] + log_home_goals;
        vector[nGames] lmu_away = offense[awayteamidx] - defense[hometeamidx] + log_away_goals;
        vector[nGames_new] lmu_home_new = offense[hometeamidx_new] - defense[awayteamidx_new] + log_home_goals;
        vector[nGames_new] lmu_away_new = offense[awayteamidx_new] - defense[hometeamidx_new] + log_away_goals;
        // Generate home/away scores for observed games
        for (i in 1:nGames) {
            homegoals_pred[i] = com_poisson_log_rng(lmu_home[i], nu);
            awaygoals_pred[i] = com_poisson_log_rng(lmu_away[i], nu);
        }
        // Generate home/away scores for unobserved games
        for (i in 1:nGames_new) {
            homegoals_new_pred[i] = com_poisson_log_rng(lmu_home_new[i], nu);
            awaygoals_new_pred[i] = com_poisson_log_rng(lmu_away_new[i], nu);
        }
    }
    
}
