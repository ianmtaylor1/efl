data {
#include std_data_result.stan
    
    // Dimenstion of multiplicative matchup effects
    int<lower=2,upper=nTeams> uvdim;
    
    // Prior mean and variance for teams parameters
    vector[nTeams] teams_prior_mean;
    cov_matrix[nTeams] teams_prior_var;
    
    // Prior location and scale for win/draw threshold parameter
    real theta_prior_loc;
    real<lower=0> theta_prior_scale;

    // Prior mean and sd for homefield advantage parameter
    real home_prior_mean;
    real<lower=0> home_prior_sd;
    
    // Scale applied to multiplicative effect magnitudes
    real<lower=0> uvscale;
}
transformed data {
    // Cholesky decomposition of prior teams variance
    cholesky_factor_cov[nTeams] teams_prior_var_chol;
    teams_prior_var_chol = cholesky_decompose(teams_prior_var);
}
parameters {
    // Team strength parameters
    vector[nTeams-1] teams_raw;
    // Homefield advantage
    real home;
    // Boundary parameter for draws
    real<lower=0> theta;
    // Multiplicative effects
    cholesky_factor_corr[uvdim] U_tri; // "offense styles" sort of
    unit_vector[uvdim] U_vec[nTeams - uvdim];
    unit_vector[uvdim] V_vec[nTeams]; // "defense styles" sort of
    // Scale of multiplicative effects: hierarchically combine to create
    // horseshoe-like distribution (hopefully shrinking multiplicative effects
    // to near zero when not necessary)
    vector<lower=0>[nTeams] sigma;
    vector<lower=0>[nTeams] mult_var;
}
transformed parameters {
    // Full team strengths, constrained to sum to zero
    vector[nTeams] teams;
    // Matrices containing offense and defense multiplicative effects
    matrix[nTeams,uvdim] U;
    matrix[nTeams,uvdim] V;
    // Multiplicative matchup effects. Equals the negative of its transpose.
    matrix[nTeams,nTeams] matchup;
    
    teams = append_row(teams_raw, -sum(teams_raw));
    for (i in 1:nTeams) {
        if (i <= uvdim) {
            U[i] = U_tri[i]';
        } else {
            U[i] = U_vec[i-uvdim]';
        }
        V[i] = V_vec[i]';
    }
    matchup = quad_form_diag(tcrossprod(U,V) - tcrossprod(V,U), sqrt(mult_var));
}
model {
    vector[nGames] apply_matchup;
    // Priors
    theta ~ logistic(theta_prior_loc, theta_prior_scale) T[0,];
    home ~ normal(home_prior_mean, home_prior_sd);
    teams ~ multi_normal_cholesky(teams_prior_mean, teams_prior_var_chol);
    // Multiplicative matchup effect priors
    // LKJ(0.5) results in the same distribution of 'angles' as would have
    // from just uniformly distributed unit vectors
    U_tri ~ lkj_corr_cholesky(0.5);
    sigma ~ cauchy(0, uvscale/sqrt(uvdim)) T[0,];
    mult_var ~ gamma(uvdim/2, 1/(2 * sigma^2)); // scaled chi-square given sigma
    // Model
    for (i in 1:nGames) {
        apply_matchup[i] = matchup[hometeamidx[i], awayteamidx[i]];
    }
    if (nGames > 0) {
        result ~ ordered_logistic(
                teams[hometeamidx] - teams[awayteamidx] + apply_matchup + home, 
                [ -theta, theta ]'
                );
    }
}
generated quantities {
    int<lower=1,upper=3> result_pred[nGames];
    int<lower=1,upper=3> result_new_pred[nGames_new];
    matrix[nTeams, nTeams] offense_similarity;
    matrix[nTeams, nTeams] defense_similarity;
    // Posterior predictions for observed games
    for (i in 1:nGames) {
        result_pred[i] = ordered_logistic_rng(
                teams[hometeamidx[i]] - teams[awayteamidx[i]] + matchup[hometeamidx[i], awayteamidx[i]] + home, 
                [ -theta, theta ]'
                );
    };
    // Posterior predictions for unobserved games
    for (i in 1:nGames_new) {
        result_new_pred[i] = ordered_logistic_rng(
                teams[hometeamidx_new[i]] - teams[awayteamidx_new[i]] + matchup[hometeamidx_new[i], awayteamidx_new[i]] + home, 
                [ -theta, theta ]'
                );
    };
    // Create offense and defense similarity
    offense_similarity = tcrossprod(U);
    defense_similarity = tcrossprod(V);
}
