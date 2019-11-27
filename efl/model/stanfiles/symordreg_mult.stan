data {
    // Number of games and number of teams
    int<lower=0> nGames;
    int<lower=1> nTeams;
    
    // Describe the known games and their outcomes
    int<lower=1,upper=nTeams> hometeamidx[nGames];
    int<lower=1,upper=nTeams> awayteamidx[nGames];
    int<lower=1,upper=3> result[nGames];
    
    // Number of new games and description of new games
    int<lower=0> nGames_new;
    int<lower=1,upper=nTeams> hometeamidx_new[nGames_new];
    int<lower=1,upper=nTeams> awayteamidx_new[nGames_new];
    
    // Dimension of multiplicative team matchup effects
    int<lower=1> uvdim;
    
    // Prior mean and variance for teams parameters
    vector[nTeams] teams_prior_mean;
    cov_matrix[nTeams] teams_prior_var;
    
    // Prior location and scale for win/draw threshold parameter
    real theta_prior_loc;
    real<lower=0> theta_prior_scale;

    // Prior mean and sd for homefield advantage parameter
    real home_prior_mean;
    real<lower=0> home_prior_sd;
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
    // Variance parameters for multiplicative effects
    positive_ordered[uvdim] uvscale;
    cholesky_factor_corr[2*uvdim] uvcorr_chol;
    // Multiplicative effect matrices
    matrix[nTeams,uvdim] U; // "offense styles" sort of
    matrix[nTeams,uvdim] V; // "defense styles" sort of
}
transformed parameters {
    // Full team strengths, constrained to sum to zero
    vector[nTeams] teams = append_row(teams_raw, -sum(teams_raw));
    // Multiplicative matchup effects. Equals the negative of its transpose.
    matrix[nTeams,nTeams] matchup = (U * V') - (V * U');
}
model {
    matrix[2*uvdim,2*uvdim] uvcov_chol;
    row_vector[2*uvdim] uv_vec[nTeams];
    vector[nGames] apply_matchup;
    // Priors
    theta ~ logistic(theta_prior_loc, theta_prior_scale) T[0,];
    home ~ normal(home_prior_mean, home_prior_sd);
    teams ~ multi_normal_cholesky(teams_prior_mean, teams_prior_var_chol);
    // Multiplicative matchup effect priors
    uvscale ~ cauchy(0.0, 1.0);
    uvcorr_chol ~ lkj_corr_cholesky(1.0);
    uvcov_chol = diag_pre_multiply(append_row(uvscale,uvscale), uvcorr_chol);
    for (i in 1:nTeams) {
        uv_vec[i] = append_col(U[i], V[i]);
    }
    uv_vec ~ multi_normal_cholesky(rep_row_vector(0.0, 2*uvdim), uvcov_chol);
    // Model
    for (i in 1:nGames) {
        apply_matchup[i] = matchup[hometeamidx[i], awayteamidx[i]];
    }
    if (nGames > 0) {
        result ~ ordered_logistic(
                teams[hometeamidx] - teams[awayteamidx] + apply_matchup + home, 
                [ -theta, theta ]'
                );
    };
}
generated quantities {
    int<lower=1,upper=3> result_pred[nGames];
    int<lower=1,upper=3> result_new_pred[nGames_new];
    corr_matrix[2*uvdim] uvcorr;
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
    // Correlation matrix for components of matchup effects
    uvcorr = multiply_lower_tri_self_transpose(uvcorr_chol);
}
