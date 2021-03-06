    // DISTRIBUTION FUNCTIONS FOR THE BIVARIATE GAUSSIAN COPULA 
    // TO BE INCLUDED IN THE functions BLOCK
    
    // Log PDF for the bivariate Gaussian copula
    real bi_gausscopula_lpdf(real[] u, real rho) {
        // x and y are the transformed components of u
        real x;
        real y;
        // check the inputs
        if (!(num_elements(u) == 2)) {
            reject("bi_gausscopula_lpdf: u must be length 2. ",
                   "(found length ", num_elements(u), ")")
        }
        if (!(fabs(rho) < 1)) {
            reject("bi_gausscopula_lpdf: rho must be between -1 and 1. ",
                   "(found phi=", rho, ")");
        }
        if (!((u[1] > 0) && (u[1] < 1) && (u[2] > 0) && (u[2] < 1))) {
            reject("bi_gausscopula_lpdf: u must have components between 0 and 1. ",
                   "(found u=(", u[1], ",", u[2], ") )");
        }
        x = inv_Phi(u[1]);
        y = inv_Phi(u[2]);
        // The joint density of x and y is equal to the marginal of x times
        // the conditional of y. The marginals of x then cancel.
        return normal_lpdf(y | rho*x, sqrt((1-rho)*(1+rho))) - std_normal_lpdf(y);
    }
    
    // CDF for the bivariate gaussian copula
    real bi_gausscopula_cdf(real[] u, real rho) {
        // check the inputs
        if (!(num_elements(u) == 2)) {
            reject("bi_gausscopula_cdf: u must be length 2. ",
                   "(found length ", num_elements(u), ")")
        }
        if (!(fabs(rho) < 1)) {
            reject("bi_gausscopula_cdf: rho must be between -1 and 1. ",
                   "(found phi=", rho, ")");
        }
        if (!((u[1] >= 0) && (u[1] <= 1) && (u[2] >= 0) && (u[2] <= 1))) {
            reject("bi_gausscopula_cdf: u must have components between 0 and 1. ",
                   "(found u=(", u[1], ",", u[2], ") )");
        }
        // Edge cases to avoid passing infinity to binormal_cdf
        if ((u[1] == 0) || (u[2] == 0)) {
            return 0;
        } else if (u[1] == 1) {  // This works if u[1] == u[2] == 1
            return u[2];
        } else if (u[2] == 1) {
            return u[1];
        }
        // Bivariate Gaussian CDF with unit variance and zero mean. Taken from
        // the Stan user's manual custom probability functions examples
        // https://mc-stan.org/docs/2_21/stan-users-guide/examples.html
        if ((u[1] != 0.5) || (u[2] != 0.5)) {
            real z1 = inv_Phi(u[1]);
            real z2 = inv_Phi(u[2]);
            real denom = sqrt((1 + rho) * (1 - rho));
            real a1 = (z2 / z1 - rho) / denom;
            real a2 = (z1 / z2 - rho) / denom;
            real product = z1 * z2;
            real delta = product < 0 || (product == 0 && (z1 + z2) < 0);
            return 0.5 * (u[1] + u[2] - delta) - owens_t(z1, a1) - owens_t(z2, a2);
        }
        return 0.25 + asin(rho) / (2 * pi());
    }
    
    // Log CDF for the bivariate guassian copula
    real bi_gausscopula_lcdf(real[] u, real rho) {
        return log(bi_gausscopula_cdf(u, rho));
    }
    
    // Random number generator for the bivariate gaussian copula
    real[] bi_gausscopula_rng(real rho) {
        // Draw u marginally, v conditionally
        real u = uniform_rng(0.0, 1.0);
        real v = Phi(normal_rng(rho*inv_Phi(u), sqrt((1-rho)*(1+rho))));
        return {u, v};
    }
