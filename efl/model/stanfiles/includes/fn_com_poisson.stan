    // DISTRIBUTION FUNCTIONS FOR THE CONWAY-MAXWELL-POISSON DISTRIBUTION
    // TO BE INCLUDED IN THE functions BLOCK
    
    // The COM Poisson functions are taken from Paul-Christian BÃ¼rkner's 'brms'
    // project: https://github.com/paul-buerkner/brms
    // As of 2020-01-24 these functions aren't officially supported in brms, but
    // are available on Github. They've been linked on Stan discussions:
    // https://discourse.mc-stan.org/t/conway-maxwell-poisson-distribution/2370
    // They have since been modified by me to fit my use case here.
    
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