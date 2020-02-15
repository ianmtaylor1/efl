    // DISTRIBUTION FUNCTIONS FOR THE CONWAY-MAXWELL-POISSON DISTRIBUTION
    // TO BE INCLUDED IN THE functions BLOCK
    
    // The COM Poisson functions are taken from Paul-Christian Buerkner's 'brms'
    // project: https://github.com/paul-buerkner/brms
    // As of 2020-01-24 these functions aren't officially supported in brms, but
    // are available on Github. They've been linked on Stan discussions:
    // https://discourse.mc-stan.org/t/conway-maxwell-poisson-distribution/2370
    // They have since been modified by me to fit my use case here.
    
    // log normalizing constant of the COM Poisson distribution
    // implementation inspired by code of Ben Goodrich
    // Args:
    //   log_mu: log location parameter
    //   nu: positive shape parameter
    //   tp: truncation point of distribution
    real log_Z_com_poisson(real log_mu, real nu, int tp) {
        real log_Z;
        real lfac;
        real term;
        int x;
        real log_eps;
        // Approximate by Poisson if possible
        if (nu == 1) {
            return exp(log_mu);
        }
        // nu == 0 or Inf will fail in this parameterization
        if (nu <= 0) {
            reject("log_Z_com_poisson: nu must be positive. ",
                   "(found nu=", nu, ")");
        }
        if (is_inf(nu)) {
            reject("log_Z_com_poisson: nu must be finite");
        }
        // direct computation of the truncated series
        log_eps = log(machine_precision()/2);
        log_Z = log1p_exp(nu * log_mu);  // first 2 terms of the series
        lfac = 0;
        term = nu * log_mu; // x=1 term
        x = 1;
        // relative tolerance threshold of last term
        while ((x < tp) && (term - log_Z > log_eps)) {
            x += 1;
            lfac += log(x);
            term = nu * (x * log_mu - lfac);
            log_Z = log_sum_exp(log_Z, term);
        }
        // Return the accumulated log normalizing constant
        return log_Z;
    }
    
    // Effective truncation point of the COM Poisson distribution. If it is
    // equal to tp, then the distribution was force truncated. Effective
    // truncation point is defined by where the normalizing constant stops
    // being calculated. It is a relative tolerance. When
    // P(X=x)/P(X<=x) <= epsilon, computation stops.
    // Args:
    //   log_mu: log location parameter
    //   nu: positive shape parameter
    //   tp: truncation point of distribution
    int com_poisson_truncpoint(real log_mu, real nu, int tp) {
        real log_Z;
        real lfac;
        real term;
        int x;
        real log_eps;
        // nu == 0 or Inf will fail in this parameterization
        if (nu <= 0) {
            reject("log_Z_com_poisson: nu must be positive. ",
                   "(found nu=", nu, ")");
        }
        if (is_inf(nu)) {
            reject("log_Z_com_poisson: nu must be finite");
        }
        // direct computation of the truncated series
        log_eps = log(machine_precision()/2);
        log_Z = log1p_exp(nu * log_mu);  // first 2 terms of the series
        lfac = 0;
        term = nu * log_mu; // x=1 term
        x = 1;
        // relative tolerance threshold of last term
        while ((x < tp) && (term - log_Z > log_eps)) {
            x += 1;
            lfac += log(x);
            term = nu * (x * log_mu - lfac);
            log_Z = log_sum_exp(log_Z, term);
        }
        // Return the last value calculated
        return x;
    }
    
    // COM Poisson log-PMF for a single response (log parameterization)
    // Args: 
    //   y: the response value 
    //   log_mu: log location parameter
    //   nu: positive shape parameter
    //   tp: truncation point of distribution
    real com_poisson_log_lpmf(int y, real log_mu, real nu, int tp) {
        if (nu == 1) return poisson_log_lpmf(y | log_mu);
        if (!(y >= 0)) {
            reject("com_poisson_log_lpmf: y must be non-negative");
        }
        if (!(y <= tp)) {
            reject("com_poisson_log_lpmf: y must be less than truncation point");
        }
        return nu * (y * log_mu - lgamma(y + 1)) - log_Z_com_poisson(log_mu, nu, tp);
    }
    
    // COM Poisson log-CDF for an array of values y[], with the same parameters
    // At least as efficient as computing the cdf multiple times, since the
    // normalizing constant only needs to be calculated once
    real[] com_poisson_log_lcdf_array(int[] y, real log_mu, real nu, int tp) {
        int n = num_elements(y);
        int max_y = max(y); // the maximum value we need to calculate for
        real lprob[max_y+1]; // array to hold log probability (numerators)
        real log_Z; // log normalizing constant
        real lfac; // log factorial tracker
        real lcdf[n];
        // Check for valid y
        if (!(min(y) >= 0)) {
            reject("com_poisson_log_lcdf_array: y must be non-negative");
        }
        if (!(max(y) <= tp)) {
            reject("com_poisson_log_lpmf: y must be less than truncation point");
        }
        // Simplify to poisson, if possible
        if (nu == 1) {
            for (j in 1:n) {
                lcdf[j] = poisson_lcdf(y[j] | exp(log_mu));
            }
            return lcdf;
        }
        // Calculate normalizing constant (checks parameters)
        log_Z = log_Z_com_poisson(log_mu, nu, tp);
        // term for i=0
        lfac = 0;
        lprob[1] = 0;
        // Terms for i = 1, ..., max_y
        for (i in 1:max_y) {
            lfac += log(i);
            lprob[i+1] = nu * (i * log_mu - lfac);
        }
        // Sum up for all y's
        for (j in 1:n) {
            lcdf[j] = log_sum_exp(lprob[1:(y[j]+1)]) - log_Z;
        }
        return lcdf;
    }
        
    // COM Poisson log-CDF for a single observation (log parametrization)
    real com_poisson_log_lcdf(int y, real log_mu, real nu, int tp) {
        return com_poisson_log_lcdf_array({y}, log_mu, nu, tp)[1];
    }
    
    // Inverse Log CDF of COM Poisson distribution (log parameterization)
    // log_u: log probability we want to invert
    // Returns the first x such that log(P(X <= x)) >= log_u
    int com_poisson_log_ilcdf(real log_u, real log_mu, real nu, int tp) {
        real log_num;  // log numerator
        real log_Z;  // log denominator
        int x;  // number that will eventually be returned
        // Find the normalizing constant, Z
        log_Z = log_Z_com_poisson(log_mu, nu, tp);
        // Start at x=0, and set the appropriate log numerator
        x = 0;
        log_num = 0;
        {
            real lfac = 0.0; // log factorial tracker
            while ((log_num - log_Z < log_u) && (x < tp)) {
                // Keep incrementing the numerator until we exceed the random value
                x += 1;
                lfac += log(x);
                log_num = log_sum_exp(log_num, nu * (x * log_mu - lfac));
            }
        }
        return x;
    }
    
    // Random number generator for COM Poisson
    int com_poisson_log_rng(real log_mu, real nu, int tp) {
        real lu;
        // Simplify to poisson, if possible
        if (nu == 1) {
            return poisson_log_rng(log_mu);
        }
        // Draw a uniform r.v. to transform via inverse cdf
        lu = log(uniform_rng(0.0, 1.0));
        // Call the inverse cdf to get the value
        return com_poisson_log_ilcdf(lu, log_mu, nu, tp);
    }