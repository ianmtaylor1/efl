    // DISTRIBUTION FUNCTIONS FOR THE GENERALIZED POISSON DISTRIBUTION
    // OF CONSUL AND JAIN (1973)
    // TO BE INCLUDED IN THE functions BLOCK
    
    // Based on a given delta, return an approximate value t such that if
    // The Generalized Poisson distribution is truncated at a value larger
    // than t, then the truncation error is negligible. Everything in this 
    // function is very empirical and approximate, with absolutely no
    // theoretical backing.
    real cj_min_noerr_trunc(real delta) {
        if (!(fabs(delta) <= 1)) {
            reject("cj_min_noerr_trunc: delta must be between -1 and 1 ",
                   "(found delta=", delta, ")")
        }
        if (delta >= 0) {
            // There's never error for nonnegative delta, so truncate anywhere
            // There's also never truncation for nonnegative delta, so this
            // doesn't matter. But returning zero is sensible.
            return 0;
        } else {
            // For a given delta, there relationship between truncation
            // point (m) and log absolute error is approximately linear.
            // The intercept of this relationship is zero (if truncated at
            // zero, the distribution has no mass, so log(abs(error)) == 0).
            // The slope of the relationship depends on delta. The relationship
            // between log(-delta) and the slope of the above linear
            // relationship is also approximately linear (woohoo!) with
            // intercept -1.491123 and slope 1.089328.
            // So to make sure the truncation error is less than machine eps...
            return log(machine_precision()) / (-1.491123 + 1.089328 * log(-delta));
        }
    }
    
    // Log normalizing constant for the Consul-Jain generalized Poisson
    // distribution, due to truncation when delta < 0
    real cj_log_norm(real lambda, real delta) {
        real logc;
        real m = positive_infinity();
        // Larger than truncpoint, errors are negligible.
        real truncpoint = cj_min_noerr_trunc(delta);
        // Check that lambda is valid (truncpoint function checks delta)
        if (!(lambda > 0)) {
            reject("cj_log_norm: lambda must be positive. ",
                   "(found lambda=", lambda, ")")
        }
        // Is there a max to the support?
        if (delta < 0) {
            m = -lambda/delta;
        }
        if (m > truncpoint) {
            // If truncation is large enough, series sums to 1 as expected
            // (this catches the case when delta is positive)
            logc = 0;
        } else { // Just sum up all the terms
            int x;
            real lprob;
            real lfac = 0;
            real log_lambda = log(lambda);
            logc = -lambda; // x=0 term
            x = 1;
            while (x < m) {
                lfac += log(x);
                lprob =  log_lambda + (x - 1) * log(lambda + x * delta) - lambda - x * delta - lfac;
                logc = log_sum_exp(logc, lprob);
                x += 1;
            }
        }
        return logc;
    }
    
    // Log PMF function for the Consul-Jain generalized Poisson distribution
    // Mean and index of dispersion parameterization:
    // mu > 0 : mean, 
    // theta > 0.25 : index of dispersion
    real consuljain_lpmf(int x, real mu, real theta) {
        // converted parameters
        real delta;
        real lambda;
        // convenient precomputation
        real lxd;
        // the gamut of integrity checks on parameters
        if (!(x >= 0)) {
            reject("consuljain_lpmf: x must be positive. ",
                   "(found x=", x, ")");
        }
        if (!(mu > 0)) {
            reject("consuljain_lpmf: mu must be positive. ",
                   "(found mu=", mu, ")");
        }
        if (!(theta > 0.25)) {
            reject("consuljain_lpmf: theta must be greater than 0.25. ",
                   " (found theta=", theta, ")");
        }
        // Calculate and return log pmf
        delta = 1 - 1 / sqrt(theta);
        lambda = mu / sqrt(theta);
        lxd = lambda + x * delta;
        if (lxd <= 0) {
            // Any x's such that this is negative have probability zero
            return negative_infinity();
        } else {
            return log(lambda) + (x - 1) * log(lxd) - lxd - lgamma(x + 1) - cj_log_norm(lambda, delta);
        }
    }
    
    // Log CDF of Consul-Jain distribution
    real consuljain_lcdf(int x, real mu, real theta) {
        real lcdf; // Keep track of total probability
        real delta; // Standard second parameter
        real lambda; // Standard first parameter
        real m = positive_infinity(); // Maximum allowable x value
        // the gamut of integrity checks on parameters
        if (!(x >= 0)) {
            reject("consuljain_lcdf: x must be positive. ",
                   "(found x=", x, ")");
        }
        if (!(mu > 0)) {
            reject("consuljain_lcdf: mu must be positive. ",
                   "(found mu=", mu, ")");
        }
        if (!(theta > 0.25)) {
            reject("consuljain_lcdf: theta must be greater than 0.25. ",
                   " (found theta=", theta, ")");
        }
        delta = 1 - 1 / sqrt(theta); // Standard second parameter
        lambda = mu * (1 - delta); // Standard first parameter
        // Is there a max to the support?
        if (delta < 0) {
            m = -lambda/delta;
        }
        // If x is high enough, probability is one
        if (x >= m) {
            lcdf = 0;
        } else {
            // Calculate probabilities for i = 0, ..., x and sum
            real logc; // Log of normalizing constant
            real lprob[x+1]; // array of log probabilities
            real lfac = 0;
            real log_lambda = log(lambda);
            logc = cj_log_norm(lambda, delta);
            lprob[1] = -lambda;  // i = 0 term
            for (i in 1:x) {  // from i = 1 ...
                lfac += log(i);
                lprob[i+1] = log_lambda + (i - 1) * log(lambda + delta * i) - lambda - delta * i - lfac;
            }
            // reduce, normalize
            lcdf = log_sum_exp(lprob) - logc;
        }
        return lcdf;
    }
    
    // Log Complementary CDF of Consul-Jain distribution
    real consuljain_lccdf(int x, real mu, real theta) {
        return log1m_exp(consuljain_lcdf(x | mu, theta));
    }
    
    // CDF of Consul-Jain distribution
    real consuljain_cdf(int x, real mu, real theta) {
        return exp(consuljain_lcdf(x | mu, theta));
    }
    
    // Generate random numbers from the Consul-Jain distribution
    int consuljain_rng(real mu, real theta) {
        real lu; // log of Uniform random variable
        real lcdf; // Keep track of total probability
        int x = 0; // Value that will eventually be returned
        real delta = 1 - 1 / sqrt(theta); // Standard second parameter
        real lambda = mu * (1 - delta); // Standard first parameter
        real logc = cj_log_norm(lambda, delta); // Log of normalizing constant
        real m = positive_infinity(); // Maximum allowable x value
        // Draw from uniform
        lu = log(uniform_rng(0.0,1.0));
        // Do we have a max?
        if (delta < 0) {
            m = -lambda/delta;
        }
        // Term for x=0
        lcdf = -lambda;
        // Accumulate probability until we're above u, then we stop
        {
            real log_lambda = log(lambda);
            real lfac = 0; // log factorial tracker
            while ((lcdf - logc < lu) && (x + 1 < m)) {
                real lprob;
                x += 1; 
                lfac += log(x);
                lprob = log_lambda + (x - 1) * log(lambda + delta*x) - lambda - delta * x - lfac;
                lcdf = log_sum_exp(lcdf, lprob);
            }
        }
        // Return the value
        return x;
    }