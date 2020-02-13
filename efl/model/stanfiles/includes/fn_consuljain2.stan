    // DISTRIBUTION FUNCTIONS FOR THE GENERALIZED POISSON DISTRIBUTION
    // OF CONSUL AND JAIN (1973)
    // TO BE INCLUDED IN THE functions BLOCK
    
    // Based on a given delta, return an approximate value mu* such that if
    // the Generalized Poisson distribution has parameter mu > mu*, then the 
    // truncation error is negligible. Everything in this function is very
    // empirical and approximate, with absolutely no
    // theoretical backing.
    real cj_min_noerr_mu(real delta) {
        if (delta >= 0) {
            // There's never truncation for nonnegative delta, so any value of
            // mu will result in zero truncation error.
            return 0;
        } else {
            // For a given delta, the relationship between truncation
            // point (t) and log absolute error is approximately linear.
            // The intercept of this relationship is zero (if truncated at
            // zero, the distribution has no mass, so log(abs(error)) == 0).
            // The slope of the relationship depends on delta. The relationship
            // between log(-delta) and the slope of the above linear
            // relationship is also approximately linear (woohoo!) with
            // intercept -1.488872 and slope 1.087622.
            // So the minimum truncation point so that the truncation error is
            // probably less than machine epsilon is...
            real t = log(machine_precision()/2.0) / fma(1.087622, log(-delta), -1.488872);
            // Truncation point is mu * (1 - delta) / (-delta). So solving
            // from truncation point to get a minimum required mu_star ...
            return t / (1 + inv(-delta)); // inv(-delta) >= 1 always
        }
    }
    
    // Log normalizing constant for the Consul-Jain generalized Poisson
    // distribution, due to truncation when delta < 0
    // Does *not* check parameters for validity
    real cj_log_norm(real log_mu, real delta) {
        real mu = exp(log_mu);
        real logc; // value to return
        // mu threshold
        if (mu > cj_min_noerr_mu(delta)) {
            // Large enough mu's result in negligible truncation error
            logc = 0;
        } else { // Just sum up all the terms
            real t = mu * (1 + inv(-delta)); // Truncation point
            int x;
            real lprob;
            real lfac = 0;
            real log_pars = log_mu + log1m(delta); // Appears in every term
            logc = -fma(-delta, mu, mu); // x=0 term
            x = 1;
            while (x < t) {
                // "Rate" for this x: lambda-like value in pmf
                real rate = fma(x - mu, delta, mu);
                lfac += log(x);
                lprob = log_pars + lmultiply(x - 1, rate) - rate - lfac;
                logc = log_sum_exp(logc, lprob);
                x += 1;
            }
        }
        return logc;
    }
    
    // Log PMF function for the Consul-Jain generalized Poisson distribution
    // Mean and index of dispersion parameterization:
    // log_mu : log mean, 
    // -1 < delta < 1 : dispersion parameter
    real consuljain_lpmf(int x, real log_mu, real delta) {
        // convenient precomputation: "lambda-like" value in pmf
        real rate;
        real mu = exp(log_mu);
        // the gamut of integrity checks on parameters
        if (!(x >= 0)) {
            reject("consuljain_lpmf: x must be non-negative. ",
                   "(found x=", x, ")");
        }
        if (!(mu > 0)) {
            reject("consuljain_lpmf: mu must be positive. ",
                   "(found mu=", mu, ")");
        }
        if (!(fabs(delta) <= 1)) {
            reject("consuljain_lpmf: delta must be between -1 and 1. ",
                   " (found delta=", delta, ")");
        }
        // Calculate and return log pmf
        rate = fma(x - mu, delta, mu);
        if (rate <= 0) {
            // Any x's such that this is negative have probability zero
            return negative_infinity();
        } else if (x == 0) {
            // Zero can be computed easily
            return -fma(-delta, mu, mu) - cj_log_norm(log_mu, delta);
        } else {
            return log_mu + log1m(delta) + lmultiply(x - 1, rate) - rate - lgamma(x + 1) - cj_log_norm(log_mu, delta);
        }
    }
    
    // Log CDF of Consul-Jain distribution for multiple values. More efficient
    // than calling separately as the normalizing constant and the log
    // numerators for each value up to the max x only need to be calculated
    // once. The only repeated calculation is log_sum_exp.
    real[] consuljain_lcdf_array(int[] x, real log_mu, real delta) {
        int n = num_elements(x);
        int min_x = min(x);
        real lcdf[n]; // Keep track of total probability
        real t = positive_infinity(); // Maximum allowable x value
        real mu = exp(log_mu);
        // the gamut of integrity checks on parameters
        if (!(min_x >= 0)) {
            reject("consuljain_lcdf_array: x must be positive. ",
                   "(found x=", min_x, ")");
        }
        if (!(mu > 0)) {
            reject("consuljain_lcdf_array: mu must be positive. ",
                   "(found mu=", mu, ")");
        }
        if (!(fabs(delta) <= 1)) {
            reject("consuljain_lcdf_array: delta must be between -1 and 1. ",
                   " (found delta=", delta, ")");
        }
        // Is there a max to the support?
        if (delta < 0) {
            t = mu * (1 + inv(-delta));
        }
        // If all x's are high enough, probability is one
        if (min_x + 1 >= t) {
            lcdf = rep_array(0, n);
        } else {
            // what is the largest x we must calculate? (prob < 1)
            // Any x's larger than max_x will have cdf = 1 (lcdf = 0).
            int max_x = min_x;
            for (xi in x) {
                if ((xi > max_x) && (xi + 1 < t)) {
                    max_x = xi;
                }
            }
            {
                // Calculate the probabilities for i = 0, ..., max_x and sum
                real logc; // Log of normalizing constant
                real lprob[max_x+1]; // array of log probabilities
                real lfac = 0;
                real log_pars = log_mu + log1m(delta); // appears in every term
                int i; // Looping index
                logc = cj_log_norm(log_mu, delta);
                // i = 0 term
                lprob[1] = -fma(-delta, mu, mu);
                i = 0;
                while (i < max_x) {
                    real rate; // "lambda-like" term in pmf
                    i += 1;
                    rate = fma(i - mu, delta, mu);
                    lfac += log(i);
                    lprob[i+1] = log_pars + lmultiply(i - 1, rate) - rate - lfac;
                }
                // Sum the first x[j]+1 terms of lprob for all elements j of x.
                for (j in 1:n) {
                    if (x[j] > max_x) {
                        // Any x larger than max_x has cdf = 1 (lcdf = 0)
                        // by construction of max_x
                        lcdf[j] = 0;
                    } else {
                        // Otherwise, return the log_sum_exp of lprob, minus
                        // normalizing constant. Cap at 1 (log = 0).
                        lcdf[j] = fmin(log_sum_exp(lprob[1:(x[j]+1)]) - logc, 0.0);
                    }
                }
            }
        }
        return lcdf;
    }
    
    // CDF of Consul-Jain distribution for multiple values
    real[] consuljain_cdf_array(int[] x, real log_mu, real delta) {
        return exp(consuljain_lcdf_array(x, log_mu, delta));
    }
    
    // Log Complementary CDF of Consul-Jain distribution for multiple values
    real[] consuljain_lccdf_array(int[] x, real log_mu, real delta) {
        return log1m_exp(consuljain_lcdf_array(x, log_mu, delta));
    }
    
    // Log CDF of Consul-Jain distribution
    real consuljain_lcdf(int x, real log_mu, real delta) {
        return consuljain_lcdf_array({x}, log_mu, delta)[1];
    }
    
    // Log Complementary CDF of Consul-Jain distribution
    real consuljain_lccdf(int x, real log_mu, real delta) {
        return log1m_exp(consuljain_lcdf(x | log_mu, delta));
    }
    
    // CDF of Consul-Jain distribution
    real consuljain_cdf(int x, real log_mu, real delta) {
        return exp(consuljain_lcdf(x | log_mu, delta));
    }
    
    // Inverse Log CDF of Consul-Jain distribution
    int consuljain_ilcdf(real log_u, real log_mu, real delta) {
        int x; // Value that will eventually be returned
        real mu = exp(log_mu);
        // Check inputs
        if (!(log_u <= 0)) {
            reject("consuljain_icdf: log_u must be non-positive. ",
                   "(found log_u=", log_u, ")");
        }
        if (!(mu > 0)) {
            reject("consuljain_ilcdf: mu must be positive. ",
                   "(found mu=", mu, ")");
        }
        if (!(fabs(delta) <= 1)) {
            reject("consuljain_lcdf: delta must be between -1 and 1. ",
                   " (found delta=", delta, ")");
        }
        // Accumulate probability until we're above u, then we stop
        {
            real lfac = 0; // log factorial tracker
            real lcdf; // Keep track of total probability
            real log_pars = log_mu + log1m(delta); // appears in every term
            real log_c = cj_log_norm(log_mu, delta); // Log of normalizing constant
            real t = positive_infinity(); // Maximum allowable x value
            // Do we have a max?
            if (delta < 0) {
                t = mu * (1 + inv(-delta));
            }
            // Term for x=0
            lcdf = -fma(-delta, mu, mu);
            x = 0;
            // Terms for x>0
            while ((lcdf - log_c < log_u) && (x + 1 < t)) {
                real lprob;
                real rate;
                x += 1; 
                rate = fma(x - mu, delta, mu);
                lfac += log(x);
                lprob = log_pars + lmultiply(x - 1, rate) - rate - lfac;
                lcdf = log_sum_exp(lcdf, lprob);
            }
        }
        // Return the value
        return x;
    }
    
    // Inverse CDF of Consul-Jain distribution
    int consuljain_icdf(real u, real log_mu, real delta) {
        return consuljain_ilcdf(log(u), log_mu, delta);
    }
    
    // Generate random numbers from the Consul-Jain distribution
    int consuljain_rng(real log_mu, real delta) {
        // draw from a uniform distribution
        real u = uniform_rng(0.0, 1.0);
        // Find inverse cdf of u
        return consuljain_icdf(u, log_mu, delta);
    }