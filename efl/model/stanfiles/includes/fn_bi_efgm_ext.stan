    // DISTRIBUTION FUNCTIONS FOR THE BIVARIATE EFGM 
    // (EYRAUD-FARLIE-GUMBEL-MORGENSTERN) COPULA, "EXTENDED" TO ALLOW HIGHER
    // CORRELATIONS BY MIXING IN COMPLETE NEGATIVE OR POSITIVE DEPENDENCE.
    // TO BE INCLUDED IN THE functions BLOCK
    
    // See Eyraud, 1936; Farlie, 1960; Gumbel, 1958,1960; Morgenstern, 1956; 
    // Nelsen, 2006; or Nadarajah, Afuecheta and Chan, 2017, for an overview. 
    // https://pdfs.semanticscholar.org/3ffe/fd157fa4f8cff5b7bb7f4efde9294f154c66.pdf
    
    // CDF for the bivariate EFGM copula
    real bi_efgm_ext_cdf(real[] x, real rho) {
        real u; // Convenience variable for first component
        real v; // Convenience variable for second component
        real phi; // Parameter for EFGM copula
        real delta; // Parameter controlling mixture
        real efgm_component;
        real dependence_component;
        if (!(num_elements(x) == 2)) {
            reject("bi_efgm_lcdf: x must be length 2. ",
                   "(found length ", num_elements(x), ")")
        }
        u = x[1];
        v = x[2];
        if (!(fabs(rho) <= 1)) {
            reject("bi_efgm_lcdf: phi must be between -1 and 1. ",
                   "(found phi=", phi, ")");
        }
        if (!((u >= 0) && (u <= 1) && (v >= 0) && (v <= 1))) {
            reject("bi_efgm_lcdf: x must have components between 0 and 1. ",
                   "(found x=(", u, ",", v, ") )");
        }
        // Break down rho into relevant pieces
        if (rho >= 0) {
            phi = fmin(3*rho, 1.0);
            delta = fmax((3*rho - 1)/2, 0.0);
        } else {
            phi = fmax(3*rho, -1.0);
            delta = fmin((3*rho + 1)/2, 0.0);
        }
        // The component of the mixture from the EFGM copula
        efgm_component = u * v * (1 + phi * (1 - u) * (1 - v));
        // The component of the mixture from complete dependence
        if (delta >= 0) {
            // Complete positive dependence
            dependence_component = fmin(u, v);
        } else {
            // Complete negative dependence
            dependence_component = v - fmin(1-u, v);
        }
        return fabs(delta) * dependence_component + (1 - fabs(delta)) * efgm_component;
    }
    
    // Log CDF for the bivariate EFGM copula
    real bi_efgm_ext_lcdf(real[] x, real rho) {
        return log(bi_efgm_cdf(x, rho));
    }
    
    // RNG for the bivariate EFGM copula
    real[] bi_efgm_ext_rng(real rho) {
        // Draw u uniformly, then draw v conditionally
        real u = uniform_rng(0.0, 1.0);
        real v;
        real phi;
        real delta;
        // Break down rho into two parameters
        if (rho >= 0) {
            phi = fmin(3*rho, 1.0);
            delta = fmax((3*rho - 1)/2, 0.0);
        } else {
            phi = fmax(3*rho, -1.0);
            delta = fmin((3*rho + 1)/2, 0.0);
        }
        // Draw v in one of three cases:
        if ((delta > 0) && (bernoulli_rng(delta) == 1)) {
            // 1) We need complete positive dependence
            v = u;
        } else if ((delta < 0) && bernoulli_rng(-delta) == 1) {
            // 2) We need complete negative dependence
            v = 1 - u;
        } else {
            // 3) Either delta == 0, or the bernoulli r.v. indicating complete
            // dependence were zero. We need to sample from the EFGM copula
            real x = uniform_rng(0.0, 1.0);
            real A = phi * (1 - 2 * u);
            real B = - 1 - A;
            if (A == 0) {
                v = x;
            } else {
                // Solve the quadratic for the conditional CDF to get v
                // We want the "minus" solution to the quadratic equation
                // We know that -1 < A < 1, so B is always negative
                // Following the algorithm outlined here:
                // https://en.wikipedia.org/wiki/Loss_of_significance#A_better_algorithm
                // Discriminant calculated according to this answer here:
                // https://stackoverflow.com/a/50065711
                real fourac = 4 * A * x;
                real disc = fma(B, B, -fourac) + fma(-4*A, x, fourac);
                real sol1 = (-B + sqrt(disc)) / (2 * A);
                v = x / (A * sol1);
            }
        }
        return {u, v};
    }