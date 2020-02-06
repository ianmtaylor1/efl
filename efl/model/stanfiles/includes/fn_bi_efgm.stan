    // DISTRIBUTION FUNCTIONS FOR THE BIVARIATE EFGM 
    // (EYRAUD-FARLIE-GUMBEL-MORGENSTERN) COPULA.
    // TO BE INCLUDED IN THE functions BLOCK
    
    // See Eyraud, 1936; Farlie, 1960; Gumbel, 1958,1960; Morgenstern, 1956; 
    // Nelsen, 2006; or Nadarajah, Afuecheta and Chan, 2017, for an overview. 
    // https://pdfs.semanticscholar.org/3ffe/fd157fa4f8cff5b7bb7f4efde9294f154c66.pdf
    
    // Log PDF for the bivariate EFGM copula
    real bi_efgm_lpdf(real[] x, real phi) {
        real u;
        real v;
        if (!(num_elements(x) == 2)) {
            reject("bi_efgm_lpdf: x must be length 2. ",
                   "(found length ", num_elements(x), ")")
        }
        u = x[1];
        v = x[2];
        if (!(fabs(phi) <= 1)) {
            reject("bi_efgm_lpdf: phi must be between -1 and 1. ",
                   "(found phi=", phi, ")");
        }
        if (!((u > 0) && (u < 1) && (v > 0) && (v < 1))) {
            reject("bi_efgm_lpdf: x must have components between 0 and 1. ",
                   "(found x=(", u, ",", v, ") )");
        }
        return log1p(phi * (1 - 2 * u) * (1 - 2 * v));
    }
    
    // CDF for the bivariate EFGM copula
    real bi_efgm_cdf(real[] x, real phi) {
        real u;
        real v;
        if (!(num_elements(x) == 2)) {
            reject("bi_efgm_lcdf: x must be length 2. ",
                   "(found length ", num_elements(x), ")")
        }
        u = x[1];
        v = x[2];
        if (!(fabs(phi) <= 1)) {
            reject("bi_efgm_lcdf: phi must be between -1 and 1. ",
                   "(found phi=", phi, ")");
        }
        if (!((u >= 0) && (u <= 1) && (v >= 0) && (v <= 1))) {
            reject("bi_efgm_lcdf: x must have components between 0 and 1. ",
                   "(found x=(", x[1], ",", x[2], ") )");
        }
        return u * v * (1 + phi * (1 - u) * (1 - v));
    }
    
    // Log CDF for the bivariate EFGM copula
    real bi_efgm_lcdf(real[] x, real phi) {
        return log(bi_efgm_cdf(x, phi));
    }
    
    // RNG for the bivariate EFGM copula
    real[] bi_efgm_rng(real phi) {
        // Draw u uniformly, then draw v conditionally
        real u = uniform_rng(0.0, 1.0);
        real x = uniform_rng(0.0, 1.0);
        real v;
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
        return {u, v};
    }