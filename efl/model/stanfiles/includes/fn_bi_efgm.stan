    // DISTRIBUTION FUNCTIONS FOR THE BIVARIATE EFGM 
    // (EYRAUD-FARLIE-GUMBEL-MORGENSTERN) COPULA.
    // TO BE INCLUDED IN THE functions BLOCK
    
    // See Eyraud, 1936; Farlie, 1960; Gumbel, 1958,1960; Morgenstern, 1956; 
    // Nelsen, 2006; or Nadarajah and Afuecheta and Chan, 2017, for an
    // overview. 
    // https://pdfs.semanticscholar.org/3ffe/fd157fa4f8cff5b7bb7f4efde9294f154c66.pdf
    
    // Log PDF for the bivariate EFGM copula
    real bi_efgm_lpdf(vector x, real phi) {
        real u = x[1];
        real v = x[2];
        if (!(fabs(phi) <= 1)) {
            reject("phi must be between -1 and 1. ",
                   "(found phi=", phi, ")");
        }
        if (!((u > 0) && (u < 1) && (v > 0) && (v < 1))) {
            reject("x must have components between 0 and 1. ",
                   "(found x=(", u, ",", v, ") )");
        }
        return log1p(phi * (1 - 2*v) * (1 - 2*u));
    }
    
    // Log CDF for the bivariate EFGM copula
    real bi_efgm_lcdf(vector x, real phi) {
        real u = x[1];
        real v = x[2];
        if (!(fabs(phi) <= 1)) {
            reject("phi must be between -1 and 1. ",
                   "(found phi=", phi, ")");
        }
        if (!((u > 0) && (u < 1) && (v > 0) && (v < 1))) {
            reject("x must have components between 0 and 1. ",
                   "(found x=(", u, ",", v, ") )");
        }
        return log(u) + log(v) + log1p(phi * (1 - u) * (1 - v));
    }
    
    // CDF for the bivariate EFGM copula
    real bi_efgm_cdf(vector x, real phi) {
        return exp(bi_efgm_lcdf(x | phi));
    }
    
    // RNG for the bivariate EFGM copula
    vector bi_efgm_rng(real phi) {
        // Draw u uniformly, then draw v conditionally
        real u = uniform_rng(0.0, 1.0);
        real x = uniform_rng(0.0, 1.0);
        real v;
        if (phi == 0) {
            v = x;
        } else {
            // Solve the quadratic for the conditional CDF to get v
            real A = phi * (1 - 2 * u);
            real B = - 1 - A;
            v = (-B - sqrt(B^2 - 4*A*x)) / (2 * A);
        }
        return [u,v]';
    }