#include "../include/SSTModel.hpp"
#include <cmath>
#include <algorithm>

// Pointwise functions

// F1 blending: switches between inner (k-omega) and outer (k-epsilon) model
// detects whether cell is near a wall or in free flow
double SSTModel::computeF1(double k, double omega, double y, double nu, double CDkw_pos) const {
    // numeric safeguards
    double sqrtK = std::sqrt(std::max(k, 0.0));
    double omSafe = std::max(omega, 1e-20);
    double ySafe  = std::max(y, 1e-20);
    double y2s    = ySafe * ySafe;

    double term1 = sqrtK / (coeffs.betaStar * omSafe * ySafe);
    double term2 = 500.0 * nu / (y2s * omSafe);
    double CDpos = std::max(CDkw_pos, 1e-20);
    double term3 = 4.0 * coeffs.sigma_w2 * std::max(k, 0.0) / (CDpos * y2s);

    // arg1 = min( max(sqrt(k)/(betaStar*omega*y), 500*nu/(y^2*omega)), 4*sigma_w2*k / (CDkw_pos * y^2))
    // F1 = tanh(arg1^4)
    double arg1 = std::min(std::max(term1, term2), term3);
    double a4   = arg1 * arg1 * arg1 * arg1;
    return std::tanh(a4);
}

// F2 blending: used in the eddy viscosity limiter
// detects how close cell is to a wall to limit turbulent eddy viscosity near walls
double SSTModel::computeF2(double k, double omega, double y, double nu) const {
    // numeric safeguards
    double sqrtK = std::sqrt(std::max(k, 0.0));
    double omSafe = std::max(omega, 1e-20);
    double ySafe  = std::max(y, 1e-20);
    double y2     = ySafe * ySafe;

    double term1 = 2.0 * sqrtK / (coeffs.betaStar * omSafe * ySafe);
    double term2 = 500.0 * nu / (y2 * omSafe);

    // arg2 = max(2*sqrt(k)/(betaStar*omega*y), 500*nu/(y^2*omega))
    // F2 = tanh(arg2^2)
    double arg2  = std::max(term1, term2);
    return std::tanh(arg2 * arg2);
}

// Cross-diffusion term: CDkw = 2*sigma_w2/omega * (gradK . gradOmega)
double SSTModel::crossDiffusion(const Vec3& gradK, const Vec3& gradOmega, double omega) const {
    double omSafe = std::max(omega, 1e-20);
    return 2.0 * coeffs.sigma_w2 / omSafe * gradK.dot(gradOmega);
}

// Eddy viscosity with Bradshaw limiter:
// ensures turbulent shear stress scales with turbulent kinetic energy
// prevents excessuve eddy viscosity in high-strain regions (e.g. near walls)
// nuT = a1*k / max(a1*omega, S*F2)
double SSTModel::eddyViscosity(double k, double omega, double S, double F2) const {
    double kSafe = std::max(k, 0.0);
    double denom = std::max(coeffs.a1 * omega, S * F2);
    denom = std::max(denom, 1e-20);
    return coeffs.a1 * kSafe / denom;
}

// Production with Menter limiter:
// prevents overprediction of turbulent kinetic energy at stagnation points (fluid velocity = 0)
// Pk = min(nuT * S^2, 10 * betaStar * k * omega)
double SSTModel::production(double nuT, double S, double k, double omega) const {
    double Pk_raw = nuT * S * S;
    double Pk_lim = 10.0 * coeffs.betaStar * std::max(k, 0.0) * std::max(omega, 1e-20);
    return std::min(Pk_raw, Pk_lim);
}

// Source term for k-equation:
// Sk = Pk - betaStar * omega * k
// production minus dissipation of turbulent kinetic energy
double SSTModel::sourceK(double Pk, double k, double omega) const {
    return Pk - coeffs.betaStar * std::max(omega, 1e-20) * std::max(k, 0.0);
}

// Source term of omega-equation
// Sw = alpha*(Pk/nuT) - beta*omega^2 + (1-F1)*max(CDkw, 0)
// combines turbulence production, dissipation, and cross-diffusion effects
double SSTModel::sourceOmega(double Pk, double nuT, double omega, double F1, double CDkw) const {
    double alphaB  = coeffs.alpha(F1);
    double betaB   = coeffs.beta(F1);
    double nuTSafe = std::max(nuT, 1e-20);
    return alphaB * Pk / nuTSafe - betaB * omega * omega + (1.0 - F1) * std::max(CDkw, 0.0);
}

// Field function
// computes 
void SSTModel::computeFields(
        const Mesh& mesh,      
        const ScalarField& k, 
        const ScalarField& omega,
        const VectorField& U, 
        double nu,
        ScalarField& nuT,                   // turbulent eddy viscosity
        ScalarField& F1field,               // iner/outer model blending
        ScalarField& F2field,               // eddy viscosity limiter
        ScalarField& Pk,                    // turbulence production
        ScalarField& CDkwField              // cross-diffusion term
    ) const {

    // velocity gradients - strain rate magnitude
    VelocityGradients vg = computeVelocityGradients(U);
    ScalarField Smag = strainRateMagnitude(vg);

    // k and omega gradients for cross-diffusion
    VectorField gradK     = greenGaussGrad(k);
    VectorField gradOmega = greenGaussGrad(omega);

    // computes distance to the nearest wall
    const auto& wd = mesh.wallDistance();

    // loops over cells
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double kc  = std::max(k[ci], 0.0);
        double wc  = std::max(omega[ci], 1e-20);
        double y   = std::max(wd[ci], 1e-20);
        double Sc  = Smag[ci];

        // cross-diffusion
        double CDkw = crossDiffusion(gradK[ci], gradOmega[ci], wc);
        CDkwField[ci] = CDkw;

        // blending functions
        double CDpos = std::max(CDkw, 1e-20);
        F1field[ci] = computeF1(kc, wc, y, nu, CDpos);
        F2field[ci] = computeF2(kc, wc, y, nu);

        // eddy viscosity
        nuT[ci] = eddyViscosity(kc, wc, Sc, F2field[ci]);

        // production
        Pk[ci] = production(nuT[ci], Sc, kc, wc);
    }
}