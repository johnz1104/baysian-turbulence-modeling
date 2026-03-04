#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <cmath>
#include <algorithm>

// Container for SST k-omega turbulence model constants
struct SSTCoefficients {
    // k-omega model coefficients (near-wall layer)
    double sigma_k1  = 0.85;
    double sigma_w1  = 0.5;
    double beta1     = 0.075;
    double alpha1    = 5.0 / 9.0;   // ~0.5556

    // k-epsilon model coefficients (expressed in k-omega form) (away-from-wall layer)
    double sigma_k2  = 1.0;
    double sigma_w2  = 0.856;
    double beta2     = 0.0828;
    double alpha2    = 0.44;

    // general SST constants
    double betaStar  = 0.09;
    double a1        = 0.31;        // Bradshaw structural parameter
    double kappa     = 0.41;        // von Karman (usually fixed)

    // blended coefficient: phi = F1*phi1 + (1-F1)*phi2
    // F1 = 1: pure k-omega
    // F1 = 0: pure k-epsilon
    double blend(double phi1,   // k-omega coefficient 
                 double phi2,   // k-epsilon coefficient    
                 double F1)     // blending function
                 const {
        return F1 * phi1 + (1.0 - F1) * phi2;
    }
    double sigma_k(double F1) const { return blend(sigma_k1, sigma_k2, F1); }
    double sigma_w(double F1) const { return blend(sigma_w1, sigma_w2, F1); }
    double beta(double F1)    const { return blend(beta1, beta2, F1); }
    double alpha(double F1)   const { return blend(alpha1, alpha2, F1); }
};

// SSTModel
// Computes F1, F2, nuT, Pk, and source terms
class SSTModel {
public:
    SSTCoefficients coeffs;

    // Constructors
    SSTModel() = default;
    explicit SSTModel(const SSTCoefficients& c) : coeffs(c) {}

    // Pointwise functions - operate on one control volume (single cell)
    double computeF1(double k, double omega, double y, double nu, double CDkw_pos) const;       // primary blending function (detects if cell is in boundary layer)
    double computeF2(double k, double omega, double y, double nu) const;                        // second blending function (used in eddy viscosity limiter)
    double crossDiffusion(const Vec3& gradK, const Vec3& gradOmega, double omega) const;        // computes cross-diffusion term in omega equation
    double eddyViscosity(double k, double omega, double S, double F2) const;                    // computes turbulent viscosity
    double production(double nuT, double S, double k, double omega) const;                      // computes turbulence production term
    double sourceK(double Pk, double k, double omega) const;                                    // computes source term in k equation
    double sourceOmega(double Pk, double nuT, double omega, double F1, double CDkw) const;      // computes RHS of omega equation

    // Computes for the entire mesh
    void computeFields(const Mesh& mesh,
                       const ScalarField& k, const ScalarField& omega,
                       const VectorField& U, double nu,
                       ScalarField& nuT, ScalarField& F1field,
                       ScalarField& F2field, ScalarField& Pk,
                       ScalarField& CDkwField) const;
};