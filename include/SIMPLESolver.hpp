#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "BoundaryCondition.hpp"
#include "LinearSolver.hpp"
#include "SSTModel.hpp"
#include <vector>
#include <string>
#include <memory>

// FlowFields - data container for all cell cenetered variables solved by SIMPLE
struct FlowFields {
    // Primary solved variables (unknowns in PDE system)
    VectorField U;          // velocity
    ScalarField p;          // pressure
    ScalarField k;          // turbulent kinetic energy
    ScalarField omega;      // specific dissipation rate

    // Turbulence model fields
    ScalarField nuT;        // eddy viscosity
    ScalarField F1, F2;     // SST blending
    ScalarField Pk;         // production of k
    ScalarField CDkw;       // cross-diffusion term

    // Constructor
    FlowFields() = default;
    explicit FlowFields(const Mesh& mesh)
        : U(mesh, "U"), p(mesh, "p"), k(mesh, "k"), omega(mesh, "omega"),
          nuT(mesh, "nuT"), F1(mesh, "F1"), F2(mesh, "F2"),
          Pk(mesh, "Pk"), CDkw(mesh, "CDkw") {}
};

// Convergence tracking
// determines whether or not solver has converged 
// solver is considered converged when all equation residuals r fall below their respective tolerance threshold
// i.e.: (R_Ux < eps_U) && (R_Uy < eps_U) && (R_Uz < eps_U) && (R_p < eps_p) && (R_k < eps_k) && (R_omega < eps_omega)

// stores residuals for one solver iteration (r = Ax - b)
struct ResidualEntry {
    int    iteration = 0;
    double Ux = 0, Uy = 0, Uz = 0;
    double p  = 0;
    double k  = 0, omega = 0;
};

// stores entire convergence history
struct ConvergenceHistory {
    std::vector<ResidualEntry> entries;
    bool converged = false;
    bool diverged  = false;
    int  finalIter = 0;
};

// SolverSettings - configuration parameters for the CFD solver
// iteration limits, convergence criteria, relaxation factors, linear solver controls, turbulence activation, and reporting)
struct SolverSettings {
    int    maxIterations   = 500;
    double convergenceTol  = 1e-5;
    double divergenceLimit = 1e6;

    // under-relaxation factors
    double alphaU     = 0.7;
    double alphaP     = 0.3;
    double alphaK     = 0.5;
    double alphaOmega = 0.5;

    // inner linear solver settings
    int    innerIterations = 200;
    double innerTolerance  = 1e-3;

    // turbulence activation
    int turbStartIter = 5;      // freeze turb for first N outer iterations

    // field clamps
    double kMin     = 1e-10;
    double omegaMin = 1e-6;

    // reporting
    int  reportInterval = 10;
    bool verbose        = true;

    // linear solver names
    std::string pressureSolver   = "PCG";
    std::string momentumSolver   = "BiCGSTAB";
    std::string turbulenceSolver = "BiCGSTAB";
};

// SIMPLE = Semi-Implicit Method for Pressure-Linked Equations
// SIMPLESolver — implements the SIMPLE algorithm for pressure–velocity coupling
// iteratively solves momentum and pressure correction equations to enforce continuity and converge the flow field
class SIMPLESolver {
public:
    // Constructor
    SIMPLESolver(const Mesh& mesh, 
                 const SSTModel& sst,
                 const FlowBoundaryConditions& bcs, 
                 double nu,
                 const SolverSettings& settings = {});

    // main entry: runs SIMPLE loop until convergence / maxIter / divergence
    ConvergenceHistory solve(FlowFields& fields);

    // initial conditions
    void initUniform(FlowFields& f, const Vec3& Uinit,
                     double pInit, double kInit, double omegaInit);

private:
    const Mesh& mesh_;
    SSTModel sst_;
    FlowBoundaryConditions bcs_;
    double nu_;
    SolverSettings settings_;

    // linear solvers
    std::unique_ptr<ILinearSolver> pSolver_;
    std::unique_ptr<ILinearSolver> mSolver_;
    std::unique_ptr<ILinearSolver> tSolver_;

    // working storage
    // stores diagonal momentum coefficients 
    // needed for Rhie-Chow interpolation (prevents pressure oscillations)
    std::vector<double> aP_;    

    // equation assembly functions
    // build the lienar system from the PDEs
    void assembleMomentum(LinearSystem& sys, const FlowFields& f, int component, std::vector<double>& aP);
    void assemblePressureCorrection(LinearSystem& sys, const FlowFields& f, const std::vector<double>& aP, ScalarField& pPrime);
    void assembleKEquation(LinearSystem& sys, const FlowFields& f);
    void assembleOmegaEquation(LinearSystem& sys, const FlowFields& f);

    // correction functions 
    void correctVelocity(FlowFields& f, const ScalarField& pPrime, const std::vector<double>& aP);
    void correctPressure(FlowFields& f, const ScalarField& pPrime);

    // residual calculation
    double computeResidual(const FlowFields& f, int component);
};