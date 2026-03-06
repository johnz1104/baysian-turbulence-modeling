#include "../include/SIMPLESolver.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

// Constructor 
// Returns initialized SIMPLESolver object
SIMPLESolver::SIMPLESolver(const Mesh& mesh, const SSTModel& sst,
                           const FlowBoundaryConditions& bcs, double nu,
                           const SolverSettings& settings)
    : mesh_(mesh), sst_(sst), bcs_(bcs), nu_(nu), settings_(settings),
      aP_(mesh.nCells(), 0.0), aPunrelaxed_(mesh.nCells(), 0.0)
{
    pSolver_ = makeSolver(settings_.pressureSolver);
    mSolver_ = makeSolver(settings_.momentumSolver);
    tSolver_ = makeSolver(settings_.turbulenceSolver);
}

// Sets uniform initial conditions for velocity, pressure, and turbulence variables (k, omega)
// initializes derived SST fields (nuT, F1, F2, Pk, CDkw), and applies boundary conditions before iteration
void SIMPLESolver::initUniform(FlowFields& f, const Vec3& Uinit, double pInit, double kInit, double omegaInit) {
    f.U.setUniform(Uinit);
    f.p.setUniform(pInit);
    f.k.setUniform(kInit);
    f.omega.setUniform(omegaInit);
    f.F1.setUniform(1.0);
    f.F2.setUniform(1.0);
    f.Pk.setUniform(0.0);
    f.CDkw.setUniform(0.0);

    // Initialize omega with the wall-distance profile omega = max(60nu/(beta1*y^2), omegaInit).
    // A uniform omega = omegaInit everywhere creates a cliff-edge gradient when wall cells are
    // later pinned to ~5800: diffusion from the wall cell drives the 2nd-row cell from 2.55 to
    // ~350 in the first turbulence iteration and the spike propagates inward until divergence.
    // Using the 1/y^2 profile from the start gives smooth, physics-consistent gradients.
    const auto& wd = mesh_.wallDistance();
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double y = std::max(wd[ci], 1e-20);
        double omWall = 60.0 * nu_ / (sst_.coeffs.beta1 * y * y);
        f.omega[ci] = std::max(omWall, omegaInit);
    }

    // nuT from the updated omega profile
    for (int ci = 0; ci < mesh_.nCells(); ++ci)
        f.nuT[ci] = sst_.coeffs.a1 * kInit / std::max(sst_.coeffs.a1 * f.omega[ci], 1e-20);

    applyAllBCs(f.U, f.p, f.k, f.omega, mesh_, bcs_, nu_);
}

// Momentum equation assembly (Ux, Uy, or Uz) 
// constructs a linear system for finite-volume discretization of momentum equation 
// builds A_U * U_component = b (where A_U is in LDU and b is source)
void SIMPLESolver::assembleMomentum(LinearSystem& sys, const FlowFields& f, int component, std::vector<double>& aP) {
    sys.zero();         // resets linear system
    int nIF = mesh_.nInternalFaces();

    // internal faces: convection + diffusion
    for (int fi = 0; fi < nIF; ++fi) {

        // face geometry
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // effective viscosity at face (molecular + eddy)
        double nuEff_o = nu_ + f.nuT[o];        // molecular viscosity
        double nuEff_n = nu_ + f.nuT[n];        // eddy viscosity
        double nuEff_f = face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n;     // interpolation

        // diffusion coefficient
        double Df = nuEff_f * Sf / delta;

        // mass flux through face (first-order upwind)
        // flux = rho * U_f . S_f   (rho = 1 for incompressible)
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;

        // upwind convection coefficients
        double cPos = std::max( mFlux, 0.0);   // flux from owner to neighbor
        double cNeg = std::max(-mFlux, 0.0);   // flux from neighbor to owner

        // owner equation: aP += Df + cPos,  aN = -(Df + cNeg)
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);   // off-diag contribution to owner from neighbor
        sys.lower[fi] = -(Df + cPos);   // off-diag contribution to neighbor from owner
    }

    // boundary faces
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {

        // face geometry
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double nuEff = nu_ + f.nuT[o];

        // diffusion term
        double Db = nuEff * Sf / delta;

        // computes boundary velocity component
        double Ub;
        if (component == 0) Ub = f.U.bface(fi).x;
        else if (component == 1) Ub = f.U.bface(fi).y;
        else Ub = f.U.bface(fi).z;

        // boundary mass flux
        Vec3 Ubv = f.U.bface(fi);
        double mFlux = (Ubv.x * face.normal.x + Ubv.y * face.normal.y
                      + Ubv.z * face.normal.z) * Sf;

        if (mFlux >= 0) {
            // outflow: convection from owner
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * Ub;
        } else {
            // inflow: fixed boundary value
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * Ub;
        }
    }

    // pressure gradient source (-dP/dx_component * V)
    VectorField gradP = greenGaussGrad(f.p);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double dpdx;
        if (component == 0) dpdx = gradP[ci].x;
        else if (component == 1) dpdx = gradP[ci].y;
        else dpdx = gradP[ci].z;
        sys.source[ci] -= dpdx * vol;
    }

    // In SIMPLE, the momentum equation is usually under-relaxed to improve stability.
    // under-relaxation: numerical stabilization technique - prevents solution from changing too drastically between iterations
    // makes small adjustments to allow for smooth convergence

    // under-relaxation:  aP_new = aP / alphaU,  source += (1-alphaU)/alphaU * aP * phi_old
    double alphaU = settings_.alphaU;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double phi_old;
        if (component == 0) phi_old = f.U[ci].x;
        else if (component == 1) phi_old = f.U[ci].y;
        else phi_old = f.U[ci].z;

        aPunrelaxed_[ci] = sys.diag[ci];  // store UNRELAXED diagonal for Rhie-Chow pressure Laplacian
        sys.source[ci] += (1.0 - alphaU) / alphaU * sys.diag[ci] * phi_old;
        sys.diag[ci] /= alphaU;
        aP[ci] = sys.diag[ci];   // store RELAXED diagonal (after /alphaU) for velocity correction
    }
}

// Pressure correction equation assembly
// constructs a finite-volume linear system for the pressure correction equation in SIMPLE algorithm
// constructs Laplacian operator using inverse momentum diagonal and adding source terms from divergence of predicted velocity field
// Laplacian(p') = div(U*)  where coefficients use rAP = 1/aP per cell
// ensures that div(U*) = 0
void SIMPLESolver::assemblePressureCorrection(LinearSystem& sys,
                                              const FlowFields& f,
                                              const std::vector<double>& aP,
                                              ScalarField& pPrime) {
    sys.zero();
    pPrime.setUniform(0.0);
    int nIF = mesh_.nInternalFaces();

    // internal faces
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // Pressure correction Laplacian: ∇·((V/aP) ∇p') = div(U*)
        // Face coefficient D_f = (V/aP)_f * A_f / delta
        // Interpolate (V/aP) from owner and neighbor cells
        double Vo = mesh_.cell(o).volume;
        double Vn = mesh_.cell(n).volume;
        double dP_o = (std::abs(aP[o]) > 1e-30) ? Vo / aP[o] : 0.0;
        double dP_n = (std::abs(aP[n]) > 1e-30) ? Vn / aP[n] : 0.0;
        double dP_f = face.weight * dP_o + (1.0 - face.weight) * dP_n;

        // pressure correction Laplacian coefficient
        double coeff = dP_f * Sf / delta;

        sys.diag[o] += coeff;
        sys.diag[n] += coeff;
        sys.upper[fi] = -coeff;
        sys.lower[fi] = -coeff;

        // mass flux source: U*.Sf
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double massFlux = (Uf.x * face.normal.x + Uf.y * face.normal.y
                         + Uf.z * face.normal.z) * Sf;

        // Source = -div(U*)*V: FV Laplacian sum(coeff*(p'_P - p'_N)) = -∇²p'*V,
        // so the Poisson equation rAP*∇²p' = div(U*) maps to source = -div(U*)*V.
        sys.source[o] -= massFlux;
        sys.source[n] += massFlux;
    }

    // boundary faces: mass flux source + Dirichlet diffusion for outlet (p'=0 there)
    // Wall and inlet use zero-gradient p' (Neumann) — no diffusion term needed.
    // Outlet uses Dirichlet p'=0 — the Laplacian coefficient must be added to the diagonal
    // so the pressure correction at the outlet-adjacent cell is actually constrained.
    // Without this, the p' Poisson system has no Dirichlet row and is singular.
    for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
        const Patch& pat = mesh_.patch(pi);
        bool isOutlet = (pat.type == "outlet");
        for (FaceID fi : pat.faces) {
            const Face& face = mesh_.face(fi);
            int o = face.owner;
            double Sf = face.area;
            double delta = std::max(face.delta, 1e-20);

            // boundary mass flux source (same sign convention as internal)
            Vec3 Ub = f.U.bface(fi);
            double massFlux = (Ub.x * face.normal.x + Ub.y * face.normal.y
                             + Ub.z * face.normal.z) * Sf;
            sys.source[o] -= massFlux;

            // outlet Dirichlet: p'=0 → add diffusion coeff, source += coeff*0 = 0
            if (isOutlet) {
                double Vo = mesh_.cell(o).volume;
                double dP_o = (std::abs(aP[o]) > 1e-30) ? Vo / aP[o] : 0.0;
                double coeff = dP_o * Sf / delta;
                sys.diag[o] += coeff;
                // sys.source[o] += coeff * 0.0 — omitted (p'_outlet = 0)
            }
        }
    }
}

// Correction step of the SIMPLE Algorithm

// Velocity correction:  U = U* - gradP' * V / aP
// adjusts predicted velocity using pressure-correction gradient and momentum diagonal coefficients
void SIMPLESolver::correctVelocity(FlowFields& f, const ScalarField& pPrime, const std::vector<double>& aP) {
    VectorField gradPp = greenGaussGrad(pPrime);
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double rAP = (std::abs(aP[ci]) > 1e-30) ? 1.0 / aP[ci] : 0.0;
        double vol = mesh_.cell(ci).volume;
        f.U[ci].x -= rAP * gradPp[ci].x * vol;
        f.U[ci].y -= rAP * gradPp[ci].y * vol;
        f.U[ci].z -= rAP * gradPp[ci].z * vol;
    }
    applyVelocityBC(f.U, mesh_, bcs_);      // boundary values must be re-enforced after modifying interior cells
}

// Pressure correction:  p = p + alphaP * p'
// updates pressure field with an under-relaxed pressure correction so that corrected U satisfies incompressible continuity equation (div U = 0)
void SIMPLESolver::correctPressure(FlowFields& f, const ScalarField& pPrime) {
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        f.p[ci] += settings_.alphaP * pPrime[ci];   // SIMPLE pressure update with under-relaxation to prevent instability
    }
    applyPressureBC(f.p, mesh_, bcs_);
}


// k-equation assembly
// converts continuous k-transport equation into a finite-volume matrix form (A_k * k = b_k)
//      discretizing convection and diffusion terms across faces, applying boundary conditions, 
//      adding production and destruction source terms, and applying under-relaxation for stability 
//      before the equation is solved
void SIMPLESolver::assembleKEquation(LinearSystem& sys, const FlowFields& f) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    // loop over internal faces
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // diffusion term
        double F1_f = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sk = sst_.coeffs.sigma_k(F1_f);
        double nuEff_o = nu_ + sk * f.nuT[o];                                       // effective diffusivity: nu + sigma_k * nuT
        double nuEff_n = nu_ + sk * f.nuT[n];
        double nuEff_f = face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n;     // interpolate to face
        double Df = nuEff_f * Sf / delta;

        // convection term
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);                                  // velocity at the face
        double mFlux = (Uf.x * face.normal.x + Uf.y * face.normal.y + Uf.z * face.normal.z) * Sf;       // mass flux
        double cPos = std::max( mFlux, 0.0);                                                            // upwind discretization
        double cNeg = std::max(-mFlux, 0.0);
        
        // add matrix coefficients
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    // boundary faces
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double sk = sst_.coeffs.sigma_k(f.F1[o]);
        double nuEff = nu_ + sk * f.nuT[o];
        double Db = nuEff * Sf / delta;         // diffusion coefficient

        double kb = f.k.bface(fi);              // boundary k value
        Vec3 Ub = f.U.bface(fi);                // boundary velocity
        double mFlux = (Ub.x * face.normal.x + 
                        Ub.y * face.normal.y + 
                        Ub.z * face.normal.z) * Sf; // mass flux

        // outflow boundary
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * kb;
        } 
        // inflow boundary
        else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * kb;
        }
    }

    // source terms: Pk (explicit), betaStar*omega*k (linearised destruction)
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        // production term (explicit)
        sys.source[ci] += f.Pk[ci] * vol;
        // destruction term (linearised): betaStar * omega * k → diag += betaStar*omega*V
        double destruction = sst_.coeffs.betaStar * std::max(f.omega[ci], 1e-20);
        sys.diag[ci] += destruction * vol;
    }

    // under-relaxation
    double alphaK = settings_.alphaK;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaK) / alphaK * sys.diag[ci] * f.k[ci];
        sys.diag[ci] /= alphaK;
    }
}

// omega equation assembly
// converts continous SST omega transport equation into finite-volume matrix form (A_ω * ω = b_ω)
//      discretize convection and diffusion fluxes across faces, applying boundary conditions, 
//      adding turbulence production, destruction, and cross-diffusion source terms, and 
//      applying under-relaxation before solving for the updated ω field
void SIMPLESolver::assembleOmegaEquation(LinearSystem& sys, const FlowFields& f, const ScalarField& Smag) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();
    // Smag is passed in (frozen at the pre-correction U from computeFields).
    // Do NOT recompute from f.U here: the post-correction velocity may carry
    // pressure-correction oscillations that would inflate S^2 and blow up omega.

    // internal face loop
    for (int fi = 0; fi < nIF; ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        int n = face.neighbor;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);

        // diffusion term div((nu + sw * nuT) * grad(omega))
        double F1_f = face.weight * f.F1[o] + (1.0 - face.weight) * f.F1[n];
        double sw = sst_.coeffs.sigma_w(F1_f);                  
        double nuEff_o = nu_ + sw * f.nuT[o];                   // effective diffusivity: nu + sigma_w * nuT
        double nuEff_n = nu_ + sw * f.nuT[n];
        double nuEff_f = face.weight * nuEff_o + (1.0 - face.weight) * nuEff_n;
        double Df = nuEff_f * Sf / delta;                       // face diffusion coefficient

        // convection term (U dot grad(omega))
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double mFlux = (Uf.x * face.normal.x + 
                        Uf.y * face.normal.y + 
                        Uf.z * face.normal.z) * Sf;             // mass flux

        // upwind discretization (determines flow direction)
        double cPos = std::max( mFlux, 0.0);
        double cNeg = std::max(-mFlux, 0.0);
        
        // matrix coefficients: 
        // builds convection-diffusion operator
        sys.diag[o] += Df + cPos;
        sys.diag[n] += Df + cNeg;
        sys.upper[fi] = -(Df + cNeg);
        sys.lower[fi] = -(Df + cPos);
    }

    // boundary faces
    // computes diffusion coefficient, boundary omega value, and mass flux
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;
        double delta = std::max(face.delta, 1e-20);
        double sw = sst_.coeffs.sigma_w(f.F1[o]);
        double nuEff = nu_ + sw * f.nuT[o];
        double Db = nuEff * Sf / delta;

        double wb = f.omega.bface(fi);
        Vec3 Ub = f.U.bface(fi);
        double mFlux = (Ub.x * face.normal.x + 
                        Ub.y * face.normal.y +
                        Ub.z * face.normal.z) * Sf;
        
        // outflow boundary
        if (mFlux >= 0) {
            sys.diag[o] += mFlux + Db;
            sys.source[o] += Db * wb;
        } 
        // inflow boundary
        else {
            sys.diag[o] += Db;
            sys.source[o] += (Db - mFlux) * wb;
        }
    }

    // source terms
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double vol = mesh_.cell(ci).volume;
        double F1  = f.F1[ci];
        double omC = std::max(f.omega[ci], 1e-20);

        // production: alpha * S^2  (standard Menter SST formulation)
        // This is naturally self-limiting: as omega grows, destruction (beta*omega^2)
        // increases quadratically while production (alpha*S^2) stays fixed, guaranteeing
        // omega convergence.  
        // After fiddling: DO NOT use Pk/nuT: when the Bradshaw limiter reduces nuT,  dividing by the 
        // reduced nuT amplifies production and creates a positive feedback.
        double alphaB = sst_.coeffs.alpha(F1);
        double S = Smag[ci];
        sys.source[ci] += alphaB * S * S * vol;

        // destruction term (linearised) beta*omega^2 → diag += beta*omega*V
        double betaB = sst_.coeffs.beta(F1);
        sys.diag[ci] += betaB * omC * vol;

        // cross-diffusion (explicit): (1-F1)*max(CDkw, 0)
        sys.source[ci] += (1.0 - F1) * std::max(f.CDkw[ci], 0.0) * vol;
    }

    // under-relaxation
    double alphaW = settings_.alphaOmega;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        sys.source[ci] += (1.0 - alphaW) / alphaW * sys.diag[ci] * f.omega[ci];
        sys.diag[ci] /= alphaW;
    }
}

// placeholder for tracking field residuals
// currently we rely on linear solver residuals 
double SIMPLESolver::computeResidual(const FlowFields& f, int component) {
    (void)f; (void)component;
    return 0.0; // actual convergence tracked through linear solver results
}

// Main SIMPLE loop     
// SIMPLE algorithm runs until flow solution converges or diverges
ConvergenceHistory SIMPLESolver::solve(FlowFields& f) { // input: FlowField object / flow variables
    ConvergenceHistory hist;
    turbNormSet_ = false;
    // Create working linear systems
    LinearSystem momSys  = makeSystem(mesh_);
    LinearSystem pSys    = makeSystem(mesh_);
    LinearSystem kSys    = makeSystem(mesh_);
    LinearSystem omSys   = makeSystem(mesh_);
    ScalarField  pPrime(mesh_, "p'");

    // Smag = Strain-rate magnitude
    // Frozen strain-rate field: computed from the pre-correction U (same U computeFields uses)
    // Passed to assembleOmegaEquation so both k and omega production see the same velocity gradients, preventing omega blow-up.
    ScalarField SmagFrozen(mesh_, "Smag");

    // SIMPLE Iteration loop
    for (int iter = 0; iter < settings_.maxIterations; ++iter) {
        // 1. Update SST fields (at turbUpdateInterval cadence after turbStartIter)
        bool turbActive = (iter >= settings_.turbStartIter);
        bool turbUpdate = turbActive &&
                          ((iter - settings_.turbStartIter) % settings_.turbUpdateInterval == 0);
        if (turbUpdate) {
            // Save old nuT for under-relaxation
            std::vector<double> nuT_old(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                nuT_old[ci] = f.nuT[ci];

            sst_.computeFields(mesh_, f.k, f.omega,
                               f.U, nu_, f.nuT, f.F1,
                               f.F2, f.Pk, f.CDkw); // Computes SST quantities (nuT (eddy viscosity), F1, F2, Pk, CDkw)

            // Freeze Smag from the same U that computeFields used (pre-correction)
            // assembleOmegaEquation will use this instead of recomputing from the post-correction U
            // Post-corrected velocity can contain numerical oscillations because it is updated by discrete pressure gradients
            // Since turbulence production depends on velocity derivatives
            //      - those oscillations can artificially inflate strain rate and destabilize the ω equation
            // Using the frozen Smag decouples the turbulence model from these numerical oscillations
            SmagFrozen = strainRateMagnitude(computeVelocityGradients(f.U));

            // nuT floor: prevents nuT from collapsing to zero in early iterations
            // Set `nuT_min = 0.1 * nu` 
            // This prevents eddy viscosity from collapsing to zero in cells where the Bradshaw limiter produces 
            // extremely small values (like near-wall cells where omega is very large and drive nuT -> 0)
            //    - prevents cases that removes turbulent diffusion from the momentum equation 
            //      and creates unphysical laminar-like velocity gradients
            const double nuTMin = 0.1 * nu_;
            for (int ci = 0; ci < mesh_.nCells(); ++ci)
                f.nuT[ci] = std::max(f.nuT[ci], nuTMin);
        }

        // 2. Assemble + solve momentum x, y, z 
        SolverResult resUx, resUy, resUz;
        {
            assembleMomentum(momSys, f, 0, aP_);
            std::vector<double> Ux(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci) Ux[ci] = f.U[ci].x;
            resUx = mSolver_->solve(momSys, Ux, settings_.innerIterations,
                                    settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].x = Ux[ci];
        }
        // store aP from Ux for corrections (diagonal dominance is similar for all components)
        std::vector<double> aPstore = aP_;              // relaxed — for velocity correction
        std::vector<double> aPrhie = aPunrelaxed_;      // unrelaxed — for pressure Laplacian
        {
            assembleMomentum(momSys, f, 1, aP_);
            std::vector<double> Uy(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci) Uy[ci] = f.U[ci].y;
            resUy = mSolver_->solve(momSys, Uy, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].y = Uy[ci];
        }
        {
            assembleMomentum(momSys, f, 2, aP_);
            std::vector<double> Uz(mesh_.nCells());
            for (int ci = 0; ci < mesh_.nCells(); ++ci) Uz[ci] = f.U[ci].z;
            resUz = mSolver_->solve(momSys, Uz, settings_.innerIterations, settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.U[ci].z = Uz[ci];
        }
        applyVelocityBC(f.U, mesh_, bcs_);

        // 3. Assemble + solve pressure correction
        // Use RELAXED aP (= aP_raw / alphaU) for both pressure Laplacian and velocity correction 
        // This is necessary for consistency in SIMPLE algorithm: the momentum equation is solved with relaxed diagonal, 
        // and the correction step must use the same coefficients (so div(U_corrected) = 0). 
        // Note: U_corrected = U* - U' = U* - r_AP * ∇p' (r_AP = V/a_p)
        //  - when you solve the momentum eq, you get provisional velocity U* which does not satisfy continuity
        //  - SIMPLE introduces a correction U = U* + U' where U' is velocity correction derived from pressure correction or U_corrected
        assemblePressureCorrection(pSys, f, aPstore, pPrime);
        std::vector<double> ppVec(mesh_.nCells(), 0.0);
        SolverResult resP = pSolver_->solve(pSys,
                                            ppVec, 
                                            settings_.innerIterations,
                                            settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci) pPrime[ci] = ppVec[ci];
        
        // Gradient of pressure correction p' was being computed using incorrect boundary face values
        // Apply p' boundary conditions before computing grad(p') in correctVelocity
        //  - At wall/inlet, grad(p') = 0, but not necessarily p' itself
        //      - to enforce grad(p') = 0, we set boundary face equal to value of adjacent interior cell
        //  - At outlet, BC is p' = 0 (Dirichlet) - bface stays at 0 (default)
        // Without this, greenGaussGrad sees bface=0 on all faces (initialization default)
        //  - this would give wrong gradients at boundary-adjacent cells and causing velocity to grow without bound
        for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
            const Patch& pat = mesh_.patch(pi);
            if (pat.type == "outlet") continue; // keep Dirichlet p'=0
            for (FaceID fi : pat.faces)
                pPrime.bface(fi) = pPrime[mesh_.face(fi).owner];
        }

        // 4. Correct velocity and pressure (same relaxed aP)
        correctVelocity(f, pPrime, aPstore);
        correctPressure(f, pPrime);

        // 5. Assembly + solve Turbulence equations (k and omega) (on turbulence update iterations)
        SolverResult resK = {}, resOm = {};
        if (turbUpdate) {
            // k equation
            assembleKEquation(kSys, f);
            std::vector<double> kVec = f.k.data();
            resK = tSolver_->solve(kSys, kVec, settings_.innerIterations,
                                   settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.k[ci] = kVec[ci];
            f.k.clamp(settings_.kMin, settings_.kMax);
            applyKBC(f.k, mesh_, bcs_);

            // omega equation (uses frozen Smag from pre-correction U)
            assembleOmegaEquation(omSys, f, SmagFrozen);
            std::vector<double> omVec = f.omega.data();
            resOm = tSolver_->solve(omSys, omVec, settings_.innerIterations,
                                    settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.omega[ci] = omVec[ci];
            f.omega.clamp(settings_.omegaMin, 1e15);
            applyOmegaBC(f.omega, mesh_, bcs_, nu_, sst_.coeffs.beta1);

            // Wall omega: re-pin near-wall cell centers to the Menter formula each iteration
            // The ω equation strongly overproduces near walls. The α·S² production term/ drives ω toward O(1e5) 
            // and boundary-face diffusion alone cannot enforce the Dirichlet condition
            // Overriding after each solve is standard practice in SST implementations
            for (int pi = 0; pi < mesh_.nPatches(); ++pi) {
                const Patch& pat = mesh_.patch(pi);
                if (pat.type != "wall") continue;
                for (FaceID fi : pat.faces) {
                    const Face& face = mesh_.face(fi);
                    double y1 = std::max(face.delta, 1e-20);
                    f.omega[face.owner] = 60.0 * nu_ / (sst_.coeffs.beta1 * y1 * y1);
                }
            }
        }

        // 6. Track residuals using absolute initial residual (equation imbalance)
        //    normalised by iter-0 values so convergence = orders-of-magnitude reduction.
        ResidualEntry entry;
        entry.iteration = iter;
        entry.Ux    = resUx.initialRes;
        entry.Uy    = resUy.initialRes;
        entry.Uz    = resUz.initialRes;
        entry.p     = resP.initialRes;
        entry.k     = resK.initialRes;
        entry.omega = resOm.initialRes;

        // store iter-0 norms for normalisation
        if (iter == 0) {
            normUx0_ = std::max(entry.Ux, 1e-30);
            normUy0_ = std::max(entry.Uy, 1e-30);
            normP0_  = std::max(entry.p,  1e-30);
        }
        // turbulence norms are set at the first iteration where they are active
        if (turbActive && !turbNormSet_) {
            normK0_  = std::max(entry.k,  1e-30);
            normOm0_ = std::max(entry.omega, 1e-30);
            turbNormSet_ = true;
        }

        // normalised residuals (skip equations with negligible iter-0 residual)
        // in 2D, Uy/Uz have zero forcing so normUy0_ ~ 1e-30.  dividing any tiny
        // Uy residual by 1e-30 produces O(1e10), triggering false divergence.
        // safeNorm returns 0 for these equations so they don't affect convergence.
        auto safeNorm = [](double val, double ref) -> double {
            if (ref < 1e-20) return 0.0; // equation has no forcing; always "converged"
            return val / ref;
        };
        double nUx = safeNorm(entry.Ux, normUx0_);
        double nUy = safeNorm(entry.Uy, normUy0_);
        double nP  = safeNorm(entry.p,  normP0_);
        double nK  = safeNorm(entry.k,  normK0_);
        double nOm = safeNorm(entry.omega, normOm0_);
        hist.entries.push_back(entry);

        // computes maximum normalised residual
        double maxRes = std::max({nUx, nUy, nP});
        if (turbActive)
            maxRes = std::max({maxRes, nK, nOm});

        if (settings_.verbose && (iter % settings_.reportInterval == 0 || iter == 0)) {
            std::cout << "  SIMPLE iter " << iter
                      << "  Ux=" << nUx << "  Uy=" << nUy
                      << "  p=" << nP;
            if (turbUpdate)
                std::cout << "  k=" << nK << "  w=" << nOm;
            std::cout << "\n";
        }

        // convergence check (all normalised residuals below tolerance)
        if (maxRes < settings_.convergenceTol && iter > 0) {
            hist.converged = true;
            hist.finalIter = iter;
            if (settings_.verbose)
                std::cout << "  SIMPLE converged at iteration " << iter << "\n";
            return hist;
        }

        // If any individual residual is NaN, force maxRes to infinity so divergence is caught
        // (std::max silently propagates NaN, masking divergence as a false convergence)
        if (std::isnan(entry.Ux) || std::isnan(entry.Uy) || std::isnan(entry.Uz)
            || std::isnan(entry.p) || std::isnan(entry.k) || std::isnan(entry.omega))
            maxRes = std::numeric_limits<double>::infinity();

        // divergence check
        if (std::isnan(maxRes) || std::isinf(maxRes) || maxRes > settings_.divergenceLimit) {
            hist.diverged = true;
            hist.finalIter = iter;
            if (settings_.verbose)
                std::cout << "  SIMPLE diverged at iteration " << iter << "\n";
            return hist;
        }
    }

    hist.finalIter = settings_.maxIterations;
    if (settings_.verbose)
        std::cout << "  SIMPLE reached maxIter (" << settings_.maxIterations << ")\n";
    return hist;
}