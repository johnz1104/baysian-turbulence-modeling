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
      aP_(mesh.nCells(), 0.0)
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
    f.nuT.setUniform(sst_.coeffs.a1 * kInit / std::max(sst_.coeffs.a1 * omegaInit, 1e-20));
    f.F1.setUniform(1.0);
    f.F2.setUniform(1.0);
    f.Pk.setUniform(0.0);
    f.CDkw.setUniform(0.0);
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

    // under-relaxation is numerical stabilization technique - prevents solution from changing too drastically between iterations
    // makes small adjustments to allow for smooth convergence

    // under-relaxation:  aP_new = aP / alphaU,  source += (1-alphaU)/alphaU * aP * phi_old
    double alphaU = settings_.alphaU;
    for (int ci = 0; ci < mesh_.nCells(); ++ci) {
        double phi_old;
        if (component == 0) phi_old = f.U[ci].x;
        else if (component == 1) phi_old = f.U[ci].y;
        else phi_old = f.U[ci].z;

        aP[ci] = sys.diag[ci];   // store un-relaxed diagonal for pressure correction
        sys.source[ci] += (1.0 - alphaU) / alphaU * sys.diag[ci] * phi_old;
        sys.diag[ci] /= alphaU;
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

        // Rhie-Chow: rAP_f is interpolated inverse of un-relaxed diagonal
        // Laplacian(p') = div(U*)  where coefficients use rAP = 1/aP per cell
        double rAP_o = (std::abs(aP[o]) > 1e-30) ? 1.0 / aP[o] : 0.0; 
        double rAP_n = (std::abs(aP[n]) > 1e-30) ? 1.0 / aP[n] : 0.0;
        double rAP_f = face.weight * rAP_o + (1.0 - face.weight) * rAP_n;

        // pressure correction Laplacian coefficient
        double coeff = rAP_f * Sf * Sf / delta;

        sys.diag[o] += coeff;
        sys.diag[n] += coeff;
        sys.upper[fi] = -coeff;
        sys.lower[fi] = -coeff;

        // mass flux source: U*.Sf
        Vec3 Uf = f.U[o] * face.weight + f.U[n] * (1.0 - face.weight);
        double massFlux = (Uf.x * face.normal.x + Uf.y * face.normal.y
                         + Uf.z * face.normal.z) * Sf;

        sys.source[o] -= massFlux;   // div(U*) contributes with sign convention
        sys.source[n] += massFlux;
    }

    // boundary faces: mass flux source only (pressure correction is zero at boundaries)
    for (int fi = nIF; fi < mesh_.nFaces(); ++fi) {
        const Face& face = mesh_.face(fi);
        int o = face.owner;
        double Sf = face.area;

        Vec3 Ub = f.U.bface(fi);
        double massFlux = (Ub.x * face.normal.x + Ub.y * face.normal.y
                         + Ub.z * face.normal.z) * Sf;
        sys.source[o] -= massFlux;
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
void SIMPLESolver::assembleOmegaEquation(LinearSystem& sys, const FlowFields& f) {
    sys.zero();
    int nIF = mesh_.nInternalFaces();

    // strain rate for omega production
    VelocityGradients vg = computeVelocityGradients(f.U);
    ScalarField Smag = strainRateMagnitude(vg);     // omega production term depends on strain rate magnitude sqrt(2S_ij*S_ij)

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
        double S   = Smag[ci];
        double omC = std::max(f.omega[ci], 1e-20);

        // production term alpha * S^2
        double alphaB = sst_.coeffs.alpha(F1);
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
    // Create working linear systems
    LinearSystem momSys  = makeSystem(mesh_);
    LinearSystem pSys    = makeSystem(mesh_);
    LinearSystem kSys    = makeSystem(mesh_);
    LinearSystem omSys   = makeSystem(mesh_);
    ScalarField  pPrime(mesh_, "p'");

    // SIMPLE Iteration loop
    for (int iter = 0; iter < settings_.maxIterations; ++iter) {
        // 1. Update SST fields (after turbulence starts)
        if (iter >= settings_.turbStartIter) {
            sst_.computeFields(mesh_, f.k, f.omega, 
                               f.U, nu_, f.nuT, f.F1, 
                               f.F2, f.Pk, f.CDkw); // Computes SST quantities (nuT (eddy viscosity), F1, F2, Pk, CDkw)
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
        // store aP (diagonal coefficients) from Ux for pressure correction (diagonal dominance is similar for all components)
        std::vector<double> aPstore = aP_;
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
        assemblePressureCorrection(pSys, f, aPstore, pPrime);
        std::vector<double> ppVec(mesh_.nCells(), 0.0);
        SolverResult resP = pSolver_->solve(pSys, ppVec, settings_.innerIterations,
                                            settings_.innerTolerance);
        for (int ci = 0; ci < mesh_.nCells(); ++ci) pPrime[ci] = ppVec[ci];

        // 4. Correct velocity and pressure
        correctVelocity(f, pPrime, aPstore);
        correctPressure(f, pPrime);

        // 5. Assembly + solve Turbulence equations (k and omega) (if turbulence is active)
        SolverResult resK = {}, resOm = {};
        if (iter >= settings_.turbStartIter) {
            // k equation
            assembleKEquation(kSys, f);
            std::vector<double> kVec = f.k.data();
            resK = tSolver_->solve(kSys, kVec, settings_.innerIterations,
                                   settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.k[ci] = kVec[ci];
            f.k.clamp(settings_.kMin, 1e10);
            applyKBC(f.k, mesh_, bcs_);

            // omega equation
            assembleOmegaEquation(omSys, f);
            std::vector<double> omVec = f.omega.data();
            resOm = tSolver_->solve(omSys, omVec, settings_.innerIterations,
                                    settings_.innerTolerance);
            for (int ci = 0; ci < mesh_.nCells(); ++ci) f.omega[ci] = omVec[ci];
            f.omega.clamp(settings_.omegaMin, 1e15);
            applyOmegaBC(f.omega, mesh_, bcs_, nu_, sst_.coeffs.beta1);
        }

        // 6. Stores residuals
        ResidualEntry entry;
        entry.iteration = iter;
        entry.Ux    = resUx.finalRes;
        entry.Uy    = resUy.finalRes;
        entry.Uz    = resUz.finalRes;
        entry.p     = resP.finalRes;
        entry.k     = resK.finalRes;
        entry.omega = resOm.finalRes;
        hist.entries.push_back(entry);

        // computes maximum residual
        double maxRes = std::max({entry.Ux, entry.Uy, entry.Uz, entry.p});
        if (iter >= settings_.turbStartIter)
            maxRes = std::max({maxRes, entry.k, entry.omega});

        if (settings_.verbose && (iter % settings_.reportInterval == 0 || iter == 0)) {
            std::cout << "  SIMPLE iter " << iter
                      << "  Ux=" << entry.Ux << "  Uy=" << entry.Uy
                      << "  p=" << entry.p;
            if (iter >= settings_.turbStartIter)
                std::cout << "  k=" << entry.k << "  w=" << entry.omega;
            std::cout << "\n";
        }

        // convergence check
        if (maxRes < settings_.convergenceTol && iter > settings_.turbStartIter) {
            hist.converged = true;
            hist.finalIter = iter;
            if (settings_.verbose)
                std::cout << "  SIMPLE converged at iteration " << iter << "\n";
            return hist;
        }

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