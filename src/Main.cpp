#include "../include/Mesh.hpp"
#include "../include/Field.hpp"
#include "../include/BoundaryCondition.hpp"
#include "../include/LinearSolver.hpp"
#include "../include/SSTModel.hpp"
#include "../include/SIMPLESolver.hpp"
#include "../include/InferenceParameters.hpp"
#include "../include/ObservationOperator.hpp"
#include "../include/ForwardModel.hpp"
#include <iostream>
#include <cmath>
#include <string>
#include <cstring>

// Channel flow validation - turbulence model validation test


static void validateChannel() {
    std::cout << "=== Channel Flow Validation (Re_b = 6800) ===\n\n";

    // Channel geometry = [0, Lx] x [0, 2h]
    // Re_b = U_b * h / nu  (bulk Reynolds number based on half-height)
    // Re_b = 6800, nu chosen so U_b = 1
    // h = half height, Lx = length of top/bottom wall
    // Cf = 0.073 * Re_b^(-0.25)   (Dean 1978)

    double h = 1.0;         // half-height
    double Lx = 10.0 * h;  // channel length (10h for developed flow)
    int nx = 60, ny = 40;
    double Ub = 1.0;
    double Re_b = 6800.0;
    double nu = Ub * h / Re_b;      // nu = U_b * h / Re_b = 1.0 * 1.0 / 6800

    std::cout << "  nu = " << nu << "  (Re_b = " << Re_b << ")\n";
    std::cout << "  Mesh: " << nx << " x " << ny << "  domain: " << Lx << " x " << 2*h << "\n";

    // build mesh (Re-adaptive wall clustering)
    Mesh mesh = Mesh::makeChannel2D(nx, ny, Lx, 2.0 * h, Re_b);
    mesh.computeWallDistance();     // SST model used wall distance in F_1 (blending function) and nu_t (turbulence viscosity)

    std::cout << "  Cells: " << mesh.nCells() << "  Faces: " << mesh.nFaces()
              << "  Internal: " << mesh.nInternalFaces() << "\n";

    // boundary conditions
    // inlet turbulence intensity estimate: Tu ~ 5% (ratio of RMS velocity fluctiation to mean velocity - u'/U)
    double Tu = 0.05;
    double kIn = 1.5 * (Ub * Tu) * (Ub * Tu);       // turbulence kinetic energy
    // nuT/nu ~ 100 at inlet.  with nuT/nu ~ 10 (nu * 10.0), turbulence never developed —
    // nuT stayed too low throughout the solve, giving Cf errors > 30%.  nuT/nu ~ 100
    // gives omega a realistic inlet value so the SST model reaches physical equilibrium.
    double omIn = kIn / (nu * 100.0);

    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, Ub, kIn, omIn);

    // SST model with default coefficients
    SSTModel sst;
    SolverSettings settings;
    settings.maxIterations  = 5000;  // increased from 2000 to 5000 for higher accuracy (pressure residual converges slowly)
    settings.convergenceTol = 1e-4;
    settings.reportInterval = 200;
    settings.innerIterations = 500;
    settings.innerTolerance  = 1e-6;
    settings.verbose        = true;
    // Standard relaxation
    settings.alphaU         = 0.7;
    settings.alphaP         = 0.5;
    settings.alphaK         = 0.5;
    settings.alphaOmega     = 0.5;
    // turbStartIter=0: couple turbulence from the first iteration so the velocity
    // field never builds unphysical laminar gradients.  A delayed start (e.g. 20)
    // lets pure-laminar SIMPLE create high-shear profiles that explode Pk the
    // moment SST activates, driving k to blow up within ~18 turbulence iterations.
    settings.turbStartIter  = 0;
    settings.turbUpdateInterval = 1;  // update every iteration (p' BC fix provides stability)

    // With `kMax = 10 * kIn = 0.0375`, k was being clamped in cells where it should have been higher.
    // Relaxed to `kMax = 0.1`
    // Physical bounds: generous limits that prevent NaN but don't interfere with the natural turbulence equilibrium.
    // An overly tight kMax prevents k from reaching its natural level, 
    // forcing all production energy into omega growth, which then collapses nuT via the Bradshaw limiter.
    settings.kMax    = 0.1;              // generous physical bound
    settings.omegaMin = 1e-6;            // default

    // running SIMPLE solver
    SIMPLESolver solver(mesh, sst, bcs, nu, settings);  // pressure-velocity coupling algorithm
    FlowFields fields(mesh);
    solver.initUniform(fields, Vec3(Ub, 0, 0), 0.0, kIn, omIn);

    std::cout << "\n  Running SIMPLE...\n";
    ConvergenceHistory hist = solver.solve(fields);

    // compute wall shear stress -> Cf at downstream wall
    // Cf = tau_wall / (0.5 * rho * Ub^2)
    // tau_wall ≈ (nu + nuT) * |dU/dy|_wall ≈ (nu + nuT) * U_cell / delta
    double tauSum = 0.0;
    int nWallFaces = 0;
    for (int pi = 0; pi < mesh.nPatches(); ++pi) {
        const Patch& pat = mesh.patch(pi);
        if (pat.type != "wall") continue;
        for (FaceID fi : pat.faces) {
            const Face& face = mesh.face(fi);
            // only downstream half of channel for developed flow
            if (face.center.x < 0.5 * Lx) continue;
            int ow = face.owner;
            double nuEff = nu + fields.nuT[ow];
            double delta = std::max(face.delta, 1e-20);
            Vec3 Uc = fields.U[ow];
            Vec3 Ut = Uc - face.normal * Uc.dot(face.normal);
            double tau = nuEff * Ut.norm() / delta;
            tauSum += tau;
            nWallFaces++;
        }
    }

    double tauAvg = (nWallFaces > 0) ? tauSum / nWallFaces : 0.0;
    double Cf_sim = tauAvg / (0.5 * Ub * Ub);
    double Cf_dean = 0.073 * std::pow(Re_b, -0.25);

    std::cout << "\n  === Results ===\n";
    std::cout << "  SIMPLE iterations: " << hist.finalIter
              << " (" << (hist.converged ? "converged" : hist.diverged ? "DIVERGED" : "maxIter") << ")\n";
    std::cout << "  Cf (simulation):   " << Cf_sim << "\n";
    std::cout << "  Cf (Dean 1978):    " << Cf_dean << "\n";
    double error = std::abs(Cf_sim - Cf_dean) / Cf_dean * 100.0;
    std::cout << "  Relative error:    " << error << "%\n";

    if (error < 30.0)
        std::cout << "  STATUS: PASS (within 30% of Dean's correlation)\n";
    else
        std::cout << "  STATUS: FAIL (error > 30% — check mesh resolution or BCs)\n";
}

//  Flat plate validation — Schoenherr Cf(x)
//  Re_L = U_inf * L / nu = 1e5
//  Cf = 0.0592 * Re_x^(-0.2)   (turbulent Blasius / Schoenherr)
static void validatePlate() {
    std::cout << "\n=== Flat Plate Validation (Re_L = 2e6) ===\n\n";

    double L = 1.0;
    double Uinf = 1.0;
    // Re-adaptive mesh clustering now computes stretch automatically for y+ ~ 1
    double Re_L = 2e6;
    double nu = Uinf * L / Re_L;
    int nx = 60, ny = 40;

    std::cout << "  nu = " << nu << "  (Re_L = " << Re_L << ")\n";
    std::cout << "  Mesh: " << nx << " x " << ny << "\n";

    Mesh mesh = Mesh::makeChannel2D(nx, ny, L, 0.1 * L, Re_L);
    mesh.computeWallDistance();

    double Tu = 0.01;
    double kIn = 1.5 * (Uinf * Tu) * (Uinf * Tu);
    // nuT/nu ~ 100 at inlet (same reasoning as channel — with nuT/nu ~ 5 turbulence never develops)
    double omIn = kIn / (nu * 100.0);

    FlowBoundaryConditions bcs = FlowBoundaryConditions::flatPlateDefaults(mesh, Uinf, kIn, omIn);

    SSTModel sst;
    SolverSettings settings;
    settings.maxIterations  = 5000;   // flat plate needs many iterations (pressure converges slowly)
    settings.convergenceTol = 1e-4;
    settings.reportInterval = 200;
    settings.innerIterations = 500;
    settings.innerTolerance  = 1e-6;
    settings.verbose        = true;
    // same relaxation and turbulence settings as channel validation
    settings.alphaU         = 0.7;
    settings.alphaP         = 0.5;
    settings.alphaK         = 0.5;
    settings.alphaOmega     = 0.5;
    settings.turbStartIter  = 0;      // couple turbulence from iter 0 (prevents laminar gradient buildup)
    settings.turbUpdateInterval = 1;
    settings.kMax            = 0.1;
    settings.omegaMin        = 1e-6;

    SIMPLESolver solver(mesh, sst, bcs, nu, settings);
    FlowFields fields(mesh);
    solver.initUniform(fields, Vec3(Uinf, 0, 0), 0.0, kIn, omIn);

    std::cout << "\n  Running SIMPLE...\n";
    ConvergenceHistory hist = solver.solve(fields);

    // skin friction at x = 0.8L (well past transition)
    double xProbe = 0.8 * L;
    double Re_x = Uinf * xProbe / nu;
    double Cf_schoenherr = 0.0592 * std::pow(Re_x, -0.2);

    // find wall face nearest to x = 0.8L
    double bestDist = 1e30;
    double Cf_sim = 0.0;
    for (int pi = 0; pi < mesh.nPatches(); ++pi) {
        const Patch& pat = mesh.patch(pi);
        if (pat.type != "wall") continue;
        for (FaceID fi : pat.faces) {
            const Face& face = mesh.face(fi);
            double dx = face.center.x - xProbe;
            if (std::abs(dx) < bestDist) {
                bestDist = std::abs(dx);
                int ow = face.owner;
                double nuEff = nu + fields.nuT[ow];
                double delta = std::max(face.delta, 1e-20);
                Vec3 Uc = fields.U[ow];
                Vec3 Ut = Uc - face.normal * Uc.dot(face.normal);
                Cf_sim = nuEff * Ut.norm() / delta / (0.5 * Uinf * Uinf);
            }
        }
    }

    std::cout << "\n  === Results at x/L = 0.8 ===\n";
    std::cout << "  SIMPLE iterations: " << hist.finalIter
              << " (" << (hist.converged ? "converged" : hist.diverged ? "DIVERGED" : "maxIter") << ")\n";
    std::cout << "  Re_x:              " << Re_x << "\n";
    std::cout << "  Cf (simulation):   " << Cf_sim << "\n";
    std::cout << "  Cf (Schoenherr):   " << Cf_schoenherr << "\n";
    double error = std::abs(Cf_sim - Cf_schoenherr) / Cf_schoenherr * 100.0;
    std::cout << "  Relative error:    " << error << "%\n";

    if (error < 30.0)
        std::cout << "  STATUS: PASS (within 30% of Schoenherr correlation)\n";
    else
        std::cout << "  STATUS: FAIL (error > 30% — check mesh resolution or BCs)\n";
}

// Forward model Bayesian inference interface test
static void demoForwardModel() {
    std::cout << "\n=== Forward Model Demo (a1_betaStar preset) ===\n\n";

    double h = 1.0, Lx = 10.0 * h;
    int nx = 30, ny = 20;
    double Ub = 1.0, Re_b = 6800.0;
    double nu = Ub * h / Re_b;

    // Demo uses relaxed y+ target — coarse 30x20 mesh can't benefit from y+=1
    // and the extreme stretching (3.4) creates terrible aspect ratios that stall
    // pressure convergence.  y+=5 gives stretch~2.0, well-conditioned cells.
    Mesh mesh = Mesh::makeChannel2D(nx, ny, Lx, 2.0 * h, Re_b, 5.0);
    mesh.computeWallDistance();

    double Tu = 0.05;
    double kIn = 1.5 * (Ub * Tu) * (Ub * Tu);
    double omIn = kIn / (nu * 100.0);   // nuT/nu ~ 100 at inlet

    FlowBoundaryConditions bcs = FlowBoundaryConditions::channelDefaults(mesh, Ub, kIn, omIn);

    // observation: skin friction on bottom wall
    ObservationOperator obsOp;
    obsOp.addSkinFriction("bottom", Vec3(7.0, 0, 0), 0.008, 0.001, Ub, 0.001);

    InferenceParameterSet params = InferenceParameterSet::a1_betaStar();
    SolverSettings settings;
    settings.maxIterations = 8000;
    settings.convergenceTol = 1e-4;
    settings.reportInterval = 500;
    settings.alphaP = 0.8;           // aggressive pressure relaxation (stable on coarse demo mesh)
    settings.innerTolerance = 1e-6;
    settings.innerIterations = 500;
    settings.verbose = true;

    ForwardModel fm(mesh, params, obsOp, bcs, nu, settings, Vec3(Ub, 0, 0), 0.0, kIn, omIn);

    // evaluate at default parameters
    std::vector<double> theta = params.pack(SSTCoefficients{});
    std::cout << "  Active params: ";
    auto names = params.activeNames();
    for (int i = 0; i < params.nActive(); ++i)
        std::cout << names[i] << "=" << theta[i] << "  ";
    std::cout << "\n\n";

    EvaluationResult res = fm.evaluate(theta);
    std::cout << "\n  Status: " << statusString(res.status) << "\n";
    std::cout << "  Log-likelihood: " << res.loglik << "\n";
    std::cout << "  SIMPLE iters:   " << res.simpleIters << "\n";
    if (!res.predictions.empty())
        std::cout << "  Cf prediction:  " << res.predictions[0] << "\n";
    std::cout << "  Cache size:     " << fm.cache().size() << "\n";
}
// main() function
int main(int argc, char* argv[]) {
    std::cout << "RANS-SST Bayesian Calibration Framework\n";
    std::cout << "========================================\n\n";

    if (argc < 2) {
        std::cout << "Usage:\n";
        std::cout << "  rans_sst --validate-channel    # Channel flow vs Dean\n";
        std::cout << "  rans_sst --validate-plate       # Flat plate vs Schoenherr\n";
        std::cout << "  rans_sst --demo                 # Forward model demo\n";
        std::cout << "  rans_sst --all                  # Run all validations\n";
        return 0;
    }

    std::string mode = argv[1];

    if (mode == "--validate-channel" || mode == "--all")
        validateChannel();
    if (mode == "--validate-plate" || mode == "--all")
        validatePlate();
    if (mode == "--demo" || mode == "--all")
        demoForwardModel();

    return 0;
}