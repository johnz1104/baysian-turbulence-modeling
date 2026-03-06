#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// Boundary condition types
enum class BCType {
    Dirichlet, Neumann,
    InletVelocity, OutletPressure,
    WallNoSlip, WallKOmega,
    Symmetry, Cyclic
};
// Boundary conditions for a single mesh patch 
struct PatchBC {
    BCType      type     = BCType::Neumann;
    double      value    = 0.0;
    Vec3        vecValue = {};
    std::string patchName;
};

// Container for boundary conditions of solved fields
// stores conditions for every boundary patch in mesh 
struct FlowBoundaryConditions {
    std::vector<PatchBC> velocityBC;    // velocity u
    std::vector<PatchBC> pressureBC;    // pressure p
    std::vector<PatchBC> kBC;           // tubulent kinetic energy k
    std::vector<PatchBC> omegaBC;       // specific dissipation rate (omega/w)
    
    // defines defaults for channle flow
    // input args: reference to mesh, inlet u, inlet k, inlet omega
    static FlowBoundaryConditions channelDefaults(
        const Mesh& mesh, double Uin, double kIn, double omIn) {
        FlowBoundaryConditions bc;
        int np = mesh.nPatches();       // np = number of patches

        // resizes boundary conditions to np
        bc.velocityBC.resize(np);  
        bc.pressureBC.resize(np);
        bc.kBC.resize(np);        
        bc.omegaBC.resize(np);
            
        // assign default BC types and values for each field on each mesh patch 
        // channel flow case
        for (int p = 0; p < np; ++p) {
            const Patch& pat = mesh.patch(p);
            bc.velocityBC[p].patchName = pat.name;
            bc.pressureBC[p].patchName = pat.name;
            bc.kBC[p].patchName        = pat.name;
            bc.omegaBC[p].patchName    = pat.name;

            if (pat.type == "wall") {
                bc.velocityBC[p].type = BCType::WallNoSlip;
                bc.pressureBC[p].type = BCType::Neumann;
                bc.kBC[p].type        = BCType::WallKOmega;
                bc.omegaBC[p].type    = BCType::WallKOmega;
            } else if (pat.type == "inlet") {
                bc.velocityBC[p] = {BCType::InletVelocity, 0.0, Vec3(Uin,0,0), pat.name};
                bc.pressureBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.kBC[p]        = {BCType::Dirichlet, kIn, {}, pat.name};
                bc.omegaBC[p]    = {BCType::Dirichlet, omIn, {}, pat.name};
            } else if (pat.type == "outlet") {
                bc.velocityBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.pressureBC[p] = {BCType::OutletPressure, 0.0, {}, pat.name};
                bc.kBC[p]        = {BCType::Neumann, 0.0, {}, pat.name};
                bc.omegaBC[p]    = {BCType::Neumann, 0.0, {}, pat.name};
            } else {
                // default zero gradient
                bc.velocityBC[p].patchName = pat.name;
                bc.pressureBC[p].patchName = pat.name;
                bc.kBC[p].patchName        = pat.name;
                bc.omegaBC[p].patchName    = pat.name;
            }
        }
        return bc;
        }
        // flat plate: bottom = wall, top = freestream (Dirichlet at inlet values), inlet/outlet same as channel
        // the top boundary must be freestream, not wall — otherwise we solve a channel problem, not a flat plate
        static FlowBoundaryConditions flatPlateDefaults(
            const Mesh& mesh, double Uinf, double kIn, double omIn) {
        FlowBoundaryConditions bc = channelDefaults(mesh, Uinf, kIn, omIn);
        // override "top" patch from wall to freestream
        for (int p = 0; p < mesh.nPatches(); ++p) {
            const Patch& pat = mesh.patch(p);
            if (pat.name == "top") {
                bc.velocityBC[p] = {BCType::Dirichlet, 0.0, Vec3(Uinf, 0, 0), pat.name};
                bc.pressureBC[p] = {BCType::Neumann, 0.0, {}, pat.name};
                bc.kBC[p]        = {BCType::Dirichlet, kIn, {}, pat.name};
                bc.omegaBC[p]    = {BCType::Dirichlet, omIn, {}, pat.name};
            }
        }
        return bc;
    }
};

// Apply functions

// Apply velocity boundary conditions to boundary faces
// inputs: U (velocity vector field), mesh, and BCs
inline void applyVelocityBC(VectorField& U, 
                            const Mesh& mesh, 
                            const FlowBoundaryConditions& bcs) {
    for (int p = 0; p < mesh.nPatches(); ++p) { // loops over every boundary patch 
        const Patch& pat = mesh.patch(p);       // faces that this patch belongs to
        const PatchBC& bc = bcs.velocityBC[p];  // velocity BC kind
        for (FaceID fi : pat.faces) {           // loops over faces in patch
            switch (bc.type) {
                case BCType::WallNoSlip:
                    U.bface(fi) = Vec3(0,0,0); break;       // enforces u = 0 for no-slip walls
                case BCType::InletVelocity:                 // enforces u = U_in
                case BCType::Dirichlet:                     // enforces u = u_boundary
                    U.bface(fi) = bc.vecValue; break;
                case BCType::Symmetry: {                    // enforces no normal velocity and unchanged tangential velocity
                    Vec3 Uo = U[mesh.face(fi).owner];
                    U.bface(fi) = Uo - mesh.face(fi).normal * Uo.dot(mesh.face(fi).normal);
                    break;
                }
                default:
                    U.bface(fi) = U[mesh.face(fi).owner]; break;    // default case
            }
        }
    }
}

// Apply pressure boundary conditions to boundary faces
// inputs: pf (scalar pressure field), mesh, and BCs
inline void applyPressureBC(ScalarField& pf, 
                            const Mesh& mesh,
                            const FlowBoundaryConditions& bcs) {
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch& pat = mesh.patch(p);
        const PatchBC& bc = bcs.pressureBC[p];
        for (FaceID fi : pat.faces) {
            switch (bc.type) {
                case BCType::OutletPressure:                    
                case BCType::Dirichlet:                      
                    pf.bface(fi) = bc.value; break;
                default:
                    pf.bface(fi) = pf[mesh.face(fi).owner]; break;  // default case
            }
        }
    }
}

// Apply turbulence kinetic energy boundary conditions to boundary faces
// inputs: k (turbulence kinetic energy field), mesh, and BCs 
inline void applyKBC(ScalarField& k, 
                    const Mesh& mesh,
                    const FlowBoundaryConditions& bcs) {
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch& pat = mesh.patch(p);
        const PatchBC& bc = bcs.kBC[p];
        for (FaceID fi : pat.faces) {
            switch (bc.type) {
                case BCType::WallKOmega:
                    k.bface(fi) = 0.0; break;       // enforces k = 0 at wall (u = 0 at no slip wall)
                case BCType::Dirichlet:             // enforces k = k_in
                    k.bface(fi) = bc.value; break;
                default:
                    k.bface(fi) = k[mesh.face(fi).owner]; break;
            }
        }
    }
}

// Menter (1994) k-omega SST model: omega_wall = 60*nu / (beta1 * y1^2)
// Apply boundary conditions to specific dissipation rate omega
// inputs: omega (dissipation scalar field), mesh, BCs, nu (kinematic viscosity), beta1 (default 0.075)
// enforces the k-omega boundary condition system
inline void applyOmegaBC(ScalarField& omega, 
                        const Mesh& mesh,
                        const FlowBoundaryConditions& bcs,
                        double nu, 
                        double beta1 = 0.075) {
    for (int p = 0; p < mesh.nPatches(); ++p) {
        const Patch& pat = mesh.patch(p);
        const PatchBC& bc = bcs.omegaBC[p];
        for (FaceID fi : pat.faces) {
            switch (bc.type) {
                case BCType::WallKOmega: {  // near-wall omega formula Menter (1994)
                    double y1 = std::max(mesh.face(fi).delta, 1e-20);
                    omega.bface(fi) = 60.0 * nu / (beta1 * y1 * y1);
                    break;
                }
                case BCType::Dirichlet:
                    omega.bface(fi) = bc.value; break;
                default:
                    omega.bface(fi) = omega[mesh.face(fi).owner]; break;
            }
        }
    }
}

// Applies all boundary conditions
inline void applyAllBCs(VectorField& U, ScalarField& p,
                         ScalarField& k, ScalarField& omega,
                         const Mesh& mesh, const FlowBoundaryConditions& bcs,
                         double nu) {
    applyVelocityBC(U, mesh, bcs);
    applyPressureBC(p, mesh, bcs);
    applyKBC(k, mesh, bcs);
    applyOmegaBC(omega, mesh, bcs, nu);
}