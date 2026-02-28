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