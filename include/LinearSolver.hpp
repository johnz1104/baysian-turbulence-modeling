#pragma once

#include "Mesh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>

// LinearSystem - stores Ax = source in LDU (Lower-Diagonal-Upper) form using mesh owner-neighbor connectivity
// assumes one equation per cell, cells only interact with cells that it shares a face with, and internal faces connect to exactly two cells 
struct LinearSystem {
    // nCells = number of control volumes (unknowns)
    // nIF    = number of internal faces (owner–neighbour connections)
    int nCells = 0, nIF = 0;

    // Diagonal entries:
    // diag[i] = A_ii - self influence on cell i
    // size = nCells
    std::vector<double> diag;    

    // Off-diagonal entries stored per internal face f:
    // upper[f] = A_ij - neighbor j effects on owner i 
    // lower[f] = A_ji - owner i effects on neighbor j
    // size = nIF
    std::vector<double> upper; 
    std::vector<double> lower;
    
    // Right-hand side vector:
    // source[i] = b_i  in A*x = b
    // size = nCells
    std::vector<double> source;

    // Mesh connectivity for each internal face f:
    // own[f] = owner cell index
    // nbr[f] = neighbour cell index
    // size = nIF
    std::vector<int> own;           
    std::vector<int> nbr;   

    LinearSystem() = default;

    // Constructor: (allocates values to match mesh topology) 
    // diag -> size nCells
    // upper/lower → size nInternalFaces
    // source → size nCells
    // own/nbr → size nInternalFaces
    LinearSystem(int nc, int nif)
        : nCells(nc), nIF(nif),
          diag(nc, 0.0), upper(nif, 0.0), lower(nif, 0.0), source(nc, 0.0),
          own(nif), nbr(nif) {}

    // resets matrix and RHS to zero
    void zero() {
        std::fill(diag.begin(), diag.end(), 0.0);
        std::fill(upper.begin(), upper.end(), 0.0);
        std::fill(lower.begin(), lower.end(), 0.0);
        std::fill(source.begin(), source.end(), 0.0);
    }

    // y = A * x
    void matvec(const std::vector<double>& x, std::vector<double>& y) const {
        for (int i = 0; i < nCells; ++i)
            y[i] = diag[i] * x[i];              // y_i = A_ii * x_i
        
        // sparse matrix multiply
        for (int f = 0; f < nIF; ++f) {         // off diag terms
            y[own[f]] += upper[f] * x[nbr[f]];  // y_i+ = A_ij * x_j
            y[nbr[f]] += lower[f] * x[own[f]];  // y_j+ = A_ji * x_i
        }                                   
    }

    // residual r = source - A*x 
    void residual(const std::vector<double>& x, std::vector<double>& r) const {
        matvec(x, r);
        for (int i = 0; i < nCells; ++i)
            r[i] = source[i] - r[i];
    }
};

// creates linear system with mesh topology
inline LinearSystem makeSystem(const Mesh& m) {
    LinearSystem A(m.nCells(), m.nInternalFaces());
    const auto& ol = m.ownerList();
    const auto& nl = m.neighborList();
    for (int f = 0; f < m.nInternalFaces(); ++f) {
        A.own[f] = ol[f];
        A.nbr[f] = nl[f];
    }
    return A;
}