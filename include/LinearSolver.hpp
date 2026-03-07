#pragma once

#include "Mesh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <numeric>

// LinearSystem - stores Ax = source in a LDU (Lower-Diagonal-Upper) matrix
// upper - owner-neighbor fluxes 
// diag - sum of magnitude of all flux coefficients
// lower - neighbor-owner fluxes (upper but negative)
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

// result container
// initialRes tracks the absolute equation imbalance (r0 = |b - Ax0|) before the linear solve
// this is used for SIMPLE outer-loop convergence instead of finalRes (= rn/r0), which measures
// only the relative drop within a single linear solve
// with uniform initial conditions the assembled systems have near-zero RHS, 
// so finalRes drops below tolerance immediately at iter 0, causing false convergence.  
// initialRes avoids this by measuring the actual equation residual
struct SolverResult {
    int    iterations = 0;
    double finalRes   = 0.0;   // relative residual (rn/r0) — used internally by linear solver
    double initialRes = 0.0;   // absolute initial residual (r0 = |b - Ax0|) — used for outer convergence
    bool   converged  = false;
};

// interface linear solver
// functions all linear solver must implement
class ILinearSolver {
public:
    virtual ~ILinearSolver() = default;
    virtual SolverResult solve(const LinearSystem& A, std::vector<double>& x, int maxIter, double tol) = 0;
    virtual std::string name() const = 0;
};

// helpers
namespace linalg {
    // dot product
    inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0.0;
        for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
        return s;
    }
    // norm
    inline double norm(const std::vector<double>& v) { return std::sqrt(dot(v, v)); }
    // Jacobi Preconditioner 
    // args: 
    // A - Matrix system
    // r - residual vector (difference between observed and predicted data)
    // z - output vector storing preconditioned residual (z = diag(A)^-1 r)
    inline void jacobiPrecond(const LinearSystem& A, const std::vector<double>& r, std::vector<double>& z) {
        for (int i = 0; i < A.nCells; ++i)
            z[i] = (std::abs(A.diag[i]) > 1e-30) ? r[i] / A.diag[i] : r[i];
    }
}

// Solvers

// Preconditioned Conjugate Gradient
// used for large spare symmetric positive definite matrices (pressure Poisson)
class PCGSolver : public ILinearSolver {
public:
    SolverResult solve( const LinearSystem& A, 
                        std::vector<double>& x, 
                        int maxIter = 500, double tol = 1e-6) override {
        int n = A.nCells;
        std::vector<double> r(n), z(n), p(n), Ap(n);
        SolverResult res;

        A.residual(x, r);
        double r0 = linalg::norm(r);
        res.initialRes = r0; 
        if (r0 < 1e-30) { res.converged = true; return res; }

        linalg::jacobiPrecond(A, r, z);
        p = z;
        double rz = linalg::dot(r, z);

        for (int it = 0; it < maxIter; ++it) {
            A.matvec(p, Ap);
            double pAp = linalg::dot(p, Ap);
            if (std::abs(pAp) < 1e-30) break;
            double alpha = rz / pAp;
            for (int i = 0; i < n; ++i) { x[i] += alpha*p[i]; r[i] -= alpha*Ap[i]; }

            double rn = linalg::norm(r);
            res.iterations = it + 1;
            res.finalRes   = rn / r0;
            if (res.finalRes < tol) { res.converged = true; return res; }

            linalg::jacobiPrecond(A, r, z);
            double rzNew = linalg::dot(r, z);
            double beta  = rzNew / (rz + 1e-30);
            for (int i = 0; i < n; ++i) p[i] = z[i] + beta*p[i];
            rz = rzNew;
        }
        return res;
    }
    std::string name() const override { return "PCG"; }
};

// BiConjugate Gradient Stabilized
// iterative Krylov subspace method 
// used for large sparse asymmetric matrices (momentum, turbulence)
class BiCGSTABSolver : public ILinearSolver {
public:
    SolverResult solve(const LinearSystem& A, 
                       std::vector<double>& x,
                       int maxIter = 500, 
                       double tol = 1e-6) override {
        int n = A.nCells;
        std::vector<double> r(n), rh(n), p(n,0), v(n,0);
        std::vector<double> s(n), t(n), ph(n), sh(n);
        SolverResult res;

        A.residual(x, r);
        double r0 = linalg::norm(r);
        res.initialRes = r0;
        if (r0 < 1e-30) { res.converged = true; return res; }

        rh = r;
        double rho = 1, alpha = 1, omega = 1;

        for (int it = 0; it < maxIter; ++it) {
            double rhoN = linalg::dot(rh, r);
            if (std::abs(rhoN) < 1e-30) break;
            double beta = (rhoN / (rho + 1e-30)) * (alpha / (omega + 1e-30));
            for (int i = 0; i < n; ++i) p[i] = r[i] + beta*(p[i] - omega*v[i]);

            linalg::jacobiPrecond(A, p, ph);
            A.matvec(ph, v);
            double rv = linalg::dot(rh, v);
            if (std::abs(rv) < 1e-30) break;
            alpha = rhoN / rv;

            for (int i = 0; i < n; ++i) s[i] = r[i] - alpha*v[i];
            double sn = linalg::norm(s);
            if (sn/r0 < tol) {
                for (int i = 0; i < n; ++i) x[i] += alpha*ph[i];
                res = {it+1, sn/r0, r0, true}; return res;
            }

            linalg::jacobiPrecond(A, s, sh);
            A.matvec(sh, t);
            double tt = linalg::dot(t, t);
            omega = (tt > 1e-30) ? linalg::dot(t, s) / tt : 0.0;

            for (int i = 0; i < n; ++i) x[i] += alpha*ph[i] + omega*sh[i];
            for (int i = 0; i < n; ++i) r[i] = s[i] - omega*t[i];

            double rn = linalg::norm(r);
            res.iterations = it + 1;
            res.finalRes   = rn / r0;
            if (res.finalRes < tol) { res.converged = true; return res; }
            if (std::abs(omega) < 1e-30) break;
            rho = rhoN;
        }
        return res;
    }
    std::string name() const override { return "BiCGSTAB"; }
};


// Gauss-Seidel iterative solver 
class GaussSeidelSolver : public ILinearSolver {
public:
    SolverResult solve(const LinearSystem& A, std::vector<double>& x,
                       int maxIter = 500, double tol = 1e-6) override {
        int n = A.nCells;
        std::vector<double> r(n);
        SolverResult res;
        double b0 = linalg::norm(A.source);
        res.initialRes = b0;
        if (b0 < 1e-30) b0 = 1.0;

        for (int it = 0; it < maxIter; ++it) {
            std::vector<double> rhs = A.source;
            for (int f = 0; f < A.nIF; ++f) {
                rhs[A.own[f]] -= A.upper[f] * x[A.nbr[f]];
                rhs[A.nbr[f]] -= A.lower[f] * x[A.own[f]];
            }
            for (int i = 0; i < n; ++i)
                if (std::abs(A.diag[i]) > 1e-30) x[i] = rhs[i] / A.diag[i];

            A.residual(x, r);
            res.iterations = it + 1;
            res.finalRes   = linalg::norm(r) / b0;
            if (res.finalRes < tol) { res.converged = true; return res; }
        }
        return res;
    }
    std::string name() const override { return "GaussSeidel"; }
};

// ============================================================
// Algebraic Multigrid (AMG) preconditioned PCG solver
// ============================================================

// General sparse matrix in CSR format for coarse AMG levels
struct CSRMatrix {
    int nRows = 0;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<double> values;
    std::vector<double> diag;

    void matvec(const std::vector<double>& x, std::vector<double>& y) const {
        for (int i = 0; i < nRows; ++i) {
            double s = 0.0;
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
                s += values[j] * x[colIdx[j]];
            y[i] = s;
        }
    }

    void residual(const std::vector<double>& x, const std::vector<double>& b,
                  std::vector<double>& r) const {
        for (int i = 0; i < nRows; ++i) {
            double s = 0.0;
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
                s += values[j] * x[colIdx[j]];
            r[i] = b[i] - s;
        }
    }

    void gaussSeidelSweep(std::vector<double>& x, const std::vector<double>& b) const {
        for (int i = 0; i < nRows; ++i) {
            double s = b[i];
            for (int j = rowPtr[i]; j < rowPtr[i + 1]; ++j) {
                if (colIdx[j] != i)
                    s -= values[j] * x[colIdx[j]];
            }
            if (std::abs(diag[i]) > 1e-30)
                x[i] = s / diag[i];
        }
    }
};

// Convert LDU LinearSystem to CSR format
inline CSRMatrix lduToCSR(const LinearSystem& A) {
    int n = A.nCells;
    // Count entries per row (diagonal + off-diagonals)
    std::vector<int> count(n, 1); // diagonal
    for (int f = 0; f < A.nIF; ++f) {
        count[A.own[f]]++;
        count[A.nbr[f]]++;
    }

    CSRMatrix csr;
    csr.nRows = n;
    csr.rowPtr.resize(n + 1, 0);
    for (int i = 0; i < n; ++i)
        csr.rowPtr[i + 1] = csr.rowPtr[i] + count[i];
    int nnz = csr.rowPtr[n];
    csr.colIdx.resize(nnz);
    csr.values.resize(nnz);
    csr.diag.resize(n);

    // Build unsorted entries per row using fill pointers
    std::vector<int> pos(n);
    for (int i = 0; i < n; ++i)
        pos[i] = csr.rowPtr[i];

    // Diagonal first
    for (int i = 0; i < n; ++i) {
        csr.colIdx[pos[i]] = i;
        csr.values[pos[i]] = A.diag[i];
        csr.diag[i] = A.diag[i];
        pos[i]++;
    }
    // Off-diagonals
    for (int f = 0; f < A.nIF; ++f) {
        int o = A.own[f], nb = A.nbr[f];
        csr.colIdx[pos[o]] = nb;
        csr.values[pos[o]] = A.upper[f];
        pos[o]++;
        csr.colIdx[pos[nb]] = o;
        csr.values[pos[nb]] = A.lower[f];
        pos[nb]++;
    }

    // Sort each row by column index for consistent access
    for (int i = 0; i < n; ++i) {
        int start = csr.rowPtr[i], end = csr.rowPtr[i + 1];
        // Simple insertion sort (rows are small)
        for (int a = start + 1; a < end; ++a) {
            int key_c = csr.colIdx[a];
            double key_v = csr.values[a];
            int b = a - 1;
            while (b >= start && csr.colIdx[b] > key_c) {
                csr.colIdx[b + 1] = csr.colIdx[b];
                csr.values[b + 1] = csr.values[b];
                b--;
            }
            csr.colIdx[b + 1] = key_c;
            csr.values[b + 1] = key_v;
        }
    }
    return csr;
}

// Greedy aggregation coarsening with strength-of-connection filter
inline void buildAggregates(const CSRMatrix& A, std::vector<int>& map, int& nCoarse) {
    int n = A.nRows;
    map.assign(n, -1);
    nCoarse = 0;

    // Pass 1: seed aggregates from unaggregated cells
    for (int i = 0; i < n; ++i) {
        if (map[i] >= 0) continue;
        int agg = nCoarse++;
        map[i] = agg;
        double aii = std::abs(A.diag[i]);
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int c = A.colIdx[j];
            if (c == i || map[c] >= 0) continue;
            double ajj = std::abs(A.diag[c]);
            double strength = std::abs(A.values[j]) / std::sqrt(aii * ajj + 1e-30);
            if (strength > 0.25)
                map[c] = agg;
        }
    }

    // Pass 2: assign orphans to strongest neighbor's aggregate
    for (int i = 0; i < n; ++i) {
        if (map[i] >= 0) continue;
        double bestStr = -1.0;
        int bestAgg = 0;
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int c = A.colIdx[j];
            if (c == i && map[c] < 0) continue;
            if (map[c] >= 0 && std::abs(A.values[j]) > bestStr) {
                bestStr = std::abs(A.values[j]);
                bestAgg = map[c];
            }
        }
        map[i] = bestAgg;
    }
}

// Galerkin coarse matrix: A_c = R * A * P (piecewise-constant prolongation)
inline CSRMatrix galerkinProduct(const CSRMatrix& A, const std::vector<int>& map, int nCoarse) {
    // Since P[i, map[i]] = 1: A_c[I,J] = sum of A[i,j] for i in I, j in J
    // Use map to accumulate
    std::vector<std::unordered_map<int, double>> rows(nCoarse);
    for (int i = 0; i < A.nRows; ++i) {
        int I = map[i];
        for (int j = A.rowPtr[i]; j < A.rowPtr[i + 1]; ++j) {
            int J = map[A.colIdx[j]];
            rows[I][J] += A.values[j];
        }
    }

    // Convert to CSR
    CSRMatrix Ac;
    Ac.nRows = nCoarse;
    Ac.rowPtr.resize(nCoarse + 1, 0);
    Ac.diag.resize(nCoarse, 0.0);

    int nnz = 0;
    for (int I = 0; I < nCoarse; ++I)
        nnz += (int)rows[I].size();
    Ac.colIdx.resize(nnz);
    Ac.values.resize(nnz);

    int pos = 0;
    for (int I = 0; I < nCoarse; ++I) {
        Ac.rowPtr[I] = pos;
        // Collect and sort entries
        std::vector<std::pair<int, double>> entries(rows[I].begin(), rows[I].end());
        std::sort(entries.begin(), entries.end());
        for (auto& [col, val] : entries) {
            Ac.colIdx[pos] = col;
            Ac.values[pos] = val;
            if (col == I) Ac.diag[I] = val;
            pos++;
        }
    }
    Ac.rowPtr[nCoarse] = pos;
    return Ac;
}

// AMG level data
struct AMGLevel {
    CSRMatrix A;
    std::vector<int> aggregateMap;
    int nCoarse = 0;
    // Working vectors
    std::vector<double> x, b, r;
};

// AMG-preconditioned PCG solver
class AMGSolver : public ILinearSolver {
    static constexpr int maxLevels_ = 20;
    static constexpr int minCoarseSize_ = 50;
    static constexpr int nPreSmooth_ = 2;
    static constexpr int nPostSmooth_ = 2;
    static constexpr int coarseSweeps_ = 50;

    std::vector<AMGLevel> levels_;

    void setup(const LinearSystem& A) {
        levels_.clear();

        // Level 0: finest level from LDU
        AMGLevel lev0;
        lev0.A = lduToCSR(A);
        levels_.push_back(std::move(lev0));

        // Coarsen until small enough
        for (int l = 0; l < maxLevels_; ++l) {
            auto& fine = levels_[l];
            if (fine.A.nRows <= minCoarseSize_) break;

            std::vector<int> map;
            int nc;
            buildAggregates(fine.A, map, nc);

            // Stop if coarsening stalls
            if (nc == 0 || nc >= fine.A.nRows) break;

            fine.aggregateMap = std::move(map);
            fine.nCoarse = nc;

            AMGLevel coarse;
            coarse.A = galerkinProduct(fine.A, fine.aggregateMap, fine.nCoarse);
            levels_.push_back(std::move(coarse));
        }

        // Allocate working vectors
        for (auto& lev : levels_) {
            int n = lev.A.nRows;
            lev.x.resize(n, 0.0);
            lev.b.resize(n, 0.0);
            lev.r.resize(n, 0.0);
        }
    }

    void vcycle(int level) {
        auto& lev = levels_[level];
        int n = lev.A.nRows;

        // Coarsest level: direct solve with many GS sweeps
        if (level == (int)levels_.size() - 1) {
            for (int s = 0; s < coarseSweeps_; ++s)
                lev.A.gaussSeidelSweep(lev.x, lev.b);
            return;
        }

        // Pre-smooth
        for (int s = 0; s < nPreSmooth_; ++s)
            lev.A.gaussSeidelSweep(lev.x, lev.b);

        // Compute residual
        lev.A.residual(lev.x, lev.b, lev.r);

        // Restrict residual to coarse level (R = P^T, piecewise-constant)
        auto& coarse = levels_[level + 1];
        int nc = lev.nCoarse;
        std::fill(coarse.b.begin(), coarse.b.end(), 0.0);
        std::fill(coarse.x.begin(), coarse.x.end(), 0.0);
        for (int i = 0; i < n; ++i)
            coarse.b[lev.aggregateMap[i]] += lev.r[i];

        // Recurse
        vcycle(level + 1);

        // Prolongate correction and add (P is piecewise-constant)
        for (int i = 0; i < n; ++i)
            lev.x[i] += coarse.x[lev.aggregateMap[i]];

        // Post-smooth
        for (int s = 0; s < nPostSmooth_; ++s)
            lev.A.gaussSeidelSweep(lev.x, lev.b);
    }

    // Apply AMG as preconditioner: solve M*z ≈ r
    void precondition(const std::vector<double>& r, std::vector<double>& z) {
        auto& lev0 = levels_[0];
        int n = lev0.A.nRows;
        lev0.b = r;
        std::fill(lev0.x.begin(), lev0.x.end(), 0.0);
        vcycle(0);
        z = lev0.x;
    }

public:
    SolverResult solve(const LinearSystem& A, std::vector<double>& x,
                       int maxIter = 500, double tol = 1e-6) override {
        // Rebuild hierarchy (pressure matrix changes each SIMPLE iteration)
        setup(A);

        int n = A.nCells;
        std::vector<double> r(n), z(n), p(n), Ap(n);
        SolverResult res;

        A.residual(x, r);
        double r0 = linalg::norm(r);
        res.initialRes = r0;
        if (r0 < 1e-30) { res.converged = true; return res; }

        precondition(r, z);
        p = z;
        double rz = linalg::dot(r, z);

        for (int it = 0; it < maxIter; ++it) {
            A.matvec(p, Ap);
            double pAp = linalg::dot(p, Ap);
            if (std::abs(pAp) < 1e-30) break;
            double alpha = rz / pAp;
            for (int i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * Ap[i]; }

            double rn = linalg::norm(r);
            res.iterations = it + 1;
            res.finalRes = rn / r0;
            if (res.finalRes < tol) { res.converged = true; return res; }

            precondition(r, z);
            double rzNew = linalg::dot(r, z);
            double beta = rzNew / (rz + 1e-30);
            for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
            rz = rzNew;
        }
        return res;
    }

    std::string name() const override { return "AMG"; }
};

// Factory Function
inline std::unique_ptr<ILinearSolver> makeSolver(const std::string& name) {
    if (name == "PCG")         return std::make_unique<PCGSolver>();
    if (name == "BiCGSTAB")    return std::make_unique<BiCGSTABSolver>();
    if (name == "GaussSeidel") return std::make_unique<GaussSeidelSolver>();
    if (name == "AMG")         return std::make_unique<AMGSolver>();
    return std::make_unique<BiCGSTABSolver>();  // defaults to BiCGSTAB
}