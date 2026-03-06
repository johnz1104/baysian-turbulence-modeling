#pragma once

#include "Mesh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <string>

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

// Factory Function
inline std::unique_ptr<ILinearSolver> makeSolver(const std::string& name) {
    if (name == "PCG")         return std::make_unique<PCGSolver>();
    if (name == "BiCGSTAB")    return std::make_unique<BiCGSTABSolver>();
    if (name == "GaussSeidel") return std::make_unique<GaussSeidelSolver>();
    return std::make_unique<BiCGSTABSolver>();  // defaults to BiCGSTAB
}