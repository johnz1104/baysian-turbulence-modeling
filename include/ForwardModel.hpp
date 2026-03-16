#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "SSTModel.hpp"
#include "SIMPLESolver.hpp"
#include "InferenceParameters.hpp"
#include "ObservationOperator.hpp"
#include <vector>
#include <string>
#include <deque>
#include <mutex>
#include <cmath>

// Failure classification for MCMC
enum class EvaluationStatus {
    Converged,           // residuals below tolerance
    Unconverged,         // hit maxIter but residuals decreasing
    Diverged,            // residuals grew or NaN
    InvalidParameters,   // bounds / physics check failed
    Unknown              // unexpected exception
};

inline const char* statusString(EvaluationStatus s) {
    switch (s) {
        case EvaluationStatus::Converged:          return "Converged";
        case EvaluationStatus::Unconverged:        return "Unconverged";
        case EvaluationStatus::Diverged:           return "Diverged";
        case EvaluationStatus::InvalidParameters:  return "InvalidParameters";
        case EvaluationStatus::Unknown:            return "Unknown";
    }
    return "Unknown";
}

// Container for what the forward model produces when you evaluate a parameter vector during inference
// read by MCMC algorithm
struct EvaluationResult {
    EvaluationStatus status = EvaluationStatus::Unknown;    // status of the CFD simulation 
    double loglik = -1e30;                                  // log likelihood computed from observation operator
    std::vector<double> predictions;                        // stores predicted measurements of H(fields)
    int              simpleIters = 0;                       // number of iterations used by SIMPLE Solver
};


// Stores recent CFD solutions and finds nearest cached solution vector in parameter space to use as initial conditions
class WarmStartCache {
public:
    // container for solution of the flow for parameter vector θ
    struct CacheEntry {
        std::vector<double> theta;      // active inference parameter vector
        FlowFields          fields;     // converged solution
    };
    
    // defines max number of stored solutions and max allowed parameter distance for reuse
    explicit WarmStartCache(int maxSize = 50, double threshold = 0.5)
        : maxSize_(maxSize), threshold_(threshold) {}

    // searches for the closest cached parameter vector; returns nullptr if none
    const CacheEntry* findNearest(const std::vector<double>& theta) const {
        std::lock_guard<std::mutex> lock(mtx_);
        const CacheEntry* best = nullptr;
        double bestDist = threshold_;
        for (const auto& e : cache_) {
            double d = euclidean(theta, e.theta);
            if (d < bestDist) { bestDist = d; best = &e; }
        }
        return best;
    }

    // store a converged solution
    void store(const std::vector<double>& theta, const FlowFields& fields) {
        std::lock_guard<std::mutex> lock(mtx_);
        cache_.push_back({theta, fields});
        while ((int)cache_.size() > maxSize_) cache_.pop_front();
    }

    int size() const { std::lock_guard<std::mutex> lock(mtx_); return (int)cache_.size(); }
    void clear() { std::lock_guard<std::mutex> lock(mtx_); cache_.clear(); }

private:
    int maxSize_;
    double threshold_;
    mutable std::mutex mtx_;
    std::deque<CacheEntry> cache_;
    
    // computes Euclidean distance
    static double euclidean(const std::vector<double>& a, const std::vector<double>& b) {
        double s = 0;
        for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
            double d = a[i] - b[i]; s += d * d;
        }
        return std::sqrt(s);
    }
};

// Forward model orchestration layer 
class ForwardModel {
public:
    // constructor
    ForwardModel(
        const Mesh& mesh,                       // mesh (cells, faces, owner-neighbor connectivity, patches, geometry)
        const InferenceParameterSet& paramSet,  // paramSet (defines inferred parameters and mapping to SST coefficients)
        const ObservationOperator& obsOp,       // contains the experimental/prior data
        const FlowBoundaryConditions& bcs,      // boundary conditions
        double nu,                              // kinematic viscosity (used in momentum, turbulence, skin friction)
        const SolverSettings& settings = {},    // contains solver settings (SIMPLESolver.hpp)
        
        // fallback initial fields (for when no warm-start is avaiable)
        const Vec3& Uinit = {1, 0, 0},         
        double pInit = 0.0,
        double kInit = 1e-4,
        double omegaInit = 1.0);

    // core forward model evaluation
    // input: θ = inferred parameters
    // output: EvaluationResult
    EvaluationResult evaluate(const std::vector<double>& theta);

    // safe wrapper for MCMC
    // automatically rejects bad parameter proposals
    double penalizedLogLikelihood(const std::vector<double>& theta);

    // access functions 
    WarmStartCache& cache() { return cache_; } // access warm-start cache (inspect, clear, resize)
    const InferenceParameterSet& paramSet() const { return paramSet_; }

    // last solved flow fields (retained for visualization / post-processing)
    const FlowFields& lastFields() const { return lastFields_; }
    bool hasLastFields() const { return hasLastFields_; }

private:
    const Mesh& mesh_;
    InferenceParameterSet paramSet_;
    ObservationOperator obsOp_;
    FlowBoundaryConditions bcs_;
    double nu_;
    SolverSettings settings_;
    Vec3 Uinit_;
    double pInit_, kInit_, omegaInit_;

    WarmStartCache cache_;
    FlowFields lastFields_;
    bool hasLastFields_ = false;
};
