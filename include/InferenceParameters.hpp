#pragma once

#include "SSTModel.hpp"
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

// InferenceParameterSet
// maps between flat parameter vector θ and SST coefficients (which is a structured object)
// The 11 SST constants are indexed as:
// 0: sigma_k1    1: sigma_w1    2: beta1     3: alpha1
// 4: sigma_k2    5: sigma_w2    6: beta2     7: alpha2
// 8: betaStar    9: a1         10: kappa
// Preset: predefined configuration of which SST parameters you want the inference algorithm to vary
// the rest are fixed at their menter values

struct InferenceParameterSet {
    std::string  name;
    std::vector<int> activeIndices;     // indices into the 11-element coefficient array
    SSTCoefficients  base;              // default values for fixed parameters

    // number of active parameters
    int nActive() const { return static_cast<int>(activeIndices.size()); }

    // extract the 11 coefficients as a flat vector
    static std::vector<double> toVector(const SSTCoefficients& c) {
        return {c.sigma_k1, c.sigma_w1, c.beta1, c.alpha1,
                c.sigma_k2, c.sigma_w2, c.beta2, c.alpha2,
                c.betaStar, c.a1, c.kappa};
    }

    // build SSTCoefficients from a flat vector
    static SSTCoefficients fromVector(const std::vector<double>& v) {
        SSTCoefficients c;
        if (v.size() != 11) throw std::runtime_error("Expected 11 SST coefficients");
        c.sigma_k1 = v[0];  c.sigma_w1 = v[1]; c.beta1    = v[2]; c.alpha1 = v[3];
        c.sigma_k2 = v[4];  c.sigma_w2 = v[5]; c.beta2    = v[6]; c.alpha2 = v[7];
        c.betaStar = v[8];  c.a1       = v[9]; c.kappa    = v[10];
        return c;
    }

    // builds a full SST parameter set from a smaller inference vector
    SSTCoefficients unpack(const std::vector<double>& theta) const {
        if ((int)theta.size() != nActive())
            throw std::runtime_error("theta size mismatch: expected "
                + std::to_string(nActive()) + ", got " + std::to_string(theta.size()));

        std::vector<double> full = toVector(base);
        for (int i = 0; i < nActive(); ++i)
            full[activeIndices[i]] = theta[i];
        return fromVector(full);
    }

    // extracts only the active parameters
    std::vector<double> pack(const SSTCoefficients& c) const {
        std::vector<double> full = toVector(c);
        std::vector<double> theta(nActive());
        for (int i = 0; i < nActive(); ++i)
            theta[i] = full[activeIndices[i]];
        return theta;
    }

    // parameter names for the active subset
    std::vector<std::string> activeNames() const {
        static const std::vector<std::string> allNames = {
            "sigma_k1", "sigma_w1", "beta1", "alpha1",
            "sigma_k2", "sigma_w2", "beta2", "alpha2",
            "betaStar", "a1", "kappa"
        };
        std::vector<std::string> names;
        for (int idx : activeIndices) names.push_back(allNames[idx]);
        return names;
    }

    // physical bounds for each parameter
    // lower bounds (all positive, some with tighter floors)
    std::vector<double> lowerBounds() const {
        static const std::vector<double> lo = {
            0.1, 0.1, 0.01, 0.1,       // inner: sigma_k1, sigma_w1, beta1, alpha1
            0.1, 0.1, 0.01, 0.1,       // outer: sigma_k2, sigma_w2, beta2, alpha2
            0.01, 0.1, 0.35            // common: betaStar, a1, kappa
        };
        std::vector<double> bounds;
        for (int idx : activeIndices) bounds.push_back(lo[idx]);
        return bounds;
    }
    std::vector<double> upperBounds() const {
        static const std::vector<double> hi = {
            2.0, 2.0, 0.2, 1.5,
            2.0, 2.0, 0.2, 1.5,
            0.2, 0.8, 0.50
        };
        std::vector<double> bounds;
        for (int idx : activeIndices) bounds.push_back(hi[idx]);
        return bounds;
    }

    // check if theta is within physical bounds
    bool inBounds(const std::vector<double>& theta) const {
        auto lo = lowerBounds();
        auto hi = upperBounds();
        for (int i = 0; i < nActive(); ++i) {
            if (theta[i] < lo[i] || theta[i] > hi[i]) return false;
            if (std::isnan(theta[i]) || std::isinf(theta[i])) return false;
        }
        return true;
    }

    // Predefined presets

    // 2 params: a1, betaStar — general separated / APG flows
    static InferenceParameterSet a1_betaStar() {
        return {"a1_betaStar", {9, 8}, SSTCoefficients{}};
    }

    // 2 params: beta1, sigma_w1 — inlet-dominated uncertainty
    static InferenceParameterSet inletTurb() {
        return {"inletTurb", {2, 1}, SSTCoefficients{}};
    }

    // 4 params: a1, betaStar, beta1, sigma_k1 — attached BL with Cf/U data
    static InferenceParameterSet nearWall4() {
        return {"nearWall4", {9, 8, 2, 0}, SSTCoefficients{}};
    }

    // all 11 — only with large multi-QoI datasets
    static InferenceParameterSet all11() {
        return {"all11", {0,1,2,3,4,5,6,7,8,9,10}, SSTCoefficients{}};
    }
};