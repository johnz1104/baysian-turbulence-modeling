#pragma once

#include "Mesh.hpp"
#include "Field.hpp"
#include "SIMPLESolver.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <stdexcept>

// Measurement object for a single observed quantity with uncertainty 
struct Measurement {
    double observed   = 0.0;    // y_obs
    double sigma_obs  = 0.01;   // observation uncertainty
    double sigma_model = 0.0;   // model discrepancy (additive)
    double totalVariance() const { return sigma_obs*sigma_obs + sigma_model*sigma_model; }  // total variance
};

// Observable type / kind of measurement
enum class ObsType { Drag, PressureTap, VelocityProfile, SkinFriction, SeparationPoint };

// Observable object - full description of one measurement
struct Observable {
    ObsType type;
    Measurement meas;

    // additional data depending on type
    std::string patchName;          // wall patch for drag / Cf / separation
    Vec3   location = {};           // spatial point for pressure tap / profile
    int    component = 0;           // velocity component for profiles (0=x,1=y,2=z)
    double referenceArea = 1.0;     // for drag normalisation
    double refVelocity = 1.0;       // for Cf / Cd normalisation
};

// maps flow fields to predicted measurements
// converts the full CFD state into the predicted measurements that correspond to experimental data
class ObservationOperator {
public:
    ObservationOperator() = default;

    // add observable types
    void addDrag(const std::string& wallPatch, double Cd_obs, double sigma, 
                 double refArea, double refVel, double sigma_model = 0.0);
    void addPressureTap(const Vec3& loc, double Cp_obs, double sigma, double refVel, double sigma_model = 0.0);
    void addVelocityProfile(const Vec3& loc, int comp, double Uobs, double sigma, double sigma_model = 0.0);
    void addSkinFriction(const std::string& wallPatch, const Vec3& loc, double Cf_obs, 
                         double sigma, double refVel, double sigma_model = 0.0);
    void addSeparationPoint(const std::string& wallPatch, double xSep_obs, double sigma, double sigma_model = 0.0);

    // evaluate all observables
    // returns predicted values H(fields) in same order as observables_
    std::vector<double> evaluate(const Mesh& mesh, const FlowFields& fields, double nu) const;

    // log-likelihood
    // sum of Gaussian log-likelihoods: sum_i -0.5*(H_i - y_i)^2 / sigma_i^2
    double logLikelihood(const Mesh& mesh, const FlowFields& fields, double nu) const;

    int nObs() const { return static_cast<int>(observables_.size()); }
    const std::vector<Observable>& observables() const { return observables_; }

private:
    std::vector<Observable> observables_;

    // helpers
    double computeDrag(const Mesh& mesh, const FlowFields& fields, double nu, const Observable& obs) const;
    double computeSkinFrictionAt(const Mesh& mesh, const FlowFields& fields, double nu, const Observable& obs) const;
    double computeVelocityAt(const Mesh& mesh, const FlowFields& fields, const Observable& obs) const;
    double computePressureAt(const Mesh& mesh, const FlowFields& fields, const Observable& obs) const;
    double computeSeparationPoint(const Mesh& mesh, const FlowFields& fields, double nu, const Observable& obs) const;
    int findNearestCell(const Mesh& mesh, const Vec3& loc) const;
    int findNearestWallFace(const Mesh& mesh, const std::string& patchName, const Vec3& loc) const;
};

//  Inline implementations functions
// create empty observable, define measurement, store observed value + uncertainties

inline void ObservationOperator::addDrag(
        const std::string& wallPatch, double Cd_obs, double sigma,
        double refArea, double refVel, double sigma_model) {
    Observable obs;
    obs.type = ObsType::Drag;
    obs.meas = {Cd_obs, sigma, sigma_model};
    obs.patchName = wallPatch;
    obs.referenceArea = refArea;
    obs.refVelocity = refVel;
    observables_.push_back(obs);
}

inline void ObservationOperator::addPressureTap(
        const Vec3& loc, double Cp_obs, double sigma,
        double refVel, double sigma_model) {
    Observable obs;
    obs.type = ObsType::PressureTap;
    obs.meas = {Cp_obs, sigma, sigma_model};
    obs.location = loc;
    obs.refVelocity = refVel;
    observables_.push_back(obs);
}

inline void ObservationOperator::addVelocityProfile(
        const Vec3& loc, int comp, double Uobs,
        double sigma, double sigma_model) {
    Observable obs;
    obs.type = ObsType::VelocityProfile;
    obs.meas = {Uobs, sigma, sigma_model};
    obs.location = loc;
    obs.component = comp;
    observables_.push_back(obs);
}

inline void ObservationOperator::addSkinFriction(
        const std::string& wallPatch, const Vec3& loc,
        double Cf_obs, double sigma, double refVel, double sigma_model) {
    Observable obs;
    obs.type = ObsType::SkinFriction;
    obs.meas = {Cf_obs, sigma, sigma_model};
    obs.patchName = wallPatch;
    obs.location = loc;
    obs.refVelocity = refVel;
    observables_.push_back(obs);
}

inline void ObservationOperator::addSeparationPoint(
        const std::string& wallPatch, double xSep_obs,
        double sigma, double sigma_model) {
    Observable obs;
    obs.type = ObsType::SeparationPoint;
    obs.meas = {xSep_obs, sigma, sigma_model};
    obs.patchName = wallPatch;
    observables_.push_back(obs);
}

// nearest cell search (brute force)
inline int ObservationOperator::findNearestCell(const Mesh& mesh, const Vec3& loc) const {
    int best = 0;
    double bestDist = 1e30;
    for (int ci = 0; ci < mesh.nCells(); ++ci) {
        double d = (mesh.cell(ci).center - loc).norm2();
        if (d < bestDist) { bestDist = d; best = ci; }
    }
    return best;
}

// nearest wall face on a named patch 
inline int ObservationOperator::findNearestWallFace(
        const Mesh& mesh, const std::string& patchName,
        const Vec3& loc) const {
    PatchID pid = mesh.patchByName(patchName);
    const Patch& pat = mesh.patch(pid);
    int best = pat.faces.empty() ? -1 : pat.faces[0];
    double bestDist = 1e30;
    for (FaceID fi : pat.faces) {
        double d = (mesh.face(fi).center - loc).norm2();
        if (d < bestDist) { bestDist = d; best = fi; }
    }
    return best;
}

// drag: ∫(p*n + tau*n) dA normalised
inline double ObservationOperator::computeDrag(
        const Mesh& mesh, const FlowFields& fields,
        double nu, const Observable& obs) const {
    PatchID pid = mesh.patchByName(obs.patchName);
    const Patch& pat = mesh.patch(pid);

    double force = 0.0;
    for (FaceID fi : pat.faces) {
        const Face& face = mesh.face(fi);
        int ow = face.owner;
        double Sf = face.area;

        // pressure contribution (x-component by convention)
        double pf = fields.p[ow];   // zero-gradient at wall
        force += pf * face.normal.x * Sf;

        // viscous contribution: tau_wall ≈ (nu + nuT) * dU/dn
        double nuEff = nu + fields.nuT[ow];
        double delta = std::max(face.delta, 1e-20);
        // wall shear in x-direction: nuEff * (U_cell - U_wall) / delta
        double Uw_x = fields.U.bface(fi).x;
        double tau_x = nuEff * (fields.U[ow].x - Uw_x) / delta;
        force += tau_x * Sf;
    }

    double dynPressure = 0.5 * obs.refVelocity * obs.refVelocity;
    return force / (dynPressure * obs.referenceArea);
}

// skin friction at a point
inline double ObservationOperator::computeSkinFrictionAt(
        const Mesh& mesh, const FlowFields& fields,
        double nu, const Observable& obs) const {
    int fi = findNearestWallFace(mesh, obs.patchName, obs.location);
    if (fi < 0) return 0.0;
    const Face& face = mesh.face(fi);
    int ow = face.owner;
    double nuEff = nu + fields.nuT[ow];
    double delta = std::max(face.delta, 1e-20);

    // wall-tangential velocity magnitude
    Vec3 Uc = fields.U[ow];
    Vec3 Un = face.normal * Uc.dot(face.normal);   // normal component
    Vec3 Ut = Uc - Un;                              // tangential
    double tau = nuEff * Ut.norm() / delta;
    double dynP = 0.5 * obs.refVelocity * obs.refVelocity;
    return tau / std::max(dynP, 1e-20);
}

// velocity at a point
inline double ObservationOperator::computeVelocityAt(
        const Mesh& mesh, const FlowFields& fields,
        const Observable& obs) const {
    int ci = findNearestCell(mesh, obs.location);
    if (obs.component == 0) return fields.U[ci].x;
    if (obs.component == 1) return fields.U[ci].y;
    return fields.U[ci].z;
}

// pressure (Cp) at a point
inline double ObservationOperator::computePressureAt(
        const Mesh& mesh, const FlowFields& fields,
        const Observable& obs) const {
    int ci = findNearestCell(mesh, obs.location);
    double dynP = 0.5 * obs.refVelocity * obs.refVelocity;
    return fields.p[ci] / std::max(dynP, 1e-20);
}

// separation point: x-location where wall shear = 0
inline double ObservationOperator::computeSeparationPoint(
        const Mesh& mesh, const FlowFields& fields,
        double nu, const Observable& obs) const {
    PatchID pid = mesh.patchByName(obs.patchName);
    const Patch& pat = mesh.patch(pid);

    // walk along the patch and find where tangential wall shear changes sign
    double xSep = 0.0;
    double prevTau = 1.0;   // assume positive (attached) at start
    for (FaceID fi : pat.faces) {
        const Face& face = mesh.face(fi);
        int ow = face.owner;
        double nuEff = nu + fields.nuT[ow];
        double delta = std::max(face.delta, 1e-20);
        Vec3 Uc = fields.U[ow];
        Vec3 Ut = Uc - face.normal * Uc.dot(face.normal);
        // signed tangential shear (use x-component as streamwise indicator)
        double tau = nuEff * Ut.x / delta;

        if (prevTau > 0 && tau <= 0) {
            xSep = face.center.x;   // first sign change
            return xSep;
        }
        prevTau = tau;
    }
    return xSep;   // no separation found - returns 0
}

// evaluate all observables
inline std::vector<double> ObservationOperator::evaluate(
        const Mesh& mesh, const FlowFields& fields, double nu) const {
    std::vector<double> pred(observables_.size());
    for (size_t i = 0; i < observables_.size(); ++i) {
        const Observable& obs = observables_[i];
        switch (obs.type) {
            case ObsType::Drag:
                pred[i] = computeDrag(mesh, fields, nu, obs); break;
            case ObsType::PressureTap:
                pred[i] = computePressureAt(mesh, fields, obs); break;
            case ObsType::VelocityProfile:
                pred[i] = computeVelocityAt(mesh, fields, obs); break;
            case ObsType::SkinFriction:
                pred[i] = computeSkinFrictionAt(mesh, fields, nu, obs); break;
            case ObsType::SeparationPoint:
                pred[i] = computeSeparationPoint(mesh, fields, nu, obs); break;
        }
    }
    return pred;
}

// Gaussian log-likelihood
inline double ObservationOperator::logLikelihood(
        const Mesh& mesh, const FlowFields& fields, double nu) const {
    auto pred = evaluate(mesh, fields, nu);
    double ll = 0.0;
    for (size_t i = 0; i < observables_.size(); ++i) {
        double var = observables_[i].meas.totalVariance();
        if (var < 1e-30) var = 1e-30;
        double diff = pred[i] - observables_[i].meas.observed;
        ll += -0.5 * diff * diff / var;
    }
    return ll;
}