#pragma once

#include "Mesh.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

// Field<T> -- cell-centered FVM field (control volume) 
// this means that physical variables are stored as average values at the centroid of each mesh cell
// T = double   -> Scalar Field   
// T = vec3     -> Vector Field   
// for storing Field<T> quantities: 
// cell-centered (interior) values and boundary-face values are stored in separate arrays.
//      interior_[0..nCells)              — cell values
//      boundary_[0..nBoundaryFaces)      — boundary face values

template <typename T>
class Field {
public: 
    std::string name;

    Field() = default; // default constructor

    // storage allocator for mesh
    explicit Field(const Mesh& mesh, const std::string& n = "", T init = T{})
        : name(n), 
        mesh_(&mesh),
        interior_(mesh.nCells(), init),
        boundary_(mesh.nFaces() - mesh.nInternalFaces(), init){}

    // cell access via operator[]
    T&       operator[](int ci)       { return interior_[ci];}
    const T& operator[](int ci) const { return interior_[ci];}

    // boundary face access by local boundary index
    T&       bnd(int bi)       { return boundary_[bi];}
    const T& bnd(int bi) const { return boundary_[bi];}

    // boundary face access by global face ID
    T&       bface(int fi)       { return boundary_[fi - mesh_->nInternalFaces()];}
    const T& bface(int fi) const { return boundary_[fi - mesh_->nInternalFaces()];}

    // returns number of cell-centered values in field (= mesh.nCells)
    int size() const { return static_cast<int>(interior_.size());} 
    // returns number of boundary-face values (equals nFaces- nInternalFaces)
    int nBnd() const { return static_cast<int>(boundary_.size());} 
    
    // direct access to interior (cell-centered) storage.
    // returns reference to underlying vector for solver/linear operations.   
    std::vector<T>&       data()       { return interior_;}         
    const std::vector<T>& data() const { return interior_;}

    // dereferenecs stored pointer; returns reference to mesh
    const Mesh& mesh() const { return *mesh_;}    
    
    // face interpolation
    // computes field value at a face
    T faceValue(FaceID fi) const {
        const Face& f = mesh_->face(fi);
        if (f.isBoundary()) return bface(fi);               // returns stored boundary value (since no neighbor)
        return interior_[f.owner] * f.weight         
             + interior_[f.neighbor] * (1.0 - f.weight);    // returns computed interpolated internal value  
    }

   // norms
   // computes root-mean-square norm; handles both scalars and vectors
    double l2Norm() const {
        double s = 0.0;
        for (auto& v : interior_) s += sqMag(v);
        return std::sqrt(s / std::max(1, size()));
    }
    // computes maximum absolute value (scalar: max absolute value; vector: max magnitude)
    double linfNorm() const {
        double m = 0.0;
        for (auto& v : interior_) m = std::max(m, mag(v));
        return m;
    }

    // arithmetic helpers
    // sets every cell and boundary face value equal to val
    void setUniform(T val) {
        std::fill(interior_.begin(), interior_.end(), val);
        std::fill(boundary_.begin(), boundary_.end(), val);
    }
    // forces all values into the range lo <= v <= hi
    void clamp(double lo, double hi) {
        for (auto& v : interior_) doClamp(v, lo, hi);
        for (auto& v : boundary_) doClamp(v, lo, hi);
    }

private: 
    const Mesh* mesh_ = nullptr;    // pointer to mesh
    std::vector<T> interior_;       // stores cell-cenetered values
    std::vector<T> boundary_;       // stores boundary face values

    // static helper functions
    // -> correct behavior for both scalar/vector field 
    static double sqMag(double v)      { return v * v;}
    static double sqMag(const Vec3& v) { return v.norm2();}         // type polymorphism by overloading
    static double mag(double v)        { return std::abs(v);}
    static double mag(const Vec3& v)   { return v.norm();}
    static void doClamp(double& v, double lo, double hi) {
        v = std::max(lo, std::min(hi, v));
    }
    static void doClamp(Vec3& v, double lo, double hi) {
        v.x = std::max(lo, std::min(hi, v.x));
        v.y = std::max(lo, std::min(hi, v.y));
        v.z = std::max(lo, std::min(hi, v.z));
    }
};

using ScalarField = Field<double>;
using VectorField = Field<Vec3>;


// Green-Gauss gradient: (phi)_C = ∇(1/V_C) Σ_f (phi_f · S_f)
// where phi_f is the field value at face, S_f is the face area vector, and V_C is the cell volume
// computes the gradient of the scalar field at each cell center using Green-Gauss theorem
inline VectorField greenGaussGrad(const ScalarField& phi) {
    // input scalar field phi, output vector field -> each cell gets 3D vector gradient
    const Mesh& m = phi.mesh();
    VectorField grad(m, "grad(" + phi.name + ")");

    // loops over internal faces
    for (int fi = 0; fi < m.nInternalFaces(); ++fi) {
        const Face& f = m.face(fi);
        double phiF = phi[f.owner] * f.weight
                    + phi[f.neighbor] * (1.0 - f.weight);   // compute face value
        Vec3 Sf = f.normal * f.area;                        // get area vector
        grad[f.owner]    = grad[f.owner]    + Sf * phiF;    // owner cell contribution (phi_f * S_f)
        grad[f.neighbor] = grad[f.neighbor] - Sf * phiF;    // neighbor cell contribution
                                            // need to correct orientation for GG grad, thus the minus sign
    }

    // loops over boundary faces
    for (int fi = m.nInternalFaces(); fi < m.nFaces(); ++fi) {
        const Face& f = m.face(fi);
        Vec3 Sf = f.normal * f.area;
        // add boundary conditions (boundary faces only contribute to the owner cell)
        grad[f.owner] = grad[f.owner] + Sf * phi.bface(fi);
    }

    // divide by volume
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double V = m.cell(ci).volume;
        if (V > 1e-30) grad[ci] = grad[ci] / V;
    }
    return grad; 
}

// Velocity gradient (component-wise Green-Gauss)
struct VelocityGradients {
    VectorField dudx;   // (du/dx, du/dy, du/dz)
    VectorField dvdx;   // (dv/dx, dv/dy, dv/dz)
    VectorField dwdx;   // (dw/dx, dw/dy, dw/dz)
};

inline VelocityGradients computeVelocityGradients(const VectorField& U){
    const Mesh& m = U.mesh();
    ScalarField Ux(m, "Ux"), Uy(m, "Uy"), Uz(m, "Uz");

    for (int ci = 0; ci < m.nCells(); ++ci) {
        Ux[ci] = U[ci].x;  Uy[ci] = U[ci].y;  Uz[ci] = U[ci].z;
    }
    for (int fi = m.nInternalFaces(); fi < m.nFaces(); ++fi) {
        Ux.bface(fi) = U.bface(fi).x;
        Uy.bface(fi) = U.bface(fi).y;
        Uz.bface(fi) = U.bface(fi).z;
    }

    return { greenGaussGrad(Ux), greenGaussGrad(Uy), greenGaussGrad(Uz) };
}

// Velocity gradient can be decomposed into: 
// Strain-Rate tensor S (symmetric part) 
// Rotation tensor W (antisymmetric part) 

// Strain Rate Magnitude
// |S| = sqrt(2 * S_ij * S_ij),  S_ij = 0.5(dUi/dxj + dUj/dxi)
inline ScalarField strainRateMagnitude(const VelocityGradients& vg) {
    const Mesh& m = vg.dudx.mesh();
    ScalarField S(m, "|S|");
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double S11 = vg.dudx[ci].x;
        double S22 = vg.dvdx[ci].y;
        double S33 = vg.dwdx[ci].z;
        double S12 = 0.5 * (vg.dudx[ci].y + vg.dvdx[ci].x);
        double S13 = 0.5 * (vg.dudx[ci].z + vg.dwdx[ci].x);
        double S23 = 0.5 * (vg.dvdx[ci].z + vg.dwdx[ci].y);
        S[ci] = std::sqrt(2.0 * (S11*S11 + S22*S22 + S33*S33 + 2.0*(S12*S12 + S13*S13 + S23*S23)));
    }
    return S;
}

// Vorticity Magnitude
// |Omega| = sqrt(2 * W_ij * W_ij),  W_ij = 0.5(dUi/dxj - dUj/dxi)
inline ScalarField vorticityMagnitude(const VelocityGradients& vg) {
    const Mesh& m = vg.dudx.mesh();
    ScalarField W(m, "|W|");
    for (int ci = 0; ci < m.nCells(); ++ci) {
        double W12 = 0.5 * (vg.dudx[ci].y - vg.dvdx[ci].x);
        double W13 = 0.5 * (vg.dudx[ci].z - vg.dwdx[ci].x);
        double W23 = 0.5 * (vg.dvdx[ci].z - vg.dwdx[ci].y);
        W[ci] = std::sqrt(2.0 * 2.0 * (W12*W12 + W13*W13 + W23*W23));
    }
    return W;
}