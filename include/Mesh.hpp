#pragma once

#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <cmath>

// 3D cartesian vector with vector operations
struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
    
    Vec3() = default; 
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}          // 3D vector constructor
    
    Vec3 operator+(const Vec3& o) const{ return {x+o.x, y+o.y, z+o.z};}     // vector addition
    Vec3 operator-(const Vec3& o) const{ return {x-o.x, y-o.y, z-o.z};}     // vector subtraction
    Vec3 operator*(double s)    const{ return {x*s, y*s, z*s};}             // vector scalor multiplication
    Vec3 operator/(double s)    const{ return {x/s, y/s, z/s};}             // vector scalor division

    double dot(const Vec3& o)  const { return x*o.x + y*o.y + z*o.z;}       // dot product
    
    Vec3   cross(const Vec3& o) const {                                     // cross product
        return {y*o.z - z*o.y, z*o.x - x*o.z, x*o.y - y*o.x };              
    }
    double norm()  const { return std::sqrt(x*x + y*y + z*z);}              // vector norm
    double norm2() const { return x*x + y*y + z*z;}                         // vector norm squared

    Vec3   unit()  const { double n = norm();                               // unit vector
        return (n > 1e-300) ? (*this)/n : Vec3{}; 
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }             // vector scaling

// Mesh Topology Indices
using CellID = int;         // volumes
using FaceID = int;         // interface between cells
using NodeID = int;         // corner points
using PatchID = int;        // boundary groups 

// Face data
struct Face{
    CellID owner = -1;      // cell normal vector points away 
    CellID neighbor = -1;   // cell that outward-from-owner normal points towards

    // each face is connected to two cells, each initialized to -1 to indicate no cell assigned

    Vec3    center;         // face centroid
    Vec3    normal;         // outward-from-owner unit normal
    double  area = 0.0;     // face area
    double  delta = 0.0;    // distance betwwen owner and neighbor cell centers                      
    Vec3    d;              // vector from owner to neighbor centers

    double  weight = 0.5;   // interpolation weight: phi_f = w*phi_P + (1-w)*phi_N
    
    // at boundaries, owner = #, neighbor = -1 since there is no neighboring cell
    bool isBoundary() const { return neighbor < 0;} 
    
    // patch index for boundary faces
    PatchID patchID = -1;
};  

// no node struct because a mesh node is just a Vec3

// Cell data
struct Cell{
    Vec3    center;
    double volume = 0.0;        // cell volume 
    std::vector<FaceID> faces;  // faces that bound this cell
}; 

// Patch data
struct Patch{
    // actual Boundary condition values will live in a different file (BoundaryCondition.hpp)
    std::string     name;
    std::string     type;   // "wall", "inlet", "outlet", ...
    std::vector<FaceID>     faces; 
};

// Mesh
// Computational grid with cells, faces, nodes, and boundary patches
class Mesh{
private:
    std::vector<Cell>       cells_;
    std::vector<Face>       faces_;
    std::vector<Vec3>       nodes_;
    std::vector<Patch>      patches_;
    std::vector<double>     wallDist_; // wallDist data -> populated by computeWallDistance
    std::vector<int>        ownerList_;
    std::vector<int>        neighborList_;

    int nInternal_ = 0;     // numner of internal faces
    
    // Sub-loaders
    static Mesh loadOpenFOAM(const std::string& polyMeshDir);
    static Mesh loadBinary(const std::string& path);

    void computeFaceGeometry();
    void computeCellCentersAndVolumes();
    void ComputeInterpolationWeights();

public: 
    Mesh() = default;   // constructor

    // returns constructed mesh from loaded file (OpenFOAM format or custom binary format)
    static Mesh loadFromFile(const std::string& path);

    // test mesh (2D channel mesh)
    // domain [0,Lx] x [0,Ly]), useful for validation before full meshes.
    static Mesh makeChannel2D(int nx, int ny, double Lx, double Ly);

    // returns count
    int nCells()  const { return static_cast<int>(cells_.size());}
    int nFaces()  const { return static_cast<int>(faces_.size());}
    int nNodes()  const { return static_cast<int>(nodes_.size());}
    int nPatches() const { return static_cast<int>(patches_.size());}

    // returns specific object by ID
    const Cell&  cell(CellID  id) const { return cells_[id];}
    const Face&  face(FaceID  id) const { return faces_[id];}
    const Vec3&  node(NodeID  id) const { return nodes_[id];}
    const Patch& patch(PatchID id) const { return patches_[id];}

    PatchID patchByName(const std::string& name) const; // patch lookup by name

    // iterators (range-based for loops )
    const std::vector<Cell>&    cells()     const { return cells_;}
    const std::vector<Face>&    faces()     const { return faces_;}
    const std::vector<Patch>&   patches()   const { return patches_;}  

    // computes geometric quantites like interpolation weights, delta vectors, etc. 
    // must be called after modifying topology
    void computeGeometry();

    // distance from cell center to nearest wall (for SST blending)
    const std::vector<double>& wallDistance() const { return wallDist_;}
    void computeWallDistance();

    // topology access for linear solver
    
    int nInternalFaces() const { return nInternal_;} // returns number of internal faces
    // internal faces [0, nInternal), boundary faces [nInternal_, nFaces_) 

    // owner/neighbor list used by LinearSolver for matrix-vector multiplcation
    const std::vector<int>& ownerList()     const { return ownerList_;}
    const std::vector<int>& neighborList() const { return neighborList_;}

    void saveBinary(const std::string& path) const; // saves mesh data to a binary file
};