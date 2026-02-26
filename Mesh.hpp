#pragma once

#include <vector>
#include <array>
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
    double norm2() const { return x*x + y*y + z*z; }                        // vector norm squared

    Vec3   unit()  const { double n = norm();                               // unit vector
        return (n > 1e-300) ? (*this)/n : Vec3{}; 
    }
};

inline Vec3 operator*(double s, const Vec3& v) { return v * s; }


// Mesh Topology Indices
using CellID = int;         // volumes
using FaceID = int;         // interface between cells
using NodeID = int;         // corner points
using PatchID = int;        // boundary groups 

// Face data
struct Face{};

// Cell data
struct Cell{}; 

// Patch data
struct Patch{};

// Mesh
class Mesh{};
