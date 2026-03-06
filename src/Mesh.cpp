#include "../include/Mesh.hpp"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <iostream>
#include <filesystem>
#include <queue>
#include <numeric>
#include <cstring>

// loadFromFile supported mesh formats: 
//      - .foam / directory path    => OpenFoam polyMesh 
//      - .msh                      => custom bindary format
// For OpenFOAM, pass case directory or the path to the the polyMesh directory

Mesh Mesh::loadFromFile(const std::string& path) {
    namespace fs = std::filesystem;

    // Directory-based dispatch 
    // checks if path is a directory (treat as OpenFOAM case) or a direct polyMesh directory
    if (fs::is_directory(path)) {
        fs::path p(path);
        if (fs::exists(p / "points") && fs::exists(p / "faces")) {
            return loadOpenFOAM(path);
        }
        fs::path polyMesh = p / "constant" / "polyMesh";
        if (fs::exists(polyMesh / "points")) {
            return loadOpenFOAM(polyMesh.string());
        }
        throw std::runtime_error("Cannot find polyMesh in: " + path);
    }

    // File-based dispatch
    auto ext = fs::path(path).extension().string();
    if (ext == ".foam" || ext == "") {
        return loadOpenFOAM(path);
    }
    if (ext == ".msh") {
        return loadBinary(path);
    }
    throw std::runtime_error("Unknown mesh format: " + ext);
}

// OpenFOAM polyMesh reader (OpenFOAM mesh usually defined by 5 files)
// points   -- node coordinates
// faces    -- face node connectives
// owner    -- owner cell for each face
// neighbor -- neighbor cell for internal faces
// boundary -- patch definitions

// Parsing Helper: skips OpenfOAM file header until we find the start of data
static void skipFoamHeader(std::istream& is){
    std::string line;
    while (std::getline(is, line)) {
        auto pos = line.find("//");
        if (pos != std::string::npos) line.erase(pos);
        line.erase(0, line.find_first_not_of(" \t\r"));
        if (line == "(") return;
    }
    throw std::runtime_error("Could not find data block");
}

// Parsing helper: reads the element count from an OpenFOAM data file
static int readFoamCount(std::istream& is){
    std::string line;
    while (std::getline(is, line)) {
        auto pos = line.find("//");
        if (pos != std::string::npos) line.erase(pos);
        line.erase(0, line.find_first_not_of(" \t\r"));
        line.erase(line.find_last_not_of(" \t\r") + 1);
        if (line.empty() || line[0] == '/' || line[0] == 'F' ||
            line[0] == 'O' || line[0] == 'o' || line[0] == 'v' ||
            line[0] == 'C' || line[0] == 'a' || line[0] == '{' ||
            line[0] == '}')
            continue;
        try { return std::stoi(line); } catch (...) { continue; }
    }
    throw std::runtime_error("Could not read count");
}

Mesh Mesh::loadOpenFOAM(const std::string& polyMeshDir){
    namespace fs = std::filesystem;
    Mesh m;

    // extracts OpenFOAM mesh point data (defines geometric node coordinates)
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "points");
        if (!ifs) throw std::runtime_error("Cannot open points file");
        int nPts = readFoamCount(ifs);
        skipFoamHeader(ifs);
        m.nodes_.resize(nPts);
        for (int i = 0; i < nPts; ++i) {
            char c;
            ifs >> c; // '('
            ifs >> m.nodes_[i].x >> m.nodes_[i].y >> m.nodes_[i].z;
            ifs >> c; // ')'
        }
    } 
    
    // extracts OpenFOAM mesh face data (defines face-node connectivity topology)
    std::vector<std::vector<int>> faceNodes;
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "faces");
        if (!ifs) throw std::runtime_error("Cannot open faces file");
        int nF = readFoamCount(ifs);
        skipFoamHeader(ifs);
        faceNodes.resize(nF);
        for (int i = 0; i < nF; ++i) {
            int nv;
            ifs >> nv;
            char c;
            ifs >> c; // '('
            faceNodes[i].resize(nv);
            for (int j = 0; j < nv; ++j) ifs >> faceNodes[i][j];
            ifs >> c; // ')'
        }
        m.faces_.resize(nF);
    }

    // extracts OpenFOAM mesh owner data (defines face-cell connectivitytopology)
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "owner");
        if (!ifs) throw std::runtime_error("Cannot open owner file in " + polyMeshDir);
        int nOwn = readFoamCount(ifs);
        skipFoamHeader(ifs);
        for (int i = 0; i < nOwn; ++i) {
            int o;
            ifs >> o;
            m.faces_[i].owner = o;
        }
    }

    // extracts OpenFOAM mesh neighbor data (defines adjacent cell assignments internal faces)
    {
        fs::path nbrPath = fs::path(polyMeshDir) / "neighbour";
        std::ifstream ifs(nbrPath);
        if (!ifs) throw std::runtime_error("Cannot open neighbour file in " + polyMeshDir);
        int nNbr = readFoamCount(ifs);
        m.nInternal_ = nNbr;
        skipFoamHeader(ifs);
        for (int i = 0; i < nNbr; ++i) {
            int n;
            ifs >> n;
            m.faces_[i].neighbor = n;
        }
    }

    // determine total cell count from face owner and neighbor indices
    int nCells = 0;
    for (auto& f : m.faces_) {
        nCells = std::max(nCells, f.owner + 1);
        if (f.neighbor >= 0) nCells = std::max(nCells, f.neighbor + 1);
    }
    m.cells_.resize(nCells);

    // build cell-face adjacency (assigns each face to its owner and neighbor cells)
    for (int fi = 0; fi < static_cast<int>(m.faces_.size()); ++fi) {
        m.cells_[m.faces_[fi].owner].faces.push_back(fi);
        if (m.faces_[fi].neighbor >= 0)
            m.cells_[m.faces_[fi].neighbor].faces.push_back(fi);
    }

    // reads OpenFOAM boundary patch definitions and construct patch objects 
    // assigns boundary faces to patches and label faces with their patch ID
    {
        std::ifstream ifs(fs::path(polyMeshDir) / "boundary");
        if (ifs) {
            int nPatches = readFoamCount(ifs);
            skipFoamHeader(ifs);
            m.patches_.resize(nPatches);
            for (int p = 0; p < nPatches; ++p) {
                std::string line;
                // read patch name
                while (std::getline(ifs, line)) {
                    auto pos = line.find("//");
                    if (pos != std::string::npos) line.erase(pos);
                    line.erase(0, line.find_first_not_of(" \t\r"));
                    line.erase(line.find_last_not_of(" \t\r") + 1);
                    if (!line.empty() && line != "{" && line != "}") {
                        m.patches_[p].name = line;
                        break;
                    }
                }
                // parse patch body { type ...; nFaces ...; startFace ...; }
                int pnFaces = 0, startFace = 0;
                while (std::getline(ifs, line)) {
                    auto pos = line.find("//");
                    if (pos != std::string::npos) line.erase(pos);
                    line.erase(0, line.find_first_not_of(" \t\r"));
                    if (line.find('}') != std::string::npos) break;
                    std::istringstream ss(line);
                    std::string key;
                    ss >> key;
                    if (key == "type") {
                        std::string val;
                        ss >> val;
                        if (!val.empty() && val.back() == ';') val.pop_back();
                        m.patches_[p].type = val;
                    } else if (key == "nFaces") {
                        ss >> pnFaces;
                    } else if (key == "startFace") {
                        ss >> startFace;
                    }
                }
                m.patches_[p].faces.resize(pnFaces);
                for (int i = 0; i < pnFaces; ++i) {
                    int fIdx = startFace + i;
                    m.patches_[p].faces[i] = fIdx;
                    m.faces_[fIdx].patchID = p;
                }
            }
        }
    }
    // Mesh structure now: (-> means maps to)
    // Topology: face -> owner, face-> neighbor, cell -> faces
    // Boundary structure: patch -> faces, face -> patche, patch -> type


    // build geometry from topology
    // computes face centroid, area, and normal vector
    for (int fi = 0; fi < static_cast<int>(m.faces_.size()); ++fi) {
        const auto& fn = faceNodes[fi];
        int nv = static_cast<int>(fn.size());

        Vec3 ctr{};                                 // compute face centroid
        for (int j = 0; j < nv; ++j) {
            ctr = ctr + m.nodes_[fn[j]];
        }
        ctr = ctr / static_cast<double>(nv);
        m.faces_[fi].center = ctr;

        Vec3 areaNormal{};                          // compute face area normal vector
        for (int j = 0; j < nv; ++j) {
            const Vec3& a = m.nodes_[fn[j]];
            const Vec3& b = m.nodes_[fn[(j + 1) % nv]];
            areaNormal = areaNormal + (a - ctr).cross(b - ctr) * 0.5;
        }
        m.faces_[fi].area   = areaNormal.norm();    // face area (scalar)    
        m.faces_[fi].normal = areaNormal.unit();    // face unit normal
    }

    // build owner/neighbor connectivity arrays
    m.ownerList_.resize(m.nInternal_);              // m.nInternal_ = number of internal faces.
    m.neighborList_.resize(m.nInternal_);
    for (int fi = 0; fi < m.nInternal_; ++fi) {
        m.ownerList_[fi]    = m.faces_[fi].owner;
        m.neighborList_[fi] = m.faces_[fi].neighbor;
    }

    // compute cell centers, volumes, interpolation weights, delta vectors
    m.computeCellCentersAndVolumes();
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    // prints summary
    std::cout << "OpenFOAM mesh loaded: " << m.nCells() << " cells, "
              << m.nFaces() << " faces, " << m.nInternal_ << " internal faces, "
              << m.nPatches() << " patches\n";
    return m;
}

// Geometry Computations

void Mesh::computeCellCentersAndVolumes(){
    // determines cell centers by averaging the centers of all cell faces
    for (int ci = 0; ci < nCells(); ++ci) {
        Vec3 ctr{};
        for (FaceID fi : cells_[ci].faces) {
            ctr = ctr + faces_[fi].center;
        }
        cells_[ci].center = ctr / static_cast<double>(cells_[ci].faces.size());
    }

    // using divergence theorem to compute polyhedral cell volumes 
    // by summing signed contributions of each face’s area vector 
    // dotted with the vector from cell center to face center.
    // V = sum_f (Sf . (cf - cC))
    // area vector dotted with vector from cell center to face center
    // summing over faces gives volume 
    for (int ci = 0; ci < nCells(); ++ci) {
        double vol = 0.0;
        for (FaceID fi : cells_[ci].faces) {
            const Face& f = faces_[fi];
            Vec3 Sf = f.normal * f.area;
            double sign = (f.owner == ci) ? 1.0 : -1.0;
            vol += sign * Sf.dot(f.center - cells_[ci].center);
        }
        cells_[ci].volume = std::abs(vol) / 3.0;
        if (cells_[ci].volume < 1e-30) {
            cells_[ci].volume = 1e-20; // fallback for degenerate or nearly flat cells
        }
    }
}

// computes face-cell distance vectors and magnitudes
void Mesh::computeFaceGeometry() {
    // delta = distance between owner and neighbor centers
    // d     = vector from owner to neighbor centers
    for (int fi = 0; fi < nFaces(); ++fi) {
        Face& f = faces_[fi];
        if (!f.isBoundary()) {
            f.d     = cells_[f.neighbor].center - cells_[f.owner].center;
            f.delta = f.d.norm();
        } else {
            f.d     = f.center - cells_[f.owner].center;
            f.delta = f.d.norm();
        }
        if (f.delta < 1e-30) f.delta = 1e-20;
    }
}

// computes linear interpolation weights for internal faces from cell-face center distances
void Mesh::ComputeInterpolationWeights() {
    // weight = |fC - N| / (|fC - P| + |fC - N|)
    // phi_f = w * phi_P + (1-w) * phi_N
    for (int fi = 0; fi < nInternal_; ++fi) {
        Face& f = faces_[fi];
        double dP = (f.center - cells_[f.owner].center).norm();
        double dN = (f.center - cells_[f.neighbor].center).norm();
        double sum = dP + dN;
        f.weight = (sum > 1e-30) ? dN / sum : 0.5;
    }
}

// converts topological mesh into a discretization-ready computational grid
void Mesh::computeGeometry() {
    computeCellCentersAndVolumes();
    computeFaceGeometry();
    ComputeInterpolationWeights();

    ownerList_.resize(nInternal_);
    neighborList_.resize(nInternal_);
    for (int fi = 0; fi < nInternal_; ++fi) {
        ownerList_[fi]    = faces_[fi].owner;
        neighborList_[fi] = faces_[fi].neighbor;
    }
}

// computeWallDistance — brute-force from wall patches
void Mesh::computeWallDistance(){
    wallDist_.assign(nCells(), std::numeric_limits<double>::max());
    
    // collects wall face centers
    std::vector<Vec3> wallPts;
    for (const auto& p : patches_) {
        if (p.type == "wall") {
            for (FaceID fi : p.faces) {
                wallPts.push_back(faces_[fi].center);
            }
        }
    }
    // handles no wall case
    if (wallPts.empty()) {
        std::fill(wallDist_.begin(), wallDist_.end(), 1e10);
        return;
    }
    // brute force distance search
    // computes minimum Euclidean distance from each cell center to any wall face center
    for (int ci = 0; ci < nCells(); ++ci) {
        double minD = std::numeric_limits<double>::max();
        const Vec3& cc = cells_[ci].center;
        for (const Vec3& wp : wallPts) {
            double d = (cc - wp).norm();
            if (d < minD) minD = d;
        }
        wallDist_[ci] = minD;
    }
}

// lookup mechanism to retrieve a boundary patch index by its name
PatchID Mesh::patchByName(const std::string& name) const {
    for (int i = 0; i < static_cast<int>(patches_.size()); ++i) {
        if (patches_[i].name == name) return i;
    }
    throw std::runtime_error("Patch not found: " + name);
}

// save/load binary mesh code
// binary serialization layer
static const uint32_t MESH_MAGIC = 0x4D534831; // "MSH1"

void Mesh::saveBinary(const std::string& path) const {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) throw std::runtime_error("Cannot open for writing: " + path);

    auto writeInt = [&](int v)    { ofs.write(reinterpret_cast<const char*>(&v), sizeof(int)); };
    auto writeDbl = [&](double v) { ofs.write(reinterpret_cast<const char*>(&v), sizeof(double)); };
    auto writeVec = [&](const Vec3& v) { writeDbl(v.x); writeDbl(v.y); writeDbl(v.z); };
    auto writeStr = [&](const std::string& s) {
        int len = static_cast<int>(s.size());
        writeInt(len);
        ofs.write(s.data(), len);
    };

    uint32_t magic = MESH_MAGIC;
    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    writeInt(nCells());
    writeInt(nFaces());
    writeInt(nNodes());
    writeInt(nPatches());
    writeInt(nInternal_);

    for (const auto& n : nodes_) writeVec(n);

    for (const auto& f : faces_) {
        writeInt(f.owner); writeInt(f.neighbor); writeInt(f.patchID);
        writeVec(f.center); writeVec(f.normal);
        writeDbl(f.area); writeDbl(f.delta);
        writeVec(f.d); writeDbl(f.weight);
    }

    for (const auto& c : cells_) {
        writeVec(c.center); writeDbl(c.volume);
        int nf = static_cast<int>(c.faces.size());
        writeInt(nf);
        for (FaceID fid : c.faces) writeInt(fid);
    }

    for (const auto& p : patches_) {
        writeStr(p.name); writeStr(p.type);
        int nf = static_cast<int>(p.faces.size());
        writeInt(nf);
        for (FaceID fid : p.faces) writeInt(fid);
    }

    std::cout << "Mesh saved: " << path << "\n";
}

Mesh Mesh::loadBinary(const std::string& path) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) throw std::runtime_error("Cannot open binary mesh: " + path);

    auto readInt = [&]() -> int    { int v;    ifs.read(reinterpret_cast<char*>(&v), sizeof(int));    return v; };
    auto readDbl = [&]() -> double { double v; ifs.read(reinterpret_cast<char*>(&v), sizeof(double)); return v; };
    auto readVec = [&]() -> Vec3   { return {readDbl(), readDbl(), readDbl()}; };
    auto readStr = [&]() -> std::string {
        int len = readInt();
        std::string s(len, '\0');
        ifs.read(s.data(), len);
        return s;
    };

    uint32_t magic;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != MESH_MAGIC)
        throw std::runtime_error("Invalid binary mesh magic number");

    Mesh m;
    int nc = readInt(), nf = readInt(), nn = readInt(), np = readInt();
    m.nInternal_ = readInt();

    m.nodes_.resize(nn);
    for (int i = 0; i < nn; ++i) m.nodes_[i] = readVec();

    m.faces_.resize(nf);
    for (int i = 0; i < nf; ++i) {
        Face& f = m.faces_[i];
        f.owner = readInt(); f.neighbor = readInt(); f.patchID = readInt();
        f.center = readVec(); f.normal = readVec();
        f.area = readDbl(); f.delta = readDbl();
        f.d = readVec(); f.weight = readDbl();
    }

    m.cells_.resize(nc);
    for (int i = 0; i < nc; ++i) {
        Cell& c = m.cells_[i];
        c.center = readVec(); c.volume = readDbl();
        int cfn = readInt();
        c.faces.resize(cfn);
        for (int j = 0; j < cfn; ++j) c.faces[j] = readInt();
    }

    m.patches_.resize(np);
    for (int i = 0; i < np; ++i) {
        m.patches_[i].name = readStr();
        m.patches_[i].type = readStr();
        int pfn = readInt();
        m.patches_[i].faces.resize(pfn);
        for (int j = 0; j < pfn; ++j) m.patches_[i].faces[j] = readInt();
    }

    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int i = 0; i < m.nInternal_; ++i) {
        m.ownerList_[i]    = m.faces_[i].owner;
        m.neighborList_[i] = m.faces_[i].neighbor;
    }

    std::cout << "Binary mesh loaded: " << nc << " cells, " << nf << " faces\n";
    return m;
}


// add validation mesh
// patches: "inlet" (x=0), "outlet" (x=Lx), "top" (y=Ly, wall), "bottom" (y=0, wall)
// y-direction - tanh stretching toward both walls for BL resolution.
Mesh Mesh::makeChannel2D(int nx, int ny, double Lx, double Ly) {
    Mesh m;
    double dz = 1.0; // unit depth for per-unit-depth quantities

    // y-coordinates with tanh stretching toward both walls
    double stretch = 2.0;
    std::vector<double> yc(ny + 1);
    for (int j = 0; j <= ny; ++j) {
        double eta = static_cast<double>(j) / ny;
        yc[j] = 0.5 * Ly * (1.0 + std::tanh(stretch * (2.0 * eta - 1.0))
                                   / std::tanh(stretch));
    }
    // uniform x-coordinates
    std::vector<double> xc(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xc[i] = Lx * static_cast<double>(i) / nx;
    }
    // nodes: (nx+1)*(ny+1) * 2 planes (front z=0, back z=dz)
    int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xc[i], yc[j], 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xc[i], yc[j], dz);
        }

    // cells
    int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };

    // faces
    int nInternalH = nx * (ny - 1);      // horizontal internal (between j, j+1)
    int nInternalV = (nx - 1) * ny;      // vertical internal   (between i, i+1)
    m.nInternal_ = nInternalH + nInternalV;
    int nBnd = 2 * nx + 2 * ny;          // bottom + top + inlet + outlet
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;

    // internal horizontal faces
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            Face& f    = m.faces_[fi];
            f.owner    = cid(i, j);
            f.neighbor = cid(i, j + 1);
            double x0 = xc[i], x1 = xc[i + 1];
            f.center = Vec3(0.5 * (x0 + x1), yc[j + 1], 0.5 * dz);
            f.area   = (x1 - x0) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // internal vertical faces
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            Face& f    = m.faces_[fi];
            f.owner    = cid(i, j);
            f.neighbor = cid(i + 1, j);
            double y0 = yc[j], y1 = yc[j + 1];
            f.center = Vec3(xc[i + 1], 0.5 * (y0 + y1), 0.5 * dz);
            f.area   = (y1 - y0) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // boundary faces
    // bottom wall (y = 0)
    Patch bottom; bottom.name = "bottom"; bottom.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, 0); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.0, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        bottom.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(bottom));

    // top wall (y = Ly)
    Patch top; top.name = "top"; top.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, ny - 1); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), Ly, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, 1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        top.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(top));

    // inlet (x = 0)
    Patch inlet; inlet.name = "inlet"; inlet.type = "inlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(0, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.0, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(-1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        inlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(inlet));

    // outlet (x = Lx)
    Patch outlet; outlet.name = "outlet"; outlet.type = "outlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(nx - 1, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(Lx, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        outlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(outlet));

    // owner/neighbor lists
    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f]    = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    for (int j = 0; j < ny; ++j) {                                                                                                                                                                                  
        for (int i = 0; i < nx; ++i) {                                                                                                                                                                              
            Cell& c = m.cells_[cid(i, j)];                                                                                                                                                                          
            c.center = Vec3(0.5*(xc[i]+xc[i+1]), 0.5*(yc[j]+yc[j+1]), 0.5*dz);                                                                                                                                      
            c.volume = (xc[i+1]-xc[i]) * (yc[j+1]-yc[j]) * dz;                                                                                                                                               
        }                                                                                                                                                                                                    
    } 
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "Channel2D mesh: " << nx << "x" << ny
              << " (" << m.nCells() << " cells, " << m.nFaces() << " faces)\n";
    return m;
}

// Re-adaptive wall clustering overload
// Computes tanh stretch parameter so the first cell targets yPlusTarget.
Mesh Mesh::makeChannel2D(int nx, int ny, double Lx, double Ly, double Re, double yPlusTarget) {
    // Estimate friction velocity from flat-plate correlation
    // Cf ~ 0.058 * Re^(-0.2)
    // u_tau = sqrt(Cf/2) * U_ref   (U_ref = 1)
    double Cf = 0.058 * std::pow(Re, -0.2);
    double u_tau = std::sqrt(Cf / 2.0);

    // nu = U_ref * L_ref / Re, with U_ref = 1, L_ref = Ly/2
    double nu = (Ly / 2.0) / Re;

    // Target first cell height from y+ definition: y1 = y+ * nu / u_tau
    double y1_target = yPlusTarget * nu / u_tau;

    // Uniform spacing for comparison
    double y1_uniform = Ly / ny;

    double stretch = 0.0; // default: uniform mesh

    if (y1_target < y1_uniform) {
        // Bisect for stretch: larger stretch => smaller first cell
        // First cell height from tanh formula:
        //   y1(s) = 0.5 * Ly * (1 + tanh(s * (2.0/ny - 1)) / tanh(s))
        double sLo = 0.1, sHi = 20.0;

        // Check if target is reachable at max stretch
        auto firstCellHeight = [&](double s) {
            return 0.5 * Ly * (1.0 + std::tanh(s * (2.0 / ny - 1.0))
                                     / std::tanh(s));
        };

        double y1_max_stretch = firstCellHeight(sHi);
        if (y1_target < y1_max_stretch) {
            std::cout << "  WARNING: y1_target=" << y1_target
                      << " unreachable with ny=" << ny
                      << "; using max stretch=" << sHi
                      << " (y1=" << y1_max_stretch << ")\n";
            stretch = sHi;
        } else {
            // Bisection: find stretch where firstCellHeight(s) == y1_target
            for (int iter = 0; iter < 100; ++iter) {
                double sMid = 0.5 * (sLo + sHi);
                double y1_mid = firstCellHeight(sMid);
                if (y1_mid > y1_target)
                    sLo = sMid; // need more stretching
                else
                    sHi = sMid;
                if (sHi - sLo < 1e-10) break;
            }
            stretch = 0.5 * (sLo + sHi);
        }
    }
    // else: y1_target >= y1_uniform => uniform mesh is fine, stretch = 0

    std::cout << "  Adaptive mesh: Re=" << Re << " y+_target=" << yPlusTarget
              << " y1_target=" << y1_target << " stretch=" << stretch << "\n";

    // Build the mesh using the same logic as the original, but with computed stretch
    Mesh m;
    double dz = 1.0;

    // y-coordinates with tanh stretching
    std::vector<double> yc(ny + 1);
    if (stretch > 1e-12) {
        for (int j = 0; j <= ny; ++j) {
            double eta = static_cast<double>(j) / ny;
            yc[j] = 0.5 * Ly * (1.0 + std::tanh(stretch * (2.0 * eta - 1.0))
                                       / std::tanh(stretch));
        }
    } else {
        for (int j = 0; j <= ny; ++j) {
            yc[j] = Ly * static_cast<double>(j) / ny;
        }
    }

    // uniform x-coordinates
    std::vector<double> xc(nx + 1);
    for (int i = 0; i <= nx; ++i) {
        xc[i] = Lx * static_cast<double>(i) / nx;
    }

    // nodes
    int ptsPerPlane = (nx + 1) * (ny + 1);
    m.nodes_.resize(2 * ptsPerPlane);
    auto nid = [&](int i, int j, int k) { return k * ptsPerPlane + j * (nx + 1) + i; };
    for (int j = 0; j <= ny; ++j)
        for (int i = 0; i <= nx; ++i) {
            m.nodes_[nid(i, j, 0)] = Vec3(xc[i], yc[j], 0.0);
            m.nodes_[nid(i, j, 1)] = Vec3(xc[i], yc[j], dz);
        }

    // cells
    int nC = nx * ny;
    m.cells_.resize(nC);
    auto cid = [&](int i, int j) { return j * nx + i; };

    // faces
    int nInternalH = nx * (ny - 1);
    int nInternalV = (nx - 1) * ny;
    m.nInternal_ = nInternalH + nInternalV;
    int nBnd = 2 * nx + 2 * ny;
    m.faces_.resize(m.nInternal_ + nBnd);

    int fi = 0;

    // internal horizontal faces
    for (int j = 0; j < ny - 1; ++j) {
        for (int i = 0; i < nx; ++i) {
            Face& f    = m.faces_[fi];
            f.owner    = cid(i, j);
            f.neighbor = cid(i, j + 1);
            double x0 = xc[i], x1 = xc[i + 1];
            f.center = Vec3(0.5 * (x0 + x1), yc[j + 1], 0.5 * dz);
            f.area   = (x1 - x0) * dz;
            f.normal = Vec3(0, 1, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // internal vertical faces
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx - 1; ++i) {
            Face& f    = m.faces_[fi];
            f.owner    = cid(i, j);
            f.neighbor = cid(i + 1, j);
            double y0 = yc[j], y1 = yc[j + 1];
            f.center = Vec3(xc[i + 1], 0.5 * (y0 + y1), 0.5 * dz);
            f.area   = (y1 - y0) * dz;
            f.normal = Vec3(1, 0, 0);
            m.cells_[f.owner].faces.push_back(fi);
            m.cells_[f.neighbor].faces.push_back(fi);
            ++fi;
        }
    }

    // boundary faces — bottom wall
    Patch bottom; bottom.name = "bottom"; bottom.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, 0); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), 0.0, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, -1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        bottom.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(bottom));

    // top wall
    Patch top; top.name = "top"; top.type = "wall";
    for (int i = 0; i < nx; ++i) {
        Face& f = m.faces_[fi];
        f.owner = cid(i, ny - 1); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.5 * (xc[i] + xc[i + 1]), Ly, 0.5 * dz);
        f.area = (xc[i + 1] - xc[i]) * dz;
        f.normal = Vec3(0, 1, 0);
        m.cells_[f.owner].faces.push_back(fi);
        top.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(top));

    // inlet
    Patch inlet; inlet.name = "inlet"; inlet.type = "inlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(0, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(0.0, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(-1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        inlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(inlet));

    // outlet
    Patch outlet; outlet.name = "outlet"; outlet.type = "outlet";
    for (int j = 0; j < ny; ++j) {
        Face& f = m.faces_[fi];
        f.owner = cid(nx - 1, j); f.neighbor = -1;
        f.patchID = static_cast<int>(m.patches_.size());
        f.center = Vec3(Lx, 0.5 * (yc[j] + yc[j + 1]), 0.5 * dz);
        f.area = (yc[j + 1] - yc[j]) * dz;
        f.normal = Vec3(1, 0, 0);
        m.cells_[f.owner].faces.push_back(fi);
        outlet.faces.push_back(fi); ++fi;
    }
    m.patches_.push_back(std::move(outlet));

    // owner/neighbor lists
    m.ownerList_.resize(m.nInternal_);
    m.neighborList_.resize(m.nInternal_);
    for (int f = 0; f < m.nInternal_; ++f) {
        m.ownerList_[f]    = m.faces_[f].owner;
        m.neighborList_[f] = m.faces_[f].neighbor;
    }

    // cell centers and volumes
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            Cell& c = m.cells_[cid(i, j)];
            c.center = Vec3(0.5*(xc[i]+xc[i+1]), 0.5*(yc[j]+yc[j+1]), 0.5*dz);
            c.volume = (xc[i+1]-xc[i]) * (yc[j+1]-yc[j]) * dz;
        }
    }
    m.computeFaceGeometry();
    m.ComputeInterpolationWeights();

    std::cout << "Channel2D mesh: " << nx << "x" << ny
              << " (" << m.nCells() << " cells, " << m.nFaces() << " faces)\n";
    return m;
}