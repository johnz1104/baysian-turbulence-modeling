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
        if (!fs::exists(nbrPath))
            nbrPath = fs::path(polyMeshDir) / "neighbor"; // US spelling fallback
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

}