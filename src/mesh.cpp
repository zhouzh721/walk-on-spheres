#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <mpi.h>
#include "wos/mesh.hpp"

namespace wos {

namespace {

[[noreturn]] void throw_mesh_error(const char *path, int line_number,
                                   const std::string &message) {
    std::string where = std::string("Mesh ") + path;
    if (line_number > 0) where += ":" + std::to_string(line_number);
    throw std::runtime_error(where + ": " + message);
}

int parse_vertex_ref(const std::string &token, int total_vertices,
                     const char *path, int line_number) {
    const std::size_t slash = token.find('/');
    const std::string index_text = token.substr(0, slash);
    if (index_text.empty()) {
        throw_mesh_error(path, line_number, "missing vertex index in '" + token + "'");
    }

    char *end = nullptr;
    errno = 0;
    const long parsed = std::strtol(index_text.c_str(), &end, 10);
    if (errno == ERANGE || end == index_text.c_str() || *end != '\0' ||
        parsed <= 0 || parsed > INT_MAX) {
        throw_mesh_error(
            path, line_number,
            "vertex index must be a positive 1-based integer: '" + token + "'");
    }

    const long zero_based = parsed - 1;
    if (zero_based >= total_vertices) {
        throw_mesh_error(path, line_number, "vertex index out of range: '" + token + "'");
    }
    return static_cast<int>(zero_based);
}

}

int peek_mesh_dim(const char *path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("Could not read mesh file ") + path);
    }

    std::string line;
    int dim = 0;
    while (std::getline(input, line)) {
        std::istringstream record(line);
        std::string kind;
        if (!(record >> kind) || kind[0] == '#') continue;
        if (kind == "l") { dim = 2; break; }
        if (kind == "f") { dim = 3; break; }
    }

    if (dim == 0) {
        throw std::runtime_error(
            std::string("Could not determine mesh dimension in ") + path +
            " (no 'l' or 'f' entries)");
    }

    return dim;
}

template<int N>
Mesh<N> load_mesh(const char *path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(std::string("Could not read mesh file ") + path);
    }

    std::string line;

    // first pass for counts
    int vert_count = 0;
    int prim_count = 0;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream record(line);
        std::string kind;
        if (!(record >> kind) || kind[0] == '#') continue;

        if (kind == "v") {
            vert_count++;
        } else if constexpr (N == 2) {
            if (kind == "l") {
                std::string vertex_ref;
                int idx_count = 0;
                while (record >> vertex_ref) {
                    if (vertex_ref[0] == '#') break;
                    idx_count++;
                }
                if (idx_count < 2) {
                    throw_mesh_error(path, line_number,
                                     "polyline has fewer than 2 vertices");
                }
                // Reserve for every consecutive segment plus a possible
                // auto-closing segment. The second pass shrinks to the exact
                // number after comparing resolved positive vertex indices.
                prim_count += idx_count;
            }
        } else {
            if (kind == "f") {
                std::string vertex_ref;
                int fv_count = 0;
                while (record >> vertex_ref) {
                    if (vertex_ref[0] == '#') break;
                    fv_count++;
                }
                if (fv_count != 3) {
                    throw_mesh_error(path, line_number,
                                     "face has " + std::to_string(fv_count) +
                                     " vertices; only triangles are supported");
                }
                prim_count++;
            }
        }
    }

    if (vert_count == 0) {
        throw_mesh_error(path, 0, "contains no vertices");
    }
    if (prim_count == 0) {
        throw_mesh_error(path, 0, N == 2
                         ? "contains no line primitives"
                         : "contains no triangle primitives");
    }

    input.clear();
    input.seekg(0, std::ios::beg);
    if (!input) {
        throw std::runtime_error(std::string("Could not rewind mesh file ") + path);
    }

    Mesh<N> mesh;
    mesh.verts.resize(vert_count);
    mesh.prims.resize(N * prim_count);
    if constexpr (N == 2) mesh.boundary_ids.resize(prim_count);

    // second pass for filling
    int v_idx = 0;
    int p_idx = 0;
    line_number = 0;
    [[maybe_unused]] int boundary_id = 0;
    while (std::getline(input, line)) {
        ++line_number;
        std::istringstream record(line);
        std::string kind;
        if (!(record >> kind) || kind[0] == '#') continue;

        if (kind == "v") {
            double x, y, z = 0.0;
            if (!(record >> x >> y)) {
                throw_mesh_error(path, line_number, "malformed vertex record");
            }
            std::string z_text;
            if (record >> z_text && z_text[0] != '#') {
                char *end = nullptr;
                errno = 0;
                z = std::strtod(z_text.c_str(), &end);
                if (errno == ERANGE || end == z_text.c_str() || *end != '\0') {
                    throw_mesh_error(path, line_number, "malformed vertex record");
                }
            }
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                throw_mesh_error(path, line_number,
                                 "vertex coordinates must be finite");
            }
            if constexpr (N == 2) {
                mesh.verts[v_idx++] = Point2D{x, y};   // discard z
            } else {
                mesh.verts[v_idx++] = Point3D{x, y, z};
            }
        } else if constexpr (N == 2) {
            if (kind == "l") {
                int first_vert = -1, prev_vert = -1;
                std::string vertex_ref;
                while (record >> vertex_ref) {
                    if (vertex_ref[0] == '#') break;
                    const int vert = parse_vertex_ref(
                        vertex_ref, vert_count, path, line_number);
                    if (first_vert < 0) first_vert = vert;
                    if (prev_vert >= 0) {
                        mesh.prims[2*p_idx + 0] = prev_vert;
                        mesh.prims[2*p_idx + 1] = vert;
                        mesh.boundary_ids[p_idx] = boundary_id;
                        p_idx++;
                    }
                    prev_vert = vert;
                }
                if (prev_vert >= 0 && prev_vert != first_vert) {
                    std::fprintf(stderr,
                                 "Warning: open polyline in %s auto-closed\n",
                                 path);
                    mesh.prims[2*p_idx + 0] = prev_vert;
                    mesh.prims[2*p_idx + 1] = first_vert;
                    mesh.boundary_ids[p_idx] = boundary_id;
                    p_idx++;
                }
                boundary_id++;
            }
        } else {
            if (kind == "f") {
                std::string refs[3];
                for (std::string &ref : refs) {
                    if (!(record >> ref) || ref[0] == '#') {
                        throw_mesh_error(path, line_number,
                                         "malformed triangle record");
                    }
                }
                mesh.prims[3*p_idx + 0] =
                    parse_vertex_ref(refs[0], vert_count, path, line_number);
                mesh.prims[3*p_idx + 1] =
                    parse_vertex_ref(refs[1], vert_count, path, line_number);
                mesh.prims[3*p_idx + 2] =
                    parse_vertex_ref(refs[2], vert_count, path, line_number);
                p_idx++;
            }
        }
    }

    if constexpr (N == 2) {
        mesh.prims.resize(2 * p_idx);
        mesh.boundary_ids.resize(p_idx);
    }

    for (int p = 0; p < mesh.n_prims(); ++p) {
        if constexpr (N == 2) {
            const Point2D a = mesh.verts[mesh.prims[2*p + 0]];
            const Point2D b = mesh.verts[mesh.prims[2*p + 1]];
            const double dx = b.x - a.x;
            const double dy = b.y - a.y;
            if (!(dx*dx + dy*dy > 0.0)) {
                throw_mesh_error(path, 0,
                                 "contains a zero-length line segment");
            }
        } else {
            const Point3D a = mesh.verts[mesh.prims[3*p + 0]];
            const Point3D b = mesh.verts[mesh.prims[3*p + 1]];
            const Point3D c = mesh.verts[mesh.prims[3*p + 2]];
            const double abx = b.x - a.x;
            const double aby = b.y - a.y;
            const double abz = b.z - a.z;
            const double acx = c.x - a.x;
            const double acy = c.y - a.y;
            const double acz = c.z - a.z;
            const double cx = aby*acz - abz*acy;
            const double cy = abz*acx - abx*acz;
            const double cz = abx*acy - aby*acx;
            if (!(cx*cx + cy*cy + cz*cz > 0.0)) {
                throw_mesh_error(path, 0,
                                 "contains a degenerate triangle");
            }
        }
    }

    return mesh;
}

template<int N>
void bcast_mesh(Mesh<N> &m, int leader_rank, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int n_verts = m.n_verts();
    int n_prims = m.n_prims();
    MPI_Bcast(&n_verts, 1, MPI_INT, leader_rank, comm);
    MPI_Bcast(&n_prims, 1, MPI_INT, leader_rank, comm);

    if (rank != leader_rank) {
        m.verts.resize(n_verts);
        m.prims.resize(N * n_prims);
        if constexpr (N == 2) m.boundary_ids.resize(n_prims);
    }

    // verts: N doubles per vertex, broadcast contiguously
    MPI_Bcast(m.verts.data(), N * n_verts, MPI_DOUBLE, leader_rank, comm);
    MPI_Bcast(m.prims.data(), N * n_prims, MPI_INT,    leader_rank, comm);
    if constexpr (N == 2) {
        MPI_Bcast(m.boundary_ids.data(), n_prims, MPI_INT, leader_rank, comm);
    }
}

template<int N>
void mesh_bbox(const Mesh<N> &m, double *xmin, double *xmax, double *ymin, double *ymax, double *zmin, double *zmax) {
    auto p0 = m.verts[0];
    double xl = p0.x, xu = p0.x;
    double yl = p0.y, yu = p0.y;
    double zl = 0.0, zu = 0.0;
    if constexpr (N == 3) {
        zl = p0.z;
        zu = p0.z;
    }

    for (int i = 1; i < m.n_verts(); i++) {
        auto v = m.verts[i];
        if (v.x < xl) xl = v.x; else if (v.x > xu) xu = v.x;
        if (v.y < yl) yl = v.y; else if (v.y > yu) yu = v.y;
        if constexpr (N == 3) {
            if (v.z < zl) zl = v.z;
            else if (v.z > zu) zu = v.z;
        }
    }
    if (xmin) *xmin = xl;
    if (xmax) *xmax = xu;
    if (ymin) *ymin = yl;
    if (ymax) *ymax = yu;
    if (zmin) *zmin = zl;
    if (zmax) *zmax = zu;
}

template<int N>
AABB<N> prim_bbox(const Mesh<N> &m, int p) {
    AABB<N> bbox{};
    if constexpr (N == 2) {
        Point2D s0 = m.verts[m.prims[2*p + 0]];
        Point2D s1 = m.verts[m.prims[2*p + 1]];
        bbox.pmin = Point2D{ std::fmin(s0.x, s1.x), std::fmin(s0.y, s1.y) };
        bbox.pmax = Point2D{ std::fmax(s0.x, s1.x), std::fmax(s0.y, s1.y) };
    } else {
        Point3D v0 = m.verts[m.prims[3*p + 0]];
        Point3D v1 = m.verts[m.prims[3*p + 1]];
        Point3D v2 = m.verts[m.prims[3*p + 2]];
        bbox.pmin = Point3D{
            std::fmin(std::fmin(v0.x, v1.x), v2.x),
            std::fmin(std::fmin(v0.y, v1.y), v2.y),
            std::fmin(std::fmin(v0.z, v1.z), v2.z),
        };
        bbox.pmax = Point3D{
            std::fmax(std::fmax(v0.x, v1.x), v2.x),
            std::fmax(std::fmax(v0.y, v1.y), v2.y),
            std::fmax(std::fmax(v0.z, v1.z), v2.z),
        };
    }
    return bbox;
}

template<int N>
Point<N> centroid(const Mesh<N> &m, int p) {
    if constexpr (N == 2) {
        Point2D s0 = m.verts[m.prims[2*p + 0]];
        Point2D s1 = m.verts[m.prims[2*p + 1]];
        return Point2D{ (s0.x+s1.x)/2.0, (s0.y+s1.y)/2.0 };
    } else {
        Point3D v0 = m.verts[m.prims[3*p + 0]];
        Point3D v1 = m.verts[m.prims[3*p + 1]];
        Point3D v2 = m.verts[m.prims[3*p + 2]];
        return Point3D{
            (v0.x+v1.x+v2.x)/3.0,
            (v0.y+v1.y+v2.y)/3.0,
            (v0.z+v1.z+v2.z)/3.0
        };
    }
}

// explicit instantiations for 2D and 3D
template Mesh<2> load_mesh<2>(const char *);
template Mesh<3> load_mesh<3>(const char *);
template void bcast_mesh<2>(Mesh<2>&, int, MPI_Comm);
template void bcast_mesh<3>(Mesh<3>&, int, MPI_Comm);
template void mesh_bbox<2>(const Mesh<2>&, double*, double*, double*, double*, double*, double*);
template void mesh_bbox<3>(const Mesh<3>&, double*, double*, double*, double*, double*, double*);
template AABB<2> prim_bbox<2>(const Mesh<2>&, int);
template AABB<3> prim_bbox<3>(const Mesh<3>&, int);
template Point2D centroid<2>(const Mesh<2>&, int);
template Point3D centroid<3>(const Mesh<3>&, int);

}
