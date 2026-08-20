#pragma once
#include <cmath>
#include <vector>
#include <mpi.h>

namespace wos {

struct Point2D { double x, y; };
struct Point3D { double x, y, z; };

// helper to select appropriately dimensioned point for Point
template<int N> struct PointSelector;
template<> struct PointSelector<2> { using type = Point2D; };
template<> struct PointSelector<3> { using type = Point3D; };
template<int N> using Point = typename PointSelector<N>::type;

inline double dist(Point2D a, Point2D b) {
    return std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y));
}
inline double dist(Point3D a, Point3D b) {
    return std::sqrt((a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z));
}

template<int N>
struct Mesh {
    std::vector<Point<N>> verts;
    std::vector<int> prims;   // 2D: segments (s0,s1) per prim; 3D: tris (v0,v1,v2) per prim
    std::vector<int> boundary_ids; // 2D: source `l` record for each segment; unused in 3D

    int n_verts() const { return (int)verts.size(); }
    int n_prims() const { return (int)(prims.size() / N); }
};

int peek_mesh_dim(const char *path);

template<int N> Mesh<N> load_mesh(const char *path);
template<int N> void bcast_mesh(Mesh<N> &m, int leader_rank, MPI_Comm comm);
template<int N> void mesh_bbox(const Mesh<N> &m, double *xmin, double *xmax, double *ymin, double *ymax, double *zmin, double *zmax);

}
