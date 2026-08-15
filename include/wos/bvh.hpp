#pragma once
#include <memory>
#include <mpi.h>
#include <vector>
#include "wos/mesh.hpp"
#include "wos/npq.hpp"

namespace wos {

template<int N>
struct BVHNode {
    AABB<N> bbox;
    int prim_count;     // 0 => internal node, >0 => leaf containing this many prims
    int idx;            // leaf: first prim in bvh.prims[idx]; internal: right child at nodes[idx]
};

template<int N>
struct BVH {
    const Mesh<N> *mesh;
    std::vector<BVHNode<N>> nodes;
    std::vector<int> prims;
    std::vector<EberlyCache> eberly_caches;     // populated only for N=3
};

template<int N> std::unique_ptr<BVH<N>> build_bvh(const Mesh<N> &m);
template<int N>
void bcast_bvh(BVH<N> &bvh, const Mesh<N> &mesh,
               int leader_rank, MPI_Comm comm);

template<int N>
struct NearestPointResult {
    double distance;
    Point<N> point;
    int primitive_id;
    int boundary_id;
};

NearestPointResult<2> bvh_npq(const BVH<2> &bvh, Point2D p);
double bvh_npq(const BVH<3> &bvh, Point3D p, Point3D *nearest);

}
