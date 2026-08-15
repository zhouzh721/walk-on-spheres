#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>
#include "wos/bvh.hpp"
#include "wos/fastmath.hpp"
#include "wos/mesh.hpp"
#include "wos/npq.hpp"

namespace wos {

namespace {

constexpr int STACK_MAX = 64;
constexpr int LEAF_SIZE = 4;

template<int N>
struct BVHBuildNode {   // temporary data used during construction
    AABB<N> bbox;
    Point<N> centroid;
    int prim_idx;
};

template<int N>
MPI_Datatype make_bvh_node_mpi_type() {
    BVHNode<N> sample{};
    MPI_Aint base = 0;
    MPI_Get_address(&sample, &base);

    int block_lengths[8];
    MPI_Aint displacements[8];
    MPI_Datatype types[8];
    int field_count = 0;

    auto add_field = [&](const void *field, MPI_Datatype type) {
        MPI_Aint address = 0;
        MPI_Get_address(field, &address);
        block_lengths[field_count] = 1;
        displacements[field_count] = address - base;
        types[field_count] = type;
        ++field_count;
    };

    add_field(&sample.bbox.pmin.x, MPI_DOUBLE);
    add_field(&sample.bbox.pmin.y, MPI_DOUBLE);
    if constexpr (N == 3) add_field(&sample.bbox.pmin.z, MPI_DOUBLE);
    add_field(&sample.bbox.pmax.x, MPI_DOUBLE);
    add_field(&sample.bbox.pmax.y, MPI_DOUBLE);
    if constexpr (N == 3) add_field(&sample.bbox.pmax.z, MPI_DOUBLE);
    add_field(&sample.prim_count, MPI_INT);
    add_field(&sample.idx, MPI_INT);

    MPI_Datatype fields_type;
    MPI_Type_create_struct(field_count, block_lengths, displacements,
                           types, &fields_type);
    MPI_Datatype node_type;
    MPI_Type_create_resized(fields_type, 0,
                            static_cast<MPI_Aint>(sizeof(BVHNode<N>)),
                            &node_type);
    MPI_Type_commit(&node_type);
    MPI_Type_free(&fields_type);
    return node_type;
}

MPI_Datatype make_eberly_cache_mpi_type() {
    EberlyCache sample{};
    MPI_Aint base = 0;
    MPI_Get_address(&sample, &base);

    int block_lengths[14];
    MPI_Aint displacements[14];
    MPI_Datatype types[14];
    int field_count = 0;

    auto add_double = [&](const double *field) {
        MPI_Aint address = 0;
        MPI_Get_address(field, &address);
        block_lengths[field_count] = 1;
        displacements[field_count] = address - base;
        types[field_count] = MPI_DOUBLE;
        ++field_count;
    };

    add_double(&sample.v0.x);
    add_double(&sample.v0.y);
    add_double(&sample.v0.z);
    add_double(&sample.edge1.x);
    add_double(&sample.edge1.y);
    add_double(&sample.edge1.z);
    add_double(&sample.edge2.x);
    add_double(&sample.edge2.y);
    add_double(&sample.edge2.z);
    add_double(&sample.a);
    add_double(&sample.b);
    add_double(&sample.c);
    add_double(&sample.det);
    add_double(&sample.denom);

    MPI_Datatype fields_type;
    MPI_Type_create_struct(field_count, block_lengths, displacements,
                           types, &fields_type);
    MPI_Datatype cache_type;
    MPI_Type_create_resized(fields_type, 0,
                            static_cast<MPI_Aint>(sizeof(EberlyCache)),
                            &cache_type);
    MPI_Type_commit(&cache_type);
    MPI_Type_free(&fields_type);
    return cache_type;
}

inline double aabb_d_sq(const AABB<2> &aabb, Point2D p) {
    double dx = dmax(0.0, dmax(aabb.pmin.x - p.x, p.x - aabb.pmax.x));
    double dy = dmax(0.0, dmax(aabb.pmin.y - p.y, p.y - aabb.pmax.y));
    return dx*dx + dy*dy;
}
inline double aabb_d_sq(const AABB<3> &aabb, Point3D p) {
    double dx = dmax(0.0, dmax(aabb.pmin.x - p.x, p.x - aabb.pmax.x));
    double dy = dmax(0.0, dmax(aabb.pmin.y - p.y, p.y - aabb.pmax.y));
    double dz = dmax(0.0, dmax(aabb.pmin.z - p.z, p.z - aabb.pmax.z));
    return dx*dx + dy*dy + dz*dz;
}

template<int N>
AABB<N> union_bbox(const std::vector<BVHBuildNode<N>> &centroids, int start, int end) {
    AABB<N> u = centroids[start].bbox;
    for (int i = start + 1; i < end; i++) {
        const AABB<N> &c = centroids[i].bbox;
        u.pmin.x = dmin(u.pmin.x, c.pmin.x);
        u.pmin.y = dmin(u.pmin.y, c.pmin.y);
        u.pmax.x = dmax(u.pmax.x, c.pmax.x);
        u.pmax.y = dmax(u.pmax.y, c.pmax.y);
        if constexpr (N == 3) {
            u.pmin.z = dmin(u.pmin.z, c.pmin.z);
            u.pmax.z = dmax(u.pmax.z, c.pmax.z);
        }
    }
    return u;
}

template<int N>
void partition_by_axis(std::vector<BVHBuildNode<N>> &centroids,
                       int start, int mid, int end, int axis) {
    auto less_by_axis = [axis](const BVHBuildNode<N> &a,
                               const BVHBuildNode<N> &b) {
        double a_coord;
        double b_coord;
        if (axis == 0) {
            a_coord = a.centroid.x;
            b_coord = b.centroid.x;
        } else if (axis == 1) {
            a_coord = a.centroid.y;
            b_coord = b.centroid.y;
        } else {
            if constexpr (N == 3) {
                a_coord = a.centroid.z;
                b_coord = b.centroid.z;
            } else {
                a_coord = 0.0;
                b_coord = 0.0;
            }
        }

        if (a_coord < b_coord) return true;
        if (a_coord > b_coord) return false;
        return a.prim_idx < b.prim_idx;
    };

    std::nth_element(centroids.begin() + start,
                     centroids.begin() + mid,
                     centroids.begin() + end,
                     less_by_axis);
}

template<int N>
int build_subtree(BVH<N> &bvh, std::vector<BVHBuildNode<N>> &centroids, int start, int end, int &node_ptr) {
    int span = end - start;
    AABB<N> node_bbox = union_bbox(centroids, start, end);
    int node = node_ptr++;
    bvh.nodes[node].bbox = node_bbox;

    if (span <= LEAF_SIZE) {
        bvh.nodes[node].prim_count = span;
        bvh.nodes[node].idx = start;

        const Mesh<N> *m = bvh.mesh;
        for (int i = 0; i < span; i++) {
            int prim_idx = centroids[start + i].prim_idx;
            bvh.prims[start + i] = prim_idx;

            if constexpr (N == 3) {
                // pre-compute values for Eberly to save work in the hot-loop
                Point3D v0 = m->verts[m->prims[3*prim_idx + 0]];
                Point3D v1 = m->verts[m->prims[3*prim_idx + 1]];
                Point3D v2 = m->verts[m->prims[3*prim_idx + 2]];

                Vec3D edge1 = Vec3D{v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
                Vec3D edge2 = Vec3D{v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
                double a = dot(edge1, edge1);
                double b = dot(edge1, edge2);
                double c = dot(edge2, edge2);
                double det = a*c - b*b;
                double denom = a - 2*b + c;

                bvh.eberly_caches[start + i] = EberlyCache{v0, edge1, edge2, a, b, c, det, denom};
            }
        }

        return node;
    }

    bvh.nodes[node].prim_count = 0;    // internal node

    double ex = node_bbox.pmax.x - node_bbox.pmin.x;
    double ey = node_bbox.pmax.y - node_bbox.pmin.y;
    int axis;
    if constexpr (N == 2) {
        axis = (ex >= ey) ? 0 : 1;
    } else {
        double ez = node_bbox.pmax.z - node_bbox.pmin.z;
        axis = (ex >= ey) ? (ex >= ez ? 0 : 2) : (ey >= ez ? 1 : 2);
    }
    int mid = (start + end) / 2;
    partition_by_axis(centroids, start, mid, end, axis);

    [[maybe_unused]] int left = build_subtree(bvh, centroids, start, mid, node_ptr);
    assert(left == node + 1);
    int right = build_subtree(bvh, centroids, mid, end, node_ptr);
    bvh.nodes[node].idx = right;
    return node;
}

}  // anonymous namespace

template<int N>
std::unique_ptr<BVH<N>> build_bvh(const Mesh<N> &m) {
    auto bvh = std::make_unique<BVH<N>>();
    bvh->mesh = &m;
    bvh->nodes.resize(2*m.n_prims() - 1);   // worst-case (LEAF_SIZE=1) node count
    bvh->prims.resize(m.n_prims());
    if constexpr (N == 3) bvh->eberly_caches.resize(m.n_prims());

    std::vector<BVHBuildNode<N>> centroids(m.n_prims());
    for (int p = 0; p < m.n_prims(); p++) {
        centroids[p] = BVHBuildNode<N>{prim_bbox(m, p), centroid(m, p), p};
    }

    int node_ptr = 0;
    build_subtree(*bvh, centroids, 0, m.n_prims(), node_ptr);
    bvh->nodes.resize(node_ptr);
    bvh->nodes.shrink_to_fit();

    return bvh;
}

template<int N>
void bcast_bvh(BVH<N> &bvh, const Mesh<N> &mesh,
               int leader_rank, MPI_Comm comm) {
    int rank = 0;
    int comm_size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);
    if (comm_size == 1) {
        bvh.mesh = &mesh;
        return;
    }

    int sizes[3] = {
        static_cast<int>(bvh.nodes.size()),
        static_cast<int>(bvh.prims.size()),
        static_cast<int>(bvh.eberly_caches.size()),
    };
    MPI_Bcast(sizes, 3, MPI_INT, leader_rank, comm);

    if (rank != leader_rank) {
        bvh.nodes.resize(sizes[0]);
        bvh.prims.resize(sizes[1]);
        if constexpr (N == 3) bvh.eberly_caches.resize(sizes[2]);
    }
    bvh.mesh = &mesh;

    MPI_Datatype node_type = make_bvh_node_mpi_type<N>();
    MPI_Bcast(bvh.nodes.data(), sizes[0], node_type, leader_rank, comm);
    MPI_Type_free(&node_type);

    MPI_Bcast(bvh.prims.data(), sizes[1], MPI_INT, leader_rank, comm);

    if constexpr (N == 3) {
        MPI_Datatype cache_type = make_eberly_cache_mpi_type();
        MPI_Bcast(bvh.eberly_caches.data(), sizes[2], cache_type,
                  leader_rank, comm);
        MPI_Type_free(&cache_type);
    }
}

NearestPointResult<2> bvh_npq(const BVH<2> &bvh, Point2D p) {
    const Mesh<2> *m = bvh.mesh;
    double closest_d_sq = INFINITY;
    NearestPointResult<2> result{INFINITY, Point2D{}, -1, -1};

    int stack[STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int curr = stack[--sp];
        const BVHNode<2> &curr_node = bvh.nodes[curr];

        if (aabb_d_sq(curr_node.bbox, p) >= closest_d_sq) continue;

        if (curr_node.prim_count > 0) {
            for (int i = 0; i < curr_node.prim_count; i++) {
                int s = bvh.prims[curr_node.idx + i];
                Point2D s0 = m->verts[m->prims[2*s + 0]];
                Point2D s1 = m->verts[m->prims[2*s + 1]];

                Point2D seg_nearest;
                double d_sq = npq_seg(s0, s1, p, &seg_nearest);

                if (d_sq < closest_d_sq) {
                    closest_d_sq = d_sq;
                    result.point = seg_nearest;
                    result.primitive_id = s;
                    result.boundary_id = m->boundary_ids[s];
                }
            }
        } else {
            // push nearer child last so it's popped first (LIFO)
            int l = curr+1;
            int r = curr_node.idx;
            double l_d_sq = aabb_d_sq(bvh.nodes[l].bbox, p);
            double r_d_sq = aabb_d_sq(bvh.nodes[r].bbox, p);

            if (l_d_sq <= r_d_sq) {
                stack[sp++] = r;
                stack[sp++] = l;
            } else {
                stack[sp++] = l;
                stack[sp++] = r;
            }
        }
    }

    result.distance = std::sqrt(closest_d_sq);
    return result;
}

double bvh_npq(const BVH<3> &bvh, Point3D p, Point3D *nearest) {
    double closest_d_sq = INFINITY;

    int stack[STACK_MAX];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        int curr = stack[--sp];
        const BVHNode<3> &curr_node = bvh.nodes[curr];

        if (aabb_d_sq(curr_node.bbox, p) >= closest_d_sq) continue;

        if (curr_node.prim_count > 0) {
            for (int i = 0; i < curr_node.prim_count; i++) {
                Point3D tri_nearest;
                double d_sq = eberly(p, &tri_nearest, &bvh.eberly_caches[curr_node.idx + i]);

                if (d_sq < closest_d_sq) {
                    closest_d_sq = d_sq;
                    *nearest = tri_nearest;
                }
            }
        } else {
            int l = curr+1;
            int r = curr_node.idx;
            double l_d_sq = aabb_d_sq(bvh.nodes[l].bbox, p);
            double r_d_sq = aabb_d_sq(bvh.nodes[r].bbox, p);

            if (l_d_sq <= r_d_sq) {
                stack[sp++] = r;
                stack[sp++] = l;
            } else {
                stack[sp++] = l;
                stack[sp++] = r;
            }
        }
    }

    return std::sqrt(closest_d_sq);
}

// explicit instantiations for 2D and 3D
template std::unique_ptr<BVH<2>> build_bvh<2>(const Mesh<2>&);
template std::unique_ptr<BVH<3>> build_bvh<3>(const Mesh<3>&);
template void bcast_bvh<2>(BVH<2>&, const Mesh<2>&, int, MPI_Comm);
template void bcast_bvh<3>(BVH<3>&, const Mesh<3>&, int, MPI_Comm);

}
