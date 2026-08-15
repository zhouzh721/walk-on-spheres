#pragma once
#include "wos/mesh.hpp"

namespace wos {

struct EberlyCache {
    Point3D v0;
    Vec3D edge1, edge2;
    double a, b, c, det, denom;
};

double npq_seg(Point2D s0, Point2D s1, Point2D p, Point2D *nearest);
double eberly(Point3D p, Point3D *nearest, const EberlyCache *eberly_cache);

double npq_naive(const Mesh<2> &m, Point2D p, Point2D *nearest);
double npq_naive(const Mesh<3> &m, Point3D p, Point3D *nearest);

}
