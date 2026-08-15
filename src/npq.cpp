#include <cmath>
#include "wos/mesh.hpp"
#include "wos/npq.hpp"

namespace wos {

// nearest point query for a line segment. Returns the nearest point distance and writes the nearest point to *nearest
double npq_seg(Point2D s0, Point2D s1, Point2D p, Point2D *nearest) {
    double vx = s1.x - s0.x;
    double vy = s1.y - s0.y;
    double wx = p.x - s0.x;
    double wy = p.y - s0.y;

    double vv = vx*vx + vy*vy;  // |v|^2
    double wv = wx*vx + wy*vy;  // w·v

    double par = (vv > 0.0) ? wv / vv : 0.0;    // parallel projection of w onto v
    // clamp projection to endpoints of line seg
    if (par < 0.0) par = 0.0;
    if (par > 1.0) par = 1.0;

    nearest->x = s0.x + par*vx;
    nearest->y = s0.y + par*vy;

    // distance p->nearest
    double dx = p.x - nearest->x;
    double dy = p.y - nearest->y;
    return dx*dx + dy*dy;
}

// deprecated by bvh npq
double npq_naive(const Mesh<2> &m, Point2D p, Point2D *nearest) {
    double closest_d_sq = INFINITY;

    for (int s = 0; s < m.n_prims(); s++) {
        Point2D s0 = m.verts[m.prims[2*s + 0]];
        Point2D s1 = m.verts[m.prims[2*s + 1]];

        Point2D seg_nearest;
        double d_sq = npq_seg(s0, s1, p, &seg_nearest);

        if (d_sq < closest_d_sq) {
            closest_d_sq = d_sq;
            *nearest = seg_nearest;
        }
    }

    return std::sqrt(closest_d_sq);
}

// nearest point query for a tri via Eberly's algorithm. Returns the nearest point squared distance and writes the nearest point to *nearest
double eberly(Point3D p, Point3D *nearest, const EberlyCache *eberly_cache) {
    Point3D v0 = eberly_cache->v0;
    Vec3D edge1 = eberly_cache->edge1;
    Vec3D edge2 = eberly_cache->edge2;
    double a = eberly_cache->a;
    double b = eberly_cache->b;
    double c = eberly_cache->c;
    double det = eberly_cache->det;
    double denom = eberly_cache->denom;

    Vec3D BP = Vec3D{v0.x - p.x, v0.y - p.y, v0.z - p.z};
    double d = dot(edge1, BP);
    double e = dot(edge2, BP);

    double s = b*e - c*d;
    double t = b*d - a*e;

    if (s + t <= det) {
        if (s < 0) {
            if (t < 0) {
                // region 4
                if (d < 0) {
                    t = 0;
                    if (-d >= a) {
                        s = 1;
                    } else {
                        s = -d/a;
                    }
                } else {
                    s = 0;
                    if (e >= 0) {
                        t = 0;
                    } else if (-e >= c) {
                        t = 1;
                    } else {
                        t = -e/c;
                    }
                }
            } else {
                // region 3
                s = 0;
                if (e >= 0) {
                    t = 0;
                } else if (-e >= c) {
                    t = 1;
                } else {
                    t = -e/c;
                }
            }
        } else if (t < 0) {
            // region 5
            t = 0;
            if (d >= 0) {
                s = 0;
            } else if (-d >= a) {
                s = 1;
            } else {
                s = -d/a;
            }
        } else {
            // region 0
            s /= det;
            t /= det;
        }
    } else {
        if (s < 0) {
            // region 2
            double bd = b+d;
            double ce = c+e;
            if (ce > bd) {
                double numer = ce - bd;
                if (numer >= denom) {
                    s = 1;
                } else {
                    s = numer / denom;
                }
                t = 1-s;
            } else {
                s = 0;
                if (ce <= 0) {
                    t = 1;
                } else if (e >= 0) {
                    t = 0;
                } else {
                    t = -e/c;
                }
            }
        } else if (t < 0) {
            // region 6
            double be = b+e;
            double ad = a+d;
            if (ad > be) {
                double numer = ad - be;
                if (numer >= denom) {
                    t = 1;
                } else {
                    t = numer / denom;
                }
                s = 1-t;
            } else {
                t = 0;
                if (ad <= 0) {
                    s = 1;
                } else if (d >= 0) {
                    s = 0;
                } else {
                    s = -d/a;
                }
            }
        } else {
            // region 1
            double numer = (c+e) - (b+d);
            if (numer <= 0) {
                s = 0;
            } else {
                if (numer >= denom) {
                    s = 1;
                } else {
                    s = numer / denom;
                }
            }
            t = 1-s;
        }
    }

    *nearest = Point3D{
        v0.x + s*edge1.x + t*edge2.x,
        v0.y + s*edge1.y + t*edge2.y,
        v0.z + s*edge1.z + t*edge2.z
    };

    Vec3D displacement = Vec3D{
        nearest->x - p.x,
        nearest->y - p.y,
        nearest->z - p.z
    };
    return dot(displacement, displacement);
}

// deprecated by bvh npq
double npq_naive(const Mesh<3> &m, Point3D p, Point3D *nearest) {
    double closest_d_sq = INFINITY;

    for (int s = 0; s < m.n_prims(); s++) {
        Point3D v0 = m.verts[m.prims[3*s + 0]];
        Point3D v1 = m.verts[m.prims[3*s + 1]];
        Point3D v2 = m.verts[m.prims[3*s + 2]];

        Point3D tri_nearest;

        Vec3D edge1 = Vec3D{v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
        Vec3D edge2 = Vec3D{v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
        double a = dot(edge1, edge1);
        double b = dot(edge1, edge2);
        double c = dot(edge2, edge2);
        double det = a*c - b*b;
        double denom = a - 2*b + c;
        EberlyCache eberly_cache{v0, edge1, edge2, a, b, c, det, denom};

        double d_sq = eberly(p, &tri_nearest, &eberly_cache);

        if (d_sq < closest_d_sq) {
            closest_d_sq = d_sq;
            *nearest = tri_nearest;
        }
    }

    return std::sqrt(closest_d_sq);
}

}
