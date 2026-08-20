#include "wos/boundary/scene.hpp"

#include "wos/boundary/fcpw_scene.hpp"

namespace wos {

std::unique_ptr<BoundaryScene2D> make_boundary_scene_2d(
    const Mesh<2> &mesh,
    const std::vector<BoundaryType> &primitive_types) {
    return std::make_unique<FcpwBoundaryScene2D>(
        mesh, primitive_types);
}

} // namespace wos
