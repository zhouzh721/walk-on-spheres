#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace wos {

// 3D box field, row-major (i, j, k). Single contiguous allocation for hdf5 write.
template<typename T>
struct Field3DData {
    int Nx, Ny, Nz;
    std::vector<T> data;

    Field3DData(int Nx, int Ny, int Nz)
        : Nx(Nx), Ny(Ny), Nz(Nz), data(static_cast<std::size_t>(Nx) * Ny * Nz) {}

    T &operator()(int i, int j, int k) {
        return data[flat(i, j, k)];
    }
    T operator()(int i, int j, int k) const {
        return data[flat(i, j, k)];
    }

private:
    std::size_t flat(int i, int j, int k) const {
        return (static_cast<std::size_t>(i) * Ny + j) * Nz + k;
    }
};

using Field3D = Field3DData<double>;
using LocationField3D = Field3DData<std::uint8_t>;

}
