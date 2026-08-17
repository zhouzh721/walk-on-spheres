#pragma once
#include <cstdint>
#include <cstdio>
#include <hdf5.h>
#include <mpi.h>
#include <vector>
#include "wos/field_3D.hpp"
#include "wos/grid.hpp"
#include "wos/source_mode.hpp"

namespace wos {

// Write WoS fields to HDF5, with each rank contributing its assigned sub-block.
// The returned status is identical on every rank.
inline bool wos_write_hdf5(const char *filename,
                           Grid grid,
                           int x_start, int y_start, int z_start,
                           const Field3D &mean,
                           const Field3D &variance,
                           const Field3D &standard_error,
                           const Field3D &mean_steps,
                           const LocationField3D &location,
                           int N_walks,
                           double epsilon,
                           int max_steps,
                           long long max_steps_hits,
                           bool write_ray_metadata,
                           int max_ray_attempts,
                           long long ambiguous_ray_retries,
                           long long indeterminate_points,
                           std::uint64_t seed,
                           bool write_screened_metadata,
                           double alpha,
                           SourceMode source_mode)
{
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto all_succeeded = [](bool local_success) {
        int local = local_success ? 1 : 0;
        int global = 0;
        MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        return global != 0;
    };
    auto report_failure = [&](const char *operation) {
        if (rank == 0) {
            std::fprintf(stderr, "HDF5 error while %s\n", operation);
        }
    };

    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (!all_succeeded(fapl >= 0)) {
        if (fapl >= 0) H5Pclose(fapl);
        report_failure("creating the file-access property list");
        return false;
    }

    herr_t status = H5Pset_fapl_mpio(fapl, MPI_COMM_WORLD, MPI_INFO_NULL);
    if (!all_succeeded(status >= 0)) {
        H5Pclose(fapl);
        report_failure("enabling parallel file access");
        return false;
    }

    hid_t file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
    const herr_t fapl_close_status = H5Pclose(fapl);
    if (!all_succeeded(file >= 0 && fapl_close_status >= 0)) {
        if (file >= 0) H5Fclose(file);
        report_failure("creating the output file");
        return false;
    }

    hid_t dxpl = H5Pcreate(H5P_DATASET_XFER);
    if (!all_succeeded(dxpl >= 0)) {
        if (dxpl >= 0) H5Pclose(dxpl);
        H5Fclose(file);
        report_failure("creating the dataset-transfer property list");
        return false;
    }

    status = H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE);
    if (!all_succeeded(status >= 0)) {
        H5Pclose(dxpl);
        H5Fclose(file);
        report_failure("enabling collective dataset writes");
        return false;
    }

    auto write_field = [&](const char *name, const auto &field,
                           hid_t memory_type, hid_t dataset_type) {
        hsize_t dims[3] = {
            static_cast<hsize_t>(grid.Nx),
            static_cast<hsize_t>(grid.Ny),
            static_cast<hsize_t>(grid.Nz),
        };
        hid_t filespace = H5Screate_simple(3, dims, nullptr);
        if (!all_succeeded(filespace >= 0)) {
            if (filespace >= 0) H5Sclose(filespace);
            return false;
        }

        hid_t dataset = H5Dcreate(file, name, dataset_type, filespace,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (!all_succeeded(dataset >= 0)) {
            if (dataset >= 0) H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        hsize_t offset[3] = {
            static_cast<hsize_t>(x_start),
            static_cast<hsize_t>(y_start),
            static_cast<hsize_t>(z_start),
        };
        hsize_t count[3] = {
            static_cast<hsize_t>(field.Nx),
            static_cast<hsize_t>(field.Ny),
            static_cast<hsize_t>(field.Nz),
        };
        status = H5Sselect_hyperslab(
            filespace, H5S_SELECT_SET, offset, nullptr, count, nullptr);
        if (!all_succeeded(status >= 0)) {
            H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        hid_t memspace = H5Screate_simple(3, count, nullptr);
        if (!all_succeeded(memspace >= 0)) {
            if (memspace >= 0) H5Sclose(memspace);
            H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        const herr_t write_status = H5Dwrite(
            dataset, memory_type, memspace, filespace, dxpl, field.data.data());
        const herr_t memspace_close_status = H5Sclose(memspace);
        const herr_t dataset_close_status = H5Dclose(dataset);
        const herr_t filespace_close_status = H5Sclose(filespace);
        return all_succeeded(write_status >= 0 &&
                             memspace_close_status >= 0 &&
                             dataset_close_status >= 0 &&
                             filespace_close_status >= 0);
    };

    if (!write_field("/mean", mean, H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE) ||
        !write_field("/variance", variance, H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE) ||
        !write_field("/standard_error", standard_error,
                     H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE) ||
        !write_field("/mean_steps", mean_steps,
                     H5T_NATIVE_DOUBLE, H5T_NATIVE_DOUBLE) ||
        !write_field("/location", location,
                     H5T_NATIVE_UCHAR, H5T_NATIVE_UCHAR)) {
        report_failure("writing field datasets");
        H5Pclose(dxpl);
        H5Fclose(file);
        return false;
    }

    // Coordinate vectors are global data. Rank 0 writes them; the other ranks
    // select no elements while still participating in each collective write.
    std::vector<double> x_arr, y_arr, z_arr;
    if (rank == 0) {
        x_arr.resize(grid.Nx);
        y_arr.resize(grid.Ny);
        z_arr.resize(grid.Nz);
        for (int i = 0; i < grid.Nx; ++i) {
            x_arr[i] = grid_coordinate(
                grid.xmin, grid.xmax, i, grid.Nx);
        }
        for (int i = 0; i < grid.Ny; ++i) {
            y_arr[i] = grid_coordinate(
                grid.ymin, grid.ymax, i, grid.Ny);
        }
        for (int i = 0; i < grid.Nz; ++i) {
            z_arr[i] = grid_coordinate(
                grid.zmin, grid.zmax, i, grid.Nz);
        }
    }

    auto write_1d = [&](const char *name, hsize_t n, const double *rank_zero_data) {
        hid_t filespace = H5Screate_simple(1, &n, nullptr);
        if (!all_succeeded(filespace >= 0)) {
            if (filespace >= 0) H5Sclose(filespace);
            return false;
        }

        hid_t dataset = H5Dcreate(file, name, H5T_NATIVE_DOUBLE, filespace,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (!all_succeeded(dataset >= 0)) {
            if (dataset >= 0) H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        hid_t memspace = H5Screate_simple(1, &n, nullptr);
        if (!all_succeeded(memspace >= 0)) {
            if (memspace >= 0) H5Sclose(memspace);
            H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        herr_t selection_status = 0;
        if (rank != 0) {
            selection_status = H5Sselect_none(filespace);
            if (selection_status >= 0) selection_status = H5Sselect_none(memspace);
        }
        if (!all_succeeded(selection_status >= 0)) {
            H5Sclose(memspace);
            H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        double unused = 0.0;
        const double *buffer = rank == 0 ? rank_zero_data : &unused;
        const herr_t write_status = H5Dwrite(
            dataset, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, buffer);
        const herr_t memspace_close_status = H5Sclose(memspace);
        const herr_t dataset_close_status = H5Dclose(dataset);
        const herr_t filespace_close_status = H5Sclose(filespace);
        return all_succeeded(write_status >= 0 &&
                             memspace_close_status >= 0 &&
                             dataset_close_status >= 0 &&
                             filespace_close_status >= 0);
    };

    if (!write_1d("/x", static_cast<hsize_t>(grid.Nx),
                  rank == 0 ? x_arr.data() : nullptr) ||
        !write_1d("/y", static_cast<hsize_t>(grid.Ny),
                  rank == 0 ? y_arr.data() : nullptr) ||
        !write_1d("/z", static_cast<hsize_t>(grid.Nz),
                  rank == 0 ? z_arr.data() : nullptr)) {
        report_failure("writing coordinate datasets");
        H5Pclose(dxpl);
        H5Fclose(file);
        return false;
    }

    // Scalar run settings are grouped under /metadata. Only rank 0 selects the
    // scalar value, while every rank participates in the collective write.
    hid_t metadata = H5Gcreate(
        file, "/metadata", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (!all_succeeded(metadata >= 0)) {
        if (metadata >= 0) H5Gclose(metadata);
        report_failure("creating the metadata group");
        H5Pclose(dxpl);
        H5Fclose(file);
        return false;
    }

    auto write_scalar = [&](const char *name, hid_t type, const void *value) {
        hid_t filespace = H5Screate(H5S_SCALAR);
        if (!all_succeeded(filespace >= 0)) {
            if (filespace >= 0) H5Sclose(filespace);
            return false;
        }

        hid_t dataset = H5Dcreate(metadata, name, type, filespace,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (!all_succeeded(dataset >= 0)) {
            if (dataset >= 0) H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        hid_t memspace = H5Screate(H5S_SCALAR);
        if (!all_succeeded(memspace >= 0)) {
            if (memspace >= 0) H5Sclose(memspace);
            H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        herr_t selection_status = 0;
        if (rank != 0) {
            selection_status = H5Sselect_none(filespace);
            if (selection_status >= 0) selection_status = H5Sselect_none(memspace);
        }
        if (!all_succeeded(selection_status >= 0)) {
            H5Sclose(memspace);
            H5Dclose(dataset);
            H5Sclose(filespace);
            return false;
        }

        const herr_t write_status =
            H5Dwrite(dataset, type, memspace, filespace, dxpl, value);
        const herr_t memspace_close_status = H5Sclose(memspace);
        const herr_t dataset_close_status = H5Dclose(dataset);
        const herr_t filespace_close_status = H5Sclose(filespace);
        return all_succeeded(write_status >= 0 &&
                             memspace_close_status >= 0 &&
                             dataset_close_status >= 0 &&
                             filespace_close_status >= 0);
    };

    bool metadata_ok =
        write_scalar("N_walks", H5T_NATIVE_INT, &N_walks) &&
        write_scalar("epsilon", H5T_NATIVE_DOUBLE, &epsilon) &&
        write_scalar("max_steps", H5T_NATIVE_INT, &max_steps) &&
        write_scalar("max_steps_hits", H5T_NATIVE_LLONG, &max_steps_hits) &&
        write_scalar("seed", H5T_NATIVE_UINT64, &seed);
    if (write_ray_metadata) {
        metadata_ok =
            metadata_ok &&
            write_scalar("max_ray_attempts", H5T_NATIVE_INT,
                         &max_ray_attempts) &&
            write_scalar("ambiguous_ray_retries", H5T_NATIVE_LLONG,
                         &ambiguous_ray_retries) &&
            write_scalar("indeterminate_points", H5T_NATIVE_LLONG,
                         &indeterminate_points);
    }
    if (write_screened_metadata) {
        const int source_mode_value = static_cast<int>(source_mode);
        metadata_ok =
            metadata_ok &&
            write_scalar("alpha", H5T_NATIVE_DOUBLE, &alpha) &&
            write_scalar("source_mode", H5T_NATIVE_INT, &source_mode_value);
    }
    const herr_t metadata_close_status = H5Gclose(metadata);
    if (!metadata_ok || !all_succeeded(metadata_close_status >= 0)) {
        report_failure("writing metadata");
        H5Pclose(dxpl);
        H5Fclose(file);
        return false;
    }

    const herr_t dxpl_close_status = H5Pclose(dxpl);
    const herr_t file_close_status = H5Fclose(file);
    if (!all_succeeded(dxpl_close_status >= 0 && file_close_status >= 0)) {
        report_failure("closing the output file");
        return false;
    }
    return true;
}

}
