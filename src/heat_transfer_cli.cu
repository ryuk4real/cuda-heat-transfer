#include "init.hpp"
#include "io.hpp"
#include "update.hpp"
#include "util.hpp"
#include <iostream>

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

void serial(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next);
void straightforward_unified(unsigned int n_steps, unsigned int grid_rows, 
                           unsigned int grid_cols, unsigned int n_hot_top_rows, 
                           unsigned int n_hot_bottom_rows, double* temperature_current, 
                           double* temperature_next);

int main()
{
    unsigned int step = 0;
    unsigned int n_steps {10000};
    unsigned int grid_rows {1 << 8};
    unsigned int grid_cols {1 << 12};
    unsigned int n_hot_top_rows {2};
    unsigned int n_hot_bottom_rows {2};
    double initial_hot_temperature {20};
    double * temperature_current = new (std::nothrow) double[grid_rows*grid_cols];
    double * temperature_next = new (std::nothrow) double[grid_rows*grid_cols];
    double elapsed_time {0.0};
    unsigned int field_width {5};
    std::string outfile_prefix {"temperature"};
    std::string outfile_extension {".dat"};
 
    init_top_bottom_temperature(temperature_current, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, initial_hot_temperature);
    init_top_bottom_temperature(temperature_next, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, initial_hot_temperature);

    std::cout << "Saving initial configuration... " << std::endl;
    save_temperature(outfile_prefix, outfile_extension, step, temperature_current, grid_rows, grid_cols, field_width);
    std::cout << "Done" << std::endl;

    std::cout << "Simulation in progress... " << std::endl;
    util::Timer clTimer;

    // Check for CUDA devices
    int device_count;
    cudaCheckError(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        return 1;
    }

    //serial(n_steps, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, temperature_current, temperature_next);
    straightforward_unified(n_steps, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, temperature_current, temperature_next);
    
    elapsed_time = static_cast<double>(clTimer.getTimeMilliseconds());
    std::cout << "Simulation loop elapsed time: " << elapsed_time << " ms (corresponding to " << (elapsed_time / 1000.0) << " s)" << std::endl;

    std::cout << "Saving final configuration... " << std::endl;
    save_temperature(outfile_prefix, outfile_extension, --step, temperature_current, grid_rows, grid_cols, field_width); 
    std::cout << "Done" << std::endl;

    /*
     * To visualize the simulation outcome, run gnuplot and use the following command:
     *
     *   plot 'temperature_step_N.dat' matrix with image
     *
     * where N is the step of the final configuration. Use quit to exit gnuplot.
     *
     */

    delete[] temperature_current, temperature_next;
    return 0;
}

void serial(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next)
{
    unsigned int step;
    for (step = 1; step <= n_steps; step++)
    {
        update_region(temperature_next, temperature_current, grid_rows, grid_cols, n_hot_top_rows, (grid_rows-1)-n_hot_bottom_rows, 1, (grid_cols-1)-1);
        swap_buffer_ptrs(temperature_next, temperature_current);    
    }
}

void straightforward_unified(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next)
{
    /** 
        This function uses automatic unified memory management, which allows the CPU and GPU to access the same memory
        without explicit memory transfers. The CPU and GPU can access the same memory location, which is allocated using
        cudaMallocManaged.
    */

    double *d_temp_current, *d_temp_next;

    cudaMallocManaged(&d_temp_current, grid_rows * grid_cols * sizeof(double));
    cudaMallocManaged(&d_temp_next, grid_rows * grid_cols * sizeof(double));

    memcpy(d_temp_current, temperature_current, grid_rows * grid_cols * sizeof(double));
    memcpy(d_temp_next, temperature_next, grid_rows * grid_cols * sizeof(double));

    //TODO: Define block and grid dimensions as input parameters
    dim3 block_dim(16, 8);
    dim3 grid_dim(
        (grid_cols + block_dim.x - 1) / block_dim.x,
        (grid_rows + block_dim.y - 1) / block_dim.y
    );

    for (unsigned int step = 1; step <= n_steps; step++) {
        straightforward_unified_kernel<<<grid_dim, block_dim>>>( d_temp_next, d_temp_current, grid_rows, grid_cols, n_hot_top_rows, (grid_rows-1)-n_hot_bottom_rows, 1, (grid_cols-1)-1);
        
        // Check for any errors launching the kernel
        cudaCheckError(cudaGetLastError());

        // Synchronize the device and check for synchronization errors
        cudaCheckError(cudaDeviceSynchronize());

        swap_buffer_ptrs(d_temp_next, d_temp_current);
    }

    memcpy(temperature_current, d_temp_current, grid_rows * grid_cols * sizeof(double));
    memcpy(temperature_next, d_temp_next, grid_rows * grid_cols * sizeof(double));

    cudaCheckError(cudaFree(d_temp_current));
    cudaCheckError(cudaFree(d_temp_next));
}