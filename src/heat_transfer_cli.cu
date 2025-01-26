#include "init.hpp"
#include "io.hpp"
#include "update.hpp"
#include "util.hpp"
#include <iostream>

#define CONFIGURATIONS_STRING "1=straighforward unified, 2=straighforward standard"

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
void straightforward_unified(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next, dim3 block_dim);
void straightforward_standard(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next, dim3 block_dim);
void tiled_no_halos(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next, dim3 block_dim);
void tiled_with_halos(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next, dim3 block_dim);

void print_usage(const char* program_name);

int main(int argc, char *argv[])
{

    unsigned int configuration;
    unsigned int block_dim_x;
    unsigned int block_dim_y;

    --argc;

    if(argc == 0)
    {
        configuration = 0;
    }
    else if(argc == 1)
    {
        if(strncmp(argv[1], "-h", 2) == 0 || strncmp(argv[1], "--help", 6) == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
    }
    else if(argc == 3 && strncmp(argv[3], "--conf=", 7) == 0)
    {
        block_dim_x = std::stoi(argv[1]);
        block_dim_y = std::stoi(argv[2]);

        if(block_dim_x == 0 || block_dim_y == 0)
        {
            std::cerr << "Invalid block dimensions" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        if(strncmp(argv[3], "--conf=", 7) != 0)
        {
            std::cerr << "Invalid configuration" << std::endl;
            print_usage(argv[0]);
            return 1;
        }

        configuration = std::stoi(argv[3] + 7);

        if(configuration == 0)
        {
            std::cerr << "Invalid configuration" << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

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

    std::cout << "Grid size: " << grid_rows << " x " << grid_cols << std::endl;
    std::cout << "Number of steps: " << n_steps << std::endl;

    if (configuration != 0)
    {
        std::cout << "Block dimensions: " << block_dim_x << " x " << block_dim_y << std::endl;
    }

    std::cout << "Configuration: " << configuration << std::endl;

    std::cout << "Saving initial configuration... " << std::endl;
    save_temperature(outfile_prefix, outfile_extension, step, temperature_current, grid_rows, grid_cols, field_width);
    std::cout << "Done" << std::endl;

    std::cout << "Simulation in progress... " << std::endl;
    util::Timer clTimer;

    // Check for CUDA devices
    int device_count;
    cudaCheckError(cudaGetDeviceCount(&device_count));

    dim3 block_dim(block_dim_x, block_dim_y);

    // Execute selected configuration
    switch (configuration) {
        case 0:
            serial(n_steps, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, temperature_current, temperature_next);
            break;
        case 1:
            straightforward_unified(n_steps, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, temperature_current, temperature_next, block_dim);
            break;
        case 2:
            straightforward_standard(n_steps, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, temperature_current, temperature_next, block_dim);
            break;
        case 3:
            tiled_no_halos(n_steps, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, temperature_current, temperature_next, block_dim);
            break;
        case 4:
            tiled_with_halos(n_steps, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, temperature_current, temperature_next, block_dim);
            break;
    }

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

void straightforward_unified(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next, dim3 block_dim)
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

    dim3 grid_dim(
        (grid_cols + block_dim.x - 1) / block_dim.x,
        (grid_rows + block_dim.y - 1) / block_dim.y
    );

    for (unsigned int step = 1; step <= n_steps; step++)
    {
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

void straightforward_standard(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next, dim3 block_dim)
{
    /**
        This function uses standard device / host memory management.
    */

    double *d_temp_current, *d_temp_next;

    cudaCheckError(cudaMalloc(&d_temp_current, grid_rows * grid_cols * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_temp_next, grid_rows * grid_cols * sizeof(double)));

    cudaCheckError(cudaMemcpy(d_temp_current, temperature_current, grid_rows * grid_cols * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_temp_next, temperature_next, grid_rows * grid_cols * sizeof(double),  cudaMemcpyHostToDevice));

    dim3 grid_dim(
        (grid_cols + block_dim.x - 1) / block_dim.x,
        (grid_rows + block_dim.y - 1) / block_dim.y
    );

    for (unsigned int step = 1; step <= n_steps; step++)
    {
        straightforward_unified_kernel<<<grid_dim, block_dim>>>(d_temp_next, d_temp_current, grid_rows, grid_cols, n_hot_top_rows, (grid_rows-1)-n_hot_bottom_rows, 1, (grid_cols-1)-1);
        
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        swap_buffer_ptrs(d_temp_next, d_temp_current);
    }

    // Copy final results back to host
    cudaCheckError(cudaMemcpy(temperature_current, d_temp_current, grid_rows * grid_cols * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(temperature_next, d_temp_next, grid_rows * grid_cols * sizeof(double), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_temp_current));
    cudaCheckError(cudaFree(d_temp_next));
}

void tiled_no_halos(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next, dim3 block_dim)
{
    /**
        This function uses standard device / host memory management. The kernel uses shared memory to store the tile so that
        the global memory accesses are reduced.
    */

    if(block_dim.x < 16 || block_dim.y < 16)
    {
        std::cerr << "Block dimensions must be at least 16 x 16" << std::endl;
        return;
    }

    double *d_temp_current, *d_temp_next;

    cudaCheckError(cudaMalloc(&d_temp_current, grid_rows * grid_cols * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_temp_next, grid_rows * grid_cols * sizeof(double)));

    cudaCheckError(cudaMemcpy(d_temp_current, temperature_current, grid_rows * grid_cols * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_temp_next, temperature_next, grid_rows * grid_cols * sizeof(double),  cudaMemcpyHostToDevice));

    dim3 grid_dim(
        (grid_cols + block_dim.x - 1) / block_dim.x,
        (grid_rows + block_dim.y - 1) / block_dim.y
    );

    for (unsigned int step = 1; step <= n_steps; step++)
    {
        tiled_no_halos_kernel<<<grid_dim, block_dim>>>(d_temp_next, d_temp_current, grid_rows, grid_cols, n_hot_top_rows, (grid_rows-1)-n_hot_bottom_rows, 1, (grid_cols-1)-1);
        
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        swap_buffer_ptrs(d_temp_next, d_temp_current);
    }

    // Copy final results back to host
    cudaCheckError(cudaMemcpy(temperature_current, d_temp_current, grid_rows * grid_cols * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(temperature_next, d_temp_next, grid_rows * grid_cols * sizeof(double), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_temp_current));
    cudaCheckError(cudaFree(d_temp_next));
}

void tiled_with_halos(unsigned int n_steps, unsigned int grid_rows, unsigned int grid_cols, unsigned int n_hot_top_rows, unsigned int n_hot_bottom_rows, double* temperature_current, double* temperature_next, dim3 block_dim)
{
    /**
        This function uses standard device / host memory management. The kernel uses shared memory to store the tile from neighboring
        and also the halo elements so that the global memory accesses are reduced. 
    */

    if(block_dim.x < 16 || block_dim.y < 16)
    {
        std::cerr << "Block dimensions must be at least 16 x 16" << std::endl;
        return;
    }

    double *d_temp_current, *d_temp_next;

    cudaCheckError(cudaMalloc(&d_temp_current, grid_rows * grid_cols * sizeof(double)));
    cudaCheckError(cudaMalloc(&d_temp_next, grid_rows * grid_cols * sizeof(double)));

    cudaCheckError(cudaMemcpy(d_temp_current, temperature_current, grid_rows * grid_cols * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_temp_next, temperature_next, grid_rows * grid_cols * sizeof(double),  cudaMemcpyHostToDevice));

    dim3 grid_dim(
        (grid_cols + block_dim.x - 1) / block_dim.x,
        (grid_rows + block_dim.y - 1) / block_dim.y
    );

    for (unsigned int step = 1; step <= n_steps; step++)
    {
        tiled_with_halos_kernel<<<grid_dim, block_dim>>>(d_temp_next, d_temp_current, grid_rows, grid_cols, n_hot_top_rows, (grid_rows-1)-n_hot_bottom_rows, 1, (grid_cols-1)-1);
        
        cudaCheckError(cudaGetLastError());
        cudaCheckError(cudaDeviceSynchronize());

        swap_buffer_ptrs(d_temp_next, d_temp_current);
    }

    // Copy final results back to host
    cudaCheckError(cudaMemcpy(temperature_current, d_temp_current, grid_rows * grid_cols * sizeof(double), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(temperature_next, d_temp_next, grid_rows * grid_cols * sizeof(double), cudaMemcpyDeviceToHost));

    cudaCheckError(cudaFree(d_temp_current));
    cudaCheckError(cudaFree(d_temp_next));
}

void print_usage(const char* program_name)
{
    std::cout << "Usage: " << program_name << " [block_dim_x] [block_dim_y] [--conf=N]\n"
            << "Options:\n"
            << "  block_dim_x     Block dimension x\n"
            << "  block_dim_y     Block dimension y\n"
            << "  --conf=N        Configuration to use ("<< CONFIGURATIONS_STRING <<")\n"
            << "  -h, --help      Display this help message\n"
            << "To run the program with serial (0) configuration, do not provide any arguments\n";
}