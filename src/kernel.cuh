#pragma once
#include <cuda_runtime.h>

#define TILE_WIDTH 16

void update_temperature_kernel(double* t_next, double* t, unsigned int i, unsigned int j, unsigned int n_cols)
{
    t_next[i*n_cols+j] = (1.0/20.0) * 
                          ( 4 * ( t[i*n_cols+(j+1)] + 
                                  t[i*n_cols+(j-1)] + 
                                  t[(i+1)*n_cols+j] + 
                                  t[(i-1)*n_cols+j] 
                                ) + 
                            t[(i+1)*n_cols+(j+1)] + 
                            t[(i+1)*n_cols+(j-1)] + 
                            t[(i-1)*n_cols+(j+1)] + 
                            t[(i-1)*n_cols+(j-1)]
                          );
}



__global__ void straightforward_unified_kernel(double* t_next, double* t, unsigned int n_rows, unsigned int n_cols, unsigned int i_start, unsigned int i_end, unsigned int j_start, unsigned int j_end) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + i_start;
    int j = blockIdx.x * blockDim.x + threadIdx.x + j_start;
    
    if (i <= i_end && j <= j_end)
    {
        t_next[i*n_cols+j] = (1.0/20.0) * 
                          ( 4 * ( t[i*n_cols+(j+1)] + 
                                 t[i*n_cols+(j-1)] + 
                                 t[(i+1)*n_cols+j] + 
                                 t[(i-1)*n_cols+j] 
                               ) + 
                            t[(i+1)*n_cols+(j+1)] + 
                            t[(i+1)*n_cols+(j-1)] + 
                            t[(i-1)*n_cols+(j+1)] + 
                            t[(i-1)*n_cols+(j-1)]
                          );
    }
}

__global__ void tiled_no_halos_kernel(double* t_next, const double* t, unsigned int n_rows, unsigned int n_cols, unsigned int i_start, unsigned int i_end, unsigned int j_start, unsigned int j_end)
{
    // Dynamic shared memory allocation to match the current block size
    extern __shared__ double tile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_width = blockDim.x;
    int block_height = blockDim.y;
    
    int i = blockIdx.y * block_height + ty + i_start;
    int j = blockIdx.x * block_width + tx + j_start;
    
    int tile_width = block_width + 2;
    
    if (i <= i_end && j <= j_end)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                // Calculate global and tile indices
                int global_y = i + dy;
                int global_x = j + dx;
                
                // Conditions for loading into shared memory
                bool valid_tile_y = (ty + dy >= 0 && ty + dy < block_height) || dy == -1 || dy == 1;
                bool valid_tile_x = (tx + dx >= 0 && tx + dx < block_width) || dx == -1 || dx == 1;
                
                if (valid_tile_y && valid_tile_x)
                {
                    int tile_idx = (ty + dy + 1) * tile_width + (tx + dx + 1);
                    int global_idx = global_y * n_cols + global_x;
                    
                    tile[tile_idx] = t[global_idx];
                }
            }
        }
    }
    
    __syncthreads();
    
    if (i <= i_end && j <= j_end && i > 1 && i < n_rows-2 && j > 1 && j < n_cols-2)
    {
        t_next[i*n_cols + j] = (1.0/20.0) * 
            (4 * (tile[(ty+1)*tile_width + (tx+2)] + 
                  tile[(ty+1)*tile_width + tx] + 
                  tile[(ty+2)*tile_width + (tx+1)] + 
                  tile[ty*tile_width + (tx+1)]) +
             tile[(ty+2)*tile_width + (tx+2)] + 
             tile[(ty+2)*tile_width + tx] + 
             tile[ty*tile_width + (tx+2)] + 
             tile[ty*tile_width + tx]);
    }
}

__global__ void tiled_with_halos_kernel(double* t_next, const double* t, unsigned int n_rows, unsigned int n_cols, unsigned int i_start, unsigned int i_end, unsigned int j_start, unsigned int j_end)
{
    // Dynamic shared memory allocation to match the current block size
    extern __shared__ double tile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_width = blockDim.x;
    int block_height = blockDim.y;
    
    int i = blockIdx.y * block_height + ty + i_start;
    int j = blockIdx.x * block_width + tx + j_start;
    
    int tile_width = block_width + 2;
    
    if (i <= i_end && j <= j_end)
    {
        // Halo center element
        int center_tile_idx = (ty + 1) * tile_width + (tx + 1);
        tile[center_tile_idx] = t[i*n_cols + j];
        
        // Halo elements loading in shared memory
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                // The center element is already loaded
                if (dy == 0 && dx == 0) continue;
                
                int halo_y = ty + dy + 1;
                int halo_x = tx + dx + 1;
                int global_y = i + dy;
                int global_x = j + dx;
                
                int halo_tile_idx = halo_y * tile_width + halo_x;
                tile[halo_tile_idx] = t[global_y*n_cols + global_x];
            }
        }
    }
    
    __syncthreads();
    
    // Compute only for interior points of the domain
    if ((i <= i_end && j <= j_end) && i > 1 && i < n_rows-2 && j > 1 && j < n_cols-2)
    {
        t_next[i*n_cols + j] = (1.0/20.0) * 
            (4 * (tile[(ty+1)*tile_width + (tx+2)] +  
                  tile[(ty+1)*tile_width + tx] +    
                  tile[(ty+2)*tile_width + (tx+1)] +   
                  tile[(ty)*tile_width + (tx+1)]) +   
             tile[(ty+2)*tile_width + (tx+2)] +       
             tile[(ty+2)*tile_width + tx] +         
             tile[(ty)*tile_width + (tx+2)] +         
             tile[(ty)*tile_width + tx]             
            );
    }
}

__global__ void tiled_with_halos_larger_block_kernel(double* t_next, const double* t, unsigned int n_rows, unsigned int n_cols, unsigned int i_start, unsigned int i_end, unsigned int j_start, unsigned int j_end)
{
    // Dynamic shared memory allocation to match the current block size
    extern __shared__ double tile[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int block_width = blockDim.x;
    int block_height = blockDim.y;
    
    int i = blockIdx.y * block_height + ty + i_start;
    int j = blockIdx.x * block_width + tx + j_start;
    
    int tile_width = (block_width * 2) + 2;
    
    if (i <= i_end && j <= j_end)
    {
        // Halo center element
        int center_tile_idx = (ty + 1) * tile_width + (tx + 1);
        tile[center_tile_idx] = t[i*n_cols + j];
        
        // Halo elements loading in shared memory
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                // The center element is already loaded
                if (dy == 0 && dx == 0) continue;
                
                int halo_y = ty + dy + 1;
                int halo_x = tx + dx + 1;
                int global_y = i + dy;
                int global_x = j + dx;
                
                int halo_tile_idx = halo_y * tile_width + halo_x;
                tile[halo_tile_idx] = t[global_y*n_cols + global_x];
            }
        }
    }
    
    __syncthreads();
    
    if ((i <= i_end && j <= j_end) && i > 1 && i < n_rows-2 && j > 1 && j < n_cols-2)
    {
        t_next[i*n_cols + j] = (1.0/20.0) * 
            (4 * (tile[(ty+1)*tile_width + (tx+2)] +  
                  tile[(ty+1)*tile_width + tx] +    
                  tile[(ty+2)*tile_width + (tx+1)] +   
                  tile[(ty)*tile_width + (tx+1)]) +   
             tile[(ty+2)*tile_width + (tx+2)] +       
             tile[(ty+2)*tile_width + tx] +         
             tile[(ty)*tile_width + (tx+2)] +         
             tile[(ty)*tile_width + tx]             
            );
    }
}