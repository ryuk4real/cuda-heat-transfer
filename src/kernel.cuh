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
    __shared__ double tile[TILE_WIDTH+2][TILE_WIDTH+2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * blockDim.y + ty + i_start;
    int j = blockIdx.x * blockDim.x + tx + j_start;
    
    // The tile is loaded into shared memory
    if (i <= i_end && j <= j_end)
    {
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                if ((ty + dy >= 0 && ty + dy < TILE_WIDTH) || dy == -1 || dy == 1)
                {
                    if ((tx + dx >= 0 && tx + dx < TILE_WIDTH) || dx == -1 || dx == 1)
                    {
                        tile[ty+dy+1][tx+dx+1] = t[(i+dy)*n_cols + (j+dx)];
                    }
                }
            }
        }
    }
    
    __syncthreads();
    
    // Skip first two rows and last two rows
    if (i <= i_end && j <= j_end && i > 1 && i < n_rows-2)
    {
        t_next[i*n_cols + j] = (1.0/20.0) * 
            (4 * (tile[ty+1][tx+2] + 
                  tile[ty+1][tx] + 
                  tile[ty+2][tx+1] + 
                  tile[ty][tx+1]) +
             tile[ty+2][tx+2] + 
             tile[ty+2][tx] + 
             tile[ty][tx+2] + 
             tile[ty][tx]);
    }
}

__global__ void tiled_with_halos_kernel(double* t_next, const double* t, unsigned int n_rows, unsigned int n_cols, unsigned int i_start, unsigned int i_end, unsigned int j_start, unsigned int j_end)
{
    __shared__ double tile[TILE_WIDTH+2][TILE_WIDTH+2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * blockDim.y + ty + i_start;
    int j = blockIdx.x * blockDim.x + tx + j_start;
    
    if (i <= i_end && j <= j_end)
    {
        // Halo center element
        tile[ty+1][tx+1] = t[i*n_cols + j];
        
        // Halo elements loading in shared memory
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                // Skip the center element because it's already loaded
                if (dy == 0 && dx == 0) continue;
                
                int halo_y = ty + dy + 1;
                int halo_x = tx + dx + 1;
                int global_y = i + dy;
                int global_x = j + dx;
                
                // Load halo element
                tile[halo_y][halo_x] = t[global_y*n_cols + global_x];
            }
        }
    }
    
    __syncthreads();
    
    if ((i <= i_end && j <= j_end) && i > 1 && i < n_rows-2 && j > 1 && j < n_cols-2)
    {
        t_next[i*n_cols + j] = (1.0/20.0) * 
            (4 * (tile[ty+1][tx+2] +  
                  tile[ty+1][tx] +    
                  tile[ty+2][tx+1] +   
                  tile[ty][tx+1]) +   
             tile[ty+2][tx+2] +       
             tile[ty+2][tx] +         
             tile[ty][tx+2] +         
             tile[ty][tx]             
            );
    }
}