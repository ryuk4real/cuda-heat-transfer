#pragma once

void update_temperature_kernel(double* t_next, double* t, unsigned int i, unsigned int j, unsigned int n_cols)
{
    t_next[i*n_cols+j] = (1.0/20.0) * \
                          ( 4 * ( t[i*n_cols+(j+1)] + \
                                  t[i*n_cols+(j-1)] + \
                                  t[(i+1)*n_cols+j] + \
                                  t[(i-1)*n_cols+j] \
                                ) + \
                            t[(i+1)*n_cols+(j+1)] + \
                            t[(i+1)*n_cols+(j-1)] + \
                            t[(i-1)*n_cols+(j+1)] + \
                            t[(i-1)*n_cols+(j-1)]
                          );
}

#include <cuda_runtime.h>

__global__ void straightforward_unified_kernel(double* t_next, double* t, unsigned int n_rows, unsigned int n_cols, unsigned int i_start, unsigned int i_end, unsigned int j_start, unsigned int j_end) 
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + i_start;
    int j = blockIdx.x * blockDim.x + threadIdx.x + j_start;
    
    if (i <= i_end && j <= j_end) {
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