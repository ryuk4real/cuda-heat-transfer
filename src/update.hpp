#pragma once

#include "kernel.cuh"

void update_region(double* next, double* curr, 
                  unsigned int n_rows, 
                  unsigned int n_cols, 
                  unsigned int i_start, 
                  unsigned int i_end,
                  unsigned int j_start, 
                  unsigned int j_end)
{
    for (unsigned int i = i_start; i <= i_end; i++)
            for (unsigned int j = j_start; j <= j_end; j++)
                update_temperature_kernel(next, curr, i, j, n_cols);
}

void swap_buffer_ptrs(double*& next, double*& curr)
{
    double* temp_ptr = next;
    next = curr;
    curr = temp_ptr;
}