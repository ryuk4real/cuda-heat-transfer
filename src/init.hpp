#pragma once

void init_top_bottom_temperature(double* grid_temperature, unsigned int n_rows, unsigned int n_cols, unsigned int n_top_rows, unsigned int n_bottom_rows, double temperature)
{
    for (unsigned int i = 0; i < n_top_rows; i++)
        for (unsigned int j = 0; j < n_cols; j++)
            grid_temperature[i*n_cols+j] = temperature;

    for (unsigned int i = n_top_rows; i < n_rows - n_bottom_rows; i++)
        for (unsigned int j = 0; j < n_cols; j++)
            grid_temperature[i*n_cols+j] = 0;

    for (unsigned int i = n_rows-n_bottom_rows; i < n_rows; i++)
        for (unsigned int j = 0; j < n_cols; j++)
            grid_temperature[i*n_cols+j] = temperature;
}
