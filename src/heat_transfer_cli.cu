#include "init.hpp"
#include "io.hpp"
#include "update.hpp"
#include "util.hpp"
#include <iostream>

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
 
    initTopBottomTemperature(temperature_current, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, initial_hot_temperature);
    initTopBottomTemperature(temperature_next, grid_rows, grid_cols, n_hot_top_rows, n_hot_bottom_rows, initial_hot_temperature);

    std::cout << "Saving initial configuration... " << std::endl;
    saveTemparature(outfile_prefix, outfile_extension, step, temperature_current, grid_rows, grid_cols, field_width);
    std::cout << "Done" << std::endl;

    std::cout << "Simulation in progress... " << std::endl;
    util::Timer clTimer;
    
    for (step = 1; step <= n_steps; step++) {
        updateRegion(temperature_next, temperature_current, grid_rows, grid_cols, n_hot_top_rows, (grid_rows-1)-n_hot_bottom_rows, 1, (grid_cols-1)-1);
        swapBufferPtrs(temperature_next, temperature_current);    
    }
    elapsed_time = static_cast<double>(clTimer.getTimeMilliseconds());
    std::cout << "Simulation loop elapsed time: " << elapsed_time << " ms (corresponding to " << (elapsed_time / 1000.0) << " s)" << std::endl;

    std::cout << "Saving final configuration... " << std::endl;
    saveTemparature(outfile_prefix, outfile_extension, --step, temperature_current, grid_rows, grid_cols, field_width); 
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
