/**
 * LICENSE TERMS

Copyright (c)2008-2010 University of Virginia
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted without royalty fees or other restrictions, provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    * Neither the name of the University of Virginia, the Dept. of Computer Science, nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY OF VIRGINIA OR THE SOFTWARE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

If you use this software or a modified version of it, please cite the most relevant among the following papers:

1) S. Che, M. Boyer, J. Meng, D. Tarjan, J. W. Sheaffer, Sang-Ha Lee and K. Skadron.
"Rodinia: A Benchmark Suite for Heterogeneous Computing". IEEE International Symposium
on Workload Characterization, Oct 2009.

2) J. Meng and K. Skadron. "Performance Modeling and Automatic Ghost Zone Optimization
for Iterative Stencil Loops on GPUs." In Proceedings of the 23rd Annual ACM International
Conference on Supercomputing (ICS), June 2009.

3) L.G. Szafaryn, K. Skadron and J. Saucerman. "Experiences Accelerating MATLAB Systems
Biology Applications." in Workshop on Biomedicine in Computing (BiC) at the International
Symposium on Computer Architecture (ISCA), June 2009.

4) M. Boyer, D. Tarjan, S. T. Acton, and K. Skadron. "Accelerating Leukocyte Tracking using CUDA:
A Case Study in Leveraging Manycore Coprocessors." In Proceedings of the International Parallel
and Distributed Processing Symposium (IPDPS), May 2009.

5) S. Che, M. Boyer, J. Meng, D. Tarjan, J. W. Sheaffer, and K. Skadron. "A Performance
Study of General Purpose Applications on Graphics Processors using CUDA" Journal of
Parallel and Distributed Computing, Elsevier, June 2008.

6) S. Che, J. Li, J. W. Sheaffer, K. Skadron, and J. Lach. "Accelerating Compute
Intensive Applications with GPUs and FPGAs" In Proceedings of the IEEE Symposium
on Application Specific Processors (SASP), June 2008.
 *
 */


/**
 * This file was converted into C99 form by Mehdi Amini
 * 05 june 2011
 */


#include <timing.h>


#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <sys/time.h>
#define STR_SIZE	256

/* maximum power density possible (say 300W for a 10mm x 10mm chip)	*/
#define MAX_PD	(3.0e6)
/* required precision in degrees	*/
#define PRECISION	0.001
#define SPEC_HEAT_SI 1.75e6
#define K_SI 100
/* capacitance fitting factor	*/
#define FACTOR_CHIP	0.5
//#define OPEN
//#define NUM_THREAD 4

/* chip parameters	*/
double t_chip = 0.0005;
double chip_height = 0.016;
double chip_width = 0.016;
/* ambient temperature, assuming no package at all	*/
double amb_temp = 80.0;

int num_omp_threads;


/* Single iteration of the transient solver in the grid model.
 * advances the solution of the discretized difference equations 
 * by one time step
 */
void single_iteration(int row,
                      int col,
                      double result[row][col],
                      double temp[row][col],
                      double power[row][col],
                      double Cap,
                      double Rx,
                      double Ry,
                      double Rz,
                      double step) {
  double delta;
  int r, c;
  //printf("num_omp_threads: %d\n", num_omp_threads);
#ifdef OPEN
  omp_set_num_threads(num_omp_threads);
#pragma omp parallel for shared(power, temp,result) private(r, c, delta) firstprivate(row, col) schedule(static)
#endif

  for (r = 0; r < row; r++) {
    for (c = 0; c < col; c++) {
      /*	Corner 1	*/
      if((r == 0) && (c == 0)) {
        delta = (step / Cap) * (power[0][0] + (temp[0][1] - temp[0][0]) / Rx
            + (temp[0][col] - temp[0][0]) / Ry + (amb_temp - temp[0][0]) / Rz);
      } /*	Corner 2	*/
      else if((r == 0) && (c == col - 1)) {
        delta = (step / Cap) * (power[0][c] + (temp[0][c - 1] - temp[0][c])
            / Rx + (temp[1][c] - temp[0][c]) / Ry + (amb_temp - temp[0][c])
            / Rz);
      } /*	Corner 3	*/
      else if((r == row - 1) && (c == col - 1)) {
        delta = (step / Cap) * (power[r][c] + (temp[r][c - 1] - temp[r][c])
            / Rx + (temp[r - 1][c] - temp[r][c]) / Ry + (amb_temp - temp[r][c])
            / Rz);
      } /*	Corner 4	*/
      else if((r == row - 1) && (c == 0)) {
        delta = (step / Cap)
            * (power[r][0] + (temp[r][1] - temp[r][0]) / Rx + (temp[r - 1][0]
                - temp[r][0]) / Ry + (amb_temp - temp[r][0]) / Rz);
      } /*	Edge 1	*/
      else if(r == 0) {
        delta = (step / Cap) * (power[0][c] + (temp[0][c + 1] + temp[0][c - 1]
            - 2.0 * temp[0][c]) / Rx + (temp[1][c] - temp[0][c]) / Ry
            + (amb_temp - temp[0][c]) / Rz);
      } /*	Edge 2	*/
      else if(c == col - 1) {
        delta = (step / Cap) * (power[r][c] + (temp[r + 1][c] + temp[r - 1][c]
            - 2.0 * temp[r][c]) / Ry + (temp[r][c - 1] - temp[r][c]) / Rx
            + (amb_temp - temp[r][c]) / Rz);
      } /*	Edge 3	*/
      else if(r == row - 1) {
        delta = (step / Cap) * (power[r][c] + (temp[r][c + 1] + temp[r][c - 1]
            - 2.0 * temp[r][c]) / Rx + (temp[r - 1][c] - temp[r][c])
            / Ry + (amb_temp - temp[r][c]) / Rz);
      } /*	Edge 4	*/
      else if(c == 0) {
        delta = (step / Cap) * (power[r][0] + (temp[r+1][0] + temp[r-1][0] - 2.0 * temp[r][0]) / Ry + (temp[r+1][0]
            - temp[r][0]) / Rx + (amb_temp - temp[r][0]) / Rz);
      } /*	Inside the chip	*/
      else {
        delta = (step / Cap) * (power[r][c] + (temp[r + 1][c]
            + temp[r-1][c] - 2.0 * temp[r][c]) / Ry + (temp[r][c + 1] + temp[r][c - 1] - 2.0 * temp[r][c])
            / Rx + (amb_temp - temp[r][c]) / Rz);
      }

      /*	Update Temperatures	*/
      result[r][c] = temp[r][c] + delta;

    }
  }

#ifdef OPEN
  omp_set_num_threads(num_omp_threads);
#pragma omp parallel for shared(result, temp) private(r, c) schedule(static)
#endif
  for (r = 0; r < row; r++) {
    for (c = 0; c < col; c++) {
      temp[r][c] = result[r][c];
    }
  }
}

/* Transient solver driver routine: simply converts the heat 
 * transfer differential equations to difference equations 
 * and solves the difference equations by iterating
 */
void compute_tran_temp(int row,
                       int col,
                       double result[row][col],
                       int num_iterations,
                       double temp[row][col],
                       double power[row][col]) {
#ifdef VERBOSE
  int i = 0;
#endif

  double grid_height = chip_height / row;
  double grid_width = chip_width / col;

  double Cap = FACTOR_CHIP * SPEC_HEAT_SI * t_chip * grid_width * grid_height;
  double Rx = grid_width / (2.0 * K_SI * t_chip * grid_height);
  double Ry = grid_height / (2.0 * K_SI * t_chip * grid_width);
  double Rz = t_chip / (K_SI * grid_height * grid_width);

  double max_slope = MAX_PD / (FACTOR_CHIP * t_chip * SPEC_HEAT_SI);
  double step = PRECISION / max_slope;
  double t;

#ifdef VERBOSE
  fprintf(stdout, "total iterations: %d s\tstep size: %g s\n", num_iterations, step);
  fprintf(stdout, "Rx: %g\tRy: %g\tRz: %g\tCap: %g\n", Rx, Ry, Rz, Cap);
#endif

  for (int i = 0; i < num_iterations; i++) {
#ifdef VERBOSE
    fprintf(stdout, "iteration %d\n", i++);
#endif
    single_iteration(row, col, result, temp, power, Cap, Rx, Ry, Rz, step);
  }

#ifdef VERBOSE
  fprintf(stdout, "iteration %d\n", i++);
#endif
}

void fatal(char *s) {
  fprintf(stderr, "error: %s\n", s);
  exit(1);
}

void read_input(int grid_rows,
                int grid_cols,
                double vect[grid_rows][grid_cols],
                char *file) {
  int i, j, index;
  FILE *fp;
  char str[STR_SIZE];
  double val;

  fp = fopen(file, "r");
  if(!fp)
    fatal("file could not be opened for reading");

  for (i = 0; i < grid_rows; i++) {
    for (j = 0; j < grid_cols; j++) {
      char *s = fgets(str, STR_SIZE, fp);
      if(feof(fp))
        fatal("not enough lines in file");
      if((sscanf(str, "%lf", &val) != 1))
        fatal("invalid file format");
      vect[i][j] = val;
    }
  }

  fclose(fp);
}

void usage(int argc, char **argv) {
  fprintf(stderr,
          "Usage: %s <grid_rows> <grid_cols> <sim_time> <no. of threads><temp_file> <power_file>\n",
          argv[0]);
  fprintf(stderr,
          "\t<grid_rows>  - number of rows in the grid (positive integer)\n");
  fprintf(stderr,
          "\t<grid_cols>  - number of columns in the grid (positive integer)\n");
  fprintf(stderr, "\t<sim_time>   - number of iterations\n");
  fprintf(stderr, "\t<no. of threads>   - number of threads\n");
  fprintf(stderr,
          "\t<temp_file>  - name of the file containing the initial temperature values of each cell\n");
  fprintf(stderr,
          "\t<power_file> - name of the file containing the dissipated power values of each cell\n");
  exit(1);
}

int main(int argc, char **argv) {
  int grid_rows, grid_cols, sim_time, i,j;
  //double *temp, *power, *result;
  char *tfile, *pfile;

  /* check validity of inputs	*/
  if(argc != 7)
    usage(argc, argv);
  if((grid_rows = atoi(argv[1])) <= 0 || (grid_cols = atoi(argv[2])) <= 0
      || (sim_time = atoi(argv[3])) <= 0 || (num_omp_threads = atoi(argv[4]))
      <= 0)
    usage(argc, argv);

  /* allocate memory for the temperature and power arrays	*/
  double temp[grid_rows][grid_cols];
  double power[grid_rows][grid_cols];
  double result[grid_rows][grid_cols];
  memset(temp,0,sizeof(temp));
  memset(power,0,sizeof(temp));
  memset(result,0,sizeof(temp));


  /* read initial temperatures and input power	*/
  tfile = argv[5];
  pfile = argv[6];
  read_input(grid_rows, grid_cols, temp, tfile);
  read_input(grid_rows, grid_cols, power, pfile);


  /* Start timer. */
  timer_start();

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[5]==0) {
    memset(temp,0,sizeof(temp));
    memset(power,0,sizeof(temp));
    memset(result,0,sizeof(temp));
  }

  // Main computation
  compute_tran_temp(grid_rows, grid_cols, result, sim_time, temp, power);

  /* Cheat the compiler to limit the scope of optimisation */
  if(argv[5]==0) {
    for(i=0; i < grid_rows; i++) {
      for(j=0; j < grid_cols; j++) {
        fprintf(stdout, "%d\t%g\n",(i*grid_cols)+j , temp[i][j]);
      }
    }
  }

  /* Stop and print timer. */
  timer_stop_display();

  /***        ***/
  /* output results	*/
#ifdef VERBOSE
  fprintf(stdout, "Final Temperatures:\n");
#endif

#ifdef OUTPUT
  for(i=0; i < grid_rows; i++)
    for(j=0; j < grid_cols; j++) {
      fprintf(stdout, "%d\t%g\n",(i*grid_cols)+j , temp[i][j]);
    }
#endif
  /* cleanup	*/
//  free(temp);
//  free(power);

  return 0;
}

