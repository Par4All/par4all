/*
 * (c) HPC Project - 2010-2011 - All rights reserved
 *
 */

#include "scilab_rt.h"


int __lv0;
int __lv1;
int __lv2;
int __lv3;

/*----------------------------------------------------*/


/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t228.sce - from mcgill/diff.sce */
  /* This function calculates the diffraction pattern of */
  /* monochromatic light through a transmission grating for */
  /* arbitrary slit sizes and slit transmission coefficients. */
  /* tic(); */
  int _u_CELLS = 2;
  double _u_SLITSIZE1 = 0.00001;
  double _u_SLITSIZE2 = 0.00001;
  int _u_T1 = 1;
  int _u_T2 = 0;
  int _u_scale = 1;
  double _u_DISTANCE = 5.0;
  double _u_WAVELENGTH = 0.000000633;
  double _tmpxx0 = (2*SCILAB_PI);
  double _u_K = (_tmpxx0 / _u_WAVELENGTH);
  double _u_CELLSIZE = (_u_SLITSIZE1+_u_SLITSIZE2);
  double _u_SLITRES = (_u_WAVELENGTH / 100);
  int _tmpxx1 = (_u_CELLS*10);
  double _tmpxx2 = (_u_DISTANCE / _tmpxx1);
  double _tmpxx3[1][2];
  _tmpxx3[0][0]=_u_SLITSIZE1;
  _tmpxx3[0][1]=_u_SLITSIZE2;
  double _tmpxx4 = (_tmpxx2*_u_WAVELENGTH);
  double _tmpxx5;
  scilab_rt_mean_d2_d0(1,2,_tmpxx3,&_tmpxx5);
  double _u_SCREENRES = (_tmpxx4 / _tmpxx5);
  double _tmpxx6 = (3*_u_DISTANCE);
  double _tmpxx7[1][2];
  _tmpxx7[0][0]=_u_SLITSIZE1;
  _tmpxx7[0][1]=_u_SLITSIZE2;
  double _tmpxx8 = (_tmpxx6*_u_WAVELENGTH);
  double _tmpxx9;
  scilab_rt_mean_d2_d0(1,2,_tmpxx7,&_tmpxx9);
  double _u_SCREENLENGTH = (_tmpxx8 / _tmpxx9);
  double _u_mag[61][2];
  scilab_rt_zeros_i0i0_d2(61,2,61,2,_u_mag);
  int _u_i = 1;
  for (double _u_screenpt=0; _u_screenpt<=_u_SCREENLENGTH; _u_screenpt+=_u_SCREENRES) {
    double complex _u_wavesum = 0;
    for (int _u_cellnum=0; _u_cellnum<=(_u_CELLS-1); _u_cellnum++) {
      for (double _u_sourcept=0; _u_sourcept<=_u_SLITSIZE1; _u_sourcept+=_u_SLITRES) {
        double _tmpxx10 = (_u_cellnum*_u_CELLSIZE);
        double _tmpxx11 = (_tmpxx10+_u_sourcept);
        double _u_horizpos = (_u_screenpt-_tmpxx11);
        double complex _tmpxx12 = (I*_u_DISTANCE);
        double _u_x = scilab_rt_abs_z0_((_u_horizpos+_tmpxx12));
        double complex _tmpxx13 = (I*_u_K);
        double complex _tmpxx14 = scilab_rt_exp_z0_((_tmpxx13*_u_x));
        double complex _tmpxx15 = (_u_T1*_tmpxx14);
        _u_wavesum = (_u_wavesum+_tmpxx15);
      }
      for (double _u_sourcept=0; _u_sourcept<=_u_SLITSIZE2; _u_sourcept+=_u_SLITRES) {
        double _tmpxx16 = (_u_cellnum*_u_CELLSIZE);
        double _tmpxx17 = (_tmpxx16+_u_SLITSIZE1);
        double _tmpxx18 = (_tmpxx17+_u_sourcept);
        double _u_horizpos = (_u_screenpt-_tmpxx18);
        double complex _tmpxx19 = (1*I);
        double complex _tmpxx20 = (_tmpxx19*_u_DISTANCE);
        double _u_x = scilab_rt_abs_z0_((_u_horizpos+_tmpxx20));
        double complex _tmpxx21 = (I*_u_K);
        double complex _tmpxx22 = scilab_rt_exp_z0_((_tmpxx21*_u_x));
        double complex _tmpxx23 = (_u_T2*_tmpxx22);
        _u_wavesum = (_u_wavesum+_tmpxx23);
      }
    }
    double _tmpxx24 = scilab_rt_abs_z0_(_u_wavesum);
    double _u_intensity = pow(_tmpxx24,2);
    double _tmpxx25 = (_u_screenpt*100);
    _u_mag[_u_i-1][0] = _tmpxx25;
    double _tmpxx26 = (_u_intensity / ((_u_CELLS*_u_CELLSIZE) / _u_SLITRES));
    _u_mag[_u_i-1][1] = _tmpxx26;
    _u_i = (_u_i+1);
  }
  /* elapsedTime = toc(); */
  /* disp("Elapsed Time"); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Mean of matrix mag: ");
  double _tmpxx27;
  scilab_rt_mean_d2_d0(61,2,_u_mag,&_tmpxx27);
  scilab_rt_disp_d0_(_tmpxx27);

  scilab_rt_terminate();
}

