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

void diff__d2(int _u_mag_n0,int _u_mag_n1,double _u_mag[_u_mag_n0][_u_mag_n1])
{
  int _u_CELLS = 2;
  double _u_SLITSIZE1 = 0.00001;
  double _u_SLITSIZE2 = 0.00001;
  int _u_T1 = 1;
  int _u_T2 = 0;
  int _u_scale = 1;
  double _u_DISTANCE = 5.0;
  double _u_WAVELENGTH = 0.000000633;
  double _tmpxx1 = (2*SCILAB_PI);
  double _u_K = (_tmpxx1 / _u_WAVELENGTH);
  double _u_CELLSIZE = (_u_SLITSIZE1+_u_SLITSIZE2);
  double _u_SLITRES = (_u_WAVELENGTH / 100);
  int _tmpxx2 = (_u_CELLS*10);
  double _tmpxx3 = (_u_DISTANCE / _tmpxx2);
  double _tmpxx4[1][2];
  _tmpxx4[0][0]=_u_SLITSIZE1;
  _tmpxx4[0][1]=_u_SLITSIZE2;
  double _tmpxx5 = (_tmpxx3*_u_WAVELENGTH);
  double _tmpxx6;
  scilab_rt_mean_d2_d0(1,2,_tmpxx4,&_tmpxx6);
  double _u_SCREENRES = (_tmpxx5 / _tmpxx6);
  double _tmpxx7 = (3*_u_DISTANCE);
  double _tmpxx8[1][2];
  _tmpxx8[0][0]=_u_SLITSIZE1;
  _tmpxx8[0][1]=_u_SLITSIZE2;
  double _tmpxx9 = (_tmpxx7*_u_WAVELENGTH);
  double _tmpxx10;
  scilab_rt_mean_d2_d0(1,2,_tmpxx8,&_tmpxx10);
  double _u_SCREENLENGTH = (_tmpxx9 / _tmpxx10);
  
  scilab_rt_zeros_i0i0_d2(61,2,_u_mag_n0,_u_mag_n1,_u_mag);
  int _u_i = 1;
  for (double _u_screenpt=0; _u_screenpt<=_u_SCREENLENGTH; _u_screenpt+=_u_SCREENRES) {
    double complex _u_wavesum = 0;
    for (int _u_cellnum=0; _u_cellnum<=(_u_CELLS-1); _u_cellnum++) {
      for (double _u_sourcept=0; _u_sourcept<=_u_SLITSIZE1; _u_sourcept+=_u_SLITRES) {
        double _tmpxx11 = (_u_cellnum*_u_CELLSIZE);
        double _tmpxx12 = (_tmpxx11+_u_sourcept);
        double _u_horizpos = (_u_screenpt-_tmpxx12);
        double complex _tmpxx13 = (I*_u_DISTANCE);
        double _u_x = scilab_rt_abs_z0_((_u_horizpos+_tmpxx13));
        double complex _tmpxx14 = (I*_u_K);
        double complex _tmpxx15 = scilab_rt_exp_z0_((_tmpxx14*_u_x));
        double complex _tmpxx16 = (_u_T1*_tmpxx15);
        _u_wavesum = (_u_wavesum+_tmpxx16);
        /*  	    tmp = %e*cos(K*x) + %e*sin(K*x)*%i; //e raised to a complex number */
        /*  	    wavesum = wavesum+T1 * tmp */
      }
      for (double _u_sourcept=0; _u_sourcept<=_u_SLITSIZE2; _u_sourcept+=_u_SLITRES) {
        double _tmpxx17 = (_u_cellnum*_u_CELLSIZE);
        double _tmpxx18 = (_tmpxx17+_u_SLITSIZE1);
        double _tmpxx19 = (_tmpxx18+_u_sourcept);
        double _u_horizpos = (_u_screenpt-_tmpxx19);
        double complex _tmpxx20 = (1*I);
        double complex _tmpxx21 = (_tmpxx20*_u_DISTANCE);
        double _u_x = scilab_rt_abs_z0_((_u_horizpos+_tmpxx21));
        double complex _tmpxx22 = (I*_u_K);
        double complex _tmpxx23 = scilab_rt_exp_z0_((_tmpxx22*_u_x));
        double complex _tmpxx24 = (_u_T2*_tmpxx23);
        _u_wavesum = (_u_wavesum+_tmpxx24);
        /*  	    tmp2 = %e*cos(K*x) + %e*sin(K*x)*%i; //e raised to a complex number */
        /*  	    wavesum = wavesum+T2 * tmp2; */
      }
    }
    double _tmpxx25 = scilab_rt_abs_z0_(_u_wavesum);
    double _u_intensity = pow(_tmpxx25,2);
    /* newdata = [screenpt*100, intensity/(CELLS*CELLSIZE/SLITRES)] */
    /*  mag = [mag; newdata]; */
    double _tmpxx26 = (_u_screenpt*100);
    _u_mag[_u_i-1][0] = _tmpxx26;
    double _tmpxx27 = (_u_intensity / ((_u_CELLS*_u_CELLSIZE) / _u_SLITRES));
    _u_mag[_u_i-1][1] = _tmpxx27;
    _u_i = (_u_i+1);
  }
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t229.sce - from mcgill/diff_function.sce */
  scilab_rt_tic__();
  double _u_mag[61][2];
  diff__d2(61,2,_u_mag);
  /* disp("Elapsed Time:"); */
  /* disp(toc()); */
  scilab_rt_disp_s0_("Mean of matrix mag: ");
  double _tmpxx0;
  scilab_rt_mean_d2_d0(61,2,_u_mag,&_tmpxx0);
  scilab_rt_disp_d0_(_tmpxx0);

  scilab_rt_terminate();
}

