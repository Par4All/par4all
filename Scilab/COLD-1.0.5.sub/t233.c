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

void fft_scilab__d2(int _u_data_n0,int _u_data_n1,double _u_data[_u_data_n0][_u_data_n1])
{
  int _u_scale = 1;
  double _tmpxx1 = scilab_rt_log2_i0_(_u_scale);
  int _u_t0 = scilab_rt_round_d0_(_tmpxx1);
  int _tmpxx2 = (1024*1024);
  int _tmpxx3 = pow(2,_u_t0);
  int _u_n = (_tmpxx2*_tmpxx3);
  int _u_t1 = 1048576;
  
  scilab_rt_rand_i0i0_d2(1,(2*_u_t1),_u_data_n0,_u_data_n1,_u_data);
  int _tmpxx4 = (_u_n<2);
  int _tmpxx5 = scilab_rt_bitand_i0i0_(_u_n,(_u_n-1));
  if ((_tmpxx4 || _tmpxx5)) {
    scilab_rt_disp_s0_("n must be power of 2 in four1");
  } else { 
    int _u_nn = (2*_u_n);
    double _u_j = 2;
    for (int _u_i=2; _u_i<=_u_nn; _u_i+=2) {
      if ((_u_j>_u_i)) {
        double _u_t = _u_data[0][((int)(_u_j-1)-1)];
        _u_data[0][((int)(_u_j-1)-1)] = _u_data[0][((_u_i-1)-1)];
        _u_data[0][((_u_i-1)-1)] = _u_t;
        _u_t = _u_data[0][((int)_u_j-1)];
        _u_data[0][((int)_u_j-1)] = _u_data[0][(_u_i-1)];
        _u_data[0][(_u_i-1)] = _u_t;
      }
      double _u_m = _u_n;
      while ( ((_u_m>=2) && (_u_j>_u_m)) ) {
        _u_j = (_u_j-_u_m);
        _u_m = (_u_m / 2);
      }
      _u_j = (_u_j+_u_m);
    }
    int _u_mmax = 2;
    int _u_isign = 1;
    while ( (_u_nn>_u_mmax) ) {
      int _u_istep = (_u_mmax*2);
      double _tmpxx6 = (6.28318530717959 / _u_mmax);
      double _u_theta = (_u_isign*_tmpxx6);
      double _u_wtemp = scilab_rt_sin_d0_((0.5*_u_theta));
      double _tmpxx7 = (-2.0);
      double _tmpxx8 = (_tmpxx7*_u_wtemp);
      double _u_wpr = (_tmpxx8*_u_wtemp);
      double _u_wpi = scilab_rt_sin_d0_(_u_theta);
      double _u_wr = 1.0;
      double _u_wi = 0.0;
      for (double _u_m=2; _u_m<=_u_mmax; _u_m+=2) {
        for (int _u_i=_u_m; _u_i<=_u_nn; _u_i+=_u_istep) {
          _u_j = (_u_i+_u_mmax);
          double _tmpxx9 = _u_data[0][((int)(_u_j-1)-1)];
          double _tmpxx10 = _u_data[0][((int)_u_j-1)];
          double _tmpxx11 = (_u_wr*_tmpxx9);
          double _tmpxx12 = (_u_wi*_tmpxx10);
          double _u_tempr = (_tmpxx11-_tmpxx12);
          double _tmpxx13 = _u_data[0][((int)_u_j-1)];
          double _tmpxx14 = _u_data[0][((int)(_u_j-1)-1)];
          double _tmpxx15 = (_u_wr*_tmpxx13);
          double _tmpxx16 = (_u_wi*_tmpxx14);
          double _u_tempi = (_tmpxx15+_tmpxx16);
          double _tmpxx17 = (_u_data[0][((_u_i-1)-1)]-_u_tempr);
          _u_data[0][((int)(_u_j-1)-1)] = _tmpxx17;
          double _tmpxx18 = (_u_data[0][(_u_i-1)]-_u_tempi);
          _u_data[0][((int)_u_j-1)] = _tmpxx18;
          double _tmpxx19 = (_u_data[0][((_u_i-1)-1)]+_u_tempr);
          _u_data[0][((_u_i-1)-1)] = _tmpxx19;
          double _tmpxx20 = (_u_data[0][(_u_i-1)]+_u_tempi);
          _u_data[0][(_u_i-1)] = _tmpxx20;
        }
        _u_wtemp = _u_wr;
        double _tmpxx21 = (_u_wtemp*_u_wpr);
        double _tmpxx22 = (_u_wi*_u_wpi);
        double _tmpxx23 = (_tmpxx21-_tmpxx22);
        _u_wr = (_tmpxx23+_u_wr);
        double _tmpxx24 = (_u_wi*_u_wpr);
        double _tmpxx25 = (_u_wtemp*_u_wpi);
        double _tmpxx26 = (_tmpxx24+_tmpxx25);
        _u_wi = (_tmpxx26+_u_wi);
      }
      _u_mmax = _u_istep;
    }
  }
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t233.sce - from mcgill/fft_function.sce */
  scilab_rt_lines_d0_(0);
  /* ----------------------------------------------  */
  /*  - fft.sce without function */
  /*  - This function computes the forward transform */
  /* ---------------------------------------------- */
  double _u_result[1][2097152];
  fft_scilab__d2(1,2097152,_u_result);
  scilab_rt_disp_s0_("Average value of the result vector");
  double _tmpxx0;
  scilab_rt_mean_d2_d0(1,2097152,_u_result,&_tmpxx0);
  scilab_rt_disp_d0_(_tmpxx0);

  scilab_rt_terminate();
}

