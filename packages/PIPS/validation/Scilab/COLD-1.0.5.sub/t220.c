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

  /*  t220.sce - from mcgill/capr.sce */
  /* This function computes the capacitance */
  /* per unit length of a coaxial pair of rectangles. */
  /* tic(); */
  int _u_scale = 1;
  double _u_a = (0.3257463*2);
  double _u_b = (8.65*0.04039);
  double _u_c = (3.29*0.55982);
  double _u_d = (0.727561*6.171);
  int _u_n = scilab_rt_floor_d0_((56.0980*0.36));
  _u_n = 20;
  double _u_tol = 1.3E-13;
  double _u_rel = 0.90;
  double _u_cap = 0.0;
  for (int _u_time=1; _u_time<=(_u_scale*10); _u_time++) {
    double _tmpxx0 = (0.5*_u_c);
    double _u_h = (_tmpxx0 / _u_n);
    double _tmpxx1 = (0.5*_u_a);
    int _u_na = scilab_rt_round_d0_((_tmpxx1 / _u_h));
    double _u_x[1][21];
    scilab_rt_linspace_d0d0i0_d2(0,(0.5*_u_c),(_u_n+1),1,21,_u_x);
    double _tmpxx2 = (0.5*_u_d);
    int _u_m = scilab_rt_round_d0_((_tmpxx2 / _u_h));
    _u_m = 49;
    double _tmpxx3 = (0.5*_u_b);
    int _u_mb = scilab_rt_round_d0_((_tmpxx3 / _u_h));
    double _u_y[1][50];
    scilab_rt_linspace_d0d0i0_d2(0,(0.5*_u_d),(_u_m+1),1,50,_u_y);
    double _u_f[21][50];
    scilab_rt_zeros_i0i0_d2((_u_n+1),(_u_m+1),21,50,_u_f);
    double _tmpxx4[21][50];
    scilab_rt_ones_i0i0_d2((_u_n+1),(_u_m+1),21,50,_tmpxx4);
    double _u_mask[21][50];
    scilab_rt_mul_d2d0_d2(21,50,_tmpxx4,_u_rel,21,50,_u_mask);
    for (int _u_ii=1; _u_ii<=(_u_na+1); _u_ii++) {
      for (int _u_jj=1; _u_jj<=(_u_mb+1); _u_jj++) {
        _u_mask[_u_ii-1][_u_jj-1] = 0;
        _u_f[_u_ii-1][_u_jj-1] = 1;
      }
    }
    double _u_oldcap = 0;
    for (int _u_iter=1; _u_iter<=1000; _u_iter++) {
      /* f = seidel(f, mask, n, m, na, mb); */
      /* /   Function seidel     ////////////// */
      for (int _u_ii=2; _u_ii<=_u_n; _u_ii++) {
        for (int _u_jj=2; _u_jj<=_u_m; _u_jj++) {
          double _tmpxx5 = (_u_f[_u_ii-1][_u_jj-1]+(_u_mask[_u_ii-1][_u_jj-1]*((0.25*(((_u_f[(_u_ii-1)-1][_u_jj-1]+_u_f[(_u_ii+1)-1][_u_jj-1])+_u_f[_u_ii-1][(_u_jj-1)-1])+_u_f[_u_ii-1][(_u_jj+1)-1]))-_u_f[_u_ii-1][_u_jj-1])));
          _u_f[_u_ii-1][_u_jj-1] = _tmpxx5;
        }
      }
      int _u_ii = 1;
      for (int _u_jj=2; _u_jj<=_u_m; _u_jj++) {
        double _tmpxx6 = (_u_f[_u_ii-1][_u_jj-1]+(_u_mask[_u_ii-1][_u_jj-1]*((0.25*(((_u_f[(_u_ii+1)-1][_u_jj-1]+_u_f[(_u_ii+1)-1][_u_jj-1])+_u_f[_u_ii-1][(_u_jj-1)-1])+_u_f[_u_ii-1][(_u_jj+1)-1]))-_u_f[_u_ii-1][_u_jj-1])));
        _u_f[_u_ii-1][_u_jj-1] = _tmpxx6;
      }
      int _u_jj = 1;
      for (_u_ii=2; _u_ii<=_u_n; _u_ii++) {
        double _tmpxx7 = (_u_f[_u_ii-1][_u_jj-1]+(_u_mask[_u_ii-1][_u_jj-1]*((0.25*(((_u_f[(_u_ii-1)-1][_u_jj-1]+_u_f[(_u_ii+1)-1][_u_jj-1])+_u_f[_u_ii-1][(_u_jj+1)-1])+_u_f[_u_ii-1][(_u_jj+1)-1]))-_u_f[_u_ii-1][_u_jj-1])));
        _u_f[_u_ii-1][_u_jj-1] = _tmpxx7;
      }
      /* ////////////////////////////////////// */
      /*     cap = gauss(n, m, h, f); */
      /* ////Function Gauss   ///////////////// */
      double _u_q = 0;
      for (_u_ii=1; _u_ii<=_u_n; _u_ii++) {
        double _tmpxx8 = _u_f[_u_ii-1][_u_m-1];
        double _tmpxx9 = _u_f[(_u_ii+1)-1][_u_m-1];
        double _tmpxx10 = (_tmpxx8+_tmpxx9);
        double _tmpxx11 = (_tmpxx10*0.5);
        _u_q = (_u_q+_tmpxx11);
      }
      for (_u_jj=1; _u_jj<=_u_m; _u_jj++) {
        double _tmpxx12 = _u_f[_u_n-1][_u_jj-1];
        double _tmpxx13 = _u_f[_u_n-1][(_u_jj+1)-1];
        double _tmpxx14 = (_tmpxx12+_tmpxx13);
        double _tmpxx15 = (_tmpxx14*0.5);
        _u_q = (_u_q+_tmpxx15);
      }
      _u_cap = (_u_q*4);
      _u_cap = (_u_cap*8.854187);
      /* ///////////////////////////////////// */
      double _tmpxx16 = scilab_rt_abs_d0_((_u_cap-_u_oldcap));
      double _tmpxx17 = (_tmpxx16 / _u_cap);
      if ((_tmpxx17<_u_tol)) {
        break;
      } else { 
        _u_oldcap = _u_cap;
      }
    }
  }
  /* elapsedTime= toc(); */
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Value of cap: ");
  scilab_rt_disp_d0_(_u_cap);

  scilab_rt_terminate();
}

