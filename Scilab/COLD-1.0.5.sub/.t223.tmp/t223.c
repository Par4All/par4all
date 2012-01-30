/*
 * (c) HPC Project - 2010-2011 - All rights reserved
 *
 */

// PIPS include guard begin: #include "scilab_rt.h" 
#include "scilab_rt.h"
// PIPS include guard end: #include "scilab_rt.h" 


int __lv0;
int __lv1;
int __lv2;
int __lv3;

/*----------------------------------------------------*/

void clos_i0_d2(int _u_N, int _u_B_n0,int _u_B_n1,double _u_B[_u_B_n0][_u_B_n1])
{
  double _u_A[_u_N][_u_N];
  scilab_rt_zeros_i0i0_d2(_u_N,_u_N,_u_N,_u_N,_u_A);
  for (int _u_ii=1; _u_ii<=_u_N; _u_ii++) {
    for (int _u_jj=1; _u_jj<=_u_N; _u_jj++) {
      int _tmpxx1 = (_u_ii*_u_jj);
      double _tmpxx2 = ((double)_u_N / 2);
      if ((_tmpxx1<_tmpxx2)) {
        _u_A[(_u_N-_u_ii)-1][(_u_ii+_u_jj)-1] = 1.0;
        _u_A[_u_ii-1][((_u_N-_u_ii)-_u_jj)-1] = 1.0;
      }
      if ((_u_ii==_u_jj)) {
        _u_A[_u_ii-1][_u_jj-1] = 1.0;
      }
    }
  }
  
  scilab_rt_assign_d2_d2(_u_N,_u_N,_u_A,_u_B_n0,_u_B_n1,_u_B);
  double _u_i = ((double)_u_N / 2);
  while ( (_u_i>=1) ) {
    double _tmpxx3[_u_N][_u_N];
    scilab_rt_mul_d2d2_d2(_u_B_n0,_u_B_n1,_u_B,_u_B_n0,_u_B_n1,_u_B,_u_N,_u_N,_tmpxx3);
    
    scilab_rt_assign_d2_d2(_u_N,_u_N,_tmpxx3,_u_B_n0,_u_B_n1,_u_B);
    _u_i = (_u_i / 2);
  }
  for (int _u_m=1; _u_m<=_u_N; _u_m++) {
    for (int _u_n=1; _u_n<=_u_N; _u_n++) {
      double _tmpxx4 = _u_B[_u_m-1][_u_n-1];
      if ((_tmpxx4>0)) {
        _u_B[_u_m-1][_u_n-1] = 1;
      } else { 
        _u_B[_u_m-1][_u_n-1] = 0;
      }
    }
  }
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t223.sce - from mcgill/clos_function.sce */
  scilab_rt_tic__();
  int _u_N = 100;
  double _u_B[100][100];
  clos_i0_d2(_u_N,100,100,_u_B);
  double _u_elapsedTime = scilab_rt_toc__();
  /*  check result */
  double _u_acc = 0;
  for (int _u_m=1; _u_m<=_u_N; _u_m++) {
    for (int _u_n=1; _u_n<=_u_N; _u_n++) {
      double _tmpxx0 = _u_B[_u_m-1][_u_n-1];
      _u_acc = (_u_acc+_tmpxx0);
    }
  }
  scilab_rt_disp_s0_("Accumulated sum of all elements of the array");
  scilab_rt_disp_d0_(_u_acc);

  scilab_rt_terminate();
}
