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

  /*  t242.sce - from mcgill/trid.sce */
  /*  trid.sce */
  /* tic(); */
  int _u_n = 1000;
  double _u_a[1][1000];
  scilab_rt_rand_i0i0_d2(1,_u_n,1,1000,_u_a);
  double _u_b[1][1000];
  scilab_rt_rand_i0i0_d2(1,_u_n,1,1000,_u_b);
  double _u_c[1][1000];
  scilab_rt_rand_i0i0_d2(1,_u_n,1,1000,_u_c);
  double _u_d[1][1000];
  scilab_rt_rand_i0i0_d2(1,_u_n,1,1000,_u_d);
  double _u_x[1][1000];
  scilab_rt_assign_d2_d2(1,1000,_u_d,1,1000,_u_x);
  int _u_l;
  scilab_rt_length_d2_i0(1,1000,_u_x,&_u_l);
  for (int _u_j=1; _u_j<=(_u_l-1); _u_j++) {
    double _tmpxx0 = _u_a[0][(_u_j-1)];
    double _tmpxx1 = _u_b[0][(_u_j-1)];
    double _u_mu = (_tmpxx0 / _tmpxx1);
    double _tmpxx2 = (_u_b[0][((_u_j+1)-1)]-(_u_mu*_u_c[0][(_u_j-1)]));
    _u_b[0][((_u_j+1)-1)] = _tmpxx2;
    double _tmpxx3 = (_u_x[0][((_u_j+1)-1)]-(_u_mu*_u_x[0][(_u_j-1)]));
    _u_x[0][((_u_j+1)-1)] = _tmpxx3;
  }
  double _tmpxx4 = (_u_x[0][(_u_l-1)] / _u_b[0][(_u_l-1)]);
  _u_x[0][(_u_l-1)] = _tmpxx4;
  for (int _u_j=(_u_l-1); _u_j>=1; _u_j+=(-1)) {
    double _tmpxx5 = ((_u_x[0][(_u_j-1)]-(_u_c[0][(_u_j-1)]*_u_x[0][((_u_j+1)-1)])) / _u_b[0][(_u_j-1)]);
    _u_x[0][(_u_j-1)] = _tmpxx5;
  }
  /* elapsedTime = toc(); */
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Mean value of matrix X: ");
  double _tmpxx6;
  scilab_rt_mean_d2_d0(1,1000,_u_x,&_tmpxx6);
  scilab_rt_disp_d0_(_tmpxx6);

  scilab_rt_terminate();
}

