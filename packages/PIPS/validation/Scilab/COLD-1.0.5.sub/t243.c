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

void trid__d2(int _u_x_n0,int _u_x_n1,double _u_x[_u_x_n0][_u_x_n1])
{
  int _u_n = 1000;
  double _u_a[1][1000];
  scilab_rt_rand_i0i0_d2(1,_u_n,1,1000,_u_a);
  double _u_b[1][1000];
  scilab_rt_rand_i0i0_d2(1,_u_n,1,1000,_u_b);
  double _u_c[1][1000];
  scilab_rt_rand_i0i0_d2(1,_u_n,1,1000,_u_c);
  double _u_d[1][1000];
  scilab_rt_rand_i0i0_d2(1,_u_n,1,1000,_u_d);
  
  scilab_rt_assign_d2_d2(1,1000,_u_d,_u_x_n0,_u_x_n1,_u_x);
  int _u_l;
  scilab_rt_length_d2_i0(_u_x_n0,_u_x_n1,_u_x,&_u_l);
  for (int _u_j=1; _u_j<=(_u_l-1); _u_j++) {
    double _tmpxx1 = _u_a[0][(_u_j-1)];
    double _tmpxx2 = _u_b[0][(_u_j-1)];
    double _u_mu = (_tmpxx1 / _tmpxx2);
    double _tmpxx3 = (_u_b[0][((_u_j+1)-1)]-(_u_mu*_u_c[0][(_u_j-1)]));
    _u_b[0][((_u_j+1)-1)] = _tmpxx3;
    double _tmpxx4 = (_u_x[0][((_u_j+1)-1)]-(_u_mu*_u_x[0][(_u_j-1)]));
    _u_x[0][((_u_j+1)-1)] = _tmpxx4;
  }
  double _tmpxx5 = (_u_x[0][(_u_l-1)] / _u_b[0][(_u_l-1)]);
  _u_x[0][(_u_l-1)] = _tmpxx5;
  for (int _u_j=(_u_l-1); _u_j>=1; _u_j+=(-1)) {
    double _tmpxx6 = ((_u_x[0][(_u_j-1)]-(_u_c[0][(_u_j-1)]*_u_x[0][((_u_j+1)-1)])) / _u_b[0][(_u_j-1)]);
    _u_x[0][(_u_j-1)] = _tmpxx6;
  }
  _u_x[0][(1-1)];
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t243.sce - from mcgill/trid_function.sce */
  /*  trid.sce */
  scilab_rt_tic__();
  double _u_x[1][1000];
  trid__d2(1,1000,_u_x);
  double _u_elapsedTime = scilab_rt_toc__();
  /* disp("Elapsed Time: "); */
  /* disp(elapsedTime); */
  scilab_rt_disp_s0_("Mean value of matrix x: ");
  double _tmpxx0;
  scilab_rt_mean_d2_d0(1,1000,_u_x,&_tmpxx0);
  scilab_rt_disp_d0_(_tmpxx0);

  scilab_rt_terminate();
}

