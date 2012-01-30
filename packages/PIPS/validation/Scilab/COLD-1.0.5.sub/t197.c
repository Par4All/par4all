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

void foo1_d2i0_d2(int _u_x_n0,int _u_x_n1,double _u_x[_u_x_n0][_u_x_n1], int _u_n, int _u_y_n0,int _u_y_n1,double _u_y[_u_y_n0][_u_y_n1])
{
  
  scilab_rt_ones_i0i0_d2(_u_n,1,_u_y_n0,_u_y_n1,_u_y);
}


void foo2_i0_(int _u_n)
{
  int _tmpxx0 = (3*_u_n);
  int _u_m = (_tmpxx0+2);
  double _u_x[_u_m][1];
  scilab_rt_zeros_i0i0_d2(_u_m,1,_u_m,1,_u_x);
  double _u_y[_u_m][1];
  foo1_d2i0_d2(_u_m,1,_u_x,_u_m,_u_m,1,_u_y);
  scilab_rt_disp_d2_(_u_m,1,_u_y);
}




/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t197.sce: dynamic array allocation */
  foo2_i0_(3);
  foo2_i0_(1);

  scilab_rt_terminate();
}

