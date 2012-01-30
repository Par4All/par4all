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

  /*  t153.sce: multiplication */
  int _u_a[3][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  _u_a[2][0]=7;
  _u_a[2][1]=8;
  _u_a[2][2]=9;
  int _u_b[3][3];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  _u_b[0][2]=3;
  _u_b[1][0]=4;
  _u_b[1][1]=5;
  _u_b[1][2]=6;
  _u_b[2][0]=7;
  _u_b[2][1]=8;
  _u_b[2][2]=9;
  int _u_c[3][3];
  scilab_rt_mul_i2i2_i2(3,3,_u_a,3,3,_u_b,3,3,_u_c);
  scilab_rt_display_s0i2_("c",3,3,_u_c);
  int _u_n = 100;
  double _u_x[100][100];
  scilab_rt_ones_i0i0_d2(_u_n,_u_n,100,100,_u_x);
  double _u_y[100][100];
  scilab_rt_ones_i0i0_d2(_u_n,_u_n,100,100,_u_y);
  for (int _u_i=1; _u_i<=_u_n; _u_i++) {
    for (int _u_j=1; _u_j<=_u_n; _u_j++) {
      _u_x[_u_i-1][_u_j-1] = _u_i;
      _u_y[_u_i-1][_u_j-1] = _u_j;
    }
  }
  /* tic(); */
  double _u_z[100][100];
  scilab_rt_mul_d2d2_d2(100,100,_u_x,100,100,_u_y,100,100,_u_z);
  /* elap = toc() */
  double _tmp0 = _u_z[69][71];
  scilab_rt_display_s0d0_("ans",_tmp0);

  scilab_rt_terminate();
}

