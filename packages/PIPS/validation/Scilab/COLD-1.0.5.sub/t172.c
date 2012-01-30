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

void foo_i0_d2(int _u_n, int _u_x_n0,int _u_x_n1,double _u_x[_u_x_n0][_u_x_n1])
{
  
  scilab_rt_zeros_i0i0_d2(10,10,_u_x_n0,_u_x_n1,_u_x);

  for(int i=0; i<_u_n; i++) {
    for(int j=0; j<_u_n; j++) {
      _u_x[i][j] = _u_n;
    }
  }
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t172.sce: user function returning an array */
  double _tmp0[10][10];
  foo_i0_d2(2,10,10,_tmp0);
  scilab_rt_display_s0d2_("ans",10,10,_tmp0);
  double _tmp1[10][10];
  foo_i0_d2(4,10,10,_tmp1);
  scilab_rt_display_s0d2_("ans",10,10,_tmp1);

  scilab_rt_terminate();
}

