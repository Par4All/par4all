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

  /*  t207.sce: temp insertion */
  int _u_b[1][2];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  int _tmpxx0;
  scilab_rt_sum_i2_i0(1,2,_u_b,&_tmpxx0);
  _u_b[0][(1-1)] = _tmpxx0;
  scilab_rt_display_s0i2_("b",1,2,_u_b);
  int _u_a[1][4];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[0][3]=4;
  scilab_rt_display_s0i2_("a",1,4,_u_a);
  
  _u_b[0][0]=10;
  _u_b[0][1]=11;
  scilab_rt_display_s0i2_("b",1,2,_u_b);
  int _tmpxx1[1][2];
  scilab_rt_add_i2i2_i2(1,2,_u_b,1,2,_u_b,1,2,_tmpxx1);

  for(int j=0; j<2; j++) {
    _u_a[0][j] = _tmpxx1[0][j];
  }
  scilab_rt_display_s0i2_("a",1,4,_u_a);

  scilab_rt_terminate();
}

