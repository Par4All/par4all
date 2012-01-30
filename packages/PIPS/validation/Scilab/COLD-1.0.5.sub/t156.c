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

  /*  t156.sce: power function */
  int _u_a[1][6];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[0][3]=4;
  _u_a[0][4]=5;
  _u_a[0][5]=6;
  int _u_apow2[1][6];
  scilab_rt_pow_i2i0_i2(1,6,_u_a,2,1,6,_u_apow2);
  scilab_rt_display_s0i2_("apow2",1,6,_u_apow2);
  int _u_b[6][1];
  _u_b[0][0]=1;
  _u_b[1][0]=2;
  _u_b[2][0]=3;
  _u_b[3][0]=4;
  _u_b[4][0]=5;
  _u_b[5][0]=6;
  int _u_bpow3[1][6];
  scilab_rt_pow_i2i0_i2(1,6,_u_a,3,1,6,_u_bpow3);
  scilab_rt_display_s0i2_("bpow3",1,6,_u_bpow3);

  scilab_rt_terminate();
}

