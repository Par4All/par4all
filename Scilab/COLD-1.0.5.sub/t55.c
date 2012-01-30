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

  /*  t55.sce: testing length function */
  int _u_a = scilab_rt_length_i0_(1);
  scilab_rt_display_s0i0_("a",_u_a);
  double _u_b[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_b);
  int _u_c;
  scilab_rt_length_d2_i0(10,10,_u_b,&_u_c);
  scilab_rt_display_s0i0_("c",_u_c);
  int _u_d = scilab_rt_length_s0_("foobar");
  scilab_rt_display_s0i0_("d",_u_d);
  double _tmpxx0[2][3][2];
  scilab_rt_rand_i0i0i0_d3(2,3,2,2,3,2,_tmpxx0);
  int _u_e;
  scilab_rt_length_d3_i0(2,3,2,_tmpxx0,&_u_e);
  scilab_rt_display_s0i0_("e",_u_e);

  scilab_rt_terminate();
}

