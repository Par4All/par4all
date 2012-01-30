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

  /* Testing function bitand */
  int _u_a[2][2];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[1][0]=3;
  _u_a[1][1]=4;
  int _u_b[2][2];
  _u_b[0][0]=5;
  _u_b[0][1]=6;
  _u_b[1][0]=7;
  _u_b[1][1]=8;
  int _u_c = 4;
  int _u_d = 5;
  int _u_e[2][2];
  scilab_rt_bitand_i2i2_i2(2,2,_u_a,2,2,_u_b,2,2,_u_e);
  scilab_rt_display_s0i2_("e",2,2,_u_e);
  int _u_f = scilab_rt_bitand_i0i0_(_u_c,_u_d);
  scilab_rt_display_s0i0_("f",_u_f);

  scilab_rt_terminate();
}

