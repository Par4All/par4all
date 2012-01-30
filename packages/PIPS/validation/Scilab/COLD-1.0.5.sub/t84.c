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

  /*  t84 - test function which returns a value that is not assigned anywhere */
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
  int _tmpc0;
  scilab_rt_length_i2_i0(3,3,_u_a,&_tmpc0);
  int _u_c;
  scilab_rt_length_i2_i0(3,3,_u_a,&_u_c);
  scilab_rt_display_s0i0_("c",_u_c);

  scilab_rt_terminate();
}

