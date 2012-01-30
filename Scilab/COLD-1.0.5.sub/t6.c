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

  int _tmpxx0 = (2*3);
  int _u_a = (_tmpxx0+4);
  scilab_rt_display_s0i0_("a",_u_a);
  /*  a=2*3+4 */
  int _tmpxx1 = (3*4);
  int _u_b = (2+_tmpxx1);
  scilab_rt_display_s0i0_("b",_u_b);
  /*  b=2+3*4 */
  int _tmpxx2 = pow(3,2);
  int _tmpxx3 = (2*_tmpxx2);
  int _u_c = (_tmpxx3+1);
  scilab_rt_display_s0i0_("c",_u_c);
  /*  c=2*3^2+1 */
  int _tmpxx4 = (1+2);
  int _tmpxx5 = (3+4);
  int _u_d = (_tmpxx4*_tmpxx5);
  scilab_rt_display_s0i0_("d",_u_d);
  /*  d=(1+2)*(3+4) */

  scilab_rt_terminate();
}

