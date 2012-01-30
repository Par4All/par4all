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

  double complex _u_a = I;
  scilab_rt_display_s0z0_("a",_u_a);
  double complex _u_b = (1.0-I);
  scilab_rt_display_s0z0_("b",_u_b);
  double complex _u_c = (1.0+I);
  scilab_rt_display_s0z0_("c",_u_c);
  double complex _u_d = (_u_b*_u_c);
  scilab_rt_display_s0z0_("d",_u_d);
  double complex _u_e = (_u_c-_u_b);
  scilab_rt_display_s0z0_("e",_u_e);

  scilab_rt_terminate();
}

