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

  /*  t174.sce: test stacksize function */
  int _u_a = 1;
  scilab_rt_display_s0i0_("a",_u_a);
  scilab_rt_stacksize_s0_("max");
  int _u_b = 2;
  scilab_rt_display_s0i0_("b",_u_b);
  scilab_rt_stacksize_d0_(1000);
  int _u_c = 3;
  scilab_rt_display_s0i0_("c",_u_c);

  scilab_rt_terminate();
}

