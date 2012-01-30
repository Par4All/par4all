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

int foo_i0i0_(int _u_a, int _u_b)
{
  int _u_res = (_u_a+_u_b);
  return _u_res;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t93.sce: testing user function */
  int _u_t1 = foo_i0i0_(1,1);
  scilab_rt_display_s0i0_("t1",_u_t1);

  scilab_rt_terminate();
}

