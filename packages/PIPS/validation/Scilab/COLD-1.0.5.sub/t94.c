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


double foo_d0d0_(double _u_a, double _u_b)
{
  double _u_res = (_u_a+_u_b);
  return _u_res;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t94.sce: testing user function */
  int _u_t1 = foo_i0i0_(1,1);
  scilab_rt_display_s0i0_("t1",_u_t1);
  double _u_t2 = foo_d0d0_(1.1,1.1);
  scilab_rt_display_s0d0_("t2",_u_t2);
  int _u_t3 = foo_i0i0_(2,2);
  scilab_rt_display_s0i0_("t3",_u_t3);

  scilab_rt_terminate();
}

