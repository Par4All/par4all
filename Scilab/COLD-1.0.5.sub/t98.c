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

void foo_i0i0_i0i0(int _u_u, int _u_v, int* _u_x, int* _u_y)
{
  *_u_x = (_u_u+_u_v);
  *_u_y = (_u_u-_u_v);
}


void foo_d0d0_d0d0(double _u_u, double _u_v, double* _u_x, double* _u_y)
{
  *_u_x = (_u_u+_u_v);
  *_u_y = (_u_u-_u_v);
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t98.sce - user function returning tuple */
  int _u_a;
  int _u_b;
  foo_i0i0_i0i0(1,2,&_u_a,&_u_b);
  scilab_rt_display_s0i0_("b",_u_b);
  scilab_rt_display_s0i0_("a",_u_a);
  double _u_r;
  double _u_s;
  foo_d0d0_d0d0(1.,2.,&_u_r,&_u_s);
  scilab_rt_display_s0d0_("s",_u_s);
  scilab_rt_display_s0d0_("r",_u_r);

  scilab_rt_terminate();
}

