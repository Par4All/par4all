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

void foo__()
{
  scilab_rt_disp_s0_("foo");
}


void quux__i0i0(int* _u_u, int* _u_v)
{
  foo__();
  *_u_u = 1;
  *_u_v = 1;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t272.sce _ fixed PR-161 */
  int _u_a;
  int _u_b;
  quux__i0i0(&_u_a,&_u_b);
  scilab_rt_display_s0i0_("b",_u_b);
  scilab_rt_display_s0i0_("a",_u_a);

  scilab_rt_terminate();
}

