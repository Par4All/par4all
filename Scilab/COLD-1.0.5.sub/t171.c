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

void palloc_i0_i0i0(int _u_n, int* _u_a, int* _u_b)
{
  *_u_a = (2*_u_n);
  *_u_b = (3*_u_n);
}


int foo_i0_(int _u_n)
{
  _u_n = 10;
  int _u_t1;
  int _u_t2;
  palloc_i0_i0i0(_u_n,&_u_t1,&_u_t2);
  int _u_res = (_u_t1+_u_t2);
  return _u_res;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t171.sce: user function */
  int _tmp0 = foo_i0_(10);
  scilab_rt_display_s0i0_("ans",_tmp0);

  scilab_rt_terminate();
}

