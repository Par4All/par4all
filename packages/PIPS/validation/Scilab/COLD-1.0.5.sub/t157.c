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

double foo_d0_(double _u_a)
{
  double _u_x = 0;
  _u_x = _u_a;
  return _u_x;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t157.sce: user function */
  double _tmp0 = foo_d0_(1.);
  scilab_rt_display_s0d0_("ans",_tmp0);

  scilab_rt_terminate();
}

