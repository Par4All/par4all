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

void foo_i0_(int _u_t)
{
  scilab_rt_disp_i0_(_u_t);
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t200.sce: parser */
  int _u_t = 1;
  scilab_rt_display_s0i0_("t",_u_t);
  foo_i0_(_u_t);

  scilab_rt_terminate();
}

