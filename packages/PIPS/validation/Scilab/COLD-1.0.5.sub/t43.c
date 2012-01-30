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

  double _u_a = scilab_rt_exp_d0_(2.);
  double _u_b = scilab_rt_log_d0_(_u_a);
  scilab_rt_display_s0d0_("b",_u_b);

  scilab_rt_terminate();
}

