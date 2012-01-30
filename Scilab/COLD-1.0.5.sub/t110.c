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

  int _tmp0 = scilab_rt_modulo_i0i0_(5,6);
  scilab_rt_display_s0i0_("ans",_tmp0);
  double _tmp1 = scilab_rt_modulo_d0i0_(5.2,6);
  scilab_rt_display_s0d0_("ans",_tmp1);

  scilab_rt_terminate();
}

