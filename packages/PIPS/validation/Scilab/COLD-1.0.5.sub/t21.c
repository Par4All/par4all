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

  /*  The ide of this template is to execute a simple operation with Scilab and to */
  /*  check the obtain result against a ptre computed one using an other tool */
  /*  (NumPy) */
  int _u_result = (-529886);
  scilab_rt_display_s0i0_("result",_u_result);
  int _tmpxx0 = (-750054);
  int _u_a = (220168+_tmpxx0);
  scilab_rt_display_s0i0_("a",_u_a);
  if ((_u_a!=_u_result)) {
    scilab_rt_disp_s0_("add_int_1.sce FAILED");
    scilab_rt_exit_i0_(1);
  }
  scilab_rt_disp_s0_("add_int_1.sce SUCCEDED");
  scilab_rt_exit_i0_(0);

  scilab_rt_terminate();
}

