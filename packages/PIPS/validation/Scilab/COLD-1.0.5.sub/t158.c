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

double f_d0_(double _u_x)
{
  double _u_y = (2*_u_x);
  return _u_y;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t158.sce: user function */
  double _u_x0 = 0;
  scilab_rt_display_s0d0_("x0",_u_x0);
  int _u_x1 = 5;
  scilab_rt_display_s0i0_("x1",_u_x1);
  double _tmpxx0 = (_u_x0+_u_x1);
  double _u_x_middle = (_tmpxx0 / 2);
  scilab_rt_display_s0d0_("x_middle",_u_x_middle);
  double _tmpxx1 = f_d0_(_u_x_middle);
  if ((_tmpxx1<=0)) {
    _u_x0 = _u_x_middle;
  }
  scilab_rt_display_s0d0_("x0",_u_x0);

  scilab_rt_terminate();
}

