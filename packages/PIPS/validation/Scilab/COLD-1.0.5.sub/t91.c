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

  int _u_lx[1][10];
  for(int __tri0=0;__tri0 < 10;__tri0++) {
    _u_lx[0][__tri0] = 1+__tri0*1;
  }
  int _u_ly[1][10];
  for(int __tri1=0;__tri1 < 10;__tri1++) {
    _u_ly[0][__tri1] = 1+__tri1*1;
  }
  double _u_x[10][10];
  double _u_y[10][10];
  scilab_rt_meshgrid_i2i2_d2d2(1,10,_u_lx,1,10,_u_ly,10,10,_u_x,10,10,_u_y);
  scilab_rt_disp_s0_("x");
  scilab_rt_display_s0d2_("x",10,10,_u_x);
  scilab_rt_disp_s0_("y");
  scilab_rt_display_s0d2_("y",10,10,_u_y);

  scilab_rt_terminate();
}

