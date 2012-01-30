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

  /*  t270.sce _ triplet notation */
  int _u_x = 1;
  scilab_rt_display_s0i0_("x",_u_x);
  int _u_y = 10;
  scilab_rt_display_s0i0_("y",_u_y);
  int _tmpxx0[1][2];
  for(int __tri0=0;__tri0 < 2;__tri0++) {
    _tmpxx0[0][__tri0] = 1+__tri0*9;
  }
  int _u_X[1][2];
  
  for(int j=0; j<2; ++j) {
    _u_X[0][j] = _tmpxx0[0][j];
  }
  scilab_rt_display_s0i2_("X",1,2,_u_X);

  scilab_rt_terminate();
}

