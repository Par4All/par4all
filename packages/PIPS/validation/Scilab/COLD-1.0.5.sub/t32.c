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

  int _u_ly = 10;
  int _tmpxx0[1][8];
  for(int __tri0=0;__tri0 < 8;__tri0++) {
    _tmpxx0[0][__tri0] = 2+__tri0*1;
  }
  int _u_a[1][8];
  
  for(int j=0; j<8; ++j) {
    _u_a[0][j] = _tmpxx0[0][j];
  }
  scilab_rt_display_s0i2_("a",1,8,_u_a);

  scilab_rt_terminate();
}

