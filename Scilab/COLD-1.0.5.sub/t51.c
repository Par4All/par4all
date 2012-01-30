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

  int _tmpxx0[1][3];
  _tmpxx0[0][0]=1;
  _tmpxx0[0][1]=2;
  _tmpxx0[0][2]=3;
  int _tmpxx1[1][3];
  _tmpxx1[0][0]=4;
  _tmpxx1[0][1]=5;
  _tmpxx1[0][2]=6;
  int _u_a[1][6];
  
  for(int j=0; j<3; ++j) {
    _u_a[0][j] = _tmpxx0[0][j];
  }
  
  for(int j=0; j<3; ++j) {
    _u_a[0][j+3] = _tmpxx1[0][j];
  }
  scilab_rt_display_s0i2_("a",1,6,_u_a);
  int _tmp0[1][2];
  scilab_rt_size_i2_i2(1,6,_u_a,1,2,_tmp0);
  scilab_rt_display_s0i2_("ans",1,2,_tmp0);

  scilab_rt_terminate();
}

