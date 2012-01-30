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

  int _tmpxx0[1][2];
  for(int __tri0=0;__tri0 < 2;__tri0++) {
    _tmpxx0[0][__tri0] = 5+__tri0*1;
  }
  int _u_a[1][3];
  
  for(int j=0; j<2; ++j) {
    _u_a[0][j] = _tmpxx0[0][j];
  }
  _u_a[0][2]=10;
  scilab_rt_display_s0i2_("a",1,3,_u_a);
  int _tmpxx1[1][6];
  for(int __tri1=0;__tri1 < 6;__tri1++) {
    _tmpxx1[0][__tri1] = 5+__tri1*1;
  }
  int _tmpxx2[1][0];
  int _u_b[1][8];
  _u_b[0][0]=0;
  
  for(int j=0; j<6; ++j) {
    _u_b[0][j+1] = _tmpxx1[0][j];
  }
  
  for(int j=0; j<0; ++j) {
    _u_b[0][j+7] = _tmpxx2[0][j];
  }
  _u_b[0][7]=25;
  scilab_rt_display_s0i2_("b",1,8,_u_b);
  int _tmpxx3[1][0];
  int _u_c[1][0];
  
  for(int j=0; j<0; ++j) {
    _u_c[0][j] = _tmpxx3[0][j];
  }
  scilab_rt_display_s0i2_("c",1,0,_u_c);
  int _tmpxx4[1][0];
  int _u_d[1][2];
  _u_d[0][0]=1;
  _u_d[0][1]=2;
  
  for(int j=0; j<0; ++j) {
    _u_d[0][j+2] = _tmpxx4[0][j];
  }
  scilab_rt_display_s0i2_("d",1,2,_u_d);

  scilab_rt_terminate();
}

