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

  /*  t265.sce _ array declaration with calls */
  double _tmpxx0[2][2];
  scilab_rt_ones_i0i0_d2(2,2,2,2,_tmpxx0);
  double _tmpxx1[2][2];
  scilab_rt_zeros_i0i0_d2(2,2,2,2,_tmpxx1);
  double _u_a[2][4];
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_a[i][j] = _tmpxx0[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_a[i][j+2] = _tmpxx1[i][j];
    }
  }
  scilab_rt_display_s0d2_("a",2,4,_u_a);
  int _tmpxx2[2][2];
  _tmpxx2[0][0]=1;
  _tmpxx2[0][1]=2;
  _tmpxx2[1][0]=3;
  _tmpxx2[1][1]=4;
  int _tmpxx3[2][2];
  _tmpxx3[0][0]=5;
  _tmpxx3[0][1]=6;
  _tmpxx3[1][0]=7;
  _tmpxx3[1][1]=8;
  double _tmpxx4[2][2];
  scilab_rt_zeros_i0i0_d2(2,2,2,2,_tmpxx4);
  double _u_b[2][6];
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_b[i][j] = _tmpxx2[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_b[i][j+2] = _tmpxx3[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_b[i][j+4] = _tmpxx4[i][j];
    }
  }
  scilab_rt_display_s0d2_("b",2,6,_u_b);

  scilab_rt_terminate();
}

