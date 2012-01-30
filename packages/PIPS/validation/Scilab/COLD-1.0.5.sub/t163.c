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

  /*  t163.sce: aref = aref and aref = adef */
  double _u_a[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_a);
  double _u_b[10][10];
  scilab_rt_zeros_i0i0_d2(10,10,10,10,_u_b);

  for(int j=0; j<10;++j) {
    _u_a[0][j] = _u_b[0][j];
  }

  for(int i=0; i<10;++i) {
    _u_a[i][0] = _u_b[i][0];
  }

  for(int i=3; i<5; i++) {
    for(int j=3; j<5; j++) {
      _u_a[i][j] = _u_b[i][j];
    }
  }
  scilab_rt_display_s0d2_("a",10,10,_u_a);
  int _tmpxx0[1][10];
  _tmpxx0[0][0]=1;
  _tmpxx0[0][1]=2;
  _tmpxx0[0][2]=3;
  _tmpxx0[0][3]=4;
  _tmpxx0[0][4]=5;
  _tmpxx0[0][5]=6;
  _tmpxx0[0][6]=7;
  _tmpxx0[0][7]=8;
  _tmpxx0[0][8]=9;
  _tmpxx0[0][9]=10;

  for(int j=0; j<10;++j) {
    _u_a[0][j] = _tmpxx0[0][j];
  }
  scilab_rt_display_s0d2_("a",10,10,_u_a);

  scilab_rt_terminate();
}

