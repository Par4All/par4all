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

  /*  t205.sce: array def */
  double _u_t1[1][4];
  scilab_rt_ones_i0i0_d2(1,4,1,4,_u_t1);
  double _u_t2[1][4];
  scilab_rt_zeros_i0i0_d2(1,4,1,4,_u_t2);
  double _tmpxx0[1][4];
  scilab_rt_assign_d2_d2(1,4,_u_t1,1,4,_tmpxx0);
  double _tmpxx1[1][4];
  scilab_rt_assign_d2_d2(1,4,_u_t2,1,4,_tmpxx1);
  double _tmp0[1][8];
  
  for(int j=0; j<4; ++j) {
    _tmp0[0][j] = _tmpxx0[0][j];
  }
  
  for(int j=0; j<4; ++j) {
    _tmp0[0][j+4] = _tmpxx1[0][j];
  }
  scilab_rt_display_s0d2_("ans",1,8,_tmp0);
  double _tmpxx2[1][4];
  scilab_rt_assign_d2_d2(1,4,_u_t1,1,4,_tmpxx2);
  double _tmpxx3[1][4];
  scilab_rt_assign_d2_d2(1,4,_u_t2,1,4,_tmpxx3);
  double _tmp1[1][8];
  
  for(int j=0; j<4; ++j) {
    _tmp1[0][j] = _tmpxx2[0][j];
  }
  
  for(int j=0; j<4; ++j) {
    _tmp1[0][j+4] = _tmpxx3[0][j];
  }
  scilab_rt_display_s0d2_("ans",1,8,_tmp1);
  double _tmpxx4[1][4];
  scilab_rt_assign_d2_d2(1,4,_u_t1,1,4,_tmpxx4);
  double _tmpxx5[1][4];
  scilab_rt_assign_d2_d2(1,4,_u_t2,1,4,_tmpxx5);
  double _tmp2[2][4];
  
  for(int j=0; j<4; ++j) {
    _tmp2[0][j] = _tmpxx4[0][j];
  }
  
  for(int j=0; j<4; ++j) {
    _tmp2[1][j] = _tmpxx5[0][j];
  }
  scilab_rt_display_s0d2_("ans",2,4,_tmp2);
  int _u_a[2][2];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[1][0]=3;
  _u_a[1][1]=4;
  int _u_b[1][2];
  _u_b[0][0]=5;
  _u_b[0][1]=6;
  int _u_bt[2][1];
  scilab_rt_transposeConjugate_i2_i2(1,2,_u_b,2,1,_u_bt);
  int _u_c[1][2];
  _u_c[0][0]=7;
  _u_c[0][1]=8;
  int _u_ct[2][1];
  scilab_rt_transposeConjugate_i2_i2(1,2,_u_c,2,1,_u_ct);
  int _tmpxx6[2][1];
  scilab_rt_assign_i2_i2(2,1,_u_bt,2,1,_tmpxx6);
  int _tmpxx7[2][2];
  scilab_rt_assign_i2_i2(2,2,_u_a,2,2,_tmpxx7);
  int _tmpxx8[2][2];
  scilab_rt_assign_i2_i2(2,2,_u_a,2,2,_tmpxx8);
  int _tmpxx9[2][1];
  scilab_rt_assign_i2_i2(2,1,_u_ct,2,1,_tmpxx9);
  int _u_d[4][3];
  
  for(int i=0; i<2; ++i) {
    _u_d[i][0] = _tmpxx6[i][0];
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_d[i][j+1] = _tmpxx7[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_d[i+2][j] = _tmpxx8[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    _u_d[i+2][2] = _tmpxx9[i][0];
  }
  scilab_rt_display_s0i2_("d",4,3,_u_d);
  double _tmpxx10[2][2];
  scilab_rt_ones_i0i0_d2(2,2,2,2,_tmpxx10);
  double _tmpxx11[2][2];
  scilab_rt_zeros_i0i0_d2(2,2,2,2,_tmpxx11);
  double _u_t3[2][4];
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_t3[i][j] = _tmpxx10[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_t3[i][j+2] = _tmpxx11[i][j];
    }
  }
  scilab_rt_display_s0d2_("t3",2,4,_u_t3);
  double _tmpxx12[2][2];
  scilab_rt_ones_i0i0_d2(2,2,2,2,_tmpxx12);
  double _tmpxx13[2][2];
  scilab_rt_zeros_i0i0_d2(2,2,2,2,_tmpxx13);
  double _u_t4[2][4];
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_t4[i][j] = _tmpxx12[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_t4[i][j+2] = _tmpxx13[i][j];
    }
  }
  scilab_rt_display_s0d2_("t4",2,4,_u_t4);
  int _tmpxx14[2][1];
  scilab_rt_transposeConjugate_i2_i2(1,2,_u_b,2,1,_tmpxx14);
  int _tmpxx15[2][2];
  scilab_rt_assign_i2_i2(2,2,_u_a,2,2,_tmpxx15);
  int _tmpxx16[2][2];
  scilab_rt_assign_i2_i2(2,2,_u_a,2,2,_tmpxx16);
  int _tmpxx17[2][1];
  scilab_rt_transposeConjugate_i2_i2(1,2,_u_c,2,1,_tmpxx17);
  int _u_d2[4][3];
  
  for(int i=0; i<2; ++i) {
    _u_d2[i][0] = _tmpxx14[i][0];
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_d2[i][j+1] = _tmpxx15[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    for(int j=0; j<2; ++j) {
      _u_d2[i+2][j] = _tmpxx16[i][j];
    }
  }
  
  for(int i=0; i<2; ++i) {
    _u_d2[i+2][2] = _tmpxx17[i][0];
  }
  scilab_rt_display_s0i2_("d2",4,3,_u_d2);

  scilab_rt_terminate();
}

