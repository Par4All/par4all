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

  /* ================================== */
  /*  scilab.sce */
  /*  */
  /*  (c) INRIA */
  /*  */
  /* ================================== */
  double _u_pi = SCILAB_PI;
  scilab_rt_display_s0d0_("pi",_u_pi);
  double complex _u_i = I;
  scilab_rt_display_s0z0_("i",_u_i);
  double _u_ee = SCILAB_E;
  scilab_rt_display_s0d0_("ee",_u_ee);
  /*  tests */
  scilab_rt_display_s0i0_("ans",1);
  int _u_a = 1;
  scilab_rt_display_s0i0_("a",_u_a);
  int _u_b[1][3];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  _u_b[0][2]=3;
  scilab_rt_display_s0i2_("b",1,3,_u_b);
  int _u_c[2][3];
  _u_c[0][0]=1;
  _u_c[0][1]=2;
  _u_c[0][2]=3;
  _u_c[1][0]=4;
  _u_c[1][1]=5;
  _u_c[1][2]=6;
  scilab_rt_display_s0i2_("c",2,3,_u_c);
  int _tmpxx0[2][3];
  _tmpxx0[0][0]=1;
  _tmpxx0[0][1]=2;
  _tmpxx0[0][2]=3;
  _tmpxx0[1][0]=4;
  _tmpxx0[1][1]=5;
  _tmpxx0[1][2]=6;
  int _u_d[3][2];
  scilab_rt_transposeConjugate_i2_i2(2,3,_tmpxx0,3,2,_u_d);
  scilab_rt_display_s0i2_("d",3,2,_u_d);
  int _u_e[4][4];
  _u_e[0][0]=1;
  _u_e[0][1]=2;
  _u_e[0][2]=3;
  _u_e[0][3]=4;
  _u_e[1][0]=4;
  _u_e[1][1]=5;
  _u_e[1][2]=6;
  _u_e[1][3]=7;
  _u_e[2][0]=8;
  _u_e[2][1]=9;
  _u_e[2][2]=10;
  _u_e[2][3]=11;
  _u_e[3][0]=12;
  _u_e[3][1]=13;
  _u_e[3][2]=14;
  _u_e[3][3]=15;
  scilab_rt_display_s0i2_("e",4,4,_u_e);
  int _u_f[2][2];
  _u_f[0][0]=1;
  _u_f[0][1]=2;
  _u_f[1][0]=3;
  _u_f[1][1]=4;
  scilab_rt_display_s0i2_("f",2,2,_u_f);
  int _tmpxx1[1][2];
  _tmpxx1[0][0]=1;
  _tmpxx1[0][1]=2;
  int _tmpxx2[1][2];
  _tmpxx2[0][0]=3;
  _tmpxx2[0][1]=4;
  int _tmpxx3[1][4];
  _tmpxx3[0][0]=5;
  _tmpxx3[0][1]=6;
  _tmpxx3[0][2]=7;
  _tmpxx3[0][3]=8;
  int _u_e1[2][4];
  
  for(int j=0; j<2; ++j) {
    _u_e1[0][j] = _tmpxx1[0][j];
  }
  
  for(int j=0; j<2; ++j) {
    _u_e1[0][j+2] = _tmpxx2[0][j];
  }
  
  for(int j=0; j<4; ++j) {
    _u_e1[1][j] = _tmpxx3[0][j];
  }
  scilab_rt_display_s0i2_("e1",2,4,_u_e1);
  double complex _u_h[1][5];
  _u_h[0][0]=_u_i;
  _u_h[0][1]=1;
  _u_h[0][2]=2;
  _u_h[0][3]=_u_i;
  _u_h[0][4]=3;
  scilab_rt_display_s0z2_("h",1,5,_u_h);
  double complex _tmpxx4[1][5];
  _tmpxx4[0][0]=_u_i;
  _tmpxx4[0][1]=1;
  _tmpxx4[0][2]=2;
  _tmpxx4[0][3]=_u_i;
  _tmpxx4[0][4]=3;
  double complex _u_o[5][1];
  scilab_rt_transposeConjugate_z2_z2(1,5,_tmpxx4,5,1,_u_o);
  scilab_rt_display_s0z2_("o",5,1,_u_o);
  _u_b[0][(2-1)] = 3;
  scilab_rt_display_s0i2_("b",1,3,_u_b);
  int _tmpxx5[1][2];
  _tmpxx5[0][0]=1;
  _tmpxx5[0][1]=2;
  int _tmpxx6[2][1];
  scilab_rt_transposeConjugate_i2_i2(1,2,_tmpxx5,2,1,_tmpxx6);
  int _u_x[2][1];
  
  for(int i=0; i<2; ++i) {
    _u_x[i][0] = _tmpxx6[i][0];
  }
  scilab_rt_display_s0i2_("x",2,1,_u_x);
  int _tmpxx7[1][3];
  scilab_rt_assign_i2_i2(1,3,_u_b,1,3,_tmpxx7);
  int _tmpxx8[1][3];
  scilab_rt_assign_i2_i2(1,3,_u_b,1,3,_tmpxx8);
  int _u_t1[4][3];
  _u_t1[0][0]=1;
  _u_t1[0][1]=2;
  _u_t1[0][2]=3;
  
  for(int j=0; j<3; ++j) {
    _u_t1[1][j] = _tmpxx7[0][j];
  }
  
  for(int j=0; j<3; ++j) {
    _u_t1[2][j] = _tmpxx8[0][j];
  }
  _u_t1[3][0]=4;
  _u_t1[3][1]=5;
  _u_t1[3][2]=6;
  scilab_rt_display_s0i2_("t1",4,3,_u_t1);
  int _u_t2[2][2];
  _u_t2[0][0]=1;
  _u_t2[0][1]=2;
  _u_t2[1][0]=3;
  _u_t2[1][1]=4;
  scilab_rt_display_s0i2_("t2",2,2,_u_t2);

  scilab_rt_terminate();
}

