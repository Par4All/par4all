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

  /*  T284.sce _ det function */
  int _u_a1[3][3];
  _u_a1[0][0]=1;
  _u_a1[0][1]=5;
  _u_a1[0][2]=9;
  _u_a1[1][0]=3;
  _u_a1[1][1]=4;
  _u_a1[1][2]=6;
  _u_a1[2][0]=8;
  _u_a1[2][1]=2;
  _u_a1[2][2]=7;
  double _u_d1;
  scilab_rt_det_i2_d0(3,3,_u_a1,&_u_d1);
  scilab_rt_display_s0d0_("d1",_u_d1);
  double _u_a2[5][5];
  _u_a2[0][0]=2.0;
  _u_a2[0][1]=4;
  _u_a2[0][2]=3;
  _u_a2[0][3]=2;
  _u_a2[0][4]=4;
  _u_a2[1][0]=8;
  _u_a2[1][1]=4;
  _u_a2[1][2]=2;
  _u_a2[1][3]=1;
  _u_a2[1][4]=7;
  _u_a2[2][0]=9;
  _u_a2[2][1]=2;
  _u_a2[2][2]=8;
  _u_a2[2][3]=2;
  _u_a2[2][4]=6;
  _u_a2[3][0]=1;
  _u_a2[3][1]=3;
  _u_a2[3][2]=5;
  _u_a2[3][3]=2;
  _u_a2[3][4]=4;
  _u_a2[4][0]=3;
  _u_a2[4][1]=1;
  _u_a2[4][2]=8;
  _u_a2[4][3]=3;
  _u_a2[4][4]=9;
  double _u_d2;
  scilab_rt_det_d2_d0(5,5,_u_a2,&_u_d2);
  scilab_rt_display_s0d0_("d2",_u_d2);
  double complex _tmpxx0 = (2*I);
  int _tmpxx1[3][3];
  _tmpxx1[0][0]=2;
  _tmpxx1[0][1]=4;
  _tmpxx1[0][2]=3;
  _tmpxx1[1][0]=8;
  _tmpxx1[1][1]=4;
  _tmpxx1[1][2]=2;
  _tmpxx1[2][0]=9;
  _tmpxx1[2][1]=2;
  _tmpxx1[2][2]=8;
  double complex _tmpxx2 = (1+_tmpxx0);
  double complex _u_a3[3][3];
  scilab_rt_mul_i2z0_z2(3,3,_tmpxx1,_tmpxx2,3,3,_u_a3);
  double complex _u_d3;
  scilab_rt_det_z2_z0(3,3,_u_a3,&_u_d3);
  scilab_rt_display_s0z0_("d3",_u_d3);
  double complex _tmpxx3 = (2*I);
  double _tmpxx4[5][5];
  _tmpxx4[0][0]=2.0;
  _tmpxx4[0][1]=4;
  _tmpxx4[0][2]=3;
  _tmpxx4[0][3]=2;
  _tmpxx4[0][4]=4;
  _tmpxx4[1][0]=8;
  _tmpxx4[1][1]=4;
  _tmpxx4[1][2]=2;
  _tmpxx4[1][3]=1;
  _tmpxx4[1][4]=7;
  _tmpxx4[2][0]=9;
  _tmpxx4[2][1]=2;
  _tmpxx4[2][2]=8;
  _tmpxx4[2][3]=2;
  _tmpxx4[2][4]=6;
  _tmpxx4[3][0]=1;
  _tmpxx4[3][1]=3;
  _tmpxx4[3][2]=5;
  _tmpxx4[3][3]=2;
  _tmpxx4[3][4]=4;
  _tmpxx4[4][0]=3;
  _tmpxx4[4][1]=1;
  _tmpxx4[4][2]=8;
  _tmpxx4[4][3]=3;
  _tmpxx4[4][4]=9;
  double complex _tmpxx5 = (3+_tmpxx3);
  double complex _u_a4[5][5];
  scilab_rt_mul_d2z0_z2(5,5,_tmpxx4,_tmpxx5,5,5,_u_a4);
  double complex _u_d4;
  scilab_rt_det_z2_z0(5,5,_u_a4,&_u_d4);
  scilab_rt_display_s0z0_("d4",_u_d4);

  scilab_rt_terminate();
}

