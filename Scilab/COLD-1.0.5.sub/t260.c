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

  double complex _tmpxx0 = (1*I);
  double complex _tmpxx1 = (2*I);
  double complex _u_a[1][2];
  _u_a[0][0]=(1+_tmpxx0);
  _u_a[0][1]=(2+_tmpxx1);
  scilab_rt_display_s0z2_("a",1,2,_u_a);
  double complex _tmpxx2 = (0.1*I);
  double complex _tmpxx3 = (0.2*I);
  double complex _tmpxx4 = (0.3*I);
  double complex _tmpxx5 = (0.4*I);
  double complex _tmpxx6 = (1*I);
  double complex _tmpxx7 = (2*I);
  double complex _tmpxx8 = (3*I);
  double complex _tmpxx9 = (4*I);
  double complex _tmpxx10[2][2];
  _tmpxx10[0][0]=(0.1+_tmpxx2);
  _tmpxx10[0][1]=(0.2+_tmpxx3);
  _tmpxx10[1][0]=(0.3+_tmpxx4);
  _tmpxx10[1][1]=(0.4+_tmpxx5);
  double complex _tmpxx11[2][2];
  _tmpxx11[0][0]=(1+_tmpxx6);
  _tmpxx11[0][1]=(2+_tmpxx7);
  _tmpxx11[1][0]=(3+_tmpxx8);
  _tmpxx11[1][1]=(4+_tmpxx9);
  double complex _u_b[2][2];
  scilab_rt_mul_z2z2_z2(2,2,_tmpxx10,2,2,_tmpxx11,2,2,_u_b);
  scilab_rt_display_s0z2_("b",2,2,_u_b);

  scilab_rt_terminate();
}

