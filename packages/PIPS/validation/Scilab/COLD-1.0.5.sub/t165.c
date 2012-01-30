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

  double complex _tmpxx0 = (3*I);
  double complex _u_a = (2+_tmpxx0);
  scilab_rt_display_s0z0_("a",_u_a);
  double complex _tmpxx1 = (2*I);
  double complex _tmpxx2 = (4*I);
  double complex _tmpxx3 = (6*I);
  double complex _tmpxx4 = (8*I);
  double complex _tmpxx5 = (10*I);
  double complex _tmpxx6 = (12*I);
  double complex _u_b[3][2];
  _u_b[0][0]=(1+_tmpxx1);
  _u_b[0][1]=(3+_tmpxx2);
  _u_b[1][0]=(5+_tmpxx3);
  _u_b[1][1]=(7+_tmpxx4);
  _u_b[2][0]=(9+_tmpxx5);
  _u_b[2][1]=(11+_tmpxx6);
  scilab_rt_display_s0z2_("b",3,2,_u_b);
  double _u_aReal = scilab_rt_real_z0_(_u_a);
  scilab_rt_display_s0d0_("aReal",_u_aReal);
  double _u_aImag = scilab_rt_imag_z0_(_u_a);
  scilab_rt_display_s0d0_("aImag",_u_aImag);
  double _u_bReal[3][2];
  scilab_rt_real_z2_d2(3,2,_u_b,3,2,_u_bReal);
  scilab_rt_display_s0d2_("bReal",3,2,_u_bReal);
  double _u_bImag[3][2];
  scilab_rt_imag_z2_d2(3,2,_u_b,3,2,_u_bImag);
  scilab_rt_display_s0d2_("bImag",3,2,_u_bImag);

  scilab_rt_terminate();
}

