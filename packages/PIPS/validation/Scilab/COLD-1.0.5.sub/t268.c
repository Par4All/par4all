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

  /*  t268.sce _ multiplication function */
  int _u_ai[3][2];
  _u_ai[0][0]=51;
  _u_ai[0][1]=39;
  _u_ai[1][0]=4;
  _u_ai[1][1]=9;
  _u_ai[2][0]=5;
  _u_ai[2][1]=18;
  int _u_bi[2][3];
  _u_bi[0][0]=2;
  _u_bi[0][1]=9;
  _u_bi[0][2]=7;
  _u_bi[1][0]=47;
  _u_bi[1][1]=3;
  _u_bi[1][2]=10;
  double _u_ar[3][2];
  _u_ar[0][0]=51.0;
  _u_ar[0][1]=39;
  _u_ar[1][0]=4;
  _u_ar[1][1]=9;
  _u_ar[2][0]=5;
  _u_ar[2][1]=18;
  double _u_br[2][3];
  _u_br[0][0]=2.0;
  _u_br[0][1]=9;
  _u_br[0][2]=7;
  _u_br[1][0]=47;
  _u_br[1][1]=3;
  _u_br[1][2]=10;
  double complex _tmpxx0 = (2.1*I);
  double complex _tmpxx1 = (4.2*I);
  double complex _tmpxx2 = (6.3*I);
  double complex _tmpxx3 = (8.4*I);
  double complex _tmpxx4 = (10.5*I);
  double complex _tmpxx5 = (12.6*I);
  double complex _u_ac[3][2];
  _u_ac[0][0]=(1.1+_tmpxx0);
  _u_ac[0][1]=(3.2+_tmpxx1);
  _u_ac[1][0]=(5.3+_tmpxx2);
  _u_ac[1][1]=(7.4+_tmpxx3);
  _u_ac[2][0]=(9.5+_tmpxx4);
  _u_ac[2][1]=(11.6+_tmpxx5);
  double complex _tmpxx6 = (1*I);
  double complex _tmpxx7 = (2*I);
  double complex _tmpxx8 = (5*I);
  double complex _tmpxx9 = (3*I);
  double complex _tmpxx10 = (4*I);
  double complex _tmpxx11 = (6*I);
  double complex _u_bc[2][3];
  _u_bc[0][0]=(1+_tmpxx6);
  _u_bc[0][1]=(2+_tmpxx7);
  _u_bc[0][2]=(5+_tmpxx8);
  _u_bc[1][0]=(3+_tmpxx9);
  _u_bc[1][1]=(4+_tmpxx10);
  _u_bc[1][2]=(6+_tmpxx11);
  int _u_c_ii[3][3];
  scilab_rt_mul_i2i2_i2(3,2,_u_ai,2,3,_u_bi,3,3,_u_c_ii);
  scilab_rt_display_s0i2_("c_ii",3,3,_u_c_ii);
  double _u_c_ir[3][3];
  scilab_rt_mul_i2d2_d2(3,2,_u_ai,2,3,_u_br,3,3,_u_c_ir);
  scilab_rt_display_s0d2_("c_ir",3,3,_u_c_ir);
  double complex _u_c_ic[3][3];
  scilab_rt_mul_i2z2_z2(3,2,_u_ai,2,3,_u_bc,3,3,_u_c_ic);
  scilab_rt_display_s0z2_("c_ic",3,3,_u_c_ic);
  double _u_c_ri[3][3];
  scilab_rt_mul_d2i2_d2(3,2,_u_ar,2,3,_u_bi,3,3,_u_c_ri);
  scilab_rt_display_s0d2_("c_ri",3,3,_u_c_ri);
  double _u_c_rr[3][3];
  scilab_rt_mul_d2d2_d2(3,2,_u_ar,2,3,_u_br,3,3,_u_c_rr);
  scilab_rt_display_s0d2_("c_rr",3,3,_u_c_rr);
  double complex _u_c_rc[3][3];
  scilab_rt_mul_d2z2_z2(3,2,_u_ar,2,3,_u_bc,3,3,_u_c_rc);
  scilab_rt_display_s0z2_("c_rc",3,3,_u_c_rc);
  double complex _u_c_ci[3][3];
  scilab_rt_mul_z2i2_z2(3,2,_u_ac,2,3,_u_bi,3,3,_u_c_ci);
  scilab_rt_display_s0z2_("c_ci",3,3,_u_c_ci);
  double complex _u_c_cr[3][3];
  scilab_rt_mul_z2d2_z2(3,2,_u_ac,2,3,_u_br,3,3,_u_c_cr);
  scilab_rt_display_s0z2_("c_cr",3,3,_u_c_cr);
  double complex _u_c_cc[3][3];
  scilab_rt_mul_z2z2_z2(3,2,_u_ac,2,3,_u_bc,3,3,_u_c_cc);
  scilab_rt_display_s0z2_("c_cc",3,3,_u_c_cc);

  scilab_rt_terminate();
}

