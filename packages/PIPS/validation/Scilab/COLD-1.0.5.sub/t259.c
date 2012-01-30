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

  /*  t259.sce: log, exp, sqrt and abs with complex */
  double complex _tmpxx0 = (2*I);
  double complex _u_a = (1+_tmpxx0);
  scilab_rt_display_s0z0_("a",_u_a);
  double complex _tmpxx1 = (2*I);
  double complex _tmpxx2 = (4*I);
  double complex _u_b[1][2];
  _u_b[0][0]=(1+_tmpxx1);
  _u_b[0][1]=(3+_tmpxx2);
  scilab_rt_display_s0z2_("b",1,2,_u_b);
  double complex _tmpxx3 = (3*I);
  double _tmpxx4[2][2][2];
  scilab_rt_ones_i0i0i0_d3(2,2,2,2,2,2,_tmpxx4);
  double complex _tmpxx5 = (1+_tmpxx3);
  double complex _u_c[2][2][2];
  scilab_rt_mul_d3z0_z3(2,2,2,_tmpxx4,_tmpxx5,2,2,2,_u_c);
  scilab_rt_display_s0z3_("c",2,2,2,_u_c);
  double complex _u_loga = scilab_rt_log_z0_(_u_a);
  scilab_rt_display_s0z0_("loga",_u_loga);
  double complex _u_logb[1][2];
  scilab_rt_log_z2_z2(1,2,_u_b,1,2,_u_logb);
  scilab_rt_display_s0z2_("logb",1,2,_u_logb);
  double complex _u_logc[2][2][2];
  scilab_rt_log_z3_z3(2,2,2,_u_c,2,2,2,_u_logc);
  scilab_rt_display_s0z3_("logc",2,2,2,_u_logc);
  double complex _u_expa = scilab_rt_exp_z0_(_u_a);
  scilab_rt_display_s0z0_("expa",_u_expa);
  double complex _u_expb[1][2];
  scilab_rt_exp_z2_z2(1,2,_u_b,1,2,_u_expb);
  scilab_rt_display_s0z2_("expb",1,2,_u_expb);
  double complex _u_expc[2][2][2];
  scilab_rt_exp_z3_z3(2,2,2,_u_c,2,2,2,_u_expc);
  scilab_rt_display_s0z3_("expc",2,2,2,_u_expc);
  double complex _u_sqrta = scilab_rt_sqrt_z0_(_u_a);
  scilab_rt_display_s0z0_("sqrta",_u_sqrta);
  double complex _u_sqrtb[1][2];
  scilab_rt_sqrt_z2_z2(1,2,_u_b,1,2,_u_sqrtb);
  scilab_rt_display_s0z2_("sqrtb",1,2,_u_sqrtb);
  double complex _u_sqrtc[2][2][2];
  scilab_rt_sqrt_z3_z3(2,2,2,_u_c,2,2,2,_u_sqrtc);
  scilab_rt_display_s0z3_("sqrtc",2,2,2,_u_sqrtc);
  double _u_absa = scilab_rt_abs_z0_(_u_a);
  scilab_rt_display_s0d0_("absa",_u_absa);
  double _u_absb[1][2];
  scilab_rt_abs_z2_d2(1,2,_u_b,1,2,_u_absb);
  scilab_rt_display_s0d2_("absb",1,2,_u_absb);
  double _u_absc[2][2][2];
  scilab_rt_abs_z3_d3(2,2,2,_u_c,2,2,2,_u_absc);
  scilab_rt_display_s0d3_("absc",2,2,2,_u_absc);

  scilab_rt_terminate();
}

