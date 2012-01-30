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

  /*  t257.sce: trigonometry with complex */
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
  /*  COS */
  scilab_rt_disp_s0_("_________cos");
  double complex _u_cos1 = scilab_rt_cos_z0_(_u_a);
  scilab_rt_display_s0z0_("cos1",_u_cos1);
  double complex _u_cos2[1][2];
  scilab_rt_cos_z2_z2(1,2,_u_b,1,2,_u_cos2);
  scilab_rt_display_s0z2_("cos2",1,2,_u_cos2);
  double complex _u_cos3[2][2][2];
  scilab_rt_cos_z3_z3(2,2,2,_u_c,2,2,2,_u_cos3);
  scilab_rt_display_s0z3_("cos3",2,2,2,_u_cos3);
  /*  COSD */
  scilab_rt_disp_s0_("_________cosd");
  /* complex not implemented in siclab */
  /*  COSH */
  scilab_rt_disp_s0_("_________cosh");
  double complex _u_acosh1 = scilab_rt_cosh_z0_(_u_a);
  scilab_rt_display_s0z0_("acosh1",_u_acosh1);
  double complex _u_acosh2[1][2];
  scilab_rt_cosh_z2_z2(1,2,_u_b,1,2,_u_acosh2);
  scilab_rt_display_s0z2_("acosh2",1,2,_u_acosh2);
  /* 3D not implemented in scilab */
  /*  ACOS */
  scilab_rt_disp_s0_("_________acos");
  double complex _u_acos1 = scilab_rt_acos_z0_(_u_a);
  scilab_rt_display_s0z0_("acos1",_u_acos1);
  double complex _u_acos2[1][2];
  scilab_rt_acos_z2_z2(1,2,_u_b,1,2,_u_acos2);
  scilab_rt_display_s0z2_("acos2",1,2,_u_acos2);
  /* 3D not implemented in scilab */
  /*  ACOSD */
  scilab_rt_disp_s0_("_________acosd");
  /* complex not implemented in siclab */
  /*  ACOSH */
  scilab_rt_disp_s0_("_________acosh");
  _u_acosh1 = scilab_rt_acosh_z0_(_u_a);
  scilab_rt_display_s0z0_("acosh1",_u_acosh1);
  
  scilab_rt_acosh_z2_z2(1,2,_u_b,1,2,_u_acosh2);
  scilab_rt_display_s0z2_("acosh2",1,2,_u_acosh2);
  /* 3D not implemented in scilab */
  /*  sin */
  scilab_rt_disp_s0_("_________sin");
  double complex _u_sin1 = scilab_rt_sin_z0_(_u_a);
  scilab_rt_display_s0z0_("sin1",_u_sin1);
  double complex _u_sin2[1][2];
  scilab_rt_sin_z2_z2(1,2,_u_b,1,2,_u_sin2);
  scilab_rt_display_s0z2_("sin2",1,2,_u_sin2);
  double complex _u_sin3[2][2][2];
  scilab_rt_sin_z3_z3(2,2,2,_u_c,2,2,2,_u_sin3);
  scilab_rt_display_s0z3_("sin3",2,2,2,_u_sin3);
  /*  sinD */
  scilab_rt_disp_s0_("_________sind");
  /* complex not implemented in siclab */
  /*  sinH */
  scilab_rt_disp_s0_("_________sinh");
  double complex _u_asinh1 = scilab_rt_sinh_z0_(_u_a);
  scilab_rt_display_s0z0_("asinh1",_u_asinh1);
  double complex _u_asinh2[1][2];
  scilab_rt_sinh_z2_z2(1,2,_u_b,1,2,_u_asinh2);
  scilab_rt_display_s0z2_("asinh2",1,2,_u_asinh2);
  /* 3D not implemented in scilab */
  /*  ASIN */
  scilab_rt_disp_s0_("_________asin");
  double complex _u_asin1 = scilab_rt_asin_z0_(_u_a);
  scilab_rt_display_s0z0_("asin1",_u_asin1);
  double complex _u_asin2[1][2];
  scilab_rt_asin_z2_z2(1,2,_u_b,1,2,_u_asin2);
  scilab_rt_display_s0z2_("asin2",1,2,_u_asin2);
  /* 3D not implemented in scilab */
  /*  ASIND */
  scilab_rt_disp_s0_("_________asind");
  /* complex not implemented in siclab */
  /*  ASINH */
  scilab_rt_disp_s0_("_________asinh");
  _u_asinh1 = scilab_rt_asinh_z0_(_u_a);
  scilab_rt_display_s0z0_("asinh1",_u_asinh1);
  
  scilab_rt_asinh_z2_z2(1,2,_u_b,1,2,_u_asinh2);
  scilab_rt_display_s0z2_("asinh2",1,2,_u_asinh2);
  /* 3D not implemented in scilab */
  /*  tan */
  scilab_rt_disp_s0_("_________tan");
  double complex _u_tan1 = scilab_rt_tan_z0_(_u_a);
  scilab_rt_display_s0z0_("tan1",_u_tan1);
  double complex _u_tan2[1][2];
  scilab_rt_tan_z2_z2(1,2,_u_b,1,2,_u_tan2);
  scilab_rt_display_s0z2_("tan2",1,2,_u_tan2);
  /* 3D not implemented in scilab */
  /*  tanD */
  scilab_rt_disp_s0_("_________tand");
  /* complex not implemented in siclab */
  /*  tanH */
  scilab_rt_disp_s0_("_________tanh");
  double complex _u_atanh1 = scilab_rt_tanh_z0_(_u_a);
  scilab_rt_display_s0z0_("atanh1",_u_atanh1);
  double complex _u_atanh2[1][2];
  scilab_rt_tanh_z2_z2(1,2,_u_b,1,2,_u_atanh2);
  scilab_rt_display_s0z2_("atanh2",1,2,_u_atanh2);
  /* 3D not implemented in scilab */
  /*  ATAN */
  scilab_rt_disp_s0_("_________atan");
  double complex _u_atan1 = scilab_rt_atan_z0_(_u_a);
  scilab_rt_display_s0z0_("atan1",_u_atan1);
  double complex _u_atan2[1][2];
  scilab_rt_atan_z2_z2(1,2,_u_b,1,2,_u_atan2);
  scilab_rt_display_s0z2_("atan2",1,2,_u_atan2);
  /* 3D not implemented in scilab */
  /*  ATAND */
  scilab_rt_disp_s0_("_________atand");
  /* complex not implemented in siclab */
  /*  ASINH */
  scilab_rt_disp_s0_("_________atanh");
  _u_atanh1 = scilab_rt_atanh_z0_(_u_a);
  scilab_rt_display_s0z0_("atanh1",_u_atanh1);
  
  scilab_rt_atanh_z2_z2(1,2,_u_b,1,2,_u_atanh2);
  scilab_rt_display_s0z2_("atanh2",1,2,_u_atanh2);
  /* 3D not implemented in scilab */

  scilab_rt_terminate();
}

