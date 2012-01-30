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

  /*  t263.sce: trigo functions */
  double _u_cos1 = scilab_rt_cos_d0_(0.5);
  scilab_rt_display_s0d0_("cos1",_u_cos1);
  double _u_cosd1 = scilab_rt_cosd_d0_(0.5);
  scilab_rt_display_s0d0_("cosd1",_u_cosd1);
  double _u_cosh1 = scilab_rt_cosh_d0_(0.5);
  scilab_rt_display_s0d0_("cosh1",_u_cosh1);
  double _u_acos1 = scilab_rt_acos_d0_(0.5);
  scilab_rt_display_s0d0_("acos1",_u_acos1);
  double _u_acosd1 = scilab_rt_acosd_d0_(0.5);
  scilab_rt_display_s0d0_("acosd1",_u_acosd1);
  double _u_acosh1 = scilab_rt_acosh_d0_(1.5);
  scilab_rt_display_s0d0_("acosh1",_u_acosh1);
  double _u_sin1 = scilab_rt_sin_d0_(0.5);
  scilab_rt_display_s0d0_("sin1",_u_sin1);
  double _u_sind1 = scilab_rt_sind_d0_(0.5);
  scilab_rt_display_s0d0_("sind1",_u_sind1);
  double _u_sinh1 = scilab_rt_sinh_d0_(0.5);
  scilab_rt_display_s0d0_("sinh1",_u_sinh1);
  double _u_asin1 = scilab_rt_asin_d0_(0.5);
  scilab_rt_display_s0d0_("asin1",_u_asin1);
  double _u_asind1 = scilab_rt_asind_d0_(0.5);
  scilab_rt_display_s0d0_("asind1",_u_asind1);
  double _u_asinh1 = scilab_rt_asinh_d0_(0.5);
  scilab_rt_display_s0d0_("asinh1",_u_asinh1);
  double _u_tan1 = scilab_rt_tan_d0_(0.5);
  scilab_rt_display_s0d0_("tan1",_u_tan1);
  double _u_tand1 = scilab_rt_tand_d0_(0.5);
  scilab_rt_display_s0d0_("tand1",_u_tand1);
  double _u_tanh1 = scilab_rt_tanh_d0_(0.5);
  scilab_rt_display_s0d0_("tanh1",_u_tanh1);
  double _u_atan1 = scilab_rt_atan_d0_(0.5);
  scilab_rt_display_s0d0_("atan1",_u_atan1);
  double _u_atand1 = scilab_rt_atand_d0_(0.5);
  scilab_rt_display_s0d0_("atand1",_u_atand1);
  double _u_atanh1 = scilab_rt_atanh_d0_(0.5);
  scilab_rt_display_s0d0_("atanh1",_u_atanh1);

  scilab_rt_terminate();
}

