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

  /* testing weekday */
  double _u_a = 732009.34;
  double _u_b[2][2];
  _u_b[0][0]=703029.76;
  _u_b[0][1]=732009.34;
  _u_b[1][0]=714547;
  _u_b[1][1]=714547.38;
  int _u_c = scilab_rt_weekday_d0_(_u_a);
  scilab_rt_display_s0i0_("c",_u_c);
  int _u_d;
  char* _u_e = NULL;
  scilab_rt_weekday_d0_i0s0(_u_a,&_u_d,&_u_e);
  scilab_rt_display_s0s0_("e",_u_e);
  scilab_rt_display_s0i0_("d",_u_d);
  int _u_f;
  char* _u_g = NULL;
  scilab_rt_weekday_d0s0_i0s0(_u_a,"long",&_u_f,&_u_g);
  scilab_rt_display_s0s0_("g",_u_g);
  scilab_rt_display_s0i0_("f",_u_f);
  int _u_h;
  char* _u_i = NULL;
  scilab_rt_weekday_d0s0_i0s0(_u_a,"short",&_u_h,&_u_i);
  scilab_rt_display_s0s0_("i",_u_i);
  scilab_rt_display_s0i0_("h",_u_h);
  int _u_j[2][2];
  scilab_rt_weekday_d2_i2(2,2,_u_b,2,2,_u_j);
  scilab_rt_display_s0i2_("j",2,2,_u_j);
  int _u_k[2][2];
  char* _u_l[2][2];
  scilab_rt_weekday_d2_i2s2(2,2,_u_b,2,2,_u_k,2,2,_u_l);
  scilab_rt_display_s0s2_("l",2,2,_u_l);
  scilab_rt_display_s0i2_("k",2,2,_u_k);
  int _u_m[2][2];
  char* _u_n[2][2];
  scilab_rt_weekday_d2s0_i2s2(2,2,_u_b,"long",2,2,_u_m,2,2,_u_n);
  scilab_rt_display_s0s2_("n",2,2,_u_n);
  scilab_rt_display_s0i2_("m",2,2,_u_m);
  int _u_o[2][2];
  char* _u_p[2][2];
  scilab_rt_weekday_d2s0_i2s2(2,2,_u_b,"short",2,2,_u_o,2,2,_u_p);
  scilab_rt_display_s0s2_("p",2,2,_u_p);
  scilab_rt_display_s0i2_("o",2,2,_u_o);

  scilab_rt_terminate();
}

