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

  /*  t141.sce: testing stdev */
  int _u_a[3][2];
  _u_a[0][0]=3;
  _u_a[0][1]=4;
  _u_a[1][0]=3;
  _u_a[1][1]=9;
  _u_a[2][0]=8;
  _u_a[2][1]=6;
  double _u_ad[3][2];
  _u_ad[0][0]=3;
  _u_ad[0][1]=4.0;
  _u_ad[1][0]=3;
  _u_ad[1][1]=9.0;
  _u_ad[2][0]=8;
  _u_ad[2][1]=6;
  double _u_a3[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_a3);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_a3[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_a3[i][1][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_a3[i][1][2] = 30;
  }
  int _u_b[1][1];
  _u_b[0][0]=1;
  double _u_c = 5.;
  double _u_d;
  scilab_rt_stdev_i2_d0(1,1,_u_b,&_u_d);
  scilab_rt_display_s0d0_("d",_u_d);
  double _u_e;
  scilab_rt_stdev_i2i0_d0(1,1,_u_b,1,&_u_e);
  scilab_rt_display_s0d0_("e",_u_e);
  double _u_f;
  scilab_rt_stdev_i2s0_d0(1,1,_u_b,"r",&_u_f);
  scilab_rt_display_s0d0_("f",_u_f);
  double _u_g = scilab_rt_stdev_d0_(_u_c);
  scilab_rt_display_s0d0_("g",_u_g);
  double _u_h = scilab_rt_stdev_d0i0_(_u_c,2);
  scilab_rt_display_s0d0_("h",_u_h);
  double _u_i = scilab_rt_stdev_d0s0_(_u_c,"c");
  scilab_rt_display_s0d0_("i",_u_i);
  double _u_j;
  scilab_rt_stdev_i2_d0(3,2,_u_a,&_u_j);
  scilab_rt_display_s0d0_("j",_u_j);
  double _u_k[3][1];
  scilab_rt_stdev_i2i0_d2(3,2,_u_a,2,3,1,_u_k);
  scilab_rt_display_s0d2_("k",3,1,_u_k);
  double _u_l[3][1];
  scilab_rt_stdev_i2s0_d2(3,2,_u_a,"c",3,1,_u_l);
  scilab_rt_display_s0d2_("l",3,1,_u_l);
  double _u_m;
  scilab_rt_stdev_d3_d0(3,2,3,_u_a3,&_u_m);
  scilab_rt_display_s0d0_("m",_u_m);
  double _u_n;
  scilab_rt_stdev_d2_d0(3,2,_u_ad,&_u_n);
  scilab_rt_display_s0d0_("n",_u_n);
  double _u_o[3][1];
  scilab_rt_stdev_d2i0_d2(3,2,_u_ad,2,3,1,_u_o);
  scilab_rt_display_s0d2_("o",3,1,_u_o);
  double _u_p[3][1];
  scilab_rt_stdev_d2s0_d2(3,2,_u_ad,"c",3,1,_u_p);
  scilab_rt_display_s0d2_("p",3,1,_u_p);

  scilab_rt_terminate();
}

