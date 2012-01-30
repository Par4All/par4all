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

  double _u_a = 1.5;
  int _u_ai = scilab_rt_int32_d0_(_u_a);
  scilab_rt_display_s0i0_("ai",_u_ai);
  double _u_b[3][3];
  _u_b[0][0]=1.1;
  _u_b[0][1]=2.2;
  _u_b[0][2]=3.3;
  _u_b[1][0]=4.4;
  _u_b[1][1]=5.5;
  _u_b[1][2]=6.6;
  _u_b[2][0]=7.7;
  _u_b[2][1]=8.8;
  _u_b[2][2]=9.9;
  int _u_bi[3][3];
  scilab_rt_int32_d2_i2(3,3,_u_b,3,3,_u_bi);
  scilab_rt_display_s0i2_("bi",3,3,_u_bi);
  double _u_c[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_c);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_c[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_c[i][1][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_c[i][1][2] = 30;
  }
  int _u_ci[3][2][3];
  scilab_rt_int32_d3_i3(3,2,3,_u_c,3,2,3,_u_ci);
  scilab_rt_display_s0i3_("ci",3,2,3,_u_ci);

  scilab_rt_terminate();
}

