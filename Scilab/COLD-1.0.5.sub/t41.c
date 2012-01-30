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

  /*  t41.sce: testing sqrt function */
  double _u_x = 4.;
  double _u_y = 3.;
  double _tmpxx0 = (_u_x*_u_x);
  double _tmpxx1 = (_u_y*_u_y);
  double _u_d = scilab_rt_sqrt_d0_((_tmpxx0+_tmpxx1));
  scilab_rt_display_s0d0_("d",_u_d);
  int _u_a = 2;
  double _u_b = scilab_rt_sqrt_i0_(_u_a);
  scilab_rt_display_s0d0_("b",_u_b);
  /* a3 = round(predicted_rand(3,2,3)*10); */
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
  double _u_c[3][2][3];
  scilab_rt_sqrt_d3_d3(3,2,3,_u_a3,3,2,3,_u_c);
  scilab_rt_display_s0d3_("c",3,2,3,_u_c);

  scilab_rt_terminate();
}

