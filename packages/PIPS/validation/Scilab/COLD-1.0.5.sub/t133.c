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

  /*  t133.sce: Testing exp; log and log2 for 2D and 3D matrices */
  double _u_a2[3][4];
  scilab_rt_ones_i0i0_d2(3,4,3,4,_u_a2);

  for(int j=0; j<4;++j) {
    _u_a2[1][j] = 10;
  }

  for(int i=0; i<3;++i) {
    _u_a2[i][1] = 20;
  }

  for(int i=0; i<3;++i) {
    _u_a2[i][1] = 30;
  }
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
  double _u_b[3][4];
  scilab_rt_exp_d2_d2(3,4,_u_a2,3,4,_u_b);
  scilab_rt_display_s0d2_("b",3,4,_u_b);
  double _u_c[3][2][3];
  scilab_rt_exp_d3_d3(3,2,3,_u_a3,3,2,3,_u_c);
  scilab_rt_display_s0d3_("c",3,2,3,_u_c);
  double _u_d[3][4];
  scilab_rt_log_d2_d2(3,4,_u_a2,3,4,_u_d);
  scilab_rt_display_s0d2_("d",3,4,_u_d);
  double _u_e[3][2][3];
  scilab_rt_log_d3_d3(3,2,3,_u_a3,3,2,3,_u_e);
  scilab_rt_display_s0d3_("e",3,2,3,_u_e);
  double _u_f[3][4];
  scilab_rt_log2_d2_d2(3,4,_u_a2,3,4,_u_f);
  scilab_rt_display_s0d2_("f",3,4,_u_f);
  double _u_g[3][2][3];
  scilab_rt_log2_d3_d3(3,2,3,_u_a3,3,2,3,_u_g);
  scilab_rt_display_s0d3_("g",3,2,3,_u_g);

  scilab_rt_terminate();
}

