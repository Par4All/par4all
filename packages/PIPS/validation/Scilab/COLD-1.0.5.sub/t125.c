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

  /*  t124.sce: testing ceil and floor for 3D matrices */
  double _u_a[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_a);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_a[1][j][k] = 10.3;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_a[i][1][k] = 20.5;
    }
  }

  for(int i=0; i<3;++i) {
    _u_a[i][1][2] = 30.1;
  }
  double _u_b[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_b);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_b[1][j][k] = 10.2;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_b[i][1][k] = 20.6;
    }
  }

  for(int i=0; i<3;++i) {
    _u_b[i][1][2] = 30.7;
  }
  int _u_c[3][2][3];
  scilab_rt_floor_d3_i3(3,2,3,_u_a,3,2,3,_u_c);
  scilab_rt_display_s0i3_("c",3,2,3,_u_c);
  int _u_d[3][2][3];
  scilab_rt_ceil_d3_i3(3,2,3,_u_b,3,2,3,_u_d);
  scilab_rt_display_s0i3_("d",3,2,3,_u_d);

  scilab_rt_terminate();
}

