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

  /*  t138.sce: testing squeeze function */
  double _u_a[3][1][3];
  scilab_rt_ones_i0i0i0_d3(3,1,3,3,1,3,_u_a);

  for(int j=0; j<1;++j) {
    for(int k=0; k<3;++k) {
      _u_a[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_a[i][0][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_a[i][0][2] = 30;
  }
  double _u_a1[1][6][3];
  scilab_rt_ones_i0i0i0_d3(1,6,3,1,6,3,_u_a1);

  for(int j=0; j<6;++j) {
    for(int k=0; k<3;++k) {
      _u_a1[0][j][k] = 10;
    }
  }

  for(int i=0; i<1;++i) {
    for(int k=0; k<3;++k) {
      _u_a1[i][3][k] = 20;
    }
  }

  for(int i=0; i<1;++i) {
    _u_a1[i][2][2] = 30;
  }
  double _u_a2[7][3][1];
  scilab_rt_ones_i0i0i0_d3(7,3,1,7,3,1,_u_a2);

  for(int j=0; j<3;++j) {
    for(int k=0; k<1;++k) {
      _u_a2[3][j][k] = 10;
    }
  }

  for(int k=0; k<1;++k) {
    _u_a2[5][1][k] = 20;
  }

  for(int i=0; i<7;++i) {
    _u_a2[i][2][0] = 30;
  }
  double _u_b[3][3];
  scilab_rt_squeeze_d3_d2(3,1,3,_u_a,3,3,_u_b);
  scilab_rt_display_s0d2_("b",3,3,_u_b);
  double _u_b1[6][3];
  scilab_rt_squeeze_d3_d2(1,6,3,_u_a1,6,3,_u_b1);
  scilab_rt_display_s0d2_("b1",6,3,_u_b1);
  double _u_b2[7][3];
  scilab_rt_squeeze_d3_d2(7,3,1,_u_a2,7,3,_u_b2);
  scilab_rt_display_s0d2_("b2",7,3,_u_b2);

  scilab_rt_terminate();
}

