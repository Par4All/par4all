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

  /*  t266.sce : hypermatrix (3D) display */
  double _u_a[3][3][3];
  scilab_rt_ones_i0i0i0_d3(3,3,3,3,3,3,_u_a);

  for(int i=0; i<3;++i) {
    for(int j=0; j<3;++j) {
      _u_a[i][j][0] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_a[i][0][k] = 20;
    }
  }

  for(int j=0; j<3;++j) {
    for(int k=0; k<3;++k) {
      _u_a[0][j][k] = 30;
    }
  }
  double _u_b[3][3][3];
  scilab_rt_ones_i0i0i0_d3(3,3,3,3,3,3,_u_b);

  for(int i=0; i<3;++i) {
    for(int j=0; j<3;++j) {
      _u_b[i][j][0] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_b[i][0][k] = 20;
    }
  }

  for(int j=0; j<3;++j) {
    for(int k=0; k<3;++k) {
      _u_b[0][j][k] = 30;
    }
  }

  for(int i=0; i<3;++i) {
    for(int j=0; j<3;++j) {
      _u_b[i][j][1] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_b[i][1][k] = 20;
    }
  }

  for(int j=0; j<3;++j) {
    for(int k=0; k<3;++k) {
      _u_b[1][j][k] = 30;
    }
  }
  double _u_c[3][3][3];
  scilab_rt_add_d3d3_d3(3,3,3,_u_a,3,3,3,_u_b,3,3,3,_u_c);
  scilab_rt_display_s0d3_("c",3,3,3,_u_c);
  scilab_rt_disp_d3_(3,3,3,_u_c);

  scilab_rt_terminate();
}

