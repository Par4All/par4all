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

  /*  126.sce: Testing sin, cos and tan with 3D matrices */
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
  double _u_s[3][2][3];
  scilab_rt_sin_d3_d3(3,2,3,_u_a,3,2,3,_u_s);
  scilab_rt_display_s0d3_("s",3,2,3,_u_s);
  double _u_c[3][2][3];
  scilab_rt_cos_d3_d3(3,2,3,_u_a,3,2,3,_u_c);
  scilab_rt_display_s0d3_("c",3,2,3,_u_c);
  /* t = tan(a) No supported on scilab */

  scilab_rt_terminate();
}

