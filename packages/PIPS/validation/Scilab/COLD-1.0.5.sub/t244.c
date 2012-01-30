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

  double _u_a[2][4][4];
  scilab_rt_ones_i0i0i0_d3(2,4,4,2,4,4,_u_a);
  double _tmpxx0[4][4];
  scilab_rt_ones_i0i0_d2(4,4,4,4,_tmpxx0);
  double _u_b[4][4];
  scilab_rt_add_d2i0_d2(4,4,_tmpxx0,1,4,4,_u_b);

  for(int j=0; j<4;++j) {
    for(int k=0; k<4;++k) {
      _u_a[0][j][k] = _u_b[j][k];
    }
  }
  scilab_rt_display_s0d3_("a",2,4,4,_u_a);

  for(int j=0; j<4;++j) {
    for(int k=0; k<4;++k) {
      _u_a[0][j][k] = _u_b[j][k];
    }
  }
  scilab_rt_display_s0d3_("a",2,4,4,_u_a);

  scilab_rt_terminate();
}

