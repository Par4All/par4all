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

  /*  t139.sce: testing conj function */
  double _u_a2[2][2];
  scilab_rt_ones_i0i0_d2(2,2,2,2,_u_a2);

  for(int j=0; j<2;++j) {
    _u_a2[1][j] = 10;
  }

  for(int i=0; i<2;++i) {
    _u_a2[i][1] = 20;
  }

  for(int i=0; i<2;++i) {
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
  double _tmp0[2][2];
  scilab_rt_conj_d2_d2(2,2,_u_a2,2,2,_tmp0);
  scilab_rt_display_s0d2_("ans",2,2,_tmp0);
  double _tmp1[3][2][3];
  scilab_rt_conj_d3_d3(3,2,3,_u_a3,3,2,3,_tmp1);
  scilab_rt_display_s0d3_("ans",3,2,3,_tmp1);

  scilab_rt_terminate();
}

