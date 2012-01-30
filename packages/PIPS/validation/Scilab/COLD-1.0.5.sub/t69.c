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

  /*  t69.sce: testing power function */
  int _u_a[1][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  /* b = a^2 */
  int _u_c[1][3];
  scilab_rt_eltpow_i2i0_i2(1,3,_u_a,3,1,3,_u_c);
  scilab_rt_display_s0i2_("c",1,3,_u_c);
  int _u_d[3][3];
  _u_d[0][0]=1;
  _u_d[0][1]=2;
  _u_d[0][2]=3;
  _u_d[1][0]=3;
  _u_d[1][1]=2;
  _u_d[1][2]=1;
  _u_d[2][0]=1;
  _u_d[2][1]=2;
  _u_d[2][2]=3;
  int _u_e[3][3];
  scilab_rt_eltpow_i2i0_i2(3,3,_u_d,3,3,3,_u_e);
  scilab_rt_display_s0i2_("e",3,3,_u_e);
  int _u_f[1][3];
  scilab_rt_pow_i0i2_i2(2,1,3,_u_a,1,3,_u_f);
  scilab_rt_display_s0i2_("f",1,3,_u_f);
  double _u_g[1][3];
  scilab_rt_pow_d0i2_d2(2.,1,3,_u_a,1,3,_u_g);
  scilab_rt_display_s0d2_("g",1,3,_u_g);
  int _u_h[1][3];
  _u_h[0][0]=4;
  _u_h[0][1]=5;
  _u_h[0][2]=6;
  int _u_i[3][3];
  _u_i[0][0]=4;
  _u_i[0][1]=5;
  _u_i[0][2]=6;
  _u_i[1][0]=4;
  _u_i[1][1]=5;
  _u_i[1][2]=6;
  _u_i[2][0]=4;
  _u_i[2][1]=5;
  _u_i[2][2]=6;
  /* j = a^h */
  int _u_k[1][3];
  scilab_rt_eltpow_i2i2_i2(1,3,_u_a,1,3,_u_h,1,3,_u_k);
  scilab_rt_display_s0i2_("k",1,3,_u_k);
  /* l = d^i */
  int _u_m[3][3];
  scilab_rt_eltpow_i2i2_i2(3,3,_u_d,3,3,_u_i,3,3,_u_m);
  scilab_rt_display_s0i2_("m",3,3,_u_m);
  double _u_n[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_n);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_n[1][j][k] = 1;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_n[i][1][k] = 2;
    }
  }

  for(int i=0; i<3;++i) {
    _u_n[i][1][2] = 3;
  }
  double _u_o[3][2][3];
  scilab_rt_eltpow_d3i0_d3(3,2,3,_u_n,2,3,2,3,_u_o);
  scilab_rt_display_s0d3_("o",3,2,3,_u_o);
  double _u_p[3][2][3];
  scilab_rt_eltpow_d3d3_d3(3,2,3,_u_n,3,2,3,_u_n,3,2,3,_u_p);
  scilab_rt_display_s0d3_("p",3,2,3,_u_p);
  /* q = d^d */
  /* r = a^h */

  scilab_rt_terminate();
}

