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

  /*  t150.sce: testing variance */
  double _u_x[2][3];
  _u_x[0][0]=0.2113249;
  _u_x[0][1]=0.0002211;
  _u_x[0][2]=0.6653811;
  _u_x[1][0]=0.7560439;
  _u_x[1][1]=0.4453586;
  _u_x[1][2]=0.6283918;
  double _u_a;
  scilab_rt_variance_d2_d0(2,3,_u_x,&_u_a);
  scilab_rt_display_s0d0_("a",_u_a);
  double _u_b[1][3];
  scilab_rt_variance_d2s0_d2(2,3,_u_x,"r",1,3,_u_b);
  scilab_rt_display_s0d2_("b",1,3,_u_b);
  double _u_c[2][1];
  scilab_rt_variance_d2s0_d2(2,3,_u_x,"c",2,1,_u_c);
  scilab_rt_display_s0d2_("c",2,1,_u_c);
  double _u_d[1][3];
  scilab_rt_variance_d2i0_d2(2,3,_u_x,1,1,3,_u_d);
  scilab_rt_display_s0d2_("d",1,3,_u_d);
  double _u_e[2][1];
  scilab_rt_variance_d2i0_d2(2,3,_u_x,2,2,1,_u_e);
  scilab_rt_display_s0d2_("e",2,1,_u_e);
  double _u_f[1][3];
  scilab_rt_variance_d2i0i0_d2(2,3,_u_x,1,0,1,3,_u_f);
  scilab_rt_display_s0d2_("f",1,3,_u_f);
  double _u_g[1][3];
  scilab_rt_variance_d2i0i0_d2(2,3,_u_x,1,1,1,3,_u_g);
  scilab_rt_display_s0d2_("g",1,3,_u_g);
  double _u_h[2][1];
  scilab_rt_variance_d2i0i0_d2(2,3,_u_x,2,0,2,1,_u_h);
  scilab_rt_display_s0d2_("h",2,1,_u_h);
  double _u_i[2][1];
  scilab_rt_variance_d2i0i0_d2(2,3,_u_x,2,1,2,1,_u_i);
  scilab_rt_display_s0d2_("i",2,1,_u_i);
  double _u_j[1][3];
  scilab_rt_variance_d2s0i0_d2(2,3,_u_x,"r",0,1,3,_u_j);
  scilab_rt_display_s0d2_("j",1,3,_u_j);
  double _u_k[1][3];
  scilab_rt_variance_d2s0i0_d2(2,3,_u_x,"r",1,1,3,_u_k);
  scilab_rt_display_s0d2_("k",1,3,_u_k);
  double _u_l[2][1];
  scilab_rt_variance_d2s0i0_d2(2,3,_u_x,"c",0,2,1,_u_l);
  scilab_rt_display_s0d2_("l",2,1,_u_l);
  double _u_m[2][1];
  scilab_rt_variance_d2s0i0_d2(2,3,_u_x,"c",1,2,1,_u_m);
  scilab_rt_display_s0d2_("m",2,1,_u_m);
  int _u_xi[2][3];
  _u_xi[0][0]=1;
  _u_xi[0][1]=2;
  _u_xi[0][2]=3;
  _u_xi[1][0]=4;
  _u_xi[1][1]=5;
  _u_xi[1][2]=6;
  double _u_n;
  scilab_rt_variance_i2_d0(2,3,_u_xi,&_u_n);
  scilab_rt_display_s0d0_("n",_u_n);
  double _u_o[1][3];
  scilab_rt_variance_i2s0i0_d2(2,3,_u_xi,"r",0,1,3,_u_o);
  scilab_rt_display_s0d2_("o",1,3,_u_o);
  double _u_x3[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_x3);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_x3[1][j][k] = 10;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_x3[i][1][k] = 20;
    }
  }

  for(int i=0; i<3;++i) {
    _u_x3[i][1][2] = 30;
  }
  double _u_p;
  scilab_rt_variance_d3_d0(3,2,3,_u_x3,&_u_p);
  scilab_rt_display_s0d0_("p",_u_p);
  double _u_x3d[3][2][3];
  scilab_rt_ones_i0i0i0_d3(3,2,3,3,2,3,_u_x3d);

  for(int j=0; j<2;++j) {
    for(int k=0; k<3;++k) {
      _u_x3[1][j][k] = 10.5;
    }
  }

  for(int i=0; i<3;++i) {
    for(int k=0; k<3;++k) {
      _u_x3[i][1][k] = 20.2;
    }
  }

  for(int i=0; i<3;++i) {
    _u_x3[i][1][2] = 30.9;
  }
  double _u_q;
  scilab_rt_variance_d3_d0(3,2,3,_u_x3d,&_u_q);
  scilab_rt_display_s0d0_("q",_u_q);

  scilab_rt_terminate();
}

