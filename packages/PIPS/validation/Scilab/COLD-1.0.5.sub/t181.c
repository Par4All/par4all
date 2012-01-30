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

  /*  t.sce: cumsum function */
  int _u_a[3][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  _u_a[1][0]=4;
  _u_a[1][1]=5;
  _u_a[1][2]=6;
  _u_a[2][0]=7;
  _u_a[2][1]=8;
  _u_a[2][2]=9;
  scilab_rt_display_s0i2_("a",3,3,_u_a);
  int _u_aCumsum[3][3];
  scilab_rt_cumsum_i2_i2(3,3,_u_a,3,3,_u_aCumsum);
  scilab_rt_display_s0i2_("aCumsum",3,3,_u_aCumsum);
  int _u_aCumsumR[3][3];
  scilab_rt_cumsum_i2s0_i2(3,3,_u_a,"r",3,3,_u_aCumsumR);
  scilab_rt_display_s0i2_("aCumsumR",3,3,_u_aCumsumR);
  int _u_aCumsumC[3][3];
  scilab_rt_cumsum_i2s0_i2(3,3,_u_a,"c",3,3,_u_aCumsumC);
  scilab_rt_display_s0i2_("aCumsumC",3,3,_u_aCumsumC);
  int _u_b[3][4];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  _u_b[0][2]=3;
  _u_b[0][3]=3;
  _u_b[1][0]=4;
  _u_b[1][1]=5;
  _u_b[1][2]=6;
  _u_b[1][3]=6;
  _u_b[2][0]=7;
  _u_b[2][1]=8;
  _u_b[2][2]=9;
  _u_b[2][3]=9;
  scilab_rt_display_s0i2_("b",3,4,_u_b);
  int _u_bCumsum[3][4];
  scilab_rt_cumsum_i2_i2(3,4,_u_b,3,4,_u_bCumsum);
  scilab_rt_display_s0i2_("bCumsum",3,4,_u_bCumsum);
  int _u_bCumsumR[3][4];
  scilab_rt_cumsum_i2s0_i2(3,4,_u_b,"r",3,4,_u_bCumsumR);
  scilab_rt_display_s0i2_("bCumsumR",3,4,_u_bCumsumR);
  int _u_bCumsumC[3][4];
  scilab_rt_cumsum_i2s0_i2(3,4,_u_b,"c",3,4,_u_bCumsumC);
  scilab_rt_display_s0i2_("bCumsumC",3,4,_u_bCumsumC);
  double _u_c[3][4];
  _u_c[0][0]=1;
  _u_c[0][1]=2;
  _u_c[0][2]=3;
  _u_c[0][3]=3;
  _u_c[1][0]=4;
  _u_c[1][1]=5.1;
  _u_c[1][2]=6;
  _u_c[1][3]=6;
  _u_c[2][0]=7;
  _u_c[2][1]=8.1;
  _u_c[2][2]=9;
  _u_c[2][3]=9;
  scilab_rt_display_s0d2_("c",3,4,_u_c);
  double _u_cCumsum[3][4];
  scilab_rt_cumsum_d2_d2(3,4,_u_c,3,4,_u_cCumsum);
  scilab_rt_display_s0d2_("cCumsum",3,4,_u_cCumsum);
  double _u_cCumsumR[3][4];
  scilab_rt_cumsum_d2s0_d2(3,4,_u_c,"r",3,4,_u_cCumsumR);
  scilab_rt_display_s0d2_("cCumsumR",3,4,_u_cCumsumR);
  double _u_cCumsumC[3][4];
  scilab_rt_cumsum_d2s0_d2(3,4,_u_c,"c",3,4,_u_cCumsumC);
  scilab_rt_display_s0d2_("cCumsumC",3,4,_u_cCumsumC);

  scilab_rt_terminate();
}

