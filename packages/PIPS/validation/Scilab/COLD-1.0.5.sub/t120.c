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

  /*  t120.sce - testing aref */
  /* Access to 2D matrix */
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
  int _u_b = _u_a[((1-1) % 3)][((1-1) / 3)];
  scilab_rt_display_s0i0_("b",_u_b);
  int _u_c = _u_a[((9-1) % 3)][((9-1) / 3)];
  scilab_rt_display_s0i0_("c",_u_c);
  int _u_d = _u_a[((6-1) % 3)][((6-1) / 3)];
  scilab_rt_display_s0i0_("d",_u_d);
  int _tmp0 = _u_a[(((int)1.1-1) % 3)][(((int)1.1-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp0);
  int _tmp1 = _u_a[(((int)2.-1) % 3)][(((int)2.-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp1);
  int _tmp2 = _u_a[(((int)3.-1) % 3)][(((int)3.-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp2);
  int _tmp3 = _u_a[(((int)4.-1) % 3)][(((int)4.-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp3);
  int _tmp4 = _u_a[(((int)5.-1) % 3)][(((int)5.-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp4);
  int _tmp5 = _u_a[(((int)6.-1) % 3)][(((int)6.-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp5);
  int _tmp6 = _u_a[(((int)7.-1) % 3)][(((int)7.-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp6);
  int _tmp7 = _u_a[(((int)8.-1) % 3)][(((int)8.-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp7);
  int _tmp8 = _u_a[(((int)9.-1) % 3)][(((int)9.-1) / 3)];
  scilab_rt_display_s0i0_("ans",_tmp8);
  /* Access to 1D matrix */
  int _u_l[1][6];
  _u_l[0][0]=1;
  _u_l[0][1]=2;
  _u_l[0][2]=3;
  _u_l[0][3]=4;
  _u_l[0][4]=5;
  _u_l[0][5]=6;
  scilab_rt_display_s0i2_("l",1,6,_u_l);
  int _tmp9 = _u_l[0][((int)4.-1)];
  scilab_rt_display_s0i0_("ans",_tmp9);
  int _tmp10 = _u_l[0][(4-1)];
  scilab_rt_display_s0i0_("ans",_tmp10);
  int _u_lt[6][1];
  _u_lt[0][0]=1;
  _u_lt[1][0]=2;
  _u_lt[2][0]=3;
  _u_lt[3][0]=4;
  _u_lt[4][0]=5;
  _u_lt[5][0]=6;
  scilab_rt_display_s0i2_("lt",6,1,_u_lt);
  int _tmp11 = _u_lt[((int)5.-1)][0];
  scilab_rt_display_s0i0_("ans",_tmp11);
  int _tmp12 = _u_lt[(5-1)][0];
  scilab_rt_display_s0i0_("ans",_tmp12);

  scilab_rt_terminate();
}

