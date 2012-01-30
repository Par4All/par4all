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

double foo_i2_(int _u_tab_n0,int _u_tab_n1,int _u_tab[_u_tab_n0][_u_tab_n1])
{
  int _u_val = 0;
  for (int _u_i=1; _u_i<=5; _u_i++) {
    int _tmpxx1 = _u_tab[_u_i-1][0];
    if ((_tmpxx1>2)) {
      _u_val = (_u_val+1);
    }
  }
  scilab_rt_disp_s0_("val");
  scilab_rt_disp_i0_(_u_val);
  double _u_tmpTab[_u_val][1];
  scilab_rt_ones_i0i0_d2(_u_val,1,_u_val,1,_u_tmpTab);
  double _u_r;
  scilab_rt_sum_d2_d0(_u_val,1,_u_tmpTab,&_u_r);
  return _u_r;
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t273.sce _ fixed PR-163.sce */
  int _tmpxx0[5][2];
  _tmpxx0[0][0]=1;
  _tmpxx0[0][1]=1;
  _tmpxx0[1][0]=2;
  _tmpxx0[1][1]=2;
  _tmpxx0[2][0]=3;
  _tmpxx0[2][1]=3;
  _tmpxx0[3][0]=4;
  _tmpxx0[3][1]=4;
  _tmpxx0[4][0]=5;
  _tmpxx0[4][1]=5;
  double _tmp0 = foo_i2_(5,2,_tmpxx0);
  scilab_rt_display_s0d0_("ans",_tmp0);

  scilab_rt_terminate();
}

