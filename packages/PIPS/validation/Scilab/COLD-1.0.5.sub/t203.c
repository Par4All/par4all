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

void setlhs_d2i0i0_d2(int _c_0_n0,int _c_0_n1,double _c_0[_c_0_n0][_c_0_n1], int _u_k, int _u_h, int _u_lhs_n0,int _u_lhs_n1,double _u_lhs[_u_lhs_n0][_u_lhs_n1])
{
  
  scilab_rt_assign_d2_d2(_u_lhs_n0,_u_lhs_n1,_c_0,_u_lhs_n0,_u_lhs_n1,_u_lhs);
  _u_lhs[0][0] = 1.0;
}


void newlhs_i0i0i0_d2(int _u_N, int _u_h, int _u_msiz, int _u_lhs_n0,int _u_lhs_n1,double _u_lhs[_u_lhs_n0][_u_lhs_n1])
{
  
  scilab_rt_zeros_i0i0_d2(_u_msiz,_u_msiz,_u_lhs_n0,_u_lhs_n1,_u_lhs);
  double _tmpxx0[_u_msiz][_u_msiz];
  setlhs_d2i0i0_d2(_u_lhs_n0,_u_lhs_n1,_u_lhs,1,_u_h,_u_msiz,_u_msiz,_tmpxx0);
  
  scilab_rt_assign_d2_d2(_u_msiz,_u_msiz,_tmpxx0,_u_lhs_n0,_u_lhs_n1,_u_lhs);
}




/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t203.sce - dyn alloc of arrays in function (from demo rlc) */
  /*     to help debug when checking array shapes in assign */
  double _tmp0[10][10];
  newlhs_i0i0i0_d2(1,300,10,10,10,_tmp0);
  scilab_rt_display_s0d2_("ans",10,10,_tmp0);

  scilab_rt_terminate();
}

