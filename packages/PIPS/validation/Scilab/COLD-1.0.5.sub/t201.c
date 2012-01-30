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

void setlhs_d2_d2(int _c_0_n0,int _c_0_n1,double _c_0[_c_0_n0][_c_0_n1], int _u_lhs_n0,int _u_lhs_n1,double _u_lhs[_u_lhs_n0][_u_lhs_n1])
{
  
  scilab_rt_assign_d2_d2(_u_lhs_n0,_u_lhs_n1,_c_0,_u_lhs_n0,_u_lhs_n1,_u_lhs);
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t201.sce: parser */
  double _u_x[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_x);
  double _u_lhs[10][10];
  setlhs_d2_d2(10,10,_u_x,10,10,_u_lhs);
  scilab_rt_display_s0d2_("lhs",10,10,_u_lhs);

  scilab_rt_terminate();
}

