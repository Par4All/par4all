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

void foo_i2_d2(int _u_b_n0,int _u_b_n1,int _u_b[_u_b_n0][_u_b_n1], int _u_a_n0,int _u_a_n1,double _u_a[_u_a_n0][_u_a_n1])
{
  double _u_d = 1;
  _u_d = 1.;
  
  scilab_rt_cos_i2_d2(_u_b_n0,_u_b_n1,_u_b,_u_a_n0,_u_a_n1,_u_a);
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t294.sce _ Fixed bug when multiple inference loop iteration in function */
  int _u_a[1][3];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[0][2]=3;
  scilab_rt_display_s0i2_("a",1,3,_u_a);
  double _u_c[1][3];
  foo_i2_d2(1,3,_u_a,1,3,_u_c);
  scilab_rt_display_s0d2_("c",1,3,_u_c);

  scilab_rt_terminate();
}

