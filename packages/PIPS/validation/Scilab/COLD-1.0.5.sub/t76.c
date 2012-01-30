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

  /* t76.sce */
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
  double _u_b[3][3];
  scilab_rt_ones_i2_d2(3,3,_u_a,3,3,_u_b);
  scilab_rt_display_s0d2_("b",3,3,_u_b);
  char* _u_d[2][2];
  _u_d[0][0]="titi";
  _u_d[0][1]="tata";
  _u_d[1][0]="tutu";
  _u_d[1][1]="toto";
  double _u_e[2][2];
  scilab_rt_zeros_s2_d2(2,2,_u_d,2,2,_u_e);
  scilab_rt_display_s0d2_("e",2,2,_u_e);

  scilab_rt_terminate();
}

