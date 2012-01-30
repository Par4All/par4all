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

  char* _u_a = (char *) malloc(5);
  strcpy(_u_a, "titi");
  scilab_rt_display_s0s0_("a",_u_a);
  double _u_b[1][1];
  _u_b[0][0]=SCILAB_PI;
  scilab_rt_display_s0d2_("b",1,1,_u_b);
  char* _u_c[1][1];
  _u_c[0][0]="tutu";
  scilab_rt_display_s0s2_("c",1,1,_u_c);
  char* _u_d[2][3];
  _u_d[0][0]="tutu";
  _u_d[0][1]="titi";
  _u_d[0][2]="tata";
  _u_d[1][0]="foo";
  _u_d[1][1]="bar";
  _u_d[1][2]="sux";
  scilab_rt_display_s0s2_("d",2,3,_u_d);
  scilab_rt_disp_s0_("Array of string : ");
  scilab_rt_disp_s2_(2,3,_u_d);

  scilab_rt_terminate();
}

