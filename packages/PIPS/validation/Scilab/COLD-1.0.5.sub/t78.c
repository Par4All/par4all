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

  int _u_a[1][3];
  _u_a[0][0]=0;
  _u_a[0][1]=(8-1);
  _u_a[0][2]=0;
  scilab_rt_display_s0i2_("a",1,3,_u_a);
  int _u_b[1][3];
  _u_b[0][0]=0;
  _u_b[0][1]=(8-1);
  _u_b[0][2]=0;
  scilab_rt_display_s0i2_("b",1,3,_u_b);

  scilab_rt_terminate();
}

