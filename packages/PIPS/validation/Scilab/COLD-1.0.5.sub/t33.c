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

  int _u_a1[1][1];
  _u_a1[0][0]=(1-2);
  scilab_rt_display_s0i2_("a1",1,1,_u_a1);
  int _u_a2[1][1];
  _u_a2[0][0]=(1-2);
  scilab_rt_display_s0i2_("a2",1,1,_u_a2);
  int _u_a3[1][2];
  _u_a3[0][0]=1;
  _u_a3[0][1]=(-2);
  scilab_rt_display_s0i2_("a3",1,2,_u_a3);
  int _u_b1 = (1-2);
  scilab_rt_display_s0i0_("b1",_u_b1);
  int _u_b2 = (1-2);
  scilab_rt_display_s0i0_("b2",_u_b2);
  int _u_b3 = (1-2);
  scilab_rt_display_s0i0_("b3",_u_b3);

  scilab_rt_terminate();
}

