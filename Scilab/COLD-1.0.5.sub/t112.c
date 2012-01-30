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

  double _u_a = scilab_rt_grand_i0i0s0d0d0_(1,1,"nor",2,1);
  scilab_rt_display_s0d0_("a",_u_a);
  double _u_b[2][3];
  scilab_rt_grand_i0i0s0d0d0_d2(2,3,"nor",2,1,2,3,_u_b);
  scilab_rt_display_s0d2_("b",2,3,_u_b);
  double _u_c[3][3];
  scilab_rt_grand_i0i0s0d0d0_d2(3,3,"nor",3,2,3,3,_u_c);
  scilab_rt_display_s0d2_("c",3,3,_u_c);

  scilab_rt_terminate();
}

