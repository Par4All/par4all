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

  /* testing function datevec. For the moment, only admits scalars as input */
  int _u_a = 730011;
  double _u_b = 734621.52;
  double _u_c[1][6];
  scilab_rt_datevec_d0_d2(_u_a,1,6,_u_c);
  scilab_rt_display_s0d2_("c",1,6,_u_c);
  double _u_d[1][6];
  scilab_rt_datevec_d0_d2(_u_b,1,6,_u_d);
  scilab_rt_display_s0d2_("d",1,6,_u_d);

  scilab_rt_terminate();
}

