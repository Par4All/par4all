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

  /* testin clock */
  double _u_t[1][6];
  scilab_rt_clock__d2(1,6,_u_t);
  scilab_rt_display_s0d2_("t",1,6,_u_t);

  scilab_rt_terminate();
}

