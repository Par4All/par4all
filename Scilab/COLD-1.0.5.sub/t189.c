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

  double _u_a[1][32];
  for(int __tri0=0;__tri0 < 32;__tri0++) {
    _u_a[0][__tri0] = 0+__tri0*0.1;
  }
  scilab_rt_display_s0d2_("a",1,32,_u_a);

  scilab_rt_terminate();
}

