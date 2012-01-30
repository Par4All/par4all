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

  int _u_x = 1;
  while ( (_u_x<10) ) {
    _u_x = (_u_x+1);
  }
  scilab_rt_display_s0i0_("x",_u_x);

  scilab_rt_terminate();
}

