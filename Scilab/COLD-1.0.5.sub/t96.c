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

  /*  t96.sce: testing loop with negative increment */
  for (int _u_i=2; _u_i<=5; _u_i++) {
    scilab_rt_display_s0i0_("i",_u_i);
  }
  for (int _u_j=(-5); _u_j<=(-2); _u_j++) {
    scilab_rt_display_s0i0_("j",_u_j);
  }
  for (int _u_k=10; _u_k>=5; _u_k+=(-1)) {
    scilab_rt_display_s0i0_("k",_u_k);
  }

  scilab_rt_terminate();
}

