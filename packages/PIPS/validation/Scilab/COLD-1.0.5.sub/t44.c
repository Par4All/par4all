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

  for (int _u_i=1; _u_i<=10; _u_i++) {
    for (int _u_j=1; _u_j<=10; _u_j++) {
      scilab_rt_disp_i0_((_u_i+_u_j));
    }
  }

  scilab_rt_terminate();
}

