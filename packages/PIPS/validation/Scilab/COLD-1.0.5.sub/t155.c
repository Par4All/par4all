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

  int _u_a[1][100];
  for(int __tri0=0;__tri0 < 100;__tri0++) {
    _u_a[0][__tri0] = 1+__tri0*1;
  }
  int _u_k = 10;
  scilab_rt_disp_i0_(_u_k);
  for (int _u_i=1; _u_i<=_u_k; _u_i++) {
    scilab_rt_disp_i0_(_u_a[0][(_u_i-1)]);
    int _u_b = 1;
  }

  scilab_rt_terminate();
}

