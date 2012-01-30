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

  /*  checking temp insertion inside for loops */
  for (int _u_i=1; _u_i<=10; _u_i++) {
    double _tmpxx0[10][10];
    scilab_rt_ones_i0i0_d2(10,10,10,10,_tmpxx0);
    double _u_a[10][10];
    scilab_rt_mul_d2i0_d2(10,10,_tmpxx0,_u_i,10,10,_u_a);
  }

  scilab_rt_terminate();
}

