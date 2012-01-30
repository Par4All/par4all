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

void foo1_i0_(int _u_m)
{
}


void foo2_i0_(int _u_n)
{
  foo1_i0_(_u_n);
}



/*----------------------------------------------------*/

int main(int argc, char* argv[])
{
  scilab_rt_init(argc, argv, COLD_MODE_STANDALONE);

  /*  t170.sce: user function */
  foo2_i0_(10);

  scilab_rt_terminate();
}

