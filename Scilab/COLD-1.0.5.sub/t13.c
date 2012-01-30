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

  int _tmpxx0 = (2*3);
  int _u_b[1][2];
  _u_b[0][0]=(1+_tmpxx0);
  _u_b[0][1]=4;
  scilab_rt_display_s0i2_("b",1,2,_u_b);

  scilab_rt_terminate();
}

