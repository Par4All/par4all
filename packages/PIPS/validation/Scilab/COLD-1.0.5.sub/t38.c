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
    scilab_rt_disp_i0_(_u_i);
  }
  scilab_rt_disp_s0_("-----------------------");
  int _tmpfor0[1][3];
  _tmpfor0[0][0]=1;
  _tmpfor0[0][1]=2;
  _tmpfor0[0][2]=3;
  for (int __n0=0; __n0 <3; __n0++) {
    int _u_i=_tmpfor0[0][__n0];
    scilab_rt_disp_i0_(_u_i);
  }
  scilab_rt_disp_s0_("-----------------------");
  int _u_a = 2;
  int _u_i=_u_a;
  {
    scilab_rt_disp_i0_(_u_i);
  }

  scilab_rt_terminate();
}

