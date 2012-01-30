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

  int _u_a[9][1];
  _u_a[0][0]=1;
  _u_a[1][0]=2;
  _u_a[2][0]=3;
  _u_a[3][0]=4;
  _u_a[4][0]=5;
  _u_a[5][0]=6;
  _u_a[6][0]=7;
  _u_a[7][0]=8;
  _u_a[8][0]=9;
  
  int _u_tmpa[1][1];
  _u_tmpa[0][0] = _u_a[1][0];
  scilab_rt_display_s0i2_("tmpa",1,1,_u_tmpa);
  int _u_b[1][9];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  _u_b[0][2]=3;
  _u_b[0][3]=4;
  _u_b[0][4]=5;
  _u_b[0][5]=6;
  _u_b[0][6]=7;
  _u_b[0][7]=8;
  _u_b[0][8]=9;
  
  int _u_tmpb[1][1];
  _u_tmpb[0][0] = _u_b[0][4];
  scilab_rt_display_s0i2_("tmpb",1,1,_u_tmpb);

  scilab_rt_terminate();
}

