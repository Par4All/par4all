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

  /*  t12.sce: array definition (see t205.sce) */
  /* a=[]; */
  int _u_b[1][2];
  _u_b[0][0]=1;
  _u_b[0][1]=2;
  int _u_c[2][3];
  _u_c[0][0]=1;
  _u_c[0][1]=2;
  _u_c[0][2]=3;
  _u_c[1][0]=4;
  _u_c[1][1]=5;
  _u_c[1][2]=6;
  int _u_e[1][2];
  _u_e[0][0]=(1*2);
  _u_e[0][1]=(2*3);

  scilab_rt_terminate();
}

