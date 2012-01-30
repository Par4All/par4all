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

  /*  t162.sce - aref */
  int _u_n = 1;
  _u_n = 2;
  int _u_a[2][2];
  _u_a[0][0]=1;
  _u_a[0][1]=2;
  _u_a[1][0]=3;
  _u_a[1][1]=4;
  
  int _u_b[1][2];
  for(int j=0; j<2; ++j) {
    _u_b[0][j] = _u_a[_u_n-1][j];
  }
  scilab_rt_display_s0i2_("b",1,2,_u_b);
  
  int _u_c[2][1];
  for(int i=0; i<2; ++i) {
    _u_c[i][0] = _u_a[i][_u_n-1];
  }
  scilab_rt_display_s0i2_("c",2,1,_u_c);

  scilab_rt_terminate();
}

