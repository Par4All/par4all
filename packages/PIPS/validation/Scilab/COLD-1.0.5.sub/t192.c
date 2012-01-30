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

  /*  t192.sce: using array as index in aref */
  int _u_a[3][3];
  _u_a[0][0]=2;
  _u_a[0][1]=1;
  _u_a[0][2]=1;
  _u_a[1][0]=1;
  _u_a[1][1]=3;
  _u_a[1][2]=1;
  _u_a[2][0]=1;
  _u_a[2][1]=1;
  _u_a[2][2]=4;
  int _u_key[1][3];
  _u_key[0][0]=2;
  _u_key[0][1]=3;
  _u_key[0][2]=1;
  
  int _u_b[3][3];
  for(int i=0; i<3; ++i) {
    for(int j=0; j<3; ++j) {
      _u_b[i][j] = _u_a[i][_u_key[0][j]-1];
    }
  }
  scilab_rt_display_s0i2_("b",3,3,_u_b);

  scilab_rt_terminate();
}

