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

  /*  t161.sce - aref */
  double _u_a[10][10];
  scilab_rt_ones_i0i0_d2(10,10,10,10,_u_a);

  for(int i=0; i<10;++i) {
    _u_a[i][0] = 0;
  }
  scilab_rt_display_s0d2_("a",10,10,_u_a);

  for(int i=0; i<10;++i) {
    _u_a[i][9] = 0;
  }
  scilab_rt_display_s0d2_("a",10,10,_u_a);

  for(int j=0; j<10;++j) {
    _u_a[0][j] = 0;
  }
  scilab_rt_display_s0d2_("a",10,10,_u_a);

  for(int j=0; j<10;++j) {
    _u_a[9][j] = 0;
  }
  scilab_rt_display_s0d2_("a",10,10,_u_a);
  
  double _u_b[1][10];
  for(int j=0; j<10; ++j) {
    _u_b[0][j] = _u_a[1][j];
  }
  scilab_rt_display_s0d2_("b",1,10,_u_b);

  scilab_rt_terminate();
}

