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

  int _u_x = 1;
  scilab_rt_display_s0i0_("x",_u_x);
  /*  Comment with ... without \n after */
  int _u_aabb = 1;
  scilab_rt_display_s0i0_("aabb",_u_aabb);
  scilab_rt_disp_i0_(_u_aabb);
  char* _u_a = (char *) malloc(19);
  strcpy(_u_a, "this is a new text");
  scilab_rt_display_s0s0_("a",_u_a);

  scilab_rt_terminate();
}

