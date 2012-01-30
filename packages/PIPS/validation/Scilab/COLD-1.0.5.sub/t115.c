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

  /*  t145.sce - testing string op == and ~= */
  if (scilab_rt_eq_s0s0_("a","b")) {
    scilab_rt_disp_s0_("oops");
  }
  char* _u_a = (char *) malloc(2);
  strcpy(_u_a, "a");
  char* _u_b = (char *) malloc(2);
  strcpy(_u_b, "b");
  if (scilab_rt_eq_s0s0_(_u_a,_u_b)) {
    scilab_rt_disp_s0_("a = b _ wrong");
  }
  if (scilab_rt_ne_s0s0_(_u_a,_u_b)) {
    scilab_rt_disp_s0_("a != b _ ok");
  }
  free(_u_b);
  _u_b = (char *) malloc(2);
  strcpy(_u_b, "a");
  if (scilab_rt_eq_s0s0_(_u_a,_u_b)) {
    scilab_rt_disp_s0_("a = b _ ok");
  }
  if (scilab_rt_ne_s0s0_(_u_a,_u_b)) {
    scilab_rt_disp_s0_("a != b _ wrong");
  }
  int _u_true_eq = scilab_rt_eq_s0s0_(_u_a,_u_b);
  scilab_rt_display_s0i0_("true_eq",_u_true_eq);
  int _u_false_eq = scilab_rt_eq_s0s0_("foo","bar");
  scilab_rt_display_s0i0_("false_eq",_u_false_eq);
  int _u_false_ne = scilab_rt_ne_s0s0_(_u_a,_u_b);
  scilab_rt_display_s0i0_("false_ne",_u_false_ne);
  int _u_true_ne = scilab_rt_ne_s0s0_("foo","bar");
  scilab_rt_display_s0i0_("true_ne",_u_true_ne);

  scilab_rt_terminate();
}

