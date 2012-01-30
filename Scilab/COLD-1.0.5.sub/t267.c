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

  double _u_man_eps = 1.0;
  double _u_save = 0.0;
  while ( ((_u_man_eps+1)!=1) ) {
    _u_save = _u_man_eps;
    _u_man_eps = (_u_man_eps / 2);
    double _tmpxx0 = (_u_man_eps+1);
    if ((_tmpxx0==1)) {
      break;
    }
  }
  _u_man_eps = _u_save;
  scilab_rt_display_s0d0_("man_eps",_u_man_eps);
  scilab_rt_display_s0d0_("%eps",SCILAB_EPS);
  double _tmpxx1 = (_u_man_eps+1);
  if ((_tmpxx1!=1)) {
    scilab_rt_disp_s0_("manual eps + 1 =>  different of 1");
  } else { 
    scilab_rt_disp_s0_("manual eps + 1 =>  equal to 1");
  }
  double _tmpxx2 = (SCILAB_EPS+1);
  if ((_tmpxx2!=1)) {
    scilab_rt_disp_s0_("%eps + 1 ~= different of 1");
  } else { 
    scilab_rt_disp_s0_("%eps + 1 => equal to 1");
  }
  double _tmpxx3 = (_u_man_eps / 2);
  double _tmpxx4 = (_tmpxx3+1);
  if ((_tmpxx4!=1)) {
    scilab_rt_disp_s0_("manual eps/2 + 1 =>  different of 1");
  } else { 
    scilab_rt_disp_s0_("manual eps/2 + 1 =>  equal to 1");
  }

  scilab_rt_terminate();
}

