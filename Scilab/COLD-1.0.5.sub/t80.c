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

  /*  t80.sce: read/write from/to file */
  int _u_fd = scilab_rt_mopen_s0s0_("tmp/foo.txt","w");
  double _u_val1 = 56.214;
  double _u_val2 = 2.254;
  double _u_val3 = 7895.12;
  scilab_rt_mfprintf_i0s0d0_(_u_fd,"%f",_u_val1);
  scilab_rt_mfprintf_i0s0_(_u_fd,"\n");
  scilab_rt_mfprintf_i0s0d0_(_u_fd,"%f",_u_val2);
  scilab_rt_mfprintf_i0s0_(_u_fd,"\n");
  scilab_rt_mfprintf_i0s0d0_(_u_fd,"%f",_u_val3);
  scilab_rt_mclose_i0_(_u_fd);
  int _u_fd2 = scilab_rt_mopen_s0s0_("tmp/foo.txt","r");
  double _u_tmp = 2.35;
  while ( 1 ) {
    int _tmpxx0 = scilab_rt_meof_i0_(_u_fd2);
    if ((_tmpxx0!=0)) {
      break;
    }
    _u_tmp = scilab_rt_mfscanf_i0i0s0_(1,_u_fd2,"%lf");
    scilab_rt_disp_d0_(_u_tmp);
  }
  scilab_rt_mclose_i0_(_u_fd2);

  scilab_rt_terminate();
}

