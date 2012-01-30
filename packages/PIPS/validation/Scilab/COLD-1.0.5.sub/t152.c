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

  scilab_rt_lines_d0_(1);
  scilab_rt_lines_d0_(6.1);
  scilab_rt_lines_d0d0_(1,2);
  scilab_rt_lines_d0d0_(1,2.1);
  scilab_rt_lines_d0d0_(1.1,2);
  scilab_rt_lines_d0d0_(1.1,2.1);

  scilab_rt_terminate();
}

