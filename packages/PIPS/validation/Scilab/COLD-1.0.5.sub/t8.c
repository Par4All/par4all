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

  char* _u_a = (char *) malloc(4);
  strcpy(_u_a, "foo");
  char* _u_b = (char *) malloc(4);
  strcpy(_u_b, "foo");

  scilab_rt_terminate();
}

