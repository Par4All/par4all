/*
  To test we have M_PI:

   If fails in C89 mode, the "gcc -E -ansi" PIPS used to use.

   Who cares? When this file is written, it's 2009. 20 years later
   afterwards... :-)

   Compatibility mode can be back anyway by using PIPS_CPP and
   PIPS_CPP_FLAGS environment variables, with PIPS_CPP="gcc -E -ansi"
*/

#include <math.h>

int
main(argc,argv)

int argc;
char *argv[];

{	return M_PI;
}
