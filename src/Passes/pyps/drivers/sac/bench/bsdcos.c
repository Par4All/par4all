/* This is the FreeBSD implementation of cosinus, from
 * svn.freebsd.org/viewvc/base/releng/8.0/lib/msun/src/k_cos.c?revision=198460&view=markup */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */


static const double
one =  1.00000000000000000000e+00, /* 0x3FF00000, 0x00000000 */
C1  =  4.16666666666666019037e-02, /* 0x3FA55555, 0x5555554C */
C2  = -1.38888888888741095749e-03, /* 0xBF56C16C, 0x16C15177 */
C3  =  2.48015872894767294178e-05, /* 0x3EFA01A0, 0x19CB1590 */
C4  = -2.75573143513906633035e-07, /* 0xBE927E4F, 0x809C52AD */
C5  =  2.08757232129817482790e-09, /* 0x3E21EE9E, 0xBDB4B1C4 */
C6  = -1.13596475577881948265e-11; /* 0xBDA8FAE9, 0xBE8838D4 */

/* x and y have been computed so that x+y = (initial argument) mod pi/2, with
 * x+y in [-pi/4,+pi/4], and y is the "tail" of x (i.e. x+y ~= x within the
 * precision of double numbers). */

double
__kernel_cos(double x, double y)
{
        double hz,z,r,w;

        z  = x*x;
        w  = z*z;
        r  = z*(C1+z*(C2+z*C3)) + w*w*(C4+z*(C5+z*C6));
        hz = 0.5*z;
        w  = one-hz;
        w = w + (((one-w)-hz) + (z*r-x*y));
	return w;
}

#include <stdio.h>

int main(int argc, char **argv)
{
	double x = -0.1, y = 0.1e-23;
	double z = 0;
	int i;
	for (i = 0; i < 2000000; i++)
		z += __kernel_cos(x, y);
	printf("%e\n", z);
	return 0;
}
