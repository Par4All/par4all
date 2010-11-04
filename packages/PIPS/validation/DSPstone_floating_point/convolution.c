#include "bench.h"
/*
 * benchmark program:   convolution.c
 * 
 * benchmark suite:     DSP-kernel
 * 
 * description:         convolution - filter benchmarking
 * 
 * reference code:      target assembly
 * 
 * f. verification:     none
 * 
 *  organization:        Aachen University of Technology - IS2 
 *                       DSP Tools Group
 *                       phone:  +49(241)807887 
 *                       fax:    +49(241)8888195
 *                       e-mail: zivojnov@ert.rwth-aachen.de 
 *
 * author:              Vojin Zivojnovic
 * 
 * history:             14-01-94 creation for fixed-point (Vojin Zivojnovic)
 *                      18-03-94 asm labels included (Martinez Velarde)
 *                      16-03-95 adaption for floating-point (Harald L. Schraut)
 *
 *                      $Author: schraut $
 *                      $Date: 1995/04/11 05:44:57 $
 *                      $Revision: 1.2 $
 */

#define STORAGE_CLASS register
#define TYPE  float
#define LENGTH 16

void pin_down(TYPE px[LENGTH], TYPE ph[LENGTH])
{
	STORAGE_CLASS int    i;

	for (i = 0; i < LENGTH; ++i) {
		px[i] = 1;
		ph[i] = 1;
	}

}


TYPE main()
{

	static TYPE     x[LENGTH];
	static TYPE     h[LENGTH];

	STORAGE_CLASS TYPE y;
	int i;
        

	pin_down(x, h);

	START_PROFILING;

	y = 0;

	for (i = 0; i < LENGTH; ++i)
		y += x[i] * h[LENGTH - 1 -i];

	END_PROFILING;


	return ((TYPE) y);

}


