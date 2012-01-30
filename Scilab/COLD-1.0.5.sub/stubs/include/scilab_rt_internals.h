/*----------------------------------------------------------- -*- H -*-
*
*  (c) HPC Project - 2010
*
*/

#ifndef __SCILAB_RT_INTERNALS_H__
#define __SCILAB_RT_INTERNALS_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "complex.h"


int vasprintf( char **sptr, char *fmt, va_list argv );

int asprintf( char **sptr, char *fmt, ... );


void _printJniVarName(char* name);

void _printJniReturn();

void _printJniVar_i0(int var);

void _printJniVar_d0(double var);

void _printJniVar_s0(char* var);

void _printJniVar_z0(double complex var);

#endif //__SCILAB_RT_INTERNALS_H__



