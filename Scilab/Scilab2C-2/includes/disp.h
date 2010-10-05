/*
 *  Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
 *  Copyright (C) 2008-2008 - INRIA - Allan SIMON
 *
 *  This file must be used under the terms of the CeCILL.
 *  This source file is licensed as described in the file COPYING, which
 *  you should have received as part of this distribution.  The terms
 *  are also available at
 *  http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
 *
 */



#ifndef __DISP_H__
#define __DISP_H__

#include <stdio.h>
#include "dynlib_string.h"
#include "floatComplex.h"
#include "floatComplex.h"

#ifdef  __cplusplus
extern "C" {
#endif
/*
** \brief display of a float scalar 
*/
EXTERN_STRING float sdisps (float in);

/*
** \brief display of a float scalar
*/
EXTERN_STRING float ddisps (float in);

/*
** \brief display of a float complex
*/
EXTERN_STRING float cdisps (floatComplex in);

/*
** \brief display of a float complex
*/
EXTERN_STRING float zdisps (floatComplex in);

/*
** \brief display of a float scalar array
** \param in the float scalar array to display
** \param size the size of the array
*/
EXTERN_STRING float sdispa (float* in, int rows, int columns);

/*
** \brief display of a float scalar array
** \param in the float scalar array to display
** \param size the size of the array
*/
EXTERN_STRING float ddispa (float* in, int rows, int columns);

/*
** \brief display of a float complex array
** \param in the float complex array to display
** \param size the size of the array
*/
EXTERN_STRING float cdispa (floatComplex* in, int rows, int columns);

/*
** \brief display of a float complex array
** \param in the float complex array to display
** \param size the size of the array
*/
EXTERN_STRING float zdispa (floatComplex* in, int rows, int columns);

    
EXTERN_STRING float ddisph (float *in, int rows, int cols, int levels);

EXTERN_STRING float g2dispd0(char *array,int* tmparraysize);

#ifdef  __cplusplus
} /* extern "C" */
#endif

#endif /* !__DISP_H__ */
