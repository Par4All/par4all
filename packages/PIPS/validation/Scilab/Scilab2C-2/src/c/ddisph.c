/*
 *  Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
 *  Copyright (C) 2010-2010 - DIGITEO - Bruno JOFRET
 *
 *  This file must be used under the terms of the CeCILL.
 *  This source file is licensed as described in the file COPYING, which
 *  you should have received as part of this distribution.  The terms
 *  are also available at
 *  http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
 *
 */

#include "disp.h"

double ddisph (double* in, int rows, int columns, int levels){
	int i = 0,j = 0,k = 0;
	
    for (k = 0; k < levels; ++k)
    { 
        printf("(:, :, %d)\n", k + 1);
        for (i = 0; i < rows; ++i)
        {
            for (j=0;j<columns;j++)
            {
                printf ("  %1.20f  ", in[i+j*rows+k*columns*rows]);
            }
            printf("\n");
        }
		printf("\n");
	}
	return 0;
}
