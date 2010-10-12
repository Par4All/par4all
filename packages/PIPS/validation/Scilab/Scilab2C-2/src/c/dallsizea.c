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

#include <stdio.h>
#include <string.h>
#include "size.h"

double dallsizea(int *size, char *select)
{
    printf("** DEBUG ** select = [%s]\n", select);
    if (strcmp(select, "*"))
    {
        return size[0] * size[1];
    }
    if (strcmp(select, "r"))
    {
        return size[0];
    }
    if (strcmp(select, "c"))
    {
        return size[1];
    }

    return 0;
}
