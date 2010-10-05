/*
 *  Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
 *  Copyright (C) 2007-2008 - INRIA - Bruno JOFRET
 *
 *  This file must be used under the terms of the CeCILL.
 *  This source file is licensed as described in the file COPYING, which
 *  you should have received as part of this distribution.  The terms
 *  are also available at
 *  http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
 *
 */

#include <math.h>
#include "implicitList.h"

void dimplicitLists(float start, float step, float end, float *out)
{
  int i = 0;
  int iNbElements = 0;
  if (start <= end)
    {
      if (start < start + step)
	{
	  iNbElements = (int)(floor((end - start) / step) + 1);
	  out[0] = start;
	}
    }
  else
    {
      if (start > start + step)
	{
	  iNbElements = (int)(floor((start - end) / step) + 1);
	  out[0] = start;
	}
    }

  for (i = 1 ; i < iNbElements ; ++i)
    {
      start += step;
      out[i] = start;
    }
}
