/*
 *  Scilab ( http://www.scilab.org/ ) - This file is part of Scilab
 *  Copyright (C) 2008-2008 - INRIA - Bruno JOFRET
 *
 *  This file must be used under the terms of the CeCILL.
 *  This source file is licensed as described in the file COPYING, which
 *  you should have received as part of this distribution.  The terms
 *  are also available at
 *  http://www.cecill.info/licences/Licence_CeCILL_V2-en.txt
 *
 */

#include "mean.h"
#include "sum.h"

void zrowmeana(doubleComplex *in, int lines, int columns, doubleComplex *out) {
  int i = 0;

  zrowsuma(in, lines, columns, out);
  for (i = 0; i < columns; ++i)
    {
      out[i] = zrdivs(out[i], DoubleComplex((double)lines, 0.0f));
    }
}
