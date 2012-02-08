/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "arithmetique.h"
#include "assert.h"

/* Same as value sign */
int value_comparison(Value v1, Value v2)
{
  int cmp = 0;
  if(value_gt(v1, v2))
    cmp = 1;
  else if(value_gt(v2, v1))
    cmp = -1;
  return cmp;
}
