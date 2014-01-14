/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

#include "arithmetique.h"
#include "assert.h"

/* it seems rather stupid to trap arithmetic errors on abs... FC.
 */

Value abs_ofl_ctrl(Value i, int ofl_ctrl)
{
    
    if ((ofl_ctrl == 1) && value_eq(i,VALUE_MIN))
	THROW(overflow_error);
        
    assert(value_ne(i,VALUE_MIN));
    return value_pos_p(i)? i: value_uminus(i);
}

/* int absval_ofl(int i): absolute value of i (SUN version)
 */
Value absval_ofl(Value i)
{
    return abs_ofl_ctrl(i, 1);
}


/* int absval(int i): absolute value of i (SUN version)
 */
Value absval(Value i)
{
    return abs_ofl_ctrl(i, 0);
}
