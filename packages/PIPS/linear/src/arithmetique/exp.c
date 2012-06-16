/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

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

 /* package arithmetique 
  */

/*LINTLIBRARY*/
/* no overflow is checked 
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>

#include "arithmetique.h"
#include "assert.h"

/* int exponentiate(x,n):  raise x to the power n
 * 
 * Precondition: 	n => 0
 */
Value exponentiate(Value x, int n)
{
    Value y;

    /* validation - n is positive 
     */
    assert(n >= 0);
    if (n == 0) return VALUE_ONE;

    /* FI: la complexite pourrait etre reduite de O(n) a O(log n) 
     */
    for(y=VALUE_ONE; n>0; n--)
	value_product(y,x);

    return y;
}
