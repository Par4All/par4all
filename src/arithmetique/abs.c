/* $Id$ */

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
