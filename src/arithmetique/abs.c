/* $RCSfile: abs.c,v $ (version $Revision$)
 * $Date: 1996/07/16 18:47:44 $, 
 */

#include <stdio.h>
/* #include <values.h> */
#include <limits.h>
#include <setjmp.h>

#include "arithmetique.h"
#include "assert.h"

Value abs_ofl_ctrl(Value i, int ofl_ctrl)
{
    extern jmp_buf overflow_error;
    
    if ((ofl_ctrl == 1) && (i == VALUE_MIN))   
	longjmp(overflow_error, 5);
        
    assert(i != VALUE_MIN);
    return VALUE_POS_P(i)? i: value_uminus(i);
}

/* int absval_ofl(int i): absolute value of i (SUN version)
 * Overflow control is made and returns to the last setjmp(overflow_error). 
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
