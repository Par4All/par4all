 /* package arithmetique */

/*LINTLIBRARY*/

#include <stdio.h>
/* #include <values.h> */
#include <limits.h>
#include <setjmp.h>

#include "arithmetique.h"
#include "assert.h"

/* int absval(int i): absolute value of i (SUN version)
 */
int absval(i)
int	i;
{
    return( abs_ofl_ctrl(i, 0));
}

/* int absval_ofl(int i): absolute value of i (SUN version)
 * Overflow control is made and returns to the last setjmp(overflow_error). 
 */
int absval_ofl(i)
int	i;
{
    return( abs_ofl_ctrl(i, 1));

}


int abs_ofl_ctrl(i, ofl_ctrl)
int i;
int ofl_ctrl;
{

    extern jmp_buf overflow_error;
    
    if ((ofl_ctrl == 1) && (i == INT_MIN))   
	longjmp(overflow_error, 5);
        
    assert(i != INT_MIN);
    return (i>0) ? i: -i;    
    
}
