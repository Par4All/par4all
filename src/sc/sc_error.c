 /* package sc */

#include <stdio.h>
#include <varargs.h>
	
/* void sc_error(va_dcl va_alist) should be called to terminate execution
 * and to core dump when data structures are corrupted or when an undefined
 * operation is requested (zero divide for instance).
 * SC_ERROR should be called as:
 * 
 *   SC_ERROR(function_name, format, expression-list)
 * 
 * where function_name is a string containing the name of the function
 * calling SC_ERROR, and where format and expression-list are passed as
 * arguments to vprintf. SC_ERROR terminates execution with abort.
 */
void sc_error(va_alist)
va_dcl
{
    va_list args;
    char *fmt;

    va_start(args);

    /* print name of function causing error */
    (void) fprintf(stderr, "sc error in %s: ", va_arg(args, char *));
    fmt = va_arg(args, char *);

    /* print out remainder of message */
    (void) vfprintf(stderr, fmt, args);
    va_end(args);

    /* create a core file for debug */
    (void) abort();
}
