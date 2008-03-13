 /* package contrainte */

/*LINTLIBRARY*/

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
	
/* void contrainte_error(va_dcl va_alist) should be called to terminate
 * execution and to core dump when data structures are corrupted or when
 * an undefined operation is requested (zero divide for instance).
 * CONTRAINTE_ERROR should be called as:
 * 
 *   CONTRAINTE_ERROR(function_name, format, expression-list)
 * 
 * where function_name is a string containing the name of the function
 * calling CONTRAINTE_ERROR, and where format and expression-list are passed as
 * arguments to vprintf. CONTRAINTE_ERROR terminates execution with abort.
 */
/*VARARGS0*/
void contrainte_error(char * name, char * fmt, ...)
{
    va_list args;

    va_start(args, fmt);

    /* print name of function causing error */
    (void) fprintf(stderr, "contrainte error in %s: ", name);

    /* print out remainder of message */
    (void) vfprintf(stderr, fmt, args);
    va_end(args);

    /* create a core file for debug */
    (void) abort();
}
