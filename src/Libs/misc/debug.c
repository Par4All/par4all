/* Debugging functions
 *
 * Modifications:
 * - set_debug_level function has been changed.
 *   Mainly the condition test, from  (l >= 0 || l <= 9) to (l >= 0 && l <= 9)
 *   and else clause has been added, that is pips_error function.
 *   (Mar. 21,91  L.Zhou )
 * - debug_off(): idls-- was replaced by (--idls)-1
 *   (April 7, 1991, Francois Irigoin)
 */

#include <stdio.h>
extern int atoi(char *nptr);
extern char *getenv();
extern int fprintf();
extern int printf();
extern int vfprintf(FILE *stream, char *format, ...);
#include <varargs.h>
	
#include "genC.h"
#include "misc.h"

/* I switched this variable from static to non-static in order to
 * avoid a function call in the pips_debug macro. This does not mean
 * that this variable is intended to be touch by anyone directly.
 * the interface remains the set/get functions. Just to reduce the
 * debuging impact on the performance (from low to very low:-)
 * FC, May 95.
 */
int the_current_debug_level = 0;

/* 
SET_DEBUG_LEVEL is a function that sets the current debuging level to
the value passed as argument.  Valid debuging values are from 0 to 9.
0 means no debug, 9 means extensive debug. Other values are illegal.
*/
void set_debug_level(l)
int l;
{
    message_assert("debug level not in 0-9", l>=0 && l<=9);
    the_current_debug_level = l;
}

/* GET_DEBUG_LEVEL returns the current debugging level. */
int get_debug_level()
{
    return(the_current_debug_level);
}

#define STACK_LENGTH 50
static debug_level_stack[STACK_LENGTH];
static int idls = 0;

void debug_off()
{
    message_assert("empty debug level stack", idls > 0);

    if(idls>1)
	set_debug_level(debug_level_stack[(--idls)-1]);
    else
	set_debug_level(0);
}

void debug_on(env)
char *env;
{
    int dl;
    char *debug_level;

    pips_assert("debug_on", idls < STACK_LENGTH-1);

    dl = ((debug_level = getenv(env)) != NULL) ? atoi(debug_level) : 0;

    set_debug_level(debug_level_stack[idls++] = dl);
}

/* function used to debug (can be called from dbx)
 * BB 6.12.91
 */
void print_debug_stack()
{
    int i;

    (void) printf("Debug stack (last debug_on first): ");
    for(i=idls-1;i>=0;i--) {
	(void) printf("%d ",debug_level_stack[i]);
    }
}

/*
DEBUG prints debuging messages. DEBUG should be called as:
     debug(debug_level, function_name, format, expression_list)
DEBUG prints a message if the value of DEBUG_LEVEL is greater or equal
to the current debuging level (see SET_DEBUG_LEVEL).  function_name is a
string containing the name of the function calling DEBUG, and format and
expression_list are passed as arguments to vprintf.
*/
/*VARARGS0*/
void debug(va_alist)
va_dcl
{
    va_list args;
    char *fmt;
    int l;

    if (the_current_debug_level == 0)
	    return;

    va_start(args);

    /* get wanted debuging level */
    l = va_arg(args, int);

    if (l > the_current_debug_level)
	    return;

    /* print name of function printing debug message */
    (void) fprintf(stderr, "[%s] ", va_arg(args, char *));
    fmt = va_arg(args, char *);

    /* print out remainder of message */
    (void) vfprintf(stderr, fmt, args);

    va_end(args);
}

/* pips_debug is a nice macro that depends on gcc to generate the
 * function name and to handle a variable number of arguments.
 * if these feature are not available, it will be this function.
 * the function name won't be available. FC May 95.
 */
void pips_debug_function(va_alist)
va_dcl
{
    va_list args;
    char *fmt;
    int l;

    if (the_current_debug_level == 0)
	    return;

    va_start(args);
    l = va_arg(args, int);
    if (l > the_current_debug_level) return;

    (void) fprintf(stderr, "[unknown] ");
    fmt = va_arg(args, char *);
    (void) vfprintf(stderr, fmt, args);

    va_end(args);
}

/* is that all? :-)
 */
