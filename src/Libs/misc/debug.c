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
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
	
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

/* The pair get_ and set_debug_stack_pointer() should never be used
except to clean up the stack fater a long jump */

int get_debug_stack_pointer()
{
    return idls;
}

void set_debug_stack_pointer(i)
{
    if(i >= 0 && i <= idls) {
	if (i!=idls) {
	    user_warning("set_debug_stack_pointer",
			 "debug level stack is set to %d\n", i);
	    idls = i;
	    if(idls>1) {
		set_debug_level(debug_level_stack[idls-1]);
	    }
	    else {
		set_debug_level(0);
	    }
	}
    }
    else
	pips_error("set_debug_stack_pointer", 
		   "value %d out of range [0..%d]\n", i, idls);
}

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
void
debug(int the_expected_debug_level,
      char * calling_function_name,
      char * a_message_format,
      ...)
{
   va_list some_arguments;

   /* If the current debug level is not high enough, do nothing: */
   if (the_expected_debug_level > the_current_debug_level)
      return;

   /* print name of function printing debug message */
   (void) fprintf(stderr, "[%s] ", calling_function_name);

   va_start(some_arguments, a_message_format);

   /* print out remainder of message */
   (void) vfprintf(stderr, a_message_format, some_arguments);

   va_end(some_arguments);
}

/* pips_debug is a nice macro that depends on gcc to generate the
 * function name and to handle a variable number of arguments.
 * if these feature are not available, it will be this function.
 * the function name won't be available. FC May 95.
 */
void
pips_debug_function(int the_expected_debug_level,
                    char * a_message_format,
                    ...)
{
   va_list some_arguments;

   /* If the current debug level is not high enough, do nothing: */
   if (the_expected_debug_level > the_current_debug_level)
      return;

   (void) fprintf(stderr, "[unknown] ");

   va_start(some_arguments, a_message_format);

   /* print out remainder of message */
   (void) vfprintf(stderr, a_message_format, some_arguments);

   va_end(some_arguments);
}

/* is that all? :-)
 */
