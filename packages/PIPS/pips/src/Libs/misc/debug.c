/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
    #include "pips_config.h"
#endif
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
void set_debug_level(const int l)
{
    message_assert("debug level not in 0-9", l>=0 && l<=9);
    the_current_debug_level = l;
}

/* GET_DEBUG_LEVEL returns the current debugging level. */
int get_debug_level(void)
{
    return(the_current_debug_level);
}

#define STACK_LENGTH 50

typedef struct 
{
    char * name;
    char * function;
    char * file;
    int line;
    int level;
} debug_level ;

/* idls points to the first free bucket in debug_stack
 */
static debug_level debug_stack[STACK_LENGTH];
static _int idls = 0;

/* The pair get_ and set_debug_stack_pointer() should never be used
except to clean up the stack after a long jump */

_int get_debug_stack_pointer(void)
{
    return idls;
}

void set_debug_stack_pointer(const int i)
{
    if(i >= 0 && i <= idls) {
	if (i!=idls) {
	    user_warning("set_debug_stack_pointer",
			 "debug level stack is set to %d\n", i);
	    idls = i;
	    set_debug_level(idls>1 ? debug_stack[idls-1].level : 0);
	}
    }
    else
	pips_internal_error("value %d out of stack range [0..%d]. "
			    "Too many calls to debug_off()\n", i, idls);
}

void 
debug_on_function(
    const char * env,
    const char * function,
    const char * file,
    const int line)
{
    int dl;
    char * level_env;

    pips_assert("stack not full", idls < STACK_LENGTH-1);
    dl = ((level_env = getenv(env)) != NULL) ? atoi(level_env) : 0;

    debug_stack[idls].name = (char*) env;
    debug_stack[idls].function = (char*) function;
    debug_stack[idls].file = (char*) file;
    debug_stack[idls].line = line;
    debug_stack[idls].level = dl;

    idls++;

    set_debug_level(dl);
}

void 
debug_off_function(
    const char * function,
    const char * file,
    const int line)
{
    debug_level *current;

    pips_assert("debug stack not empty", idls > 0);

    idls--;
    current = &debug_stack[idls];

    if (!same_string_p(current->file, file) ||
	!same_string_p(current->function, function))
    {
	pips_internal_error("\ndebug %s (level is %d)"
			    "[%s] (%s:%d) debug on and\n"
			    "[%s] (%s:%d) debug off don't match\n",
			    current->name, current->level,
			    current->function, current->file, current->line,
			    function, file, line);
    }

    set_debug_level(idls>0 ? debug_stack[idls-1].level : 0);
}

/* function used to debug (can be called from dbx)
 * BB 6.12.91
 */
void print_debug_stack(void)
{
    int i;

    (void) fprintf(stderr, "Debug stack (last debug_on first): ");

    for(i=idls-1;i>=0;i--)
	(void) fprintf(stderr, "%s=%d [%s] (%s:%d)\n",
		       debug_stack[i].name,
		       debug_stack[i].level,
		       debug_stack[i].function,
		       debug_stack[i].file,
		       debug_stack[i].line);
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
debug(const int the_expected_debug_level,
      const char * calling_function_name,
      const char * a_message_format,
      ...)
{
   va_list some_arguments;
#define MAX_MARGIN (8)
   static char * margin = "        ";
   /* margin_length is the length of the part of the margin that is not used */
   int margin_length;

   /* If the current debug level is not high enough, do nothing: */
   if (the_expected_debug_level > the_current_debug_level)
      return;

   /* print name of function printing debug message */
   margin_length = MAX_MARGIN+1-the_expected_debug_level;
   (void) fprintf(stderr, "%s[%s] ",
		  margin + (margin_length>0 ? margin_length : 0),
		  calling_function_name);

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
pips_debug_function(const int the_expected_debug_level,
                    const char * a_message_format,
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

double get_process_memory_size(void)
{
    /* This is about the always increasing swap space */
    /* etext is not portable. it is not even documented on SUN:-) */
    /* extern etext; 
    double memory_size = (sbrk(0) - etext)/(double)(1 << 20);
    */
    return 0.0;
}

double get_process_gross_heap_size(void)
{
    /* mallinfo is not portable */
    /* This is *used* part of the heap, but it may be bigger */
    /* struct mallinfo heap_info = mallinfo();  */
    /* double memory_size = (heap_info.uordbytes)/(double)(1 << 20); */
    double memory_size = -1.0 /*(sbrk(0))/(double)(1 << 20)*/;
    return memory_size;
}

/* is that all? :-)
 */
