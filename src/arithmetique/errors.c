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

/*
  Exception management. See "arithmetic_errors.h".
*/

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "boolean.h"
#include "arithmetique.h"


/* This can be overrided in the PolyLib. And seems to be according to
   gdb who cannot find the corresponding source code. */
char const *  __attribute__ ((weak))
get_exception_name(const linear_exception_t exception)
{
  if (exception==overflow_error)
    return "overflow_error exception";
  if (exception==simplex_arithmetic_error)
    return "simplex_arithmetic_error exception";
  if (exception==user_exception_error)
    return "user_exception_error exception";
  if (exception==parser_exception_error)
    return "parser_exception_error exception";
  if (exception==timeout_error)
    return "timeout_error exception";
  if (exception==any_exception_error)
    return "all exceptions mask";

  return "unknown or mixed exception";
}

/* keep track of last thrown exception for RETHROW()
 */
/* This can be overrided in the PolyLib */
linear_exception_t the_last_just_thrown_exception __attribute__ ((weak)) = 0;

/* whether to run in debug mode (that is to trace catch/uncatch/throw)
 */
static bool linear_exception_debug_mode = false;
static unsigned int linear_exception_verbose = 1 | 2 | 16 ;

/* A structure for the exception stack.
 */
typedef struct 
{
  /* exception type.
   */
  int what;

  /* where to jump to.
   */
  jmp_buf where;

  /* location of the CATCH to be matched against the UNCATCH.
   */
  char const * function;
  char const * file;
  int          line;
} 
  linear_exception_holder;

/* exception stack.
   maximum extension.
-   current index (next available bucket)
 */
#define MAX_STACKED_CONTEXTS 64
static linear_exception_holder exception_stack[MAX_STACKED_CONTEXTS];
static int exception_index = 0;

/* callbacks...
 */
static exception_callback_t push_callback = NULL;
static exception_callback_t pop_callback = NULL;

/* This can be overrided in the PolyLib */
void __attribute__ ((weak)) set_exception_callbacks(exception_callback_t push, 
						    exception_callback_t pop)
{
  if (push_callback!=NULL || pop_callback!=NULL)
  {
    fprintf(stderr, "exception callbacks already defined! (%p, %p)\n",
	    push_callback, pop_callback);
    abort();
  }

  push_callback = push;
  pop_callback = pop;
}

/* total number of exceptions thrown, for statistics.
 */
/* This can be overrided in the PolyLib */
int linear_number_of_exception_thrown __attribute__ ((weak)) = 0;

/* dump stack
 */
/* This can be overrided in the PolyLib */
void __attribute__ ((weak)) dump_exception_stack_to_file(FILE * f)
{
  int i;
  fprintf(f, "[dump_exception_stack_to_file] size=%d\n", exception_index);
  for (i=0; i<exception_index; i++)
  {
    fprintf(f, 
	    "%d: [%s:%d in %s (%d)]\n",
	    i, 
	    exception_stack[i].file,
	    exception_stack[i].line,
	    exception_stack[i].function,
	    exception_stack[i].what);
  }
  fprintf(f, "\n");
}

/* This can be overrided in the PolyLib */
void __attribute__ ((weak)) dump_exception_stack()
{
  dump_exception_stack_to_file(stderr);
}

#define exception_debug_message(type) 				  \
    fprintf(stderr, "%s[%s:%d %s (%d)/%d]\n", 			  \
	    type, file, line, function, what, exception_index) 

#define exception_debug_trace(type) \
    if (linear_exception_debug_mode) { exception_debug_message(type); }


/* push a what exception on stack.
 */
/* This can be overrided in the PolyLib */
jmp_buf * __attribute__ ((weak))
push_exception_on_stack(
    const int what,
    const char * function,
    const char * file,
    const int line)
{
  exception_debug_trace("PUSH ");

  if (exception_index==MAX_STACKED_CONTEXTS)
  {
    exception_debug_message("push");
    fprintf(stderr, "exception stack overflow\n");
    dump_exception_stack();
    abort();
  }

  if (push_callback != NULL)
    push_callback(file, function, line);

  the_last_just_thrown_exception = 0;

  exception_stack[exception_index].what = what;
  exception_stack[exception_index].function = function;
  exception_stack[exception_index].file = file;
  exception_stack[exception_index].line = line;

  return & exception_stack[exception_index++].where;
}

#if !defined(same_string_p)
#define same_string_p(s1, s2) (strcmp((s1),(s2))==0)
#endif

/* pop a what exception.
   check for any mismatch!
 */
/* This can be overrided in the PolyLib */
void __attribute__ ((weak))
pop_exception_from_stack(
    const int what,
    const char * function,
    const char * file,
    const int line)
{  
  exception_debug_trace("POP  ");

  if (exception_index==0)
  {
    exception_debug_message("pop");
    fprintf(stderr, "exception stack underflow\n");
    dump_exception_stack();
    abort();
  }

  if (pop_callback) pop_callback(file, function, line);

  exception_index--;
  the_last_just_thrown_exception = 0;

  if ((exception_stack[exception_index].what != what) ||
      !same_string_p(exception_stack[exception_index].file, file) ||
      !same_string_p(exception_stack[exception_index].function, function))
  {
    exception_debug_message("pop");
    fprintf(stderr, 
	    "exception stack mismatch at depth=%d:\n"
	    "   CATCH: %s:%d in %s (%d)\n"
	    " UNCATCH: %s:%d in %s (%d)\n",
	    exception_index,
	    exception_stack[exception_index].file,
	    exception_stack[exception_index].line,
	    exception_stack[exception_index].function,
	    exception_stack[exception_index].what,
	    file, line, function, what);
    dump_exception_stack();
    abort();
  }
}

/* throws an exception of a given type by searching for 
   the specified 'what' in the current exception stack.
*/
/* This can be overridden in the PolyLib */
void __attribute__ ((weak)) throw_exception(const int what,
					    const char * function,
					    const char * file,
					    const int line)

{
  int i;
  
  exception_debug_trace("THROW");

  the_last_just_thrown_exception = what; /* for rethrow */

  for (i=exception_index-1; i>=0; i--)
  {
    if (pop_callback) 
      /* pop with push parameters! */
      pop_callback(exception_stack[i].file,
		   exception_stack[i].function,
		   exception_stack[i].line);

    if (exception_stack[i].what & what) 
    {
      exception_index = i;
      linear_number_of_exception_thrown++;

      if (linear_exception_debug_mode)
	fprintf(stderr, "---->[%s:%d %s (%d)/%d]\n", 
		exception_stack[i].file,
		exception_stack[i].line,
		exception_stack[i].function,
		exception_stack[i].what,
		i);
  
      /* trace some exceptions... */
      if (linear_exception_verbose & what)
	fprintf(stderr, "exception %d/%d: %s(%s:%d) -> %s(%s:%d)\n",
		what, exception_stack[i].what,
		function, file, line,
		exception_stack[i].function, 
		exception_stack[i].file,
		exception_stack[i].line);

      longjmp(exception_stack[i].where, 0);
    }
  }

  /* error. */
  exception_debug_message("throw");
  fprintf(stderr,
	  "exception not found in stack:\n"
	  "an exception was THROWN without a proper matching CATCH\n");
  dump_exception_stack();
  abort();
}

/* This can be overridden in the PolyLib */
void __attribute__ ((weak))
 linear_initialize_exception_stack(unsigned int verbose_exceptions,
				   exception_callback_t push, 
				   exception_callback_t pop)
{
  linear_exception_verbose = verbose_exceptions;
  set_exception_callbacks(push, pop);
}
