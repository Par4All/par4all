/*
  $Id: errors.c,v 1.2 2002/08/12 13:11:27 loechner Exp $

  Exception management.
  See "arithmetic_errors.h".

  $Log: errors.c,v $
  Revision 1.2  2002/08/12 13:11:27  loechner
  union ehrhart, first complete version

  Revision 1.1.1.1  2001/07/16 15:00:31  risset
  initial import into CVS

  Revision 1.16  2000/10/27 13:26:03  ancourt
  exception_thrown -> linear_number_of_exception_thrown

  Revision 1.15  2000/07/27 15:21:55  coelho
  message++

  Revision 1.14  2000/07/27 14:59:59  coelho
  trace added.

  Revision 1.13  2000/07/26 08:41:23  coelho
  the_last_just_thrown_exception management added.

  Revision 1.12  1998/10/26 18:48:34  coelho
  message++.

  Revision 1.11  1998/10/26 14:38:06  coelho
  constants back in.

  Revision 1.10  1998/10/26 14:35:47  coelho
  constants in .h.

  Revision 1.9  1998/10/24 15:36:22  coelho
  better exception error messages...

  Revision 1.8  1998/10/24 15:19:17  coelho
  more verbose throw...

  Revision 1.7  1998/10/24 14:31:13  coelho
  new stack implementation.
  checks for matching catch/uncatch.
  debug mode.

  Revision 1.6  1998/10/24 09:31:06  coelho
  RCS headers.
  'const' tried.

*/

#include <stdio.h>

#include "arithmetique.h"

/* global constants to designate exceptions.
   to be put in the type field bellow.
*/
unsigned int overflow_error = 1;
unsigned int simplex_arithmetic_error = 2;
unsigned int user_exception_error = 4;
unsigned int parser_exception_error = 8;
unsigned int any_exception_error = ~0;

/* keep track of last thrown exception for RETHROW()
 */
unsigned int the_last_just_thrown_exception = 0;

/* whether to run in debug mode (that is to trace catch/uncatch/throw)
 */
static int linear_exception_debug_mode = 0;
static int linear_exception_verbose_mode = 1;

/* A structure for the exception stack.
 */
typedef struct {
  /* exception type.
   */
  int what;

  /* where to jump to.
   */
  jmp_buf where;

  /* location of the CATCH to be matched against the UNCATCH.
   */
  char * function;
  char * file;
  int    line;
} 
  linear_exception_holder;

/* exception stack.
   maximum extension.
   current index (next available bucket)
 */
#define MAX_STACKED_CONTEXTS 50
static linear_exception_holder exception_stack[MAX_STACKED_CONTEXTS];
static int exception_index = 0;

/* total number of exceptions thrown, for statistics.
 */
int linear_number_of_exception_thrown = 0;

/* dump stack
 */
void dump_exception_stack_to_file(FILE * f)
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

void dump_exception_stack()
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
jmp_buf * 
push_exception_on_stack(
    int what,
    char * function,
    char * file,
    int line)
{
  exception_debug_trace("PUSH ");

  if (exception_index==MAX_STACKED_CONTEXTS)
  {
    exception_debug_message("push");
    fprintf(stderr, "exception stack overflow\n");
    dump_exception_stack();
    abort();
  }

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
void
pop_exception_from_stack(
    int what,
    char * function,
    char * file,
    int line)
{  
  exception_debug_trace("POP  ");

  if (exception_index==0)
  {
    exception_debug_message("pop");
    fprintf(stderr, "exception stack underflow\n");
    dump_exception_stack();
    abort();
  }

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
void throw_exception(
    int what,
    char * function,
    char * file,
    int line)
{
  int i;
  
  exception_debug_trace("THROW");

  the_last_just_thrown_exception = what; /* for rethrow */

  for (i=exception_index-1; i>=0; i--)
  {
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
  
      if (linear_exception_verbose_mode)
	fprintf(stderr, "exception %d/%d: %s(%s:%d) -> %s(%s:%d)\n",
		what, exception_stack[i].what,
		function, file, line,
		exception_stack[i].function, 
		exception_stack[i].file,
		exception_stack[i].line);

      longjmp(exception_stack[i].where,0);
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
