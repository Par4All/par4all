/*
  $Id$

  Exception management.
  See "arithmetic_errors.h".

  $Log: errors.c,v $
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

#include "boolean.h"
#include "arithmetique.h"

/* global constants to designate exceptions.
   to be put in the type field bellow.
 */
unsigned int overflow_error = 1;
unsigned int simplex_arithmetic_error = 2;
unsigned int user_exception_error = 4;
unsigned int parser_exception_error = 8;
unsigned int any_exception_error = ~0;

/* whether to run in debug mode (that is to trace catch/uncatch/throw)
 */
int linear_exception_debug_mode = 0;

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
  char * function;
  char * file;
  int    line;
} 
  linear_exception_holder;

/* exception stack.
 */
#define MAX_STACKED_CONTEXTS 50
static linear_exception_holder exception_stack[MAX_STACKED_CONTEXTS];
static int exception_index = 0;

/* total number of exceptions thrown.
 */
static int exception_thrown = 0;

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

/* push a what exception on stack.
 */
jmp_buf * 
push_exception_on_stack(
    int what,
    char * function,
    char * file,
    int line)
{
  if (linear_exception_debug_mode)
    fprintf(stderr, "PUSH [%s:%d %s (%d)]\n", file, line, function, what);

  if (exception_index==MAX_STACKED_CONTEXTS)
  {
    fprintf(stderr, 
	    "exception stack overflow at %s:%d in %s for %d\n",
	    file, line, function, what);
    dump_exception_stack();
    abort();
  }

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
  if (linear_exception_debug_mode)
    fprintf(stderr, "POP  [%s:%d %s (%d)]\n", file, line, function, what);

  if (exception_index==0)
  {
    fprintf(stderr, 
	    "exception stack underflow at %s:%d in %s for %d\n",
	    file, line, function, what);
    dump_exception_stack();
    abort();
  }

  exception_index--;

  if ((exception_stack[exception_index].what != what) ||
      !same_string_p(exception_stack[exception_index].file, file) ||
      !same_string_p(exception_stack[exception_index].function, function))
  {
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
  int i=exception_index-1;
  
  if (linear_exception_debug_mode)
    fprintf(stderr, "THROW[%s:%d %s (%d)/%d]\n",
	    file, line, function, what, exception_index);

  for (; i>=0 ;i--)
  {
    if (exception_stack[i].what & what) 
    {
      exception_index = i;
      exception_thrown++;

      if (linear_exception_debug_mode)
	fprintf(stderr, "---->[%s:%d %s (%d)/%d]\n", 
		exception_stack[i].file,
		exception_stack[i].line,
		exception_stack[i].function,
		exception_stack[i].what,
		i);
  
      longjmp(exception_stack[i].where,0);
    }
  }

  /* error. */
  fprintf(stderr,"exception %d not found in stack\n", what);
  dump_exception_stack();
  abort();
}
