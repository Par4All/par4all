/*
  $Id$

  Exception management.
  See arithmetic_errors.h

  $Log: errors.c,v $
  Revision 1.6  1998/10/24 09:31:06  coelho
  RCS headers.
  'const' tried.

*/

#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"

/* exception stack (could/should be dynamic).
 * MAX_STACKED_CONTEXTS == 40
 */
jmp_buf global_exception_stack[MAX_STACKED_CONTEXTS];
int     global_exception_type[MAX_STACKED_CONTEXTS];
int     global_exception_index = 0;
int     global_exception_thrown = 0;

/* global constants to designate exceptions.
 */
unsigned int overflow_error = 1;
unsigned int simplex_arithmetic_error = 2;
unsigned int user_exception_error = 4;
unsigned int any_exception_error = ~0;

/* throws an exception of a given type by searching for 
   the specified 'what' in the current exception stack.
*/
void throw_exception(int what)
{
    int i=global_exception_index_decr;
    for (; i>=0 ;i--)
	if (global_exception_type[i]&what) 
	{
	    global_exception_index = i;
	    longjmp(global_exception_stack[i],0);
	}
    fprintf(stderr,"stack index error \n");
    abort();
}
  
/* aborts on errors.
 */
void print_exception_stack_error(int error)
{
    switch (error) {
    case 0:
	fprintf(stderr,"global exception stack underflow\n");
	break;
    case 1: 
	fprintf(stderr,"global exception stack overflow\n") ;
	break;
    case 2: 
	fprintf(stderr, "non matching uncatch found\n");
	break;
    default: 
	fprintf(stderr, "unexpected error tag %d\n", error);
    }

    abort();
}
