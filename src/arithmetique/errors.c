

#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"

#define MAX_STACKED_CONTEXTS 20
jmp_buf global_exception_stack[MAX_STACKED_CONTEXTS];
int     global_exception_type[MAX_STACKED_CONTEXTS];
int     global_exception_index = 0;
int     global_exception_thrown = 0;

unsigned int overflow_error = 1;
unsigned int simplex_arithmetic_error = 2;
unsigned int user_exception_error = 4;
unsigned int any_exception_error = ~0;

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
  
void print_exception_stack_error(int error)
{
    switch (error) {
    case 1: 
	fprintf(stderr,"global exception stack overflow\n") ;
	break;
    case 0:
	fprintf(stderr,"global exception stack underflow\n");
	break;
    case 2: 
	fprintf(stderr, "non matching uncatch found\n");
	break;
    default: 
	fprintf(stderr, "unexpected error tag %d\n", error);
    }

    abort();
}
