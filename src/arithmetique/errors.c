

#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"

void throw_exception(what)
int what;
{
    int i=global_exception_index_decr;
    for (; i>=0 ;i--)
	if (global_exception_type[i]==what) 
	{
	    global_exception_index = i;
	    longjmp(global_exception_stack[i],0);
	}
    fprintf(stderr,"stack index error \n");
    abort();
}
  
void print_exception_stack_error(overflow)
boolean overflow;
{
    overflow ? fprintf(stderr,"global exception stack overflow\n"):
	fprintf(stderr,"global exception stack underflow\n");
    abort();
}
