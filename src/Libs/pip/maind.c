/* This file comes directly from maind.c, from which main() 
 * has been removed. All globals definitions has also been removed.
 * This file is necessary for compilation.
 *	AL 9/12/93
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include "pip__type.h"
#include "pip__sol.h"
#include "pip__tab.h"
#include <sys/types.h>

#define INLENGTH 1024
extern 	long    int cross_product, limit;
extern	int     allocation, comptage;
extern	char    inbuff[];
extern	int     inptr,	proviso, verbose;
extern	FILE    *dump;
extern	char    dump_name[];


Tableau *expanser();

int dgetc(foo)
FILE *foo;
{
 char *p;
 if(inptr >= proviso)
   {p = fgets(inbuff, INLENGTH, foo);
    if(p == NULL) return EOF;
    proviso = strlen(inbuff);
    if(INLENGTH - proviso <= 1) {
      fprintf(stderr, "troncature %d\n",proviso);
      exit(12);
    }
    inptr = 0;
    if(verbose > 0) fprintf(dump, "-- %s", inbuff);
  }
 return inbuff[inptr++];
}

int dscanf(foo, format, val)
FILE *foo;
char * format;
Entier * val;
{
 char * p;
 int c;
 for(;inptr < proviso; inptr++)
   if(inbuff[inptr] != ' ' && inbuff[inptr] != '\n' && inbuff[inptr] != '\t')
   				break;
 while(inptr >= proviso)
   {p = fgets(inbuff, 256, foo);
    if(p == NULL) return EOF;
    proviso = strlen(inbuff);
    if(verbose > 0) {
      fprintf(dump, ".. %s", inbuff);
      fflush(dump);
    }
    for(inptr = 0; inptr < proviso; inptr++)
       if(inbuff[inptr] != ' ' && inbuff[inptr] != '\n' && inbuff[inptr] != '\t')
   				break;
  }
 if(sscanf(inbuff+inptr, format, val) != 1) return -1;
 for(; inptr < proviso; inptr++)
	if((c = inbuff[inptr]) != '-' && !isdigit(c)) break;
 return 0;
}

void balance(foo, bar)
FILE *foo, *bar;
{
 int level = 0;
 int c;
 while((c = dgetc(foo)) != EOF)
     {putc(c, bar);
      switch(c)
	  {case '(' : level++; break;
	   case ')' : if(--level == 0) return;
	  }
     }
}

void escape(foo, bar, level)
FILE * foo, * bar;
int level;
{int c;
 while((c = dgetc(foo)) != EOF)
   switch(c)
     {case '(' : level ++; break;
     case ')' : if(--level == 0)
                     { fprintf(bar, "\nerror\n)\n");
		       return;
		     }
     }
}

