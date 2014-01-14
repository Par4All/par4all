%{
/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of NewGen.

  NewGen is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or any later version.

  NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU General Public License along with
  NewGen.  If not, see <http://www.gnu.org/licenses/>.

*/
/*
  When a new keyword is added, update the KEYWORDS array in build.c
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "genC.h"
#include "genspec.h"

#ifdef FLEX_SCANNER
#undef yywrap
#endif

extern int Current_start;

#ifdef FLEX_SCANNER

/* We may parse strings or files...
 */
static char * string_to_parse = (char*) 0; /* shared pointer! */

#define YY_INPUT(buffer, result, max_size)		\
{							\
    size_t i = 0;						\
    if (string_to_parse) /* we're parsing a string */	\
    {							\
	while(i<max_size && *string_to_parse)		\
	    buffer[i++]=*string_to_parse++;		\
    }							\
    else /* it's a file */				\
    {							\
	int c;						\
	while (i<max_size && (c=getc(yyin))!=EOF)	\
	    buffer[i++]=(char) c;			\
    }							\
    result = i==0? YY_NULL: i;				\
}

void yyset_string_to_parse(char *s) { string_to_parse = s;}
void yyreset_string_to_parse() { string_to_parse = (char *) 0;}

#else /* ATT/POSIX just seem to like parsing yyin... */
// I guess this code won't be use again anyway... RK
void genspec_set_string_to_parse(char *s)
{
  if ((genspec_in =  tmpfile()) == NULL) {
	user( "Cannot create temp spec file in read/write mode\n" ) ;
	return ;
    }
    fprintf( genspec_in , "%s", s ) ;
}

void genspec_reset_string_to_parse()
{
  // The file created by tmpfile() is automatically deleted on close:
    fclose(genspec_in);
}
     
}
#endif

/* cache parsed string instead of strduplicating them
 * valid because those strign are never released
 * SG, feb. 2009
 */
#ifndef DISABLE_NEWGEN_PARSER_CACHE

    #include <search.h>
    char* strcache(char *s)
    {
        static void* rootp = NULL;
        char ** res = NULL;
        if( !(res = tfind(s,&rootp,(int(*)(const void*,const void*))strcmp)) )
        {
            char * dup = strdup(s);
            res = tsearch(dup,&rootp,(int(*)(const void*,const void*))strcmp);
            assert(*res == dup);
        }
        return *res;
    }

#else

    #define strcache strdup

#endif

%}
%option nounput

%%
external		{return GRAM_EXTERNAL ;}
import                  {return GRAM_IMPORT ;}
tabulated		{return TABULATED ;}
from			{return FROM ;}
persistant              {return PERSISTANT ;}
persistent              {return PERSISTANT ;}
\,			{return COMMA ;}
\:			{return COLUMN ;}
\;			{return SEMI_COLUMN ;}
x			{return AND ;}
\+			{return OR ;}
"->"                    {return ARROW ;}
\*			{return STAR ;}
\[			{return LB ;}
\]			{return RB ;}
\{			{return LR ;}
\}			{return RR ;}
\"[_a-zA-Z0-9/\.~]+\"	{ ++yytext; yytext[+strlen(yytext)-1] = '\0';
                          yylval.name = strcache(yytext);
                          return GRAM_FILE ;}
=			{return EQUAL ;}
^--NEWGEN-START\ [0-9]*\n {
                        Current_start = 
                             atoi( yytext+strlen( "--NEWGEN-START " )) ;
			}
^--NEWGEN-INDEX\ [0-9]*\n ;
--.*\n			{}
[_a-zA-Z][_a-zA-Z0-9]*	{ yylval.name = strcache(yytext); return IDENT ;}
[0-9]+                  { yylval.val = atoi(yytext); return GRAM_INT ;}
[ \t\n]*		;
%%

int yywrap()
{
#ifdef FLEX_SCANNER
  yy_init = 1;
#endif
  return 1;
}

#ifdef FLEX_SCANNER
int yyinput()
{
  return input();
}
#endif
