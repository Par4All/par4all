%{
/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@isatis.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*/


/* token.l 

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
#endif

%}

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
\"[_a-zA-Z0-9/\.~]+\"	{ yylval.name = strdup(yytext+1);
                          *(yylval.name+strlen(yytext)-2) = '\0';
                          return GRAM_FILE ;}
=			{return EQUAL ;}
^--NEWGEN-START\ [0-9]*\n {
                        Current_start = 
                             atoi( yytext+strlen( "--NEWGEN-START " )) ;
			}
^--NEWGEN-INDEX\ [0-9]*\n ;
--.*\n			{}
[_a-zA-Z][_a-zA-Z0-9]*	{ yylval.name = strdup(yytext); return IDENT ;}
[0-9]+                  { yylval.val = atoi(yytext); return GRAM_INT ;}
[ \t\n]*		;
%%

int yywrap()
{
#ifdef FLEX_SCANNER
  yy_init = 1;
#endif
  message_assert("function not used", yyunput==yyunput);
  return 1;
}

#ifdef FLEX_SCANNER
int yyinput()
{
  return input();
}
#endif
