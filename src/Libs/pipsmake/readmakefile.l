%START COMMENT TEXT

%{
#include <stdio.h>
#include "genC.h"
#include "makefile.h"
#include "y.tab.h"
%}

%%
<TEXT>[ \t\n]*		;
<TEXT>"/*"		{BEGIN COMMENT;}
<TEXT>"//[^\n]*\n"	;
<COMMENT>"*/"		{BEGIN TEXT;}
<COMMENT>(.|\n)      	;
<TEXT>PROGRAM		{return(PROGRAM);}
<TEXT>MODULE		{return(MODULE);}
<TEXT>MAIN		{return(MAIN);}
<TEXT>COMMON		{return(COMMON);}
<TEXT>CALLEES		{return(CALLEES);}
<TEXT>CALLERS		{return(CALLERS);}
<TEXT>ALL		{return(ALL);}
<TEXT>\<		{return(REQUIRED);}
<TEXT>\>		{return(PRODUCED);}
<TEXT>\#		{return(MODIFIED);}
<TEXT>=		        {return(PRESERVED);}
<TEXT>\!	        {return(PRE_TRANSFORMATION);}
<TEXT>\.		{return(DOT);}
<TEXT>[-_a-zA-Z0-9]+	{return(NAME);}
<TEXT,COMMENT>.               	{
			fprintf(stderr, 
				"[readmakefile] unknown char %c (%d)\n", 
				*yytext, *yytext);
                        exit(1);
			}
%%
yywrap()
{
    return(1) ;
}

init_lex()
{
	BEGIN TEXT;
}
