%start COMMENT TEXT

%{
#include <stdio.h>
#include <string.h>
#include "genC.h"
#include "makefile.h"
#include "pipsmake_yacc.h"
%}

%%
<TEXT>[ \t\n]*		;
<TEXT>"/*"		{BEGIN COMMENT;}
<TEXT>"//[^\n]*\n"	;
<COMMENT>"*/"		{BEGIN TEXT;}
<COMMENT>(.|\n)      	;
<TEXT>PROGRAM		{return PROGRAM;}
<TEXT>MODULE		{return MODULE;}
<TEXT>MAIN		{return MAIN;}
<TEXT>COMMON		{return COMMON;}
<TEXT>CALLEES		{return CALLEES;}
<TEXT>CALLERS		{return CALLERS;}
<TEXT>ALL		{return ALL;}
<TEXT>SELECT		{return SELECT;}
<TEXT>COMPILATION_UNIT	{return COMPILATION_UNIT;}
<TEXT>\<		{return REQUIRED;}
<TEXT>\>		{return PRODUCED;}
<TEXT>\#		{return MODIFIED;}
<TEXT>=		        {return PRESERVED;}
<TEXT>\!	        {return PRE_TRANSFORMATION;}
<TEXT>\.		{return DOT;}
<TEXT>[-_a-zA-Z0-9]+	{ yylval.name=strdup(yytext); return NAME;}
<TEXT,COMMENT>.               	{
			fprintf(stderr, 
				"[readmakefile] unknown char %c (%d)\n", 
				*yytext, *yytext);
                        exit(1);
			}
%%

int yywrap()
{
    return 1;
}

int init_lex()
{
    BEGIN TEXT;
    return 1;
}

void yyerror_lex_part(char *s)
{
    fprintf(stderr, "[yyerror_lex_part] %s near %s\n", s, yytext);
}

