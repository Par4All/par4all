/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

%option nounput
%option noinput
%start COMMENT TEXT

%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <string.h>
#include "genC.h"
#include "makefile.h"
#include "readmakefile.h"
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
<TEXT>CALLEES		{return TK_CALLEES;}
<TEXT>CALLERS		{return CALLERS;}
<TEXT>ALL		{return ALL;}
<TEXT>SELECT		{return SELECT;}
<TEXT>COMPILATION_UNIT	{return COMPILATION_UNIT;}
<TEXT>\<		{return REQUIRED;}
<TEXT>\>		{return PRODUCED;}
<TEXT>\#		{return MODIFIED;}
<TEXT>=		        {return PRESERVED;}
<TEXT>\!	        {return PRE_TRANSFORMATION;}
<TEXT>\*	        {return POST_TRANSFORMATION;}
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

