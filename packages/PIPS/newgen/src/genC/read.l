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
   The syntax of tokens written by GEN_WRITE.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "genC.h"
#include "newgen_include.h"
#include "genread.h"

#ifdef FLEX_SCANNER
#undef yywrap
#endif

static string literal = (char*) NULL;
static int literal_size = 0;
static void append_to_literal(string s)
{
    int total;

    if (literal_size == 0) {
	literal_size = 50;
	literal = (char*) malloc(sizeof(char)*literal_size);
	message_assert("malloc ok", literal);
	literal[0]='\0';
    }

    total = strlen(literal)+strlen(s)+1;
    if (total >= literal_size) {
	literal_size = total+100;
	literal = (char*) realloc(literal, sizeof(char)*literal_size);
	message_assert("realloc ok", literal);
    }

    strcat(literal, s);
}

static size_t
  newgen_line_number,
  newgen_char_number,
  newgen_col_number,
  newgen_lex_number;

void newgen_start_lexer(FILE * in)
{
  if (in) { yyin = in; }
  newgen_char_number = 0;
  newgen_line_number = 1;
  newgen_col_number = 0;
  newgen_lex_number = 0;
}

void newgen_lexer_position(FILE * out)
{
  fprintf(out, "newgen lexer token %zd at line %zd (column %zd, char %zd)\n",
	  newgen_lex_number, newgen_line_number,
	  newgen_col_number, newgen_char_number);
}

%}

%Start a_string
%option nounput

%%
<a_string>\\\\          { append_to_literal("\\"); }
<a_string>\\\"		{ append_to_literal("\""); }
<a_string>[^\\"\n]*     { append_to_literal(yytext); } /* " */
<a_string>[ \t\n]*      { append_to_literal(yytext); }
<a_string>\"		{
			  BEGIN 0 ;
			  yylval.s = strdup(literal);
			  free(literal); literal=(char*)NULL; literal_size=0;
			  newgen_lex_number++;
			  return READ_STRING ;
			}
_			{
                          yylval.s = NULL;
			  newgen_lex_number++;
                          return READ_STRING;
                        }
!			{
                          yylval.s = string_undefined;
			  newgen_lex_number++;
			  return READ_STRING;
                        }
\[                      { newgen_lex_number++; return LB; }
T			{ newgen_lex_number++; return CHUNK_BEGIN; }
\$			{ newgen_lex_number++; return VECTOR_BEGIN; }
%			{ newgen_lex_number++; return ARROW_BEGIN; }
\*			{ newgen_lex_number++; return TABULATED_BEGIN; }
\(		        { newgen_lex_number++; return LP; }
\)			{ newgen_lex_number++; return RP; }
\{		        { newgen_lex_number++; return LC; }
\}			{ newgen_lex_number++; return RC; }
H			{ newgen_lex_number++; return SHARED_POINTER; }
U			{ newgen_lex_number++; return READ_UNIT; }
B			{ newgen_lex_number++; return READ_BOOL; }
E	 		{ newgen_lex_number++; return READ_EXTERNAL; }
D			{ newgen_lex_number++; return READ_DEF; }
R			{ newgen_lex_number++; return READ_REF; }
N			{ newgen_lex_number++; return READ_NULL; }
L	                { newgen_lex_number++; return READ_LIST_UNDEFINED; }
S	                { newgen_lex_number++; return READ_SET_UNDEFINED; }
A			{ newgen_lex_number++; return READ_ARRAY_UNDEFINED; }
\#\\.			{
                          yylval.c=*(yytext+2);
			  newgen_lex_number++;
			  return READ_CHAR;
                        }
[+-]?[ \t]*[0-9]+       {
                          yylval.val=atoi(yytext);
			  newgen_lex_number++;
			  return READ_INT;
                        }
[+-]?[ \t]*[0-9]+\.[0-9]+ {
                          yylval.d=atof(yytext);
			  newgen_lex_number++;
			  return READ_FLOAT;
                        }
\"			{ BEGIN a_string ; append_to_literal(""); }
[ \t]*	                ;
\n 			{ newgen_line_number++; newgen_col_number=0; }
.                       { fatal("gen_read: Unknown character %x\n", 
				 (int) yytext[0]);}
%%

#ifdef FLEX_SCANNER
int yyinput()
{
    newgen_char_number++;
    newgen_col_number++;
    return input();
}
#endif

int yywrap(void)
{
#ifdef FLEX_SCANNER
    yy_init = 1 ;
#endif
    newgen_start_lexer(NULL);
    return 1;
}
