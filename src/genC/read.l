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


/* read.l

   The syntax of tokens written by GEN_WRITE. */

#include <stdio.h>
#include <stdint.h>
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

static size_t newgen_line_number;
static size_t newgen_char_number;
static size_t newgen_col_number;

void newgen_start_lexer(FILE * in)
{
  if (in) { yyin = in; }
  newgen_char_number = 0;
  newgen_line_number = 1;
  newgen_col_number = 0;
}

void newgen_lexer_position(FILE * out)
{
  fprintf(out, "newgen lexer at line %zd (column %zd, char %zd)\n", 
	  newgen_line_number, newgen_col_number, newgen_char_number);
}

%}

%Start a_string

%%
<a_string>\\\\          { append_to_literal("\\"); }
<a_string>\\\"		{ append_to_literal("\""); }
<a_string>[^\\"\n]*     { append_to_literal(yytext); } /* " */
<a_string>[ \t\n]*      { append_to_literal(yytext); }
<a_string>\"		{
			  BEGIN 0 ;
			  yylval.s = strdup(literal);
			  free(literal); literal=(char*)NULL; literal_size=0;
			  return READ_STRING ;
			}
_			{ yylval.s = NULL; return READ_STRING; }
!			{ yylval.s = string_undefined; return READ_STRING; }
\[                      { return LB; }
T			{ return CHUNK_BEGIN; }
\$			{ return VECTOR_BEGIN; }
%			{ return ARROW_BEGIN; }
\*			{ return TABULATED_BEGIN; }
\(		        { return LP; }
\)			{ return RP; }
\{		        { return LC; }
\}			{ return RC; }
H			{ return SHARED_POINTER; }
U			{ return READ_UNIT; }
B			{ return READ_BOOL; }
E	 		{ return READ_EXTERNAL; }
D			{ return READ_DEF; }
R			{ return READ_REF; }
N			{ return READ_NULL; }
L	                { return READ_LIST_UNDEFINED; }
S	                { return READ_SET_UNDEFINED; }
A			{ return READ_ARRAY_UNDEFINED; }
\#\\.			{ yylval.c=*(yytext+2); return READ_CHAR; }
[+-]?[ \t]*[0-9]+       { yylval.val=atoi(yytext); return READ_INT; }
[+-]?[ \t]*[0-9]+\.[0-9]+ { yylval.d=atof(yytext); return READ_FLOAT; }
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
    message_assert("function not used", yyunput==yyunput);
    newgen_start_lexer(NULL);
    return 1;
}
