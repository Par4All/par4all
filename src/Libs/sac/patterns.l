%{
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "patterns.tab.h"

int patterns_yyerror(char* s);
%}
%%
"REFERENCE"             return REFERENCE_TOK;
"CONSTANT"              return CONSTANT_TOK;
"="			return ASSIGN_OPERATOR_TOK;
"+"			return PLUS_OPERATOR_TOK;
"-"			return MINUS_OPERATOR_TOK;
"UNARY_MINUS"		return UNARY_MINUS_OPERATOR_TOK;
"*"      		return MULTIPLY_OPERATOR_TOK;
"/"			return DIVIDE_OPERATOR_TOK;
"INVERSE"		return INVERSE_OPERATOR_TOK;
"^"			return POWER_OPERATOR_TOK;
"%"			return MODULO_OPERATOR_TOK;
"MIN"			return MIN_OPERATOR_TOK;
"MIN0"			return MIN0_OPERATOR_TOK;
"AMIN1"			return AMIN1_OPERATOR_TOK;
"DMIN1"			return DMIN1_OPERATOR_TOK;
"MAX"			return MAX_OPERATOR_TOK;
"MAX0"			return MAX0_OPERATOR_TOK;
"AMAX1"			return AMAX1_OPERATOR_TOK;
"DMAX1"			return DMAX1_OPERATOR_TOK;
"ABS"			return ABS_OPERATOR_TOK;
"IABS"			return IABS_OPERATOR_TOK;
"DABS"			return DABS_OPERATOR_TOK;
"CABS"			return CABS_OPERATOR_TOK;
"&"			return AND_OPERATOR_TOK;
"|"			return OR_OPERATOR_TOK;
"!"			return NOT_OPERATOR_TOK;
"!="     		return NON_EQUAL_OPERATOR_TOK;
"EQUIV"			return EQUIV_OPERATOR_TOK;
"NON_EQUIV"		return NON_EQUIV_OPERATOR_TOK;
"TRUE"			return TRUE_OPERATOR_TOK;
"FALSE"			return FALSE_OPERATOR_TOK;
">="                 	return GREATER_OR_EQUAL_OPERATOR_TOK;
">"     		return GREATER_THAN_OPERATOR_TOK;
"<="		        return LESS_OR_EQUAL_OPERATOR_TOK;
"<"      		return LESS_THAN_OPERATOR_TOK;
"=="			return EQUAL_OPERATOR_TOK;
[0-9]+                  { patterns_yylval.iVal = atoi(patterns_yytext); return INTEGER_TOK; }
[A-Za-z][0-9A-Za-z_]*   { patterns_yylval.strVal = strdup(patterns_yytext); return IDENTIFIER_TOK; }
[:;,]                   return *patterns_yytext;
[ \t\n]                 ;
.                       { patterns_yyerror("Unknown token\n"); return UNKNOWN_TOK; }
%%
