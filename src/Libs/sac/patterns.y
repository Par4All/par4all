%{

#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "sac-local.h"
#include "sac.h"

int patterns_yyerror(char* s);
int patterns_yywrap(void);
int patterns_yylex();
static int* make_integer_argument(int x);
static int* make_empty_argument();

%}


%union {
      int tokenId;
      list tokenList;

      int* argument;
      list argsList;

      int iVal;
      char * strVal;

      opcode opVal;
      list opcodesList;

      list mappingsList;
}

%type <tokenId> token;
%type <tokenList> tokens_list;

%type <argument> argument;
%type <argsList> arguments_list;
%type <argsList> merge_arguments;

%type <opVal> opcode;
%type <opcodesList> opcodes_list;
%type <mappingsList> mappings;

%token UNKNOWN_TOK
%token REFERENCE_TOK
%token CONSTANT_TOK

%token ASSIGN_OPERATOR_TOK
%token PLUS_OPERATOR_TOK
%token MINUS_OPERATOR_TOK
%token UNARY_MINUS_OPERATOR_TOK
%token MULTIPLY_OPERATOR_TOK
%token DIVIDE_OPERATOR_TOK
%token INVERSE_OPERATOR_TOK
%token POWER_OPERATOR_TOK
%token MODULO_OPERATOR_TOK
%token MIN_OPERATOR_TOK
%token MIN0_OPERATOR_TOK
%token AMIN1_OPERATOR_TOK
%token DMIN1_OPERATOR_TOK
%token MAX_OPERATOR_TOK
%token MAX0_OPERATOR_TOK
%token AMAX1_OPERATOR_TOK
%token DMAX1_OPERATOR_TOK
%token ABS_OPERATOR_TOK
%token IABS_OPERATOR_TOK
%token DABS_OPERATOR_TOK
%token CABS_OPERATOR_TOK
   
%token AND_OPERATOR_TOK
%token OR_OPERATOR_TOK
%token NOT_OPERATOR_TOK
%token NON_EQUAL_OPERATOR_TOK
%token EQUIV_OPERATOR_TOK
%token NON_EQUIV_OPERATOR_TOK
   
%token TRUE_OPERATOR_TOK
%token FALSE_OPERATOR_TOK
   
%token GREATER_OR_EQUAL_OPERATOR_TOK
%token GREATER_THAN_OPERATOR_TOK
%token LESS_OR_EQUAL_OPERATOR_TOK
%token LESS_THAN_OPERATOR_TOK
%token EQUAL_OPERATOR_TOK

%token <iVal> INTEGER_TOK
%token <strVal> IDENTIFIER_TOK

%%

definitions:
       definitions_list
       ;

definitions_list:
       definitions_list definition
     | 
       ;

definition:
       operation
     | pattern
     | transformation
       ;
 
operation:
       IDENTIFIER_TOK '[' INTEGER_TOK ']' '{' opcodes_list '}' 
                                        { insert_opcodeClass($1, $3, $6); }

opcodes_list:
       opcodes_list opcode              { $$ = CONS(OPCODE, $2, $1); }
     |                                  { $$ = NIL; }

opcode:
       IDENTIFIER_TOK ':' INTEGER_TOK ',' INTEGER_TOK ';'      
                                        { $$ = make_opcode($1, $3, $5); }

pattern:
       IDENTIFIER_TOK ':' tokens_list merge_arguments ';'      
                                        { insert_pattern($1, $3, $4); }

tokens_list:
       token tokens_list                { $$ = CONS(TOKEN, $1, $2); }
     |                                  { $$ = NIL; }

token:
       REFERENCE_TOK                    { $$ = REFERENCE_TOK; }
     | CONSTANT_TOK 			{ $$ = CONSTANT_TOK; }
     | ASSIGN_OPERATOR_TOK 		{ $$ = ASSIGN_OPERATOR_TOK; }
     | PLUS_OPERATOR_TOK		{ $$ = PLUS_OPERATOR_TOK; }
     | MINUS_OPERATOR_TOK		{ $$ = MINUS_OPERATOR_TOK; }
     | UNARY_MINUS_OPERATOR_TOK		{ $$ = UNARY_MINUS_OPERATOR_TOK; }
     | MULTIPLY_OPERATOR_TOK		{ $$ = MULTIPLY_OPERATOR_TOK; }
     | DIVIDE_OPERATOR_TOK		{ $$ = DIVIDE_OPERATOR_TOK; }
     | INVERSE_OPERATOR_TOK		{ $$ = INVERSE_OPERATOR_TOK; }
     | POWER_OPERATOR_TOK		{ $$ = POWER_OPERATOR_TOK; }
     | MODULO_OPERATOR_TOK		{ $$ = MODULO_OPERATOR_TOK; }
     | MIN_OPERATOR_TOK			{ $$ = MIN_OPERATOR_TOK; }
     | MIN0_OPERATOR_TOK		{ $$ = MIN0_OPERATOR_TOK; }
     | AMIN1_OPERATOR_TOK		{ $$ = AMIN1_OPERATOR_TOK; }
     | DMIN1_OPERATOR_TOK		{ $$ = DMIN1_OPERATOR_TOK; }
     | MAX_OPERATOR_TOK			{ $$ = MAX_OPERATOR_TOK; }
     | MAX0_OPERATOR_TOK		{ $$ = MAX0_OPERATOR_TOK; }
     | AMAX1_OPERATOR_TOK		{ $$ = AMAX1_OPERATOR_TOK; }
     | DMAX1_OPERATOR_TOK		{ $$ = DMAX1_OPERATOR_TOK; }
     | ABS_OPERATOR_TOK			{ $$ = ABS_OPERATOR_TOK; }
     | IABS_OPERATOR_TOK		{ $$ = IABS_OPERATOR_TOK; }
     | DABS_OPERATOR_TOK		{ $$ = DABS_OPERATOR_TOK; }
     | CABS_OPERATOR_TOK		{ $$ = CABS_OPERATOR_TOK; }
     | AND_OPERATOR_TOK			{ $$ = AND_OPERATOR_TOK; }
     | OR_OPERATOR_TOK			{ $$ = OR_OPERATOR_TOK; }
     | NOT_OPERATOR_TOK			{ $$ = NOT_OPERATOR_TOK; }
     | NON_EQUAL_OPERATOR_TOK		{ $$ = NON_EQUAL_OPERATOR_TOK; }
     | EQUIV_OPERATOR_TOK		{ $$ = EQUIV_OPERATOR_TOK; }
     | NON_EQUIV_OPERATOR_TOK		{ $$ = NON_EQUIV_OPERATOR_TOK; }
     | TRUE_OPERATOR_TOK		{ $$ = TRUE_OPERATOR_TOK; }
     | FALSE_OPERATOR_TOK		{ $$ = FALSE_OPERATOR_TOK; }
     | GREATER_OR_EQUAL_OPERATOR_TOK	{ $$ = GREATER_OR_EQUAL_OPERATOR_TOK; }
     | GREATER_THAN_OPERATOR_TOK	{ $$ = GREATER_THAN_OPERATOR_TOK; }
     | LESS_OR_EQUAL_OPERATOR_TOK	{ $$ = LESS_OR_EQUAL_OPERATOR_TOK; }
     | LESS_THAN_OPERATOR_TOK		{ $$ = LESS_THAN_OPERATOR_TOK; }
     | EQUAL_OPERATOR_TOK		{ $$ = EQUAL_OPERATOR_TOK; }

merge_arguments:
       ':' arguments_list               { $$ = $2; }
     |                                  { $$ = NIL; }
       ;

arguments_list:
       arguments_list ',' argument      { $$ = CONS(ARGUMENT, $3, $1); }
     | argument                         { $$ = CONS(ARGUMENT, $1, NIL); }

argument:
       INTEGER_TOK                      { $$ = make_integer_argument($1); }
     |                                  { $$ = make_empty_argument(); }

transformation:
     IDENTIFIER_TOK '[' INTEGER_TOK ',' INTEGER_TOK ',' INTEGER_TOK ',' INTEGER_TOK ',' INTEGER_TOK ']' '{' mappings '}'
                                        { insert_transformation( $1, $3, $5, $7, $9, $11, $14); }

mappings:
       mappings ',' INTEGER_TOK         { $$ = CONS(INT, $3, $1); }
     | INTEGER_TOK                      { $$ = CONS(INT, $1, NIL); }

%%

static int* make_integer_argument(int x)
{
   int * res;
   
   res = (int*)malloc(sizeof(int));
   *res = x;
   return res;
}

static int* make_empty_argument()
{
   return NULL;
}

int patterns_yywrap(void)
{
   return 1;
}

int patterns_yyerror(char* s)
{
   fprintf(stderr, "%s\n", s);
   return 0;
}
