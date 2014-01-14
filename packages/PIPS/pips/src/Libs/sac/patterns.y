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

%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"

#include "sac.h"

static size_t opcode_argc =0;
size_t sac_lineno = 0;

/* fake helpers */
#define TOKEN_NEWGEN_DOMAIN (-1)
#define ARGUMENT_NEWGEN_DOMAIN (-1)
#define gen_TOKEN_cons(t,l) gen_cons(t,l)
#define gen_ARGUMENT_cons(a,l) gen_cons(a,l)


%}


%union {
      intptr_t tokenId;
      intptr_t typeId;
      list tokenList;
      list typeList;

      patternArg argument;
      list argsList;

      int iVal;
      float fVal;
      char * strVal;

      opcode opVal;
      list opcodesList;

      list mappingsList;
}

%type <tokenId> token;
%type <typeId> type;
%type <tokenList> tokens_list;
%type <typeList> types_list;

%type <argument> argument;
%type <argsList> arguments_list;
%type <argsList> merge_arguments;

%type <opVal> opcode;
%type <opcodesList> opcodes_list;
%type <mappingsList> mappings;

%token UNKNOWN_TOK
%token REFERENCE_TOK
%token QI_REF_TOK
%token HI_REF_TOK
%token SI_REF_TOK
%token DI_REF_TOK
%token SF_REF_TOK
%token DF_REF_TOK
%token SC_REF_TOK
%token DC_REF_TOK
%token LOG_REF_TOK
%token CONSTANT_TOK

%token ASSIGN_OPERATOR_TOK
%token PLUS_OPERATOR_TOK
%token MINUS_OPERATOR_TOK
%token UNARY_MINUS_OPERATOR_TOK
%token MULTIPLY_OPERATOR_TOK
%token MULADD_OPERATOR_TOK
%token DIVIDE_OPERATOR_TOK
%token INVERSE_OPERATOR_TOK
%token POWER_OPERATOR_TOK
%token MODULO_OPERATOR_TOK
%token MIN_OPERATOR_TOK
%token COS_OPERATOR_TOK
%token SIN_OPERATOR_TOK
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
%token PHI_TOK

%token <iVal> INTEGER_TOK
%token <fVal> FLOAT_TOK
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
       IDENTIFIER_TOK  '{' opcodes_list '}' 
                                        {
                                           insert_opcodeClass($1, opcode_argc, $3);
											opcode_argc=0;
                                        }

opcodes_list:
       opcodes_list opcode              { $$ = CONS(OPCODE, $2, $1); }
     |                                  { $$ = NIL; }

opcode:
       IDENTIFIER_TOK ':'  types_list ',' INTEGER_TOK ';'      
                                        { 
											if(opcode_argc<=0)opcode_argc=gen_length($3);
											else pips_assert("all opcode of the same operation have the same lenght\n",opcode_argc == gen_length($3));
                                           $$ = generate_opcode($1, $3, $5);
                                        }

pattern:
       IDENTIFIER_TOK ':' tokens_list merge_arguments ';'      
                                        {
                                           insert_pattern($1, $3, $4);
                                        }

types_list:
       type types_list                  { $$ = CONS(int, $1, $2); }
     |                                  { $$ = NIL; }

type:
       QI_REF_TOK                       { $$ = QI_REF_TOK; }
     | HI_REF_TOK                       { $$ = HI_REF_TOK; }
     | SI_REF_TOK                       { $$ = SI_REF_TOK; }
     | DI_REF_TOK                       { $$ = DI_REF_TOK; }
     | SF_REF_TOK                       { $$ = SF_REF_TOK; }
     | DF_REF_TOK                       { $$ = DF_REF_TOK; }
	 | SC_REF_TOK						{ $$ = SC_REF_TOK; }
	 | DC_REF_TOK						{ $$ = DC_REF_TOK; }
     | LOG_REF_TOK                      { $$ = LOG_REF_TOK; }

tokens_list:
       token tokens_list                {
                                           $$ = CONS(TOKEN, (void*)$1, $2);
                                        }
     |                                  { $$ = NIL; }

token:
       REFERENCE_TOK                    { $$ = REFERENCE_TOK; }
     | CONSTANT_TOK 			{ $$ = CONSTANT_TOK; }
     | ASSIGN_OPERATOR_TOK 		{ $$ = ASSIGN_OPERATOR_TOK; }
     | PLUS_OPERATOR_TOK		{ $$ = PLUS_OPERATOR_TOK; }
     | MINUS_OPERATOR_TOK		{ $$ = MINUS_OPERATOR_TOK; }
     | UNARY_MINUS_OPERATOR_TOK		{ $$ = UNARY_MINUS_OPERATOR_TOK; }
     | MULTIPLY_OPERATOR_TOK		{ $$ = MULTIPLY_OPERATOR_TOK; }
     | MULADD_OPERATOR_TOK		{ $$ = MULADD_OPERATOR_TOK; }
     | DIVIDE_OPERATOR_TOK		{ $$ = DIVIDE_OPERATOR_TOK; }
     | INVERSE_OPERATOR_TOK		{ $$ = INVERSE_OPERATOR_TOK; }
     | POWER_OPERATOR_TOK		{ $$ = POWER_OPERATOR_TOK; }
     | MODULO_OPERATOR_TOK		{ $$ = MODULO_OPERATOR_TOK; }
     | MIN_OPERATOR_TOK			{ $$ = MIN_OPERATOR_TOK; }
     | COS_OPERATOR_TOK			{ $$ = COS_OPERATOR_TOK; }
     | SIN_OPERATOR_TOK			{ $$ = SIN_OPERATOR_TOK; }
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
     | PHI_TOK		                { $$ = PHI_TOK; }

merge_arguments:
       ':' arguments_list               { $$ = $2; }
     |                                  { $$ = NIL; }
       ;

arguments_list:
       arguments_list ',' argument      { $$ = CONS(ARGUMENT, $3, $1); }
     | argument                         { $$ = CONS(ARGUMENT, $1, NIL); }

argument:
       INTEGER_TOK                      { $$ = make_patternArg_static($1); }
     |                                  { $$ = make_patternArg_dynamic(); }

transformation:
     IDENTIFIER_TOK '[' INTEGER_TOK ',' INTEGER_TOK ',' INTEGER_TOK ',' INTEGER_TOK ',' INTEGER_TOK ']' '{' mappings '}'
                                        { insert_transformation( $1, $3, $5, $7, $9, $11, $14); }

mappings:
       mappings ',' INTEGER_TOK         { $$ = CONS(INT, $3, $1); }
     | INTEGER_TOK                      { $$ = CONS(INT, $1, NIL); }

%%

int yywrap(void)
{
   return 1;
}

void yyerror(const char* s)
{
   pips_internal_error("patterns parser:%zd: %s\n",sac_lineno, s);
}
