/* $Id$ 
   $Log: cyacc.y,v $
   Revision 1.2  2003/08/04 14:20:24  nguyen
   Preliminary version of the C parser

   Revision 1.1  2003/06/24 08:45:56  nguyen
   Initial revision
 */

/******************** SYNTAX ANALYZER ************************************

  Here are the parsing rules, based on the work of people from 
  Open Source Quality projects (http://osq.cs.berkeley.edu/), used 
  by the CCured source-to-source translator for C

*****************************************************************/

/*(*
 *
 * Copyright (c) 2001-2003,
 *  George C. Necula    <necula@cs.berkeley.edu>
 *  Scott McPeak        <smcpeak@cs.berkeley.edu>
 *  Wes Weimer          <weimer@cs.berkeley.edu>
 *  Ben Liblit          <liblit@cs.berkeley.edu>
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * 3. The names of the contributors may not be used to endorse or promote
 * products derived from this software without specific prior written
 * permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
 * TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **)/
/*(* NOTE: This parser is based on a parser written by Hugues Casse. Since
   * then I have changed it in numerous ways to the point where it probably
   * does not resemble Hugues's original one at all  *)*/

%{
 /* C declarations */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "transformations.h"
  
#include "c_syntax.h"


#define C_ERROR_VERBOSE 1 /* much clearer error messages with bison */

extern int c_lex(void);
extern void c_error(char *);

/* The following global variables are used to store the information such as 
   the scope, type and storage of an entity, given by the decl_spec_list, 
   which are used later by direct_decl to create the entity. 

   For the moment, block scope is not considered. CurrentScope can be File, 
   Module, File%Module or TOP-LEVEL*/

static string CurrentScope = NULL;
static type CurrentType = type_undefined; 
static storage CurrentStorage = storage_undefined;
static list CurrentQualifiers = NIL;
static string CurrentDerivedName = NULL; /* to remember the name of a struct and add it to the member prefix name*/
static int CurrentMode = 0; /* to know the mode of the formal parameter : by value or by reference*/

static bool is_external = TRUE; /* to know if the variable is declared inside or outside a function */
static bool is_typedef = FALSE; /* to know if this is a typedef name or not */
static bool is_static = FALSE; /* to know if the variable/function is declared static */
static bool is_formal = FALSE; /* to know if the entity is a formal parameter or not */

static int offset = 0; /* to know the offset of the formal argument*/

static int enum_counter = 0;

extern int derived_counter;
extern int loop_counter; 
extern bool is_loop;
extern expression CurrentSwitchController; 
extern list CurrentSwitchGotoStatements;

/* All the following global variables must be replaced by functions, once we have the preprocessor for C */
extern string CurrentSourceFile; 
extern entity CurrentSourceFileEntity;
extern entity CurrentSourceFileStaticArea;
extern int CurrentSourceFileStaticAreaOffset;

extern entity DynamicArea;
extern int CurrentDynamicAreaOffset;
%}

/* Bison declarations */

%union {
	basic basic;        
        char character;
	cons * liste;
	dimension dimension;
	entity entity;
	expression expression;
	statement statement;
	int integer;
	string string;
	syntax syntax;
	tag tag;
	type type; 
	value value;
  parameter parameter;
}

%token <string> TK_IDENT
%token <string> TK_CHARCON
%token <string> TK_INTCON
%token <string> TK_FLOATCON
%token <string> TK_NAMED_TYPE

%token <string> TK_STRINGCON   
%token <string> TK_WSTRINGCON

%token TK_EOF
%token TK_CHAR TK_INT TK_DOUBLE TK_FLOAT TK_VOID 
%token TK_ENUM TK_STRUCT TK_TYPEDEF TK_UNION
%token TK_SIGNED TK_UNSIGNED TK_LONG TK_SHORT
%token TK_VOLATILE TK_EXTERN TK_STATIC TK_CONST TK_RESTRICT TK_AUTO TK_REGISTER

%token TK_SIZEOF TK_ALIGNOF

%token TK_EQ TK_PLUS_EQ TK_MINUS_EQ TK_STAR_EQ TK_SLASH_EQ TK_PERCENT_EQ
%token TK_AND_EQ TK_PIPE_EQ TK_CIRC_EQ TK_INF_INF_EQ TK_SUP_SUP_EQ
%token TK_ARROW TK_DOT

%token TK_EQ_EQ TK_EXCLAM_EQ TK_INF TK_SUP TK_INF_EQ TK_SUP_EQ
%token TK_PLUS TK_MINUS TK_STAR
%token TK_SLASH TK_PERCENT
%token TK_TILDE TK_AND
%token TK_PIPE TK_CIRC
%token TK_EXCLAM TK_AND_AND
%token TK_PIPE_PIPE
%token TK_INF_INF TK_SUP_SUP
%token TK_PLUS_PLUS TK_MINUS_MINUS

%token TK_RPAREN 
%token TK_LPAREN TK_RBRACE
%token TK_LBRACE
%token TK_LBRACKET TK_RBRACKET
%token TK_COLON
%token TK_SEMICOLON
%token TK_COMMA TK_ELLIPSIS TK_QUEST

%token TK_BREAK TK_CONTINUE TK_GOTO TK_RETURN
%token TK_SWITCH TK_CASE TK_DEFAULT
%token TK_WHILE TK_DO TK_FOR
%token TK_IF
%token TK_ELSE

%token TK_ATTRIBUTE TK_INLINE TK_ASM TK_TYPEOF TK_FUNCTION__ TK_PRETTY_FUNCTION__
%token TK_LABEL__
%token TK_BUILTIN_VA_ARG
%token TK_BUILTIN_VA_LIST
%token TK_BLOCKATTRIBUTE
%token TK_DECLSPEC
%token TK_MSASM TK_MSATTR
%token TK_PRAGMA

/* sm: cabs tree transformation specification keywords */
%token TK_AT_TRANSFORM TK_AT_TRANSFORMEXPR TK_AT_SPECIFIER TK_AT_EXPR
%token TK_AT_NAME

/* operator precedence */
%nonassoc TK_IF
%nonassoc TK_ELSE
%left TK_COMMA
%right TK_EQ TK_PLUS_EQ TK_MINUS_EQ TK_STAR_EQ TK_SLASH_EQ TK_PERCENT_EQ TK_AND_EQ TK_PIPE_EQ TK_CIRC_EQ TK_INF_INF_EQ TK_SUP_SUP_EQ
%right TK_QUEST TK_COLON
%left TK_PIPE_PIPE
%left TK_AND_AND
%left TK_PIPE
%left TK_CIRC
%left TK_AND
%left TK_EQ_EQ TK_EXCLAM_EQ
%left TK_INF TK_SUP TK_INF_EQ TK_SUP_EQ
%left TK_INF_INF TK_SUP_SUP
%left TK_PLUS TK_MINUS
%left TK_STAR TK_SLASH TK_PERCENT TK_CONST TK_RESTRICT TK_VOLATILE
%right TK_EXCLAM TK_TILDE TK_PLUS_PLUS TK_MINUS_MINUS TK_CAST TK_RPAREN TK_ADDROF TK_SIZEOF TK_ALIGNOF
%left TK_LBRACKET
%left TK_DOT TK_ARROW TK_LPAREN TK_LBRACE
%right TK_NAMED_TYPE  
%left TK_IDENT

/* Non-terminals informations */
%start interpret
%type <liste> file interpret globals
%type <> global
%type <liste> attributes attributes_with_asm asmattr
%type <statement> statement
%type <entity> constant
%type <string> string_constant
%type <expression> expression
%type <expression> opt_expression
%type <expression> init_expression
%type <liste> comma_expression
%type <liste> paren_comma_expression
%type <liste> arguments
%type <liste> bracket_comma_expression
%type <liste> string_list 
%type <liste> wstring_list

%type <expression> initializer
%type <liste> initializer_list
%type <liste> init_designators init_designators_opt

%type <entity> type_spec
%type <liste> struct_decl_list


%type <> old_proto_decl
%type <parameter> parameter_decl
%type <entity> enumerator
%type <liste> enum_list
%type <liste> declaration
%type <> function_def
%type <> function_def_start
%type <type> type_name
%type <statement> block
%type <liste> local_labels local_label_names
%type <liste> old_parameter_list_ne

%type <entity> init_declarator
%type <liste> init_declarator_list
%type <entity> declarator
%type <entity> field_decl
%type <liste> field_decl_list
%type <entity> direct_decl
%type <> abs_direct_decl abs_direct_decl_opt
%type <> abstract_decl
%type <> pointer pointer_opt 
%type <> location

%type <string> id_or_typename
%type <liste> comma_expression_opt
%type <liste> initializer_list_opt
%type <string> one_string_constant
%type <string> one_string

%type <liste> rest_par_list rest_par_list1
%type <liste> declaration_list
%type <liste> statement_list
%type <expression> for_clause
%type <liste> decl_spec_list /* to store the list of entities such as struct, union and enum, typedef*/
%type <liste> decl_spec_list_opt_no_named
%type <liste> decl_spec_list_opt 
%%

interpret: file TK_EOF
                        {YYACCEPT};
file: globals			 /* do nothing */	
;
globals:
    /* empty */         {}       /* do nothing */             
|   global globals      {}       /* do nothing */           
|   TK_SEMICOLON globals{}       /* do nothing */            
;

location:
    /* empty */         {}  %prec TK_IDENT


/*** Global Definition ***/
global:
    declaration         { }                 
|   function_def        { }
|   TK_ASM TK_LPAREN string_constant TK_RPAREN TK_SEMICOLON
                        { CParserError("ASM not implemented\n"); }
|   TK_PRAGMA attr			
                        { 
			  CParserError("PRAGMA not implemented\n"); 
			}
/* Old-style function prototype. This should be somewhere else, like in
   "declaration". For now we keep it at global scope only because in local
   scope it looks too much like a function call */
|   TK_IDENT TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list TK_SEMICOLON
                        { 
			  /* Make a function entity here 
			     entity ent = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			     list paras = make_parameter_list($3,$5);
			     functional f = make_functional(paras,make_type_unknown());
			     entity_type(ent) = make_type_functional(f);
			  */
			  CParserError("Old-style function prototype not implemented\n");			 
			}
/* Old style function prototype, but without any arguments */
|   TK_IDENT TK_LPAREN TK_RPAREN TK_SEMICOLON
                        {
			  /* Make a function entity here 
			     entity ent = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			     functional f = make_functional(NIL,make_type_unknown());*/
			  CParserError("Old-style function prototype not implemented\n"); 
			}
/* transformer for a toplevel construct */
|   TK_AT_TRANSFORM TK_LBRACE global TK_RBRACE TK_IDENT /*to*/ TK_LBRACE globals TK_RBRACE 
                        { 
			  CParserError("CIL AT not implemented\n"); 
			}
/* transformer for an expression */
|   TK_AT_TRANSFORMEXPR TK_LBRACE expression TK_RBRACE TK_IDENT /*to*/ TK_LBRACE expression TK_RBRACE 
                        { 
			  CParserError("CIL AT not implemented\n"); 
			}
|   location error TK_SEMICOLON 
                        { 
			  CParserError("Parse error: location error TK_SEMICOLON \n");
			}
;

id_or_typename:
    TK_IDENT			
                        {}
|   TK_NAMED_TYPE				
                        {}
|   TK_AT_NAME TK_LPAREN TK_IDENT TK_RPAREN         
                        {
			   CParserError("CIL AT not implemented\n"); 
			}    
;

maybecomma:
    /* empty */ 
|   TK_COMMA      /* do nothing */
;

/* *** Expressions *** */

expression:
    constant
		        { 
			  $$ = MakeNullaryCall($1); 
			}
|   TK_IDENT
		        {
			  /* Create the expression corresponding to this identifier */
			  $$ = IdentifierToExpression($1);
                        }
|   TK_SIZEOF expression
		        {
                          $$ = MakeSizeofExpression($2);
                        }
|   TK_SIZEOF TK_LPAREN type_name TK_RPAREN
		        {
			  $$ = MakeSizeofType($3);
                        }
|   TK_ALIGNOF expression
		        { 
			  CParserError("ALIGNOF not implemented\n"); 
			}
|   TK_ALIGNOF TK_LPAREN type_name TK_RPAREN
		        { 
			  CParserError("ALIGNOF not implemented\n"); 
			}
|   TK_PLUS expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("+unary"), $2);
			}
|   TK_MINUS expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("-unary"), $2);
			}
|   TK_STAR expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("*indirection"), $2);
			}
|   TK_AND expression				%prec TK_ADDROF
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("&"), $2);
			}
|   TK_EXCLAM expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("!"), $2);
			}
|   TK_TILDE expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("~"), $2);
			}
|   TK_PLUS_PLUS expression                    %prec TK_CAST
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("++pre"), $2);
			}
|   expression TK_PLUS_PLUS
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("post++"), $1);
			}
|   TK_MINUS_MINUS expression                  %prec TK_CAST
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("--pre"), $2);
			}
|   expression TK_MINUS_MINUS
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic("post--"), $1);
			}
|   expression TK_ARROW id_or_typename
		        {	
			  /* TO BE DONE
			     Have to find the struct/union type of the expression
			     then the struct/union member entity and transform it to expression */
		       
			  expression exp = IdentifierToExpression($3); 
			  $$ = MakeBinaryCall(CreateIntrinsic("->"),$1,exp);
			}
|   expression TK_DOT id_or_typename
		        {
			  /* TO BE DONE
			     Have to find the struct/union type of the expression
			     then the struct/union member entity and transform it to expression */
			  expression exp = IdentifierToExpression($3); 
			  $$ = MakeBinaryCall(CreateIntrinsic("."),$1,exp);
			}
|   TK_LPAREN block TK_RPAREN
		        {
			  CParserError("GNU extension not implemented\n");
			}
|   paren_comma_expression
		        {
			  /* paren_comma_expression is a list of expressions*/ 
			  $$ = MakeCommaExpression($1);
			}
|   expression TK_LPAREN arguments TK_RPAREN
			{
			  /* Call expression: the function name is an expression !!!*/
			  entity ent = ExpressionToEntity($1);
			  $$ = make_call_expression(ent,$3);
			}
|   TK_BUILTIN_VA_ARG TK_LPAREN expression TK_COMMA type_name TK_RPAREN
                        {
			  CParserError("BUILTIN_VA_ARG not implemented\n");
			}
|   expression bracket_comma_expression
			{
			  /* pips_assert("The expression must be an array name or pointer")*/
			  $$ = MakeReferenceExpression($1,$2);		
			}
|   expression TK_QUEST opt_expression TK_COLON expression
			{
			  MakeTernaryCallExpr(CreateIntrinsic("?"), $1, $3, $5);
			}
|   expression TK_PLUS expression
			{ 
			  $$ = MakeBinaryCall(CreateIntrinsic("+C"), $1, $3); 
			}
|   expression TK_MINUS expression
			{ 
			  $$ = MakeBinaryCall(CreateIntrinsic("-C"), $1, $3); 
			}
|   expression TK_STAR expression
			{ 
			  $$ = MakeBinaryCall(CreateIntrinsic("*"), $1, $3); 
			}
|   expression TK_SLASH expression
			{ 
			  $$ = MakeBinaryCall(CreateIntrinsic("/"), $1, $3); 
			}
|   expression TK_PERCENT expression
			{ 
			  $$ = MakeBinaryCall(CreateIntrinsic("%"), $1, $3); 
			}
|   expression TK_AND_AND expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("&&"), $1, $3); 
			}
|   expression TK_PIPE_PIPE expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("||"), $1, $3); 
			}
|   expression TK_AND expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("&bitand"), $1, $3); 
			}
|   expression TK_PIPE expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("|"), $1, $3); 
			}
|   expression TK_CIRC expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("^"), $1, $3); 
			}
|   expression TK_EQ_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("=="), $1, $3); 
			}
|   expression TK_EXCLAM_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("!="), $1, $3); 
			}
|   expression TK_INF expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("<"), $1, $3); 
			}
|   expression TK_SUP expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(">"), $1, $3); 
			}
|   expression TK_INF_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("<="), $1, $3); 
			}
|   expression TK_SUP_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(">="), $1, $3); 
			}
|   expression TK_INF_INF expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("<<"), $1, $3); 
			}
|   expression TK_SUP_SUP expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(">>"), $1, $3); 
			}
|   expression TK_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("="), $1, $3); 
			}
|   expression TK_PLUS_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("+="), $1, $3); 
			}
|   expression TK_MINUS_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("-="), $1, $3); 
			}
|   expression TK_STAR_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("*="), $1, $3); 
			}
|   expression TK_SLASH_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("/="), $1, $3); 
			}
|   expression TK_PERCENT_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("%="), $1, $3); 
			}
|   expression TK_AND_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("&="), $1, $3); 
			}
|   expression TK_PIPE_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("|="), $1, $3); 
			}
|   expression TK_CIRC_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("^="), $1, $3); 
			}
|   expression TK_INF_INF_EQ expression	
			{
			  $$ = MakeBinaryCall(CreateIntrinsic("<<="), $1, $3); 
			}
|   expression TK_SUP_SUP_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(">>="), $1, $3); 
			}
|   TK_LPAREN type_name TK_RPAREN expression
		        {
			  $$ = MakeCastExpression($2,$4);
			}
/* (* We handle GCC constructor expressions *) */
|   TK_LPAREN type_name TK_RPAREN TK_LBRACE initializer_list_opt TK_RBRACE
		        {
			  CParserError("GCC constructor expressions not implemented\n");
			}
/* (* GCC's address of labels *)  */
|   TK_AND_AND TK_IDENT  
                        {
			  CParserError("GCC's address of labels not implemented\n");
			}
|   TK_AT_EXPR TK_LPAREN TK_IDENT TK_RPAREN         /* expression pattern variable */
                        {
			  CParserError("GCC's address of labels not implemented\n");
			}
;

constant:
    TK_INTCON			
                        {
			  $$ = MakeConstant($1,is_basic_int);
			}
|   TK_FLOATCON	
                        {
			  $$ = MakeConstant($1,is_basic_float); 
			}
|   TK_CHARCON				
                        {
			  $$ = MakeConstant($1, is_basic_int);
			}
|   string_constant	
                        {
			  $$ = MakeConstant($1, is_basic_string);
                        }
/*add a nul to strings.  We do this here (rather than in the lexer) to make
  concatenation easy below.*/
|   wstring_list	
                        {
			  $$ = MakeConstant(list_to_string($1),is_basic_string);
                        }
;

string_constant:
/* Now that we know this constant isn't part of a wstring, convert it
   back to a string for easy viewing. */
    string_list         
                        {
			  $$ = list_to_string($1);
			}
;
one_string_constant:
/* Don't concat multiple strings.  For asm templates. */
    TK_STRINGCON       
                        {}
;
string_list:
    one_string          
                        {
			  $$ = CONS(STRING,$1,NIL); 
			}
|   string_list one_string
                        {
			  $$ = CONS(STRING,$2,$1);
			}
;

wstring_list:
    TK_WSTRINGCON           
                        {
			  $$ = CONS(STRING,$1,NIL);
			}
|   wstring_list one_string
                        {
			  $$ = CONS(STRING,$2,$1);
			}
|   wstring_list TK_WSTRINGCON  
                        {
			  $$ = gen_nconc($1,CONS(STRING,$2,NIL));
			}
/* Only the first string in the list needs an L, so L"a" "b" is the same
 * as L"ab" or L"a" L"b". */

one_string: 
    TK_STRINGCON	{ }
|   TK_FUNCTION__       
                        { CParserError("TK_FUNCTION not implemented\n"); }
|   TK_PRETTY_FUNCTION__
                        { CParserError("TK_PRETTY_FUNCTION not implemented\n"); }
;    

init_expression:
    expression          { }
|   TK_LBRACE initializer_list_opt TK_RBRACE
			{ $$ = MakeBraceExpression($2); }

initializer_list:    /* ISO 6.7.8. Allow a trailing COMMA */
    initializer 
                        { 
			  $$ = CONS(EXPRESSION,$1,NIL);
			}
|   initializer TK_COMMA initializer_list_opt
                        { 
			  $$ = gen_nconc(CONS(EXPRESSION,$1,NIL),$3);
			}
;
initializer_list_opt:
    /* empty */         { $$ = NIL; }
|   initializer_list    { }
;
initializer: 
    init_designators eq_opt init_expression
                        { 
			  CParserError("Complicated initialization not implemented\n");
			}
|   gcc_init_designators init_expression
                        { 
			  CParserError("gcc init designators not implemented\n");
			}
|   init_expression     { }
;
eq_opt: 
    TK_EQ
                        { }
    /*(* GCC allows missing = *)*/
|   /*(* empty *)*/     
                        { 
			  CParserError("gcc missing = not implemented\n");
			}
;
init_designators: 
    TK_DOT id_or_typename init_designators_opt
                        { }
|   TK_LBRACKET expression TK_RBRACKET init_designators_opt
                        { }
|   TK_LBRACKET expression TK_ELLIPSIS expression TK_RBRACKET
                        { }
;         
init_designators_opt:
    /* empty */         { $$ = NIL; }
|   init_designators    {}
;

gcc_init_designators:  /*(* GCC supports these strange things *)*/
    id_or_typename TK_COLON   
                        { 
			  CParserError("gcc init designators not implemented\n");
			}
;

arguments: 
    /* empty */         { $$ = NIL; }
|   comma_expression    { }
;

opt_expression:
    /* empty */
	        	{ $$ = expression_undefined; }
|   comma_expression
	        	{ $$ = MakeCommaExpression($1); }
;

comma_expression:
    expression                        
                        {
			  $$ = CONS(EXPRESSION, $1, NIL);
			}
|   expression TK_COMMA comma_expression 
                        {
			  $$ = gen_nconc(CONS(EXPRESSION,$1,NIL),$3);
			}
|   error TK_COMMA comma_expression      
                        {
			  CParserError("Parse error: error TK_COMMA comma_expression \n");
			}
;

comma_expression_opt:
    /* empty */         { $$ = NIL; }
|   comma_expression    { }
;

paren_comma_expression:
    TK_LPAREN comma_expression TK_RPAREN
                        {
			  $$ = $2;
			}
|   TK_LPAREN error TK_RPAREN                         
                        {
			  CParserError("Parse error: TK_LPAREN error TK_RPAREN \n");
			}
;

bracket_comma_expression:
    TK_LBRACKET comma_expression TK_RBRACKET 
                        {
			  $$ = $2;
			}
|   TK_LBRACKET error TK_RBRACKET  
                        {
			  CParserError("Parse error: TK_LBRACKET error TK_RBRACKET\n");
			}
;

/*** statements ***/
block: /* ISO 6.8.2 */
    TK_LBRACE local_labels block_attrs declaration_list statement_list TK_RBRACE   
                        {
			  $$ = MakeBlock($4,$5);
			} 
|   error location TK_RBRACE 
                        { CParserError("Parse error: error location TK_RBRACE \n"); } 
;

block_attrs:
   /* empty */          {}
|  TK_BLOCKATTRIBUTE paren_attr_list_ne
                        { CParserError("BLOCKATTRIBUTE not implemented\n"); }
;

declaration_list: 
    /* empty */         { $$ = NIL; }
|   declaration declaration_list
                        {
			  $$ = gen_nconc($1,$2);
			}

;
statement_list:
    /* empty */         { $$ = NIL; }
|   statement statement_list
                        {
			  $$ = gen_nconc(CONS(STATEMENT,$1,NIL),$2);
			}
/*(* GCC accepts a label at the end of a block *)*/
|   TK_IDENT TK_COLON	{ CParserError("gcc not implemented\n"); }
;

local_labels: 
   /* empty */          {}
|  TK_LABEL__ local_label_names TK_SEMICOLON local_labels
                        { CParserError("LABEL__ not implemented\n"); }
;

local_label_names: 
   TK_IDENT             {}
|  TK_IDENT TK_COMMA local_label_names {}
;

statement:
    TK_SEMICOLON
                    	{
			  $$ = MakeNullStatement();
			}
|   comma_expression TK_SEMICOLON
	        	{
			  if (gen_length($1)==1)
			    $$ = call_to_statement(syntax_call(expression_syntax(EXPRESSION(CAR($1)))));
			  else 
			    $$ = call_to_statement(make_call(CreateIntrinsic(","),$1));
			}
|   block               { }
|   TK_IF paren_comma_expression statement                    %prec TK_IF
                	{
			  $$ = test_to_statement(make_test(MakeCommaExpression($2), $3,
							   make_empty_block_statement())); 
			}
|   TK_IF paren_comma_expression statement TK_ELSE statement
	                {
			  $$ = test_to_statement(make_test(MakeCommaExpression($2),$3,$5));
			}
|   TK_SWITCH 
                        {
			  loop_counter++; 
			  is_loop = FALSE; 
			  CurrentSwitchGotoStatements = NIL;
			} 
    paren_comma_expression 
                        {
			  CurrentSwitchController = MakeCommaExpression($3);
			} 
    statement
                        {
			  $$ = MakeSwitchStatement($5);
			}
|   TK_WHILE 
                        {
			  loop_counter++; 
			  is_loop = TRUE;
			} 
    paren_comma_expression statement
	        	{
			  $$ = MakeWhileLoop($3,$4,TRUE);
			}
|   TK_DO
                        {
			  loop_counter++; 
			  is_loop = TRUE;
			} 
    statement TK_WHILE paren_comma_expression TK_SEMICOLON
	        	{
			  $$ = MakeWhileLoop($5,$3,FALSE);
			}
|   TK_FOR
                        {
			  loop_counter++; 
			  is_loop = TRUE;
			} 
    TK_LPAREN for_clause opt_expression TK_SEMICOLON opt_expression TK_RPAREN statement
	                {
			  /*  The for clause may contain declarations*/
			  $$ = MakeForloop($4,$5,$7,$9);
			}
|   TK_IDENT TK_COLON statement
                        {
			  $$ = MakeLabelledStatement($1,$3);
			}
|   TK_CASE expression TK_COLON
                        {
			  $$ = MakeCaseStatement($2);
			}
|   TK_CASE expression TK_ELLIPSIS expression TK_COLON
                        {
			  CParserError("case e1..e2 : not implemented\n");
			}
|   TK_DEFAULT TK_COLON
	                {
			  $$ = MakeDefaultStatement();
			}
|   TK_RETURN TK_SEMICOLON		 
                        {
			  /* This kind of instruction must be added to controlize 
			     $$ = instruction_to_statement(make_instruction_return(expression_undefined));*/
			  $$ = call_to_statement(make_call(CreateIntrinsic("return"),NIL));
			}
|   TK_RETURN comma_expression TK_SEMICOLON
	                {
			  /*$$ = instruction_to_statement(make_instruction_return(MakeCommaExpression($2)));	*/	  
			  $$ =  call_to_statement(make_call(CreateIntrinsic("return"),$2));
			}
|   TK_BREAK TK_SEMICOLON
                        {
			  $$ = MakeBreakStatement();
			}
|   TK_CONTINUE TK_SEMICOLON
                 	{
			  $$ = MakeContinueStatement();
			}
|   TK_GOTO TK_IDENT TK_SEMICOLON
		        {
			  $$ = MakeGotoStatement($2); 
			}
|   TK_GOTO TK_STAR comma_expression TK_SEMICOLON 
                        {
			  CParserError("GOTO * exp not implemented\n");
			}
|   TK_ASM asmattr TK_LPAREN asmtemplate asmoutputs TK_RPAREN TK_SEMICOLON
                        { CParserError("ASM not implemented\n"); }
|   TK_MSASM            
                        { CParserError("ASM not implemented\n"); }
|   error location TK_SEMICOLON  
                        {
			  CParserError("Parse error: error location TK_SEMICOLON\n");
			} 
;

for_clause: 
    opt_expression TK_SEMICOLON   { }
|   declaration
                        {
			  CParserError("For clause containing declaration not implemented\n");
			}
;

declaration:                                /* ISO 6.7.*/
    decl_spec_list init_declarator_list TK_SEMICOLON
                        {
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  $$ = gen_nconc($1,$2);
			}
|   decl_spec_list TK_SEMICOLON	
                        {
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  $$ = $1;
			}
;

init_declarator_list:                       /* ISO 6.7 */
    init_declarator
                        {
			  $$ = CONS(ENTITY,$1,NIL);
			}
|   init_declarator TK_COMMA init_declarator_list
                        {
			  $$ = gen_nconc(CONS(ENTITY,$1,NIL),$3);
			}

;
init_declarator:                             /* ISO 6.7 */
    declarator          { }
|   declarator TK_EQ init_expression
                        { 
			  /* Put init_expression in the initial value of entity declarator*/
			  entity_initial($1) = make_value_expression($3);
			}
;

decl_spec_list:                         /* ISO 6.7 */
                                        /* ISO 6.7.1 */
    TK_TYPEDEF decl_spec_list_opt          
                        {
			  /* Add TYPEDEF_PREFIX to entity name prefix and make it a rom storage */
			  is_typedef = TRUE;
			  CurrentStorage = make_storage_rom();
			  $$ = $2;
			}    
|   TK_EXTERN decl_spec_list_opt           
                        {
			  /* This can be a variable or a function, whose storage is ram or return  */
			  CurrentScope = strdup(concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,NULL)); 
			  $$ = $2;
			}    
|   TK_STATIC decl_spec_list_opt    
                        {
			  is_static = TRUE;
			  $$ = $2;
			}
|   TK_AUTO decl_spec_list_opt           
                        {
			  /* Make dynamic storage for current entity */
			  ram r = make_ram(get_current_module_entity(),
					   DynamicArea,
					   CurrentDynamicAreaOffset,
					   NIL);
			  /* the offset must be recomputed lately, when we know the size of the variable */
			  /* block number ? */
			  CurrentStorage = make_storage_ram(r);
			  $$ = $2;
			}
|   TK_REGISTER decl_spec_list_opt        
                        {
			  /* Add to type variable qualifiers */
			  CurrentQualifiers = gen_nconc(CurrentQualifiers, 
							CONS(QUALIFIER,make_qualifier_register(),NIL));
			  $$ = $2;
			  /* Storage can be formal or dynamic ram of the current entity 
			     Scope depends if the current module is static or not */
			}
                                        /* ISO 6.7.2 */
|   type_spec decl_spec_list_opt_no_named
                        {
			  if (!entity_undefined_p($1))
			    $$ = gen_nconc(CONS(ENTITY,$1,NIL),$2);
			  else
			    $$ = $2;
			}	
                                        /* ISO 6.7.4 */
|   TK_INLINE decl_spec_list_opt
                        { 
			  CParserError("INLINE not implemented\n"); 
			  $$ = $2;
			}	 
|   attribute decl_spec_list_opt        
                        { 
			  $$ = $2;
			}	
/* specifier pattern variable (must be last in spec list) */
|   TK_AT_SPECIFIER TK_LPAREN TK_IDENT TK_RPAREN  
                        { 
			  CParserError("CIL AT not implemented\n"); 
			}	
;

/* (* In most cases if we see a NAMED_TYPE we must shift it. Thus we declare 
    * NAMED_TYPE to have right associativity  *) */
decl_spec_list_opt: 
    /* empty */         { $$ = NIL; } %prec TK_NAMED_TYPE
|   decl_spec_list      { }
;

/* (* We add this separate rule to handle the special case when an appearance 
    * of NAMED_TYPE should not be considered as part of the specifiers but as 
    * part of the declarator. IDENT has higher precedence than NAMED_TYPE  *)
 */
decl_spec_list_opt_no_named: 
    /* empty */         { $$ = NIL;} %prec TK_IDENT
|   decl_spec_list      { }
;


type_spec:   /* ISO 6.7.2 */
    TK_VOID             
                        {
			  CurrentType = make_type_void();
			  $$ = entity_undefined;
                        } 
|   TK_CHAR          
                        {
			  CurrentType = make_standard_integer_type(CurrentType,DEFAULT_CHARACTER_TYPE_SIZE);
			  $$ = entity_undefined;
			}
|   TK_SHORT      
                        {
			  CurrentType = make_standard_integer_type(CurrentType,DEFAULT_SHORT_INTEGER_TYPE_SIZE);
			  $$ = entity_undefined;
			}    
|   TK_INT  
                        {
			  if (CurrentType == type_undefined)
			    {
			      variable v = make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);	
			      CurrentType = make_type_variable(v);
			    }
			  $$ = entity_undefined;
			}  
|   TK_LONG
                        {
			  CurrentType = make_standard_long_integer_type(CurrentType);
			  $$ = entity_undefined;
			}   
|   TK_FLOAT           
                        {
			  variable v = make_variable(make_basic_float(DEFAULT_REAL_TYPE_SIZE),NIL,NIL);
			  CurrentType = make_type_variable(v);
			  $$ = entity_undefined;
			}
|   TK_DOUBLE           
                        {
			  variable v = make_variable(make_basic_float(DEFAULT_DOUBLEPRECISION_TYPE_SIZE),NIL,NIL);
			  CurrentType = make_type_variable(v);
			  $$ = entity_undefined;
			}
|   TK_SIGNED     
                        {
			  /* see the RI document or ri-util.h for explanation */
			  variable v = make_variable(make_basic_int(DEFAULT_SIGNED_TYPE_SIZE*10+
								    DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			  CurrentType = make_type_variable(v);
			  $$ = entity_undefined;
			}
|   TK_UNSIGNED          
                        {
			  variable v = make_variable(make_basic_int(DEFAULT_UNSIGNED_TYPE_SIZE*10+
								    DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			  CurrentType = make_type_variable(v);
			  $$ = entity_undefined;
			}
|   TK_STRUCT id_or_typename                           
                        {
			  /* Find the entity associated to the struct, current scope can be [file%][module:][block]*/
			  entity ent = FindEntityFromLocalNameAndPrefix($2,STRUCT_PREFIX);
			  /* Specify the type of the variable that follows this declaration specifier */
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v);
			  $$ = entity_undefined;
			}
|   TK_STRUCT id_or_typename TK_LBRACE { CurrentDerivedName = $2; } struct_decl_list TK_RBRACE
                        {
			  /* Create the struct entity */
			  entity ent = MakeDerivedEntity($2,$5,is_external,1);
			  /* Specify the type of the variable that follows this declaration specifier*/
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v); 
			  $$ = ent;
			}
|   TK_STRUCT TK_LBRACE {
			  CurrentDerivedName = strdup(concatenate("PIPS_STRUCT_",int_to_string(derived_counter++),NULL)); 
			  $<string>$ = CurrentDerivedName;
                        }
    struct_decl_list TK_RBRACE
                        {
			  /* Create the struct entity with unique name s */
			  entity ent = MakeDerivedEntity($<string>3,$4,is_external,1);	
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v); 
			  $$ = ent;
			}
|   TK_UNION id_or_typename 
                        {
			  /* Find the entity associated to the union, current scope can be [file%][module:][block]*/
			  entity ent = FindEntityFromLocalNameAndPrefix($2,UNION_PREFIX);
			  /* Specify the type of the variable that follows this declaration specifier */
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v); 
			  $$ = entity_undefined;
			}
|   TK_UNION id_or_typename TK_LBRACE { CurrentDerivedName = $2; } struct_decl_list TK_RBRACE
                        {
			  entity ent = MakeDerivedEntity($2,$5,is_external,2);
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v);
			  $$ = ent;
			}
|   TK_UNION TK_LBRACE  { 
			  CurrentDerivedName =  strdup(concatenate("PIPS_UNION_",int_to_string(derived_counter++),NULL));  
			  $<string>$ = CurrentDerivedName;
                        }
    struct_decl_list TK_RBRACE
                        {
			  /* Create the union entity with unique name */
			  entity ent = MakeDerivedEntity($<string>3,$4,is_external,2);	
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v);
			  $$ = ent;
			}
|   TK_ENUM id_or_typename   
                        {
                          /* Find the entity associated to the enum */
			  entity ent = FindEntityFromLocalNameAndPrefix($2,ENUM_PREFIX);
			  /* Specify the type of the variable that follows this declaration specifier */
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v);
			  $$ = entity_undefined;  
			}
|   TK_ENUM id_or_typename TK_LBRACE enum_list maybecomma TK_RBRACE
                        {
                          /* Create the enum entity */
			  entity ent = MakeDerivedEntity($2,$4,is_external,3);
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v);  
			  $$ = ent;
			}                   
|   TK_ENUM TK_LBRACE enum_list maybecomma TK_RBRACE
                        {
			  /* Create the enum entity with unique name */
			  string s = strdup(concatenate("PIPS_ENUM_",int_to_string(derived_counter++),NULL));
			  entity ent = MakeDerivedEntity(s,$3,is_external,3);
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  CurrentType = make_type_variable(v);  
			  $$ = ent;
			}
|   TK_NAMED_TYPE  
                        {
			  entity ent = FindEntityFromLocalNameAndPrefix($1,TYPEDEF_PREFIX);	
			  /* Specify the type of the variable that follows this declaration specifier */
			  if (is_typedef)
			    {
			      /* typedef T1 T2 => the type of T2 will be that of T1*/
			      pips_debug(5,"typedef T1 T2 where T1 =  %s\n",entity_name(ent));
			      CurrentType = entity_type(ent);
			      $$ = ent;
			    }
			  else
			    {
			      /* T1 var => the type of var is basic typedef */
			      variable v = make_variable(make_basic_typedef(ent),NIL,NIL);
			      pips_debug(5,"T1 var where T1 =  %s\n",entity_name(ent));
			      CurrentType = make_type_variable(v);  
			      $$ = entity_undefined;
			    }
			}
|   TK_TYPEOF TK_LPAREN expression TK_RPAREN  
                        {
                          CParserError("TYPEOF not implemented\n");
			}
|   TK_TYPEOF TK_LPAREN type_name TK_RPAREN    
                        {
			  CParserError("TYPEOF not implemented\n");
			}
;

struct_decl_list: /* (* ISO 6.7.2. Except that we allow empty structs. We 
                      * also allow missing field names. *)
                   */
    /* empty */         { $$ = NIL; }
|   decl_spec_list TK_SEMICOLON struct_decl_list
                        {
			  /* Create the struct member entity with unique name, the name of the 
			     struct/union is added to the member name prefix */
			  string s = strdup(concatenate("PIPS_MEMBER_",int_to_string(derived_counter++),NULL));  
			  entity ent = CreateEntityFromLocalNameAndPrefix(s,strdup(concatenate(CurrentDerivedName,
											       MEMBER_SEP_STRING,NULL)),
									  is_external);
			  pips_debug(4,"Current derived name: %s\n",CurrentDerivedName);
			  pips_debug(4,"Member name: %s\n",entity_name(ent));
			  entity_storage(ent) = make_storage_rom();
			  entity_type(ent) = CurrentType; 
			  $$ = gen_nconc(CONS(ENTITY,ent,NIL),$3); 
			  
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			}             
|   decl_spec_list
                        {
			  /* Add struct/union name and MEMBER_SEP_STRING to entity name */
                          CurrentScope = CreateMemberScope(CurrentDerivedName,is_external);
			  CurrentStorage = make_storage_rom();
                        }
    field_decl_list TK_SEMICOLON struct_decl_list
                        {
			  /* Create the list of member entities */
			  $$ = gen_nconc($3,$5);
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			}
|   error TK_SEMICOLON struct_decl_list
                        {
			  CParserError("Parse error: error TK_SEMICOLON struct_decl_list\n");
			}
;

field_decl_list: /* (* ISO 6.7.2 *) */
    field_decl          
                        {
			  $$ = CONS(ENTITY,$1,NIL);
			}
|   field_decl TK_COMMA field_decl_list    
                        {
			  $$ = gen_nconc(CONS(ENTITY,$1,NIL),$3);
			}
;

field_decl: /* (* ISO 6.7.2. Except that we allow unnamed fields. *) */
    declarator          { }
|   declarator TK_COLON expression   
                        {
			  variable v = make_variable(make_basic_bit(integer_constant_expression_value($3)),NIL,NIL);
			  pips_assert("Width of bit-field must be a constant integer", 
				      integer_constant_expression_p($3));
			  /* Ignore for this moment if the bit is signed or unsigned */
			  entity_type($1) = make_type_variable(v);
			  $$ = $1;
			}  
|   TK_COLON expression 
                        {
			  /* Unnamed bit-field : special and unique name */
			  string s = strdup(concatenate("PIPS_MEMBER_",int_to_string(derived_counter++),NULL));  
			  entity ent = CreateEntityFromLocalNameAndPrefix(s,CurrentScope,is_external);
			  variable v = make_variable(make_basic_bit(integer_constant_expression_value($2)),NIL,NIL);
			  pips_assert("Width of bit-field must be a constant integer", 
				      integer_constant_expression_p($2));
			  entity_type(ent) = make_type_variable(v);
			  entity_storage(ent) = make_storage_rom();
			  $$ = ent;
			}
;

enum_list: /* (* ISO 6.7.2.2 *) */
    enumerator	
                        {
			  /* initial_value = 0 or $3*/
			  $$ = CONS(ENTITY,$1,NIL);
			  enum_counter = 1; 
			}
|   enum_list TK_COMMA enumerator	       
                        {
			  /* Attention to the reverse recursive definition*/
			  $$ = gen_nconc($1,CONS(ENTITY,$3,NIL));
			  enum_counter ++;
			}
|   enum_list TK_COMMA error      
                        {
			  CParserError("Parse error: enum_list TK_COMMA error\n");
			}
;

enumerator:	
    TK_IDENT	
                        {
			  /* Create an entity of is_basic_int, storage rom 
			     initial_value = 0 if it is the first member
			     initial_value = intial_value(precedessor) + 1

			  No need to add CurrentDerivedName to the name's scope of the member entity 
			  for ENUM, as in the case of STRUCT and UNION */

			  entity ent = CreateEntityFromLocalNameAndPrefix($1,"",is_external);
			  variable v = make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			  entity_storage(ent) = make_storage_rom();
			  entity_type(ent) = make_type_variable(v);
			  /*  entity_initial(ent) = MakeEnumeratorInitialValue(enum_list,enum_counter);*/
			  $$ = ent;
			}
|   TK_IDENT TK_EQ expression	
                        {
			  /* Create an entity of is_basic_int, storage rom, initial_value = $3 */
			  int i; 
			  entity ent = CreateEntityFromLocalNameAndPrefix($1,"",is_external);
			  variable v = make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			  pips_assert("Enumerated value must be a constant integer", 
				      integer_constant_expression_p($3));
			  i = integer_constant_expression_value($3);
			  entity_storage(ent) = make_storage_rom();
			  entity_type(ent) = make_type_variable(v);
			  entity_initial(ent) = make_value_constant(make_constant_int(i));
			  $$ = ent;
			}
;

declarator:  /* (* ISO 6.7.5. Plus Microsoft declarators.*) */
    pointer_opt direct_decl attributes_with_asm
                        { $$ = $2; }
;

direct_decl: /* (* ISO 6.7.5 *) */
                                   /* (* We want to be able to redefine named
                                    * types as variable names *) */
    id_or_typename
                       {
			 $$  = MakeCurrentEntity($1,CurrentScope,CurrentStorage,CurrentType,CurrentQualifiers,
						 is_typedef,is_static,is_external,is_formal,offset);
		       }
|   TK_LPAREN attributes declarator TK_RPAREN
                        {
			  /* Add attributes such as const, restrict, ... to variable's qualifiers */
			  $$ = $3;
			}
|   direct_decl TK_LBRACKET attributes comma_expression_opt TK_RBRACKET
                        { 
			 /* Declare the variable array, what can be attributes  ??? 
			    Why comma_expression ??? when this comma_expression is empty,
			    the array is of unknown size => can be determined by the intialization ?*/
			  type t = entity_type($1);
			  ifdebug(3) 
			    {
			      pips_debug(3,"List of expressions:\n");
			      print_expressions($4);
			    }
			  if (type_variable_p(t))
			    {
			      variable v = type_variable(t);
			      list l = NIL; 
			      MAP(EXPRESSION,e,
			      {
				int up;
				dimension d; 
				if (expression_integer_value(e,&up))
				  d = make_dimension(int_to_expression(0),int_to_expression(up-1));
				else 
				  d = make_dimension(int_to_expression(0),MakeBinaryCall(CreateIntrinsic("-C"),e,
											 int_to_expression(1)));
				l = gen_nconc(CONS(DIMENSION,d,NIL),l);
			      },$4);
			      if (l == NIL)
				l = CONS(DIMENSION,make_dimension(int_to_expression(0),make_unbounded_expression()),NIL);
			      variable_dimensions(v) = gen_nconc(variable_dimensions(v),l);
			    }
			  else 
			    {
			      CParserError("Parse error: not variable type\n");
			    }
			  $$ = $1;
			}
|   direct_decl TK_LBRACKET attributes error TK_RBRACKET
                        {
			  CParserError("Parse error: direct_decl TK_LBRACKET attributes error TK_RBRACKET\n");
			}
|   direct_decl parameter_list_startscope rest_par_list TK_RPAREN
                        {
			  /* Update the function entity */ 
			  functional f = make_functional($3,CurrentType);
			  entity_type($1) = make_type_functional(f);
			  entity_storage($1) = make_storage_return(entity_undefined);
			  $$ = $1;
			}
;

parameter_list_startscope: 
    TK_LPAREN           { offset = 1;}
;

rest_par_list:
    /* empty */         { $$ = NIL; }
|   parameter_decl rest_par_list1
                        {
			  $$ = gen_nconc(CONS(PARAMETER,$1,NIL),$2);
			}
;
rest_par_list1: 
    /* empty */         { $$ = NIL; }
|   TK_COMMA TK_ELLIPSIS 
                        {
			  CParserError("Variable parameter list not implemented\n");
			}
|   TK_COMMA { offset++; } parameter_decl rest_par_list1 
                        {
			  $$ = gen_nconc(CONS(PARAMETER,$3,NIL),$4);
			}  
;    

parameter_decl: /* (* ISO 6.7.5 *) */
    decl_spec_list {is_formal = TRUE; } declarator 
                        {
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  is_formal = FALSE;
			  $$ = make_parameter(entity_type($3),make_mode(CurrentMode,UU));
			 /* Set CurentMode where ???? */
			}
|   decl_spec_list abstract_decl 
                        {
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  $$ = make_parameter(CurrentType,make_mode(CurrentMode,UU));
			}
|   decl_spec_list              
                        {
			  /* function prototype*/
			    /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  $$ = make_parameter(CurrentType,make_mode(CurrentMode,UU));
			}
|   TK_LPAREN parameter_decl TK_RPAREN    
                        { $$ = $2; } 
;

/* (* Old style prototypes. Like a declarator *) */
old_proto_decl:
    pointer_opt direct_old_proto_decl 
                        {
			  CParserError("Old-style function prototype not implemented\n");
			}
;
direct_old_proto_decl:
    direct_decl TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list
                        { 
			  CParserError("Old-style function prototype not implemented\n");
			}
|   direct_decl TK_LPAREN TK_RPAREN
                        {
			  CParserError("Old-style function prototype not implemented\n");
			}
;

old_parameter_list_ne:
    TK_IDENT            
                        {
			  CParserError("Old-style function prototype not implemented\n");
			}
|   TK_IDENT TK_COMMA old_parameter_list_ne   
                        {
			  CParserError("Old-style function prototype not implemented\n");
			}
;

old_pardef_list: 
    /* empty */         {}
|   decl_spec_list old_pardef TK_SEMICOLON TK_ELLIPSIS
                        {
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  CParserError("Old-style function prototype not implemented\n");
			}
|   decl_spec_list old_pardef TK_SEMICOLON old_pardef_list  
                        {
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  CParserError("Old-style function prototype not implemented\n");
			} 
;

old_pardef: 
    declarator            
                        {
			  CParserError("Old-style function prototype not implemented\n");
			}
|   declarator TK_COMMA old_pardef   
                        {
			  CParserError("Old-style function prototype not implemented\n");
			}
|   error       
                        {
			  CParserError("Parse error: error \n");
			}
;

pointer: /* (* ISO 6.7.5 *) */ 
    TK_STAR attributes pointer_opt 
                        {
			  variable v = make_variable(make_basic_pointer(CurrentType),NIL,NIL);
			  CurrentType = make_type_variable(v);
			}
;

pointer_opt:
    /* empty */         {}
|   pointer             
                        {}
;

type_name: /* (* ISO 6.7.6 *) */
    decl_spec_list abstract_decl
                        {
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			}
|   decl_spec_list      
                        {
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			}
;

abstract_decl: /* (* ISO 6.7.6. *) */
    pointer_opt abs_direct_decl attributes  
                        {
			 
			}
|   pointer                     
                        { }
;

abs_direct_decl: /* (* ISO 6.7.6. We do not support optional declarator for 
                     * functions. Plus Microsoft attributes. See the 
                     * discussion for declarator. *) */
    TK_LPAREN attributes abstract_decl TK_RPAREN
                        {
			}
|   TK_LPAREN error TK_RPAREN
                        {
			  CParserError("Parse error: TK_LPAREN error TK_RPAREN\n");
			}
            
|   abs_direct_decl_opt TK_LBRACKET comma_expression_opt TK_RBRACKET
                        {
			  if (type_variable_p(CurrentType))
			    {
			      variable v = type_variable(CurrentType);
			      variable_dimensions(v) = $3;
			    }
			  else 
			    CParserError("Parse error: current type is not variable\n");
			}
/*(* The next shoudl be abs_direct_decl_opt but we get conflicts *)*/
|   abs_direct_decl parameter_list_startscope rest_par_list TK_RPAREN
                        {
			  /* function */
			  functional f = make_functional($3,CurrentType);
			  CurrentType = make_type_functional(f);
			}  
;

abs_direct_decl_opt:
    abs_direct_decl    
                        {
			}
|   /* empty */         {}
;

function_def:  /* (* ISO 6.9.1 *) */
    function_def_start block   
                        {
			  /* Make value_code for current module here 
			     Attention with execution order ? */
			  InitializeBlock($2);	  
			  ResetCurrentModule(); 
			}	

function_def_start:  /* (* ISO 6.9.1 *) */
    decl_spec_list declarator   
                        { 
			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  is_external = FALSE; 
			  /* The current module entity is created with declarator */
			  pips_debug(2,"Case 1 Create module entity: %s\n",entity_local_name($2));
			  CreateCurrentModule($2);
			}	
/* (* Old-style function prototype *) */
|   decl_spec_list old_proto_decl 
                        { 
			  /* Where to put is_external = FALSE; ???
			     ATTENTION: conflicts with the above rule when using mid-rule actions */

			  /* Reset the global variables */
			  CurrentScope = NULL;
			  CurrentType = type_undefined;
			  CurrentStorage = storage_undefined; 
			  CurrentQualifiers = NIL;
			  is_typedef = FALSE;
			  is_static = FALSE;
			  is_external = FALSE; 
			  CParserError("Old-style function prototype not implemented\n"); 
			}	
/* (* New-style function that does not have a return type *) */
|   TK_IDENT parameter_list_startscope rest_par_list TK_RPAREN 
                        { 
			  /* Create the current function */
			  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			  functional f = make_functional($3,make_type_unknown());
			  entity_type(e) = make_type_functional(f);
			  is_external = FALSE; 
			  pips_debug(2,"Case 2 Create module entity: %s\n",$1);
			  CreateCurrentModule(e);
			}	
/* (* No return type and old-style parameter list *) */
|   TK_IDENT TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list
                        { 
			  /* Where to put is_external = FALSE; ???
			     ATTENTION: conflicts with the above rule when using mid-rule actions */
			  is_external = FALSE; 
			  CParserError("Old-style function prototype not implemented\n"); 
			}	
/* (* No return type and no parameters *) */
|   TK_IDENT TK_LPAREN TK_RPAREN
                        { 
			  /* Create the current function */
			  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			  functional f = make_functional(NIL,make_type_unknown());
			  entity_type(e) = make_type_functional(f);
			  is_external = FALSE; 
			  pips_debug(2,"Case 3 Create module entity: %s\n",$1);
			  CreateCurrentModule(e);
			}	
;

/*** GCC attributes ***/
attributes:
    /* empty */				
                        { }	
|   attribute attributes
                        { }	
;

/* (* In some contexts we can have an inline assembly to specify the name to 
    * be used for a global. We treat this as a name attribute *) */
attributes_with_asm:
    /* empty */                         
                        { }	
|   attribute attributes_with_asm       
                        { }	
|   TK_ASM TK_LPAREN string_constant TK_RPAREN attributes        
                        { CParserError("ASM not implemented\n"); }                                        
;
 
attribute:
    TK_ATTRIBUTE TK_LPAREN paren_attr_list_ne TK_RPAREN	
                        { CParserError("ATTRIBUTE not implemented\n"); }	                                       
|   TK_DECLSPEC paren_attr_list_ne       
                        { CParserError("ATTRIBUTE not implemented\n"); }	
|   TK_MSATTR                             
                        { CParserError("ATTRIBUTE not implemented\n"); }	
                                        /* ISO 6.7.3 */
|   TK_CONST                              
                        { 
			  /* Add CONST to type variable qualifiers */
			  CurrentQualifiers = gen_nconc(CurrentQualifiers, 
							CONS(QUALIFIER,make_qualifier_const(),NIL));
			}	
|   TK_RESTRICT                            
                        { 
			  /* Add RESTRICT to type variable qualifiers */
			  CurrentQualifiers = gen_nconc(CurrentQualifiers, 
							CONS(QUALIFIER,make_qualifier_restrict(),NIL));
			}	
|   TK_VOLATILE                            
                        { 
			  /* Add VOLATILE to type variable qualifiers */ 
			  CurrentQualifiers = gen_nconc(CurrentQualifiers, 
							CONS(QUALIFIER,make_qualifier_volatile(),NIL));
			}	
;

/** (* PRAGMAS and ATTRIBUTES *) ***/
/* (* We want to allow certain strange things that occur in pragmas, so we 
    * cannot use directly the language of expressions *) */ 
attr: 
|   id_or_typename                       
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_IDENT TK_COLON TK_INTCON                  
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_DEFAULT TK_COLON TK_INTCON               
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_IDENT TK_LPAREN  TK_RPAREN                 
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_IDENT paren_attr_list_ne             
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_INTCON                              
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   string_constant                      
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_CONST                                
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_SIZEOF expression                     
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_SIZEOF TK_LPAREN type_name TK_RPAREN	                         
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	

|   TK_ALIGNOF expression                   
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_ALIGNOF TK_LPAREN type_name TK_RPAREN      
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_PLUS expression    	                 
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_MINUS expression 		        
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_STAR expression		       
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_AND expression				                 %prec TK_ADDROF
	                                
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_EXCLAM expression		       
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_TILDE expression		        
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_PLUS attr                      
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	 
|   attr TK_MINUS attr                    
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_STAR expression               
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_SLASH attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_PERCENT attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_AND_AND attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_PIPE_PIPE attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_AND attr			
                        { 
			  CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); 
		        }
|   attr TK_PIPE attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_CIRC attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_EQ_EQ attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_EXCLAM_EQ attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_INF attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_SUP attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_INF_EQ attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_SUP_EQ attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_INF_INF attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_SUP_SUP attr			
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_ARROW id_or_typename          
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_DOT id_or_typename            
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_LPAREN attr TK_RPAREN                 
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	 
;

attr_list_ne:
|   attr                                  
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_COMMA attr_list_ne               
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   error TK_COMMA attr_list_ne              
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
;
paren_attr_list_ne: 
    TK_LPAREN attr_list_ne TK_RPAREN            
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_LPAREN error TK_RPAREN                   
                        { CParserError("PRAGMAS and ATTRIBUTES not implemented\n"); }	
;
/*** GCC TK_ASM instructions ***/
asmattr:
    /* empty */                        
                        { CParserError("ASM not implemented\n"); }
|   TK_VOLATILE  asmattr                  
                        { CParserError("ASM not implemented\n"); }
|   TK_CONST asmattr                      
                        { CParserError("ASM not implemented\n"); } 
;
asmtemplate: 
    one_string_constant                          
                        { CParserError("ASM not implemented\n"); }
|   one_string_constant asmtemplate              
                        { CParserError("ASM not implemented\n"); }
;
asmoutputs: 
    /* empty */           
                        { CParserError("ASM not implemented\n"); }
|   TK_COLON asmoperands asminputs                       
                        { CParserError("ASM not implemented\n"); }
;
asmoperands:
    /* empty */                       
                        { CParserError("ASM not implemented\n"); }
|   asmoperandsne                      
                        { CParserError("ASM not implemented\n"); }
;
asmoperandsne:
    asmoperand                         
                        { CParserError("ASM not implemented\n"); }
|   asmoperandsne TK_COMMA asmoperand    
                        { CParserError("ASM not implemented\n"); }
;
asmoperand:
    string_constant TK_LPAREN expression TK_RPAREN    
                        { CParserError("ASM not implemented\n"); }
|   string_constant TK_LPAREN error TK_RPAREN         
                        { CParserError("ASM not implemented\n"); }
; 
asminputs: 
    /* empty */               
                        { CParserError("ASM not implemented\n"); }
|   TK_COLON asmoperands asmclobber
                        { CParserError("ASM not implemented\n"); }                        
;
asmclobber:
    /* empty */                         
                        { CParserError("ASM not implemented\n"); }
|   TK_COLON asmcloberlst_ne            
                        { CParserError("ASM not implemented\n"); }
;
asmcloberlst_ne:
    one_string_constant 
                        { CParserError("ASM not implemented\n"); }                          
|   one_string_constant TK_COMMA asmcloberlst_ne 
                        { CParserError("ASM not implemented\n"); }
;
  
%%

