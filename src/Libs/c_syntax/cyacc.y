/* $Id$ 
   $Log: cyacc.y,v $
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

%{ /* C declarations */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#include "c_syntax.h"

extern int c_lex(void);
extern void c_error(char *);

#define C_ERROR_VERBOSE 1 /* much clearer error messages with bison */

static type CurrentType = type_undefined; /* the type of an entity */
static int CurrentTypeSize; /* number of bytes to store a value of that type */

static entity expression_to_entity(e)
expression e;
{
    syntax s = expression_syntax(e);
    
    switch (syntax_tag(s))
    {
    case is_syntax_call:
	return call_function(syntax_call(s));
    case is_syntax_reference:
	return reference_variable(syntax_reference(s));
    case is_syntax_range:
    default: 
	pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
	return entity_undefined; 
    }
}

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
}

%token <string> TK_IDENT
%token <string> TK_CHARCON
%token <string> TK_INTCON
%token <string> TK_FLOATCON
%token <string> TK_NAMED_TYPE

%token <liste> TK_STRINGCON   
%token <liste> TK_WSTRINGCON

%token TK_EOF
%token TK_CHAR TK_INT TK_DOUBLE TK_FLOAT TK_VOID TK_INT64 TK_INT32
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

%type <instruction> initializer
%type <liste> initializer_list
%type <> init_designators init_designators_opt

%type <liste> decl_spec_list
%type <tag> type_spec
%type <liste> struct_decl_list


%type <> old_proto_decl
%type <> parameter_decl
%type <> enumerator
%type <> enum_list
%type <> declaration function_def
%type <> function_def_start
%type <type> type_name
%type <> block
%type <liste> local_labels local_label_names
%type <liste> old_parameter_list_ne

%type <> init_declarator
%type <liste> init_declarator_list
%type <> declarator
%type <> field_decl
%type <liste> field_decl_list
%type <> direct_decl
%type <> abs_direct_decl abs_direct_decl_opt
%type <> abstract_decl
%type <> pointer pointer_opt 
%type <> location

%type <string> id_or_typename
%type <liste> comma_expression_opt
%type <liste> initializer_list_opt
%type <string> one_string_constant
%type <string> one_string

%%

interpret: file TK_EOF		 /* do nothing */	
;
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
    declaration                          
|   function_def                           
|   TK_ASM TK_LPAREN string_constant TK_RPAREN TK_SEMICOLON
                        { printf("ASM not implemented\n"); }
|   TK_PRAGMA attr			
                        { 
			  printf("PRAGMA not implemented\n"); 
			}
/* Old-style function prototype. This should be somewhere else, like in
   "declaration". For now we keep it at global scope only because in local
   scope it looks too much like a function call */
|   TK_IDENT TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list TK_SEMICOLON
                        { 
			  /* Make a function entity here 
			     MakeFunction($1,$3,$5);*/
			  printf("Old-style function prototype not implemented\n");
			 
			}
/* Old style function prototype, but without any arguments */
|   TK_IDENT TK_LPAREN TK_RPAREN TK_SEMICOLON
                        {
			  /* Make a function entity here 
			     MakeFunction($1);*/
			  printf("Old-style function prototype not implemented\n"); 
			}
/* transformer for a toplevel construct */
|   TK_AT_TRANSFORM TK_LBRACE global TK_RBRACE TK_IDENT /*to*/ TK_LBRACE globals TK_RBRACE 
                        { 
			  printf("CIL AT not implemented\n"); 
			}
/* transformer for an expression */
|   TK_AT_TRANSFORMEXPR TK_LBRACE expression TK_RBRACE TK_IDENT /*to*/ TK_LBRACE expression TK_RBRACE 
                        { 
			  printf("CIL AT not implemented\n"); 
			}
|   location error TK_SEMICOLON 
                        { 
			  /* Parse error, do what ? */
			  printf("Parse error: location error TK_SEMICOLON \n");
			}
;

id_or_typename:
    TK_IDENT			
                        {
			  /* FindOrCreateEntity(CurrentPackage, $1); for identifier */
			}
|   TK_NAMED_TYPE				
                        {
			   /* FindOrCreateEntity(CurrentPackage, $1); for a typedef entity */
			}
|   TK_AT_NAME TK_LPAREN TK_IDENT TK_RPAREN         
                        {
			   printf("CIL AT not implemented\n"); 
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
			  /* entity ent =  FindOrCreateEntity(CurrentPackage, $1);*/
                          /* $$ = MakeNullaryCall(ent);  */
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
			  printf("ALIGNOF not implemented\n"); 
			}
|   TK_ALIGNOF TK_LPAREN type_name TK_RPAREN
		        { 
			  printf("ALIGNOF not implemented\n"); 
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
			  /* entity ent = FindOrCreateEntity(CurrentPackage,$3);
			     expression exp = entity_to_expression(ent);
			    $$ = MakeBinaryCall(CreateIntrinsic("->"),$1,exp);*/
			}
|   expression TK_DOT id_or_typename
		        {
			  /* entity ent = FindOrCreateEntity(CurrentPackage,$3);
			     expression exp = entity_to_expression(ent);
			     $$ = MakeBinaryCall(CreateIntrinsic("."),$1,exp);*/
			}
|   TK_LPAREN block TK_RPAREN
		        {
			  printf("GNU extension not implemented\n");
			}
|   paren_comma_expression
		        {
			  /* paren_comma_expression is a list of expressions*/ 
			  $$ = MakeCommaExpression($1);
			}
|   expression TK_LPAREN arguments TK_RPAREN
			{
			  $$ = make_call_expression(expression_to_entity($1),$3);
			}
|   TK_BUILTIN_VA_ARG TK_LPAREN expression TK_COMMA type_name TK_RPAREN
                        {
			  printf("BUILTIN_VA_ARG not implemented\n");
			}
|   expression bracket_comma_expression
			{
			  /* pips_assert("The expression must be an array name or pointer")*/
			  entity ent = expression_to_entity($1);
			  $$ = reference_to_expression(make_reference(ent,$2));
			}
|   expression TK_QUEST opt_expression TK_COLON expression
			{
			  MakeTernaryCallExpr(CreateIntrinsic("?"), $1, $3, $5);
			}
|   expression TK_PLUS expression
			{ 
			  $$ = MakeBinaryCall(CreateIntrinsic("+"), $1, $3); 
			}
|   expression TK_MINUS expression
			{ 
			  $$ = MakeBinaryCall(CreateIntrinsic("-"), $1, $3); 
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
			  printf("GCC constructor expressions not implemented\n");
			}
/* (* GCC's address of labels *)  */
|   TK_AND_AND TK_IDENT  
                        {
			  printf("GCC's address of labels not implemented\n");
			}
|   TK_AT_EXPR TK_LPAREN TK_IDENT TK_RPAREN         /* expression pattern variable */
                        {
			  printf("GCC's address of labels not implemented\n");
			}
;

constant:
    TK_INTCON			
                        {
			  $$ = MakeConstant($1,is_basic_int);
			  free($1);
			}
|   TK_FLOATCON	
                        {
			  $$ = MakeConstant($1,is_basic_float); 
			}
|   TK_CHARCON				
                        {
			  $$ = MakeConstant($1, is_basic_char);
			  free($1);	 
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
                        {
			  $$ = list_to_string($1);
			}
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
			  $$ = $1;
			}
|   wstring_list one_string
                        {
			  $$ = CONS(STRING,$2,$1);
			}
|   wstring_list TK_WSTRINGCON  
                        {
			  $$ = gen_nconc($1,$2);
			}
/* Only the first string in the list needs an L, so L"a" "b" is the same
 * as L"ab" or L"a" L"b". */

one_string: 
    TK_STRINGCON				
                        {
			  $$ = list_to_string($1);    
			}
|   TK_FUNCTION__       
                        { printf("TK_FUNCTION not implemented\n"); }
|   TK_PRETTY_FUNCTION__
                        { printf("TK_PRETTY_FUNCTION not implemented\n"); }
;    

/* How to put all these initializations into intial_value ???*/

init_expression:
    expression          { $$ = $1; }
|   TK_LBRACE initializer_list_opt TK_RBRACE
			{}

initializer_list:    /* ISO 6.7.8. Allow a trailing COMMA */
    initializer                             {}
|   initializer TK_COMMA initializer_list_opt  {}
;
initializer_list_opt:
    /* empty */         {}
|   initializer_list
                        { $$ = $1; }
;
initializer: 
    init_designators eq_opt init_expression {}
|   gcc_init_designators init_expression
                        { 
			  printf("gcc init designators not implemented\n");
			}
|   init_expression     
                        {
			}
;
eq_opt: 
    TK_EQ
                        {
			}
    /*(* GCC allows missing = *)*/
|   /*(* empty *)*/     
                        { 
			  printf("gcc missing = not implemented\n");
			}
;
init_designators: 
    TK_DOT id_or_typename init_designators_opt      {}
|   TK_LBRACKET expression TK_RBRACKET init_designators_opt
                                        {}
|   TK_LBRACKET expression TK_ELLIPSIS expression TK_RBRACKET
                                        {}
;         
init_designators_opt:
    /* empty */                          {}
|   init_designators                     {}
;

gcc_init_designators:  /*(* GCC supports these strange things *)*/
    id_or_typename TK_COLON   
                        { 
			  printf("gcc init designators not implemented\n");
			}
;

arguments: 
    /* empty */         {}
|   comma_expression    { $$ = $1; }
;

opt_expression:
    /* empty */
	        	{}
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
			  $$ = gen_nconc($3, CONS(EXPRESSION, $1, NIL));
			}
|   error TK_COMMA comma_expression      
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: error TK_COMMA comma_expression \n");
			}
;

comma_expression_opt:
    /* empty */         {}
|   comma_expression    { $$ = $1; }
;

paren_comma_expression:
    TK_LPAREN comma_expression TK_RPAREN
                        {
			  $$ = $2;
			}
|   TK_LPAREN error TK_RPAREN                         
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: TK_LPAREN error TK_RPAREN \n");
			}
;

bracket_comma_expression:
    TK_LBRACKET comma_expression TK_RBRACKET 
                        {
			  $$ = $2;
			}
|   TK_LBRACKET error TK_RBRACKET  
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: TK_LBRACKET error TK_RBRACKET\n");
			}
;


/*** statements ***/
block: /* ISO 6.8.2 */
    TK_LBRACE local_labels block_attrs declaration_list statement_list TK_RBRACE   
                        { } 
|   error location TK_RBRACE 
                        { printf("Parse error: error location TK_RBRACE \n"); } 
;

block_attrs:
   /* empty */          {}
|  TK_BLOCKATTRIBUTE paren_attr_list_ne
                        { printf("BLOCKATTRIBUTE not implemented\n"); }
;

declaration_list: 
    /* empty */         {}
|   declaration declaration_list {}

;
statement_list:
|   /* empty */         {}
|   statement statement_list {}
/*(* GCC accepts a label at the end of a block *)*/
|   TK_IDENT TK_COLON	{ printf("gcc not implemented\n"); }
;

local_labels: 
   /* empty */          {}
|  TK_LABEL__ local_label_names TK_SEMICOLON local_labels
                        { printf("LABEL__ not implemented\n"); }
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
			  /*sequence s = sequence_undefined;
			  MAP(EXPRESSION,exp,
			  {
			    call c = expression_call(exp);
			    instruction i = call_to_instruction(c);
			    sequence_statement(s) = CONS(STATEMENT,si,sequence_statement(s)); 
			  },comma_expression);
			  $$ = sequence_to_instruction(s);*/
			}
|   block               {}
|   TK_IF paren_comma_expression statement                    %prec TK_IF
                	{
			  
			  $$ = MakeLogicalIfInst(MakeCommaExpression($2), $3); 
			}
|   TK_IF paren_comma_expression statement TK_ELSE statement
	                {
			  $$ = test_to_instruction(make_test(MakeCommaExpression($2),$3,$5));
			}
|   TK_SWITCH paren_comma_expression statement
                        {
			  multitest mt = make_multitest(MakeCommaExpression($2),$3);
			  $$ = make_instruction(is_instruction_multitest,mt);
			}
|   TK_WHILE paren_comma_expression statement
	        	{
			  whileloop w = make_whileloop(MakeCommaExpression($2),
						       $3,entity_undefined,make_evaluation_before());
			  $$ = make_instruction(is_instruction_whileloop,w);
			}
|   TK_DO statement TK_WHILE paren_comma_expression TK_SEMICOLON
	        	{
			  whileloop w = make_whileloop(MakeCommaExpression($4),
						       $2,entity_undefined,make_evaluation_after());
			  $$ = make_instruction(is_instruction_whileloop,w);
			}
|   TK_FOR TK_LPAREN for_clause opt_expression TK_SEMICOLON opt_expression TK_RPAREN statement
	                {
			  /*  The for clause may contain declarations 
			      make_forloop($3,$4,$6,$8);*/
			}
|   TK_IDENT TK_COLON statement
                        {
			  /* make_statement_label($1) */
			  
			}
|   TK_CASE expression TK_COLON
                        {
			  /* Create corresponding goto */
			}
|   TK_CASE expression TK_ELLIPSIS expression TK_COLON
                        {
			  /* Create corresponding goto */
			}
|   TK_DEFAULT TK_COLON
	                {
			  /* Create corresponding goto */
			}
|   TK_RETURN TK_SEMICOLON		 
                        {
			  $$ = MakeReturn(expression_undefined);
			}
|   TK_RETURN comma_expression TK_SEMICOLON
	                {
			  $$ = MakeReturn(MakeCommaExpression($2));
			}
|   TK_BREAK TK_SEMICOLON
                        {
			  /* Create corresponding goto */
			}
|   TK_CONTINUE TK_SEMICOLON
                 	{
			  /* Create corresponding goto */
			}
|   TK_GOTO TK_IDENT TK_SEMICOLON
		        {
			  /* Create goto statement */
			}
|   TK_GOTO TK_STAR comma_expression TK_SEMICOLON 
                        {
			  /* Create goto statement */
			}
|   TK_ASM asmattr TK_LPAREN asmtemplate asmoutputs TK_RPAREN TK_SEMICOLON
                        { printf("ASM not implemented\n"); }
|   TK_MSASM            
                        { printf("ASM not implemented\n"); }
|   error location TK_SEMICOLON  
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: error location TK_SEMICOLON\n");
			} 
;

for_clause: 
    opt_expression TK_SEMICOLON
                        {
			}
|   declaration
                        {
			}
;

declaration:                                /* ISO 6.7.*/
    decl_spec_list init_declarator_list TK_SEMICOLON
                        {
			}
|   decl_spec_list TK_SEMICOLON	
                        {
			}
;
init_declarator_list:                       /* ISO 6.7 */
    init_declarator     {}
|   init_declarator TK_COMMA init_declarator_list   {}

;
init_declarator:                             /* ISO 6.7 */
    declarator          {}
|   declarator TK_EQ init_expression
                        { 
			  /* Put things in intial_value of the entity*/
			}
;

decl_spec_list:                         /* ISO 6.7 */
                                        /* ISO 6.7.1 */
    TK_TYPEDEF decl_spec_list_opt          
                        {
			  /* Add TYPEDEF_PREFIX to current entity name */
			}    
|   TK_EXTERN decl_spec_list_opt           
                        {
			  /* Add to decls_text in order to generate code */
			}    
|   TK_STATIC decl_spec_list_opt    
                        {
			  /* Make static storage for current entity */
			}
|   TK_AUTO decl_spec_list_opt           
                        {
			  /* Make dynamic storage for current entity */
			}
|   TK_REGISTER decl_spec_list_opt        
                        {
			  /* Add to type variable qualifiers */
			}
                                        /* ISO 6.7.2 */
|   type_spec decl_spec_list_opt_no_named
                        { 
			  /* $$ = CurrentType = MakeType($1);*/
			}	
                                        /* ISO 6.7.4 */
|   TK_INLINE decl_spec_list_opt
                        { 
			  printf("INLINE not implemented\n"); 
			}	 
|   attribute decl_spec_list_opt        
                        { printf("ATTRIBUTE not implemented\n"); }	
/* specifier pattern variable (must be last in spec list) */
|   TK_AT_SPECIFIER TK_LPAREN TK_IDENT TK_RPAREN  
                        { 
			  printf("CIL AT not implemented\n"); 
			}	
;

/* (* In most cases if we see a NAMED_TYPE we must shift it. Thus we declare 
    * NAMED_TYPE to have right associativity  *) */
decl_spec_list_opt: 
    /* empty */         {} %prec TK_NAMED_TYPE
|   decl_spec_list      {}
;

/* (* We add this separate rule to handle the special case when an appearance 
    * of NAMED_TYPE should not be considered as part of the specifiers but as 
    * part of the declarator. IDENT has higher precedence than NAMED_TYPE  *)
 */
decl_spec_list_opt_no_named: 
    /* empty */         {} %prec TK_IDENT
|   decl_spec_list      {}
;
type_spec:   /* ISO 6.7.2 */
    TK_VOID             
                        {
			  $$ = is_type_void;
                        }
|   TK_CHAR            
                        {
			  $$ = is_basic_char;
			  CurrentTypeSize = DEFAULT_CHARACTER_TYPE_SIZE;
                        }
|   TK_SHORT            
                        {
			  $$ = is_basic_int;
			  CurrentTypeSize = DEFAULT_INTEGER_TYPE_SIZE;
                        }
|   TK_INT            
                        {
                          $$ = is_basic_int;
			  CurrentTypeSize = DEFAULT_INTEGER_TYPE_SIZE;
                        }
|   TK_LONG             
                        {
                          $$ = is_basic_int;
			  CurrentTypeSize = DEFAULT_INTEGER_TYPE_SIZE;
                        }
|   TK_INT64           
                        {
                          $$ = is_basic_int;
			  CurrentTypeSize = DEFAULT_INTEGER_TYPE_SIZE;
                        }
|   TK_FLOAT           
                        {
                          $$ = is_basic_float; 
		          CurrentTypeSize = DEFAULT_REAL_TYPE_SIZE;
			}
|   TK_DOUBLE           
                        {
                          $$ = is_basic_float; 
		          CurrentTypeSize = DEFAULT_DOUBLEPRECISION_TYPE_SIZE;
			}
|   TK_SIGNED            
                        {
			  /* if basic = int + signed + unsigned*/
                          $$ = is_basic_signed; 
		          CurrentTypeSize = DEFAULT_INTEGER_TYPE_SIZE;
			}
|   TK_UNSIGNED           
                        {
			   /* if basic = int + signed + unsigned*/
                          $$ = is_basic_unsigned; 
		          CurrentTypeSize = DEFAULT_INTEGER_TYPE_SIZE;
			}
|   TK_STRUCT id_or_typename                           
                        {
			  /* Specify a variable of type struct */
                          $$ = is_basic_derived; 
			  /* Specify the associated struct entity */
			}
|   TK_STRUCT id_or_typename TK_LBRACE struct_decl_list TK_RBRACE
                        {
			  /* Specify a struct entity */
			  /* Add STRUCT_PREFIX to current entity name */
                          $$ = is_type_struct; 
			  /* Create the corresponding entity*/
			}
|   TK_STRUCT TK_LBRACE struct_decl_list TK_RBRACE
                        {
			  /* Specify a struct entity */
			  /* Add STRUCT_PREFIX to current entity name */
			  $$ = is_type_struct; 
			  /* Create the corresponding entity with a unique name*/
			}
|   TK_UNION id_or_typename 
                        {
                          /* Specify a variable of type union */
                          $$ = is_basic_derived; 
			  /* Specify the associated union entity */
			}
|   TK_UNION id_or_typename TK_LBRACE struct_decl_list TK_RBRACE
                        {
                          /* Specify an union entity */
			  /* Add UNION_PREFIX to current entity name */
			  $$ = is_type_union; 
			  /* Create the corresponding entity with a unique name*/
			}
|   TK_UNION TK_LBRACE struct_decl_list TK_RBRACE
                        {
                          /* Specify an union entity */
			  /* Add UNION_PREFIX to current entity name */
			  $$ = is_type_union; 
			  /* Create the corresponding entity with a unique name*/
			}
|   TK_ENUM id_or_typename   
                        {
                          /* Specify a variable of type enum */
                          $$ = is_basic_derived; 
			  /* Specify the associated enum entity */
			}
|   TK_ENUM id_or_typename TK_LBRACE enum_list maybecomma TK_RBRACE
                        {
                           /* Specify an enum entity */
			  /* Add ENUM_PREFIX to current entity name */
			  $$ = is_type_enum; 
			  /* Create the corresponding entity with a unique name*/
			}                   
|   TK_ENUM TK_LBRACE enum_list maybecomma TK_RBRACE
                        {
			  /* Specify an enum entity */
			  /* Add ENUM_PREFIX to current entity name */
			  $$ = is_type_enum; 
			  /* Create the corresponding entity with a unique name*/
			}
|   TK_NAMED_TYPE  
                        {
			  /* Specify a variable of type typedef*/
                          $$ = is_basic_typedef; 
			  /* Specify the associated typedef entity */
			}
|   TK_TYPEOF TK_LPAREN expression TK_RPAREN  
                        {
                          printf("TYPEOF not implemented\n");
			}
|   TK_TYPEOF TK_LPAREN type_name TK_RPAREN    
                        {
			  printf("TYPEOF not implemented\n");
			}
;

struct_decl_list: /* (* ISO 6.7.2. Except that we allow empty structs. We 
                      * also allow missing field names. *)
                   */
    /* empty */                           
                        {}
|   decl_spec_list TK_SEMICOLON struct_decl_list
                        {
			  /* Exist missing field name ?  */
			  printf("Missing field name  not implemented\n");
			}             
|   decl_spec_list field_decl_list TK_SEMICOLON struct_decl_list
                        {
			  /*  */
			}
|   error TK_SEMICOLON struct_decl_list
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: error TK_SEMICOLON struct_decl_list\n");
			}
;

field_decl_list: /* (* ISO 6.7.2 *) */
    field_decl          
                        {
			}
|   field_decl TK_COMMA field_decl_list    
                        {
			}

;
field_decl: /* (* ISO 6.7.2. Except that we allow unnamed fields. *) */
    declarator                
                        {
			  /* Add struct/union name and MEMBER_PREFIX to entity name 
			     Update the list of members of the struct/union entity*/
			}
|   declarator TK_COLON expression   
                        {
			  /* Add struct/union name and MEMBER_PREFIX to entity name 
			     Update the list of members of the struct/union entity
			     entity of is_basic_bit, size = expression_to_int($3) */
			  pips_assert("Width of bit-field must be a constant integer", 
				      integer_constant_expression_p($3));
			}  
|   TK_COLON expression 
                        {
			  /* Unnamed bit-field : special and unique name */
			  /* Add struct/union name and MEMBER_PREFIX to entity name 
			     Update the list of members of the struct/union entity
			     entity of is_basic_bit, size = expression_to_int($3) */
			  pips_assert("Width of bit-field must be a constant integer", 
				      integer_constant_expression_p($2));
			}
;

enum_list: /* (* ISO 6.7.2.2 *) */
    enumerator	
                        {
			  /* initial_value = 0 or $3*/
			}
|   enum_list TK_COMMA enumerator	       
                        {
			  /* Attention to the reverse recursive definition*/
			}
|   enum_list TK_COMMA error      
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: enum_list TK_COMMA error\n");
			}
;

enumerator:	
    TK_IDENT	
                        {
			  /* Update the list of members of the enum entity
			     Create an entity of is_basic_int, storage rom 
			     initial_value = 0 if it is the first member
			     initial_value = intial_value(precedessor) + 1 */
			}
|   TK_IDENT TK_EQ expression	
                        {
			  /* Update the list of members of the enum entity
			     Create an entity of is_basic_int, storage rom 
			     initial_value = $3 */
			  pips_assert("Enumerated value must be a constant integer", 
				      integer_constant_expression_p($3));
			}
;

declarator:  /* (* ISO 6.7.5. Plus Microsoft declarators.*) */
    pointer_opt direct_decl attributes_with_asm
                        {
			 
			}
;

direct_decl: /* (* ISO 6.7.5 *) */
                                   /* (* We want to be able to redefine named
                                    * types as variable names *) */
    id_or_typename
                       {
			 /* Declare the variable 
			    Take CurrentType
			    entity =>  $1
			    type variable ? dimension, storage, inital value, 
			    type functional ?
			 */
		       }
|   TK_LPAREN attributes declarator TK_RPAREN
                        {
			  /* Add attributes such as const, restrict, ... to variable's qualifiers */
			}

|   direct_decl TK_LBRACKET attributes comma_expression_opt TK_RBRACKET
                        { 
			 /* Declare the variable 
			    Take CurrentType
			    entity =>  $1
			    type variable ? dimension, storage, inital value, 
			     type functional ?
			 */
			}
|   direct_decl TK_LBRACKET attributes error TK_RBRACKET
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: direct_decl TK_LBRACKET attributes error TK_RBRACKET\n");
			}
|   direct_decl parameter_list_startscope {} rest_par_list TK_RPAREN
                        {
			  /*type functional ?*/
			}
;

parameter_list_startscope: 
    TK_LPAREN           {}
;
rest_par_list:
|   /* empty */         {}
|   parameter_decl rest_par_list1  {}
;
rest_par_list1: 
    /* empty */         {}
|   TK_COMMA TK_ELLIPSIS 
                        {}
|   TK_COMMA parameter_decl rest_par_list1 
                        {}  
;    


parameter_decl: /* (* ISO 6.7.5 *) */
    decl_spec_list declarator 
                        {
			  /* Declare the variable 
			    Take CurrentType
			    entity =>  $1
			    type variable ? dimension, storage, inital value, 
			    formel 
			 */
			}
|   decl_spec_list abstract_decl 
                        {}
|   decl_spec_list              
                        {
			  /* function prototype*/
			}
|   TK_LPAREN parameter_decl TK_RPAREN    
                        { } 
;

/* (* Old style prototypes. Like a declarator *) */
old_proto_decl:
    pointer_opt direct_old_proto_decl 
                        {
			  printf("Old-style function prototype not implemented\n");
			}
;
direct_old_proto_decl:
    direct_decl TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list
                        { 
			  printf("Old-style function prototype not implemented\n");
			}
|   direct_decl TK_LPAREN TK_RPAREN
                        {
			  printf("Old-style function prototype not implemented\n");
			}
;

old_parameter_list_ne:
    TK_IDENT            
                        {
			  printf("Old-style function prototype not implemented\n");
			}
|   TK_IDENT TK_COMMA old_parameter_list_ne   
                        {
			  printf("Old-style function prototype not implemented\n");
			}
;

old_pardef_list: 
    /* empty */         {}
|   decl_spec_list old_pardef TK_SEMICOLON TK_ELLIPSIS
                        {
			  printf("Old-style function prototype not implemented\n");
			}
|   decl_spec_list old_pardef TK_SEMICOLON old_pardef_list  
                        {
			  printf("Old-style function prototype not implemented\n");
			} 
;

old_pardef: 
    declarator            
                        {
			  printf("Old-style function prototype not implemented\n");
			}
|   declarator TK_COMMA old_pardef   
                        {
			  printf("Old-style function prototype not implemented\n");
			}
|   error       
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: error \n");
			}
;

pointer: /* (* ISO 6.7.5 *) */ 
    TK_STAR attributes pointer_opt 
                        {
			  /* Add attibutes to variable's fields */
			}
;

pointer_opt:
    /**/                {}
|   pointer             
                        {
			  /* is_basic_pointer*/
			}
;

type_name: /* (* ISO 6.7.6 *) */
    decl_spec_list abstract_decl
                        {
			 
			}
|   decl_spec_list      
                        {
			 
			}
;

abstract_decl: /* (* ISO 6.7.6. *) */
    pointer_opt abs_direct_decl attributes  
                        {
			 
			}
|   pointer                     
                        {
			 
			}
;

abs_direct_decl: /* (* ISO 6.7.6. We do not support optional declarator for 
                     * functions. Plus Microsoft attributes. See the 
                     * discussion for declarator. *) */
    TK_LPAREN attributes abstract_decl TK_RPAREN
                        {
			 
			}
|   TK_LPAREN error TK_RPAREN
                        {
			  /* Parse error, do what ? */
			  printf("Parse error: TK_LPAREN error TK_RPAREN\n");
			}
            
|   abs_direct_decl_opt TK_LBRACKET comma_expression_opt TK_RBRACKET
                        {
			 
			}
/*(* The next shoudl be abs_direct_decl_opt but we get conflicts *)*/
|   abs_direct_decl parameter_list_startscope rest_par_list TK_RPAREN
                        {
			 
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
			   
			}	

function_def_start:  /* (* ISO 6.9.1 *) */
    decl_spec_list declarator   
                        { 
			   
			}	
/* (* Old-style function prototype *) */
|   decl_spec_list old_proto_decl 
                        { 
			  printf("Old-style function prototype not implemented\n"); 
			}	
/* (* New-style function that does not have a return type *) */
|   TK_IDENT parameter_list_startscope rest_par_list TK_RPAREN 
                        { 
			   
			}	
/* (* No return type and old-style parameter list *) */
|   TK_IDENT TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list
                        { 
			  printf("Old-style function prototype not implemented\n"); 
			}	
/* (* No return type and no parameters *) */
|   TK_IDENT TK_LPAREN TK_RPAREN
                        { 
			   
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
                        { printf("ASM not implemented\n"); }                                        
;
 
attribute:
    TK_ATTRIBUTE TK_LPAREN paren_attr_list_ne TK_RPAREN	
                        { printf("ATTRIBUTE not implemented\n"); }	                                       
|   TK_DECLSPEC paren_attr_list_ne       
                        { printf("ATTRIBUTE not implemented\n"); }	
|   TK_MSATTR                             
                        { printf("ATTRIBUTE not implemented\n"); }	
                                        /* ISO 6.7.3 */
|   TK_CONST                              
                        { 
			  /* Add CONST to type variable qualifiers */
			}	
|   TK_RESTRICT                            
                        { 
			  /* Add RESTRICT to type variable qualifiers */
			}	
|   TK_VOLATILE                            
                        { 
			  /* Add VOLATILE to type variable qualifiers */ 
			}	
;

/** (* PRAGMAS and ATTRIBUTES *) ***/
/* (* We want to allow certain strange things that occur in pragmas, so we 
    * cannot use directly the language of expressions *) */ 
attr: 
|   id_or_typename                       
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_IDENT TK_COLON TK_INTCON                  
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_DEFAULT TK_COLON TK_INTCON               
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_IDENT TK_LPAREN  TK_RPAREN                 
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_IDENT paren_attr_list_ne             
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_INTCON                              
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   string_constant                      
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_CONST                                
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_SIZEOF expression                     
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_SIZEOF TK_LPAREN type_name TK_RPAREN	                         
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	

|   TK_ALIGNOF expression                   
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_ALIGNOF TK_LPAREN type_name TK_RPAREN      
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_PLUS expression    	                 
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_MINUS expression 		        
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_STAR expression		       
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_AND expression				                 %prec TK_ADDROF
	                                
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_EXCLAM expression		       
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_TILDE expression		        
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_PLUS attr                      
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	 
|   attr TK_MINUS attr                    
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_STAR expression               
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_SLASH attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_PERCENT attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_AND_AND attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_PIPE_PIPE attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_AND attr			
                        { 
			  printf("PRAGMAS and ATTRIBUTES not implemented\n"); 
		        }
|   attr TK_PIPE attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_CIRC attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_EQ_EQ attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_EXCLAM_EQ attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_INF attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_SUP attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_INF_EQ attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_SUP_EQ attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_INF_INF attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_SUP_SUP attr			
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_ARROW id_or_typename          
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_DOT id_or_typename            
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_LPAREN attr TK_RPAREN                 
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	 
;

attr_list_ne:
|   attr                                  
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   attr TK_COMMA attr_list_ne               
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   error TK_COMMA attr_list_ne              
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
;
paren_attr_list_ne: 
    TK_LPAREN attr_list_ne TK_RPAREN            
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
|   TK_LPAREN error TK_RPAREN                   
                        { printf("PRAGMAS and ATTRIBUTES not implemented\n"); }	
;
/*** GCC TK_ASM instructions ***/
asmattr:
    /* empty */                        
                        { printf("ASM not implemented\n"); }
|   TK_VOLATILE  asmattr                  
                        { printf("ASM not implemented\n"); }
|   TK_CONST asmattr                      
                        { printf("ASM not implemented\n"); } 
;
asmtemplate: 
    one_string_constant                          
                        { printf("ASM not implemented\n"); }
|   one_string_constant asmtemplate              
                        { printf("ASM not implemented\n"); }
;
asmoutputs: 
    /* empty */           
                        { printf("ASM not implemented\n"); }
|   TK_COLON asmoperands asminputs                       
                        { printf("ASM not implemented\n"); }
;
asmoperands:
    /* empty */                       
                        { printf("ASM not implemented\n"); }
|   asmoperandsne                      
                        { printf("ASM not implemented\n"); }
;
asmoperandsne:
    asmoperand                         
                        { printf("ASM not implemented\n"); }
|   asmoperandsne TK_COMMA asmoperand    
                        { printf("ASM not implemented\n"); }
;
asmoperand:
    string_constant TK_LPAREN expression TK_RPAREN    
                        { printf("ASM not implemented\n"); }
|   string_constant TK_LPAREN error TK_RPAREN         
                        { printf("ASM not implemented\n"); }
; 
asminputs: 
    /* empty */               
                        { printf("ASM not implemented\n"); }
|   TK_COLON asmoperands asmclobber
                        { printf("ASM not implemented\n"); }                        
;
asmclobber:
    /* empty */                         
                        { printf("ASM not implemented\n"); }
|   TK_COLON asmcloberlst_ne            
                        { printf("ASM not implemented\n"); }
;
asmcloberlst_ne:
    one_string_constant 
                        { printf("ASM not implemented\n"); }                          
|   one_string_constant TK_COMMA asmcloberlst_ne 
                        { printf("ASM not implemented\n"); }
;
  
%%

