/* $Id$
   $Log: splitc.y,v $
   Revision 1.1  2003/07/29 15:12:37  irigoin
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

extern string strdup(string);

#define C_ERROR_VERBOSE 1 /* much clearer error messages with bison */

extern int splitc_lex(void);
extern void splitc_error(char *);
 extern string splitc_text;

extern hash_table keyword_typedef_table;

/* The following global variables are used to store the information such as 
   the scope, type and storage of an entity, given by the decl_spec_list, 
   which are used later by direct_decl to create the entity. 

   For the moment, block scope is not considered. CurrentScope can be File, 
   Module, File%Module or TOP-LEVEL*/

static string CurrentScope = NULL;
/*
static type CurrentType = type_undefined; 
static storage CurrentStorage = storage_undefined;
static list CurrentQualifiers = NIL; */
static string CurrentDerivedName = NULL; /* to remember the name of a struct and add it to the member prefix name*/
/* static int CurrentMode = 0; */ /* to know the mode of the formal parameter : by value or by reference*/

int csplit_is_external = 0; /* to know if the variable is declared inside or outside a function */
int csplit_is_function = 0; /* to know if this is the declaration of a function or not, to distinguish between 
			      a static variable and a static function */
/* static int is_typedef = 0; */ /* to know if this is a typedef name or not */
/* static int enum_counter = 0; */


/* All the following global variables must be replaced by functions, once we have the preprocessor for C */

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
    declaration     
                        {
			  csplit_is_external = 1; /* the variable is declared outside of any function */
			}                 
|   function_def
                        {
			  csplit_is_external = 0; /* the variable is declared inside a function */
			}
|   TK_ASM TK_LPAREN string_constant TK_RPAREN TK_SEMICOLON
                        { pips_internal_error("ASM not implemented\n"); }
|   TK_PRAGMA attr			
                        { 
			  pips_internal_error("PRAGMA not implemented\n"); 
			}
/* Old-style function prototype. This should be somewhere else, like in
   "declaration". For now we keep it at global scope only because in local
   scope it looks too much like a function call */
|   TK_IDENT TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list TK_SEMICOLON
                        { 
			}
/* Old style function prototype, but without any arguments */
|   TK_IDENT TK_LPAREN TK_RPAREN TK_SEMICOLON
                        {
			}
/* transformer for a toplevel construct */
|   TK_AT_TRANSFORM TK_LBRACE global TK_RBRACE TK_IDENT /*to*/ TK_LBRACE globals TK_RBRACE 
                        { 
			}
/* transformer for an expression */
|   TK_AT_TRANSFORMEXPR TK_LBRACE expression TK_RBRACE TK_IDENT /*to*/ TK_LBRACE expression TK_RBRACE 
                        { 
			}
|   location error TK_SEMICOLON 
                        { 
			  pips_user_error("Parse error: location error TK_SEMICOLON \n");
			}
;

id_or_typename:
    TK_IDENT			
                        {}
|   TK_NAMED_TYPE				
                        {}
|   TK_AT_NAME TK_LPAREN TK_IDENT TK_RPAREN         
                        {
			   pips_internal_error("CIL AT not implemented\n"); 
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
			}
|   TK_IDENT
		        {
                        }
|   TK_SIZEOF expression
		        {
                        }
|   TK_SIZEOF TK_LPAREN type_name TK_RPAREN
		        {
                        }
|   TK_ALIGNOF expression
		        { 
			}
|   TK_ALIGNOF TK_LPAREN type_name TK_RPAREN
		        { 
			}
|   TK_PLUS expression
		        {
			}
|   TK_MINUS expression
		        {
			}
|   TK_STAR expression
		        {
			}
|   TK_AND expression				%prec TK_ADDROF
		        {
			}
|   TK_EXCLAM expression
		        {
			}
|   TK_TILDE expression
		        {
			}
|   TK_PLUS_PLUS expression                    %prec TK_CAST
		        {
			}
|   expression TK_PLUS_PLUS
		        {
			}
|   TK_MINUS_MINUS expression                  %prec TK_CAST
		        {
			}
|   expression TK_MINUS_MINUS
		        {
			}
|   expression TK_ARROW id_or_typename
		        {	
			}
|   expression TK_DOT id_or_typename
		        {
			}
|   TK_LPAREN block TK_RPAREN
		        {
			}
|   paren_comma_expression
		        {
			}
|   expression TK_LPAREN arguments TK_RPAREN
			{
			}
|   TK_BUILTIN_VA_ARG TK_LPAREN expression TK_COMMA type_name TK_RPAREN
                        {
			}
|   expression bracket_comma_expression
			{
			}
|   expression TK_QUEST opt_expression TK_COLON expression
			{
			}
|   expression TK_PLUS expression
			{ 
			}
|   expression TK_MINUS expression
			{ 
			}
|   expression TK_STAR expression
			{ 
			}
|   expression TK_SLASH expression
			{ 
			}
|   expression TK_PERCENT expression
			{ 
			}
|   expression TK_AND_AND expression
			{
			}
|   expression TK_PIPE_PIPE expression
			{
			}
|   expression TK_AND expression
			{
			}
|   expression TK_PIPE expression
			{
			}
|   expression TK_CIRC expression
			{
			}
|   expression TK_EQ_EQ expression
			{
			}
|   expression TK_EXCLAM_EQ expression
			{
			}
|   expression TK_INF expression
			{
			}
|   expression TK_SUP expression
			{
			}
|   expression TK_INF_EQ expression
			{
			}
|   expression TK_SUP_EQ expression
			{
			}
|   expression TK_INF_INF expression
			{
			}
|   expression TK_SUP_SUP expression
			{
			}
|   expression TK_EQ expression
			{
			}
|   expression TK_PLUS_EQ expression
			{
			}
|   expression TK_MINUS_EQ expression
			{
			}
|   expression TK_STAR_EQ expression
			{
			}
|   expression TK_SLASH_EQ expression
			{
			}
|   expression TK_PERCENT_EQ expression
			{
			}
|   expression TK_AND_EQ expression
			{
			}
|   expression TK_PIPE_EQ expression
			{
			}
|   expression TK_CIRC_EQ expression
			{
			}
|   expression TK_INF_INF_EQ expression	
			{
			}
|   expression TK_SUP_SUP_EQ expression
			{
			}
|   TK_LPAREN type_name TK_RPAREN expression
		        {
			}
/* (* We handle GCC constructor expressions *) */
|   TK_LPAREN type_name TK_RPAREN TK_LBRACE initializer_list_opt TK_RBRACE
		        {
			}
/* (* GCC's address of labels *)  */
|   TK_AND_AND TK_IDENT  
                        {
			}
|   TK_AT_EXPR TK_LPAREN TK_IDENT TK_RPAREN         /* expression pattern variable */
                        {
			}
;

constant:
    TK_INTCON			
                        {
			}
|   TK_FLOATCON	
                        {
			}
|   TK_CHARCON				
                        {
			}
|   string_constant	
                        {
                        }
/*add a nul to strings.  We do this here (rather than in the lexer) to make
  concatenation easy below.*/
|   wstring_list	
                        {
                        }
;

string_constant:
/* Now that we know this constant isn't part of a wstring, convert it
   back to a string for easy viewing. */
    string_list         
                        {
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
			}
|   string_list one_string
                        {
			}
;

wstring_list:
    TK_WSTRINGCON           
                        {
			}
|   wstring_list one_string
                        {
			}
|   wstring_list TK_WSTRINGCON  
                        {
			}
/* Only the first string in the list needs an L, so L"a" "b" is the same
 * as L"ab" or L"a" L"b". */

one_string: 
    TK_STRINGCON	{ }
|   TK_FUNCTION__       
                        { }
|   TK_PRETTY_FUNCTION__
                        { }
;    

init_expression:
    expression          { }
|   TK_LBRACE initializer_list_opt TK_RBRACE
			{ }

initializer_list:    /* ISO 6.7.8. Allow a trailing COMMA */
    initializer 
                        { 
			}
|   initializer TK_COMMA initializer_list_opt
                        { 
			}
;
initializer_list_opt:
    /* empty */         { }
|   initializer_list    { }
;
initializer: 
    init_designators eq_opt init_expression
                        { 
			}
|   gcc_init_designators init_expression
                        { 
			}
|   init_expression     { }
;
eq_opt: 
    TK_EQ
                        { }
    /*(* GCC allows missing = *)*/
|   /*(* empty *)*/     
                        { 
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
    /* empty */         { }
|   init_designators    {}
;

gcc_init_designators:  /*(* GCC supports these strange things *)*/
    id_or_typename TK_COLON   
                        { 
			}
;

arguments: 
    /* empty */         { }
|   comma_expression    { }
;

opt_expression:
    /* empty */
	        	{ }
|   comma_expression
	        	{ }
;

comma_expression:
    expression                        
                        {
			}
|   expression TK_COMMA comma_expression 
                        {
			}
|   error TK_COMMA comma_expression      
                        {
			}
;

comma_expression_opt:
    /* empty */         { }
|   comma_expression    { }
;

paren_comma_expression:
    TK_LPAREN comma_expression TK_RPAREN
                        {
			}
|   TK_LPAREN error TK_RPAREN                         
                        {
			}
;

bracket_comma_expression:
    TK_LBRACKET comma_expression TK_RBRACKET 
                        {
			}
|   TK_LBRACKET error TK_RBRACKET  
                        {
			}
;

/*** statements ***/
block: /* ISO 6.8.2 */
    TK_LBRACE local_labels block_attrs declaration_list statement_list TK_RBRACE   
                        {
			} 
|   error location TK_RBRACE 
                        { } 
;

block_attrs:
   /* empty */          {}
|  TK_BLOCKATTRIBUTE paren_attr_list_ne
                        { }
;

declaration_list: 
    /* empty */         { }
|   declaration declaration_list
                        {
			}

;
statement_list:
    /* empty */         { }
|   statement statement_list
                        {
			}
/*(* GCC accepts a label at the end of a block *)*/
|   TK_IDENT TK_COLON	{ }
;

local_labels: 
   /* empty */          {}
|  TK_LABEL__ local_label_names TK_SEMICOLON local_labels
                        { }
;

local_label_names: 
   TK_IDENT             {}
|  TK_IDENT TK_COMMA local_label_names {}
;

statement:
    TK_SEMICOLON
                    	{
			}
|   comma_expression TK_SEMICOLON
	        	{
			}
|   block               { }
|   TK_IF paren_comma_expression statement                    %prec TK_IF
                	{
			}
|   TK_IF paren_comma_expression statement TK_ELSE statement
	                {
			}
|   TK_SWITCH 
                        {
			} 
    paren_comma_expression 
                        {
			} 
    statement
                        {
			}
|   TK_WHILE 
                        {
			} 
    paren_comma_expression statement
	        	{
			}
|   TK_DO
                        {
			} 
    statement TK_WHILE paren_comma_expression TK_SEMICOLON
	        	{
			}
|   TK_FOR
                        {
			} 
    TK_LPAREN for_clause opt_expression TK_SEMICOLON opt_expression TK_RPAREN statement
	                {
			}
|   TK_IDENT TK_COLON statement
                        {
			}
|   TK_CASE expression TK_COLON
                        {
			}
|   TK_CASE expression TK_ELLIPSIS expression TK_COLON
                        {
			}
|   TK_DEFAULT TK_COLON
	                {
			}
|   TK_RETURN TK_SEMICOLON		 
                        {
			}
|   TK_RETURN comma_expression TK_SEMICOLON
	                {
			}
|   TK_BREAK TK_SEMICOLON
                        {
			}
|   TK_CONTINUE TK_SEMICOLON
                 	{
			}
|   TK_GOTO TK_IDENT TK_SEMICOLON
		        {
			}
|   TK_GOTO TK_STAR comma_expression TK_SEMICOLON 
                        {
			}
|   TK_ASM asmattr TK_LPAREN asmtemplate asmoutputs TK_RPAREN TK_SEMICOLON
                        { }
|   TK_MSASM            
                        { }
|   error location TK_SEMICOLON  
                        {
			} 
;

for_clause: 
    opt_expression TK_SEMICOLON   { }
|   declaration
                        {
			}
;

declaration:                                /* ISO 6.7.*/
    decl_spec_list init_declarator_list TK_SEMICOLON
                        {
			  csplit_is_function = 0; /* not function's declaration */
			}
|   decl_spec_list TK_SEMICOLON	
                        {
			  csplit_is_function = 0; /* not function's declaration */
			}
;

init_declarator_list:                       /* ISO 6.7 */
    init_declarator
                        {
			}
|   init_declarator TK_COMMA init_declarator_list
                        {
			}

;
init_declarator:                             /* ISO 6.7 */
    declarator          { }
|   declarator TK_EQ init_expression
                        { 
			}
;

decl_spec_list:                         /* ISO 6.7 */
                                        /* ISO 6.7.1 */
    TK_TYPEDEF decl_spec_list_opt          
                        {
			}    
|   TK_EXTERN decl_spec_list_opt           
                        {
			}    
|   TK_STATIC decl_spec_list_opt    
                        {
			  /* There are 3 cases: static function, external and internal static variable*/
			  if (csplit_is_function) {
			    /* CurrentScope = strdup(concatenate(CurrentSourceFile,FILE_SEP_STRING,NULL)); */
			    ;
			  }
			}
|   TK_AUTO decl_spec_list_opt           
                        {
			}
|   TK_REGISTER decl_spec_list_opt        
                        {
			}
                                        /* ISO 6.7.2 */
|   type_spec decl_spec_list_opt_no_named
                        {
			}	
                                        /* ISO 6.7.4 */
|   TK_INLINE decl_spec_list_opt
                        { 
			}	 
|   attribute decl_spec_list_opt        
                        { 
			}	
/* specifier pattern variable (must be last in spec list) */
|   TK_AT_SPECIFIER TK_LPAREN TK_IDENT TK_RPAREN  
                        { 
			}	
;

/* (* In most cases if we see a NAMED_TYPE we must shift it. Thus we declare 
    * NAMED_TYPE to have right associativity  *) */
decl_spec_list_opt: 
    /* empty */         { } %prec TK_NAMED_TYPE
|   decl_spec_list      { }
;

/* (* We add this separate rule to handle the special case when an appearance 
    * of NAMED_TYPE should not be considered as part of the specifiers but as 
    * part of the declarator. IDENT has higher precedence than NAMED_TYPE  *)
 */
decl_spec_list_opt_no_named: 
    /* empty */         { } %prec TK_IDENT
|   decl_spec_list      { }
;


type_spec:   /* ISO 6.7.2 */
    TK_VOID             
                        {
			  $$ = strdup(splitc_text);
                        } 
|   TK_CHAR          
                        {
			  $$ = entity_undefined;
			}
|   TK_SHORT      
                        {
			  $$ = entity_undefined;
			}    
|   TK_INT  
                        {
			  $$ = entity_undefined;
			}  
|   TK_LONG
                        {
			  $$ = entity_undefined;
			}   
|   TK_FLOAT           
                        {
			  $$ = entity_undefined;
			}
|   TK_DOUBLE           
                        {
			  $$ = entity_undefined;
			}
|   TK_SIGNED     
                        {
			  $$ = entity_undefined;
			}
|   TK_UNSIGNED          
                        {
			  $$ = entity_undefined;
			}
|   TK_STRUCT id_or_typename                           
                        {
			  $$ = entity_undefined;
			}
|   TK_STRUCT id_or_typename TK_LBRACE { } struct_decl_list TK_RBRACE
                        {
			}
|   TK_STRUCT TK_LBRACE {
                        }
    struct_decl_list TK_RBRACE
                        {
			}
|   TK_UNION id_or_typename 
                        {
			}
|   TK_UNION id_or_typename TK_LBRACE { CurrentDerivedName = $2; } struct_decl_list TK_RBRACE
                        {
			}
|   TK_UNION TK_LBRACE  { 
                        }
    struct_decl_list TK_RBRACE
                        {
			}
|   TK_ENUM id_or_typename   
                        {
			}
|   TK_ENUM id_or_typename TK_LBRACE enum_list maybecomma TK_RBRACE
                        {
			}                   
|   TK_ENUM TK_LBRACE enum_list maybecomma TK_RBRACE
                        {
			}
|   TK_NAMED_TYPE  
                        {
			}
|   TK_TYPEOF TK_LPAREN expression TK_RPAREN  
                        {
			}
|   TK_TYPEOF TK_LPAREN type_name TK_RPAREN    
                        {
			}
;

struct_decl_list: /* (* ISO 6.7.2. Except that we allow empty structs. We 
                      * also allow missing field names. *)
                   */
    /* empty */         { }
|   decl_spec_list TK_SEMICOLON struct_decl_list
                        {
			}             
|   decl_spec_list      { 
                        }
    field_decl_list TK_SEMICOLON struct_decl_list
                        {
			}
|   error TK_SEMICOLON struct_decl_list
                        {
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
    declarator          { }
|   declarator TK_COLON expression   
                        {
			}  
|   TK_COLON expression 
                        {
			}
;

enum_list: /* (* ISO 6.7.2.2 *) */
    enumerator	
                        {
			}
|   enum_list TK_COMMA enumerator	       
                        {
			}
|   enum_list TK_COMMA error      
                        {
			}
;

enumerator:	
    TK_IDENT	
                        {
			}
|   TK_IDENT TK_EQ expression	
                        {
			}
;

declarator:  /* (* ISO 6.7.5. Plus Microsoft declarators.*) */
    pointer_opt direct_decl attributes_with_asm
                        { }
;

direct_decl: /* (* ISO 6.7.5 *) */
                                   /* (* We want to be able to redefine named
                                    * types as variable names *) */
    id_or_typename
                       {
		       }
|   TK_LPAREN attributes declarator TK_RPAREN
                        {
			}
|   direct_decl TK_LBRACKET attributes comma_expression_opt TK_RBRACKET
                        { 
			}
|   direct_decl TK_LBRACKET attributes error TK_RBRACKET
                        {
			}
|   direct_decl parameter_list_startscope rest_par_list TK_RPAREN
                        {
			}
;

parameter_list_startscope: 
    TK_LPAREN           {}
;

rest_par_list:
    /* empty */         { }
|   parameter_decl rest_par_list1
                        {
			}
;
rest_par_list1: 
    /* empty */         { }
|   TK_COMMA TK_ELLIPSIS 
                        {
			}
|   TK_COMMA parameter_decl rest_par_list1 
                        {
			}  
;    

parameter_decl: /* (* ISO 6.7.5 *) */
    decl_spec_list declarator 
                        {
			}
|   decl_spec_list abstract_decl 
                        {
			}
|   decl_spec_list              
                        {
			}
|   TK_LPAREN parameter_decl TK_RPAREN    
                        {  } 
;

/* (* Old style prototypes. Like a declarator *) */
old_proto_decl:
    pointer_opt direct_old_proto_decl 
                        {
			}
;
direct_old_proto_decl:
    direct_decl TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list
                        { 
			}
|   direct_decl TK_LPAREN TK_RPAREN
                        {
			}
;

old_parameter_list_ne:
    TK_IDENT            
                        {
			}
|   TK_IDENT TK_COMMA old_parameter_list_ne   
                        {
			}
;

old_pardef_list: 
    /* empty */         {}
|   decl_spec_list old_pardef TK_SEMICOLON TK_ELLIPSIS
                        {
			}
|   decl_spec_list old_pardef TK_SEMICOLON old_pardef_list  
                        {
			} 
;

old_pardef: 
    declarator            
                        {
			}
|   declarator TK_COMMA old_pardef   
                        {
			}
|   error       
                        {
			}
;

pointer: /* (* ISO 6.7.5 *) */ 
    TK_STAR attributes pointer_opt 
                        {
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
			  pips_user_error("Parse error: TK_LPAREN error TK_RPAREN\n");
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
			  csplit_is_function = 1; /* function's declaration */
			}	
/* (* Old-style function prototype *) */
|   decl_spec_list old_proto_decl 
                        { 
			  csplit_is_function = 1; /* function's declaration */
			}	
/* (* New-style function that does not have a return type *) */
|   TK_IDENT parameter_list_startscope rest_par_list TK_RPAREN 
                        { 
			  /* Create the current function */
			}	
/* (* No return type and old-style parameter list *) */
|   TK_IDENT TK_LPAREN old_parameter_list_ne TK_RPAREN old_pardef_list
                        { 
			}	
/* (* No return type and no parameters *) */
|   TK_IDENT TK_LPAREN TK_RPAREN
                        { 
			  /* MakeCurrentFunction*/
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
                        { }                                        
;
 
attribute:
    TK_ATTRIBUTE TK_LPAREN paren_attr_list_ne TK_RPAREN	
                        { }	                                       
|   TK_DECLSPEC paren_attr_list_ne       
                        { }	
|   TK_MSATTR                             
                        { }	
                                        /* ISO 6.7.3 */
|   TK_CONST                              
                        { 
			}	
|   TK_RESTRICT                            
                        { 
			}	
|   TK_VOLATILE                            
                        { 
			}	
;

/** (* PRAGMAS and ATTRIBUTES *) ***/
/* (* We want to allow certain strange things that occur in pragmas, so we 
    * cannot use directly the language of expressions *) */ 
attr: 
|   id_or_typename                       
                        { }	
|   TK_IDENT TK_COLON TK_INTCON                  
                        { }	
|   TK_DEFAULT TK_COLON TK_INTCON               
                        { }	
|   TK_IDENT TK_LPAREN  TK_RPAREN                 
                        { }	
|   TK_IDENT paren_attr_list_ne             
                        { }	
|   TK_INTCON                              
                        { }	
|   string_constant                      
                        { }	
|   TK_CONST                                
                        { }	
|   TK_SIZEOF expression                     
                        { }	
|   TK_SIZEOF TK_LPAREN type_name TK_RPAREN	                         
                        { }	

|   TK_ALIGNOF expression                   
                        { }	
|   TK_ALIGNOF TK_LPAREN type_name TK_RPAREN      
                        { }	
|   TK_PLUS expression    	                 
                        { }	
|   TK_MINUS expression 		        
                        { }	
|   TK_STAR expression		       
                        { }	
|   TK_AND expression				                 %prec TK_ADDROF
	                                
                        { }	
|   TK_EXCLAM expression		       
                        { }	
|   TK_TILDE expression		        
                        { }	
|   attr TK_PLUS attr                      
                        { }	 
|   attr TK_MINUS attr                    
                        { }	
|   attr TK_STAR expression               
                        { }	
|   attr TK_SLASH attr			
                        { }	
|   attr TK_PERCENT attr			
                        { }	
|   attr TK_AND_AND attr			
                        { }	
|   attr TK_PIPE_PIPE attr			
                        { }	
|   attr TK_AND attr			
                        { 
		        }
|   attr TK_PIPE attr			
                        { }	
|   attr TK_CIRC attr			
                        { }	
|   attr TK_EQ_EQ attr			
                        { }	
|   attr TK_EXCLAM_EQ attr			
                        { }	
|   attr TK_INF attr			
                        { }	
|   attr TK_SUP attr			
                        { }	
|   attr TK_INF_EQ attr			
                        { }	
|   attr TK_SUP_EQ attr			
                        { }	
|   attr TK_INF_INF attr			
                        { }	
|   attr TK_SUP_SUP attr			
                        { }	
|   attr TK_ARROW id_or_typename          
                        { }	
|   attr TK_DOT id_or_typename            
                        { }	
|   TK_LPAREN attr TK_RPAREN                 
                        { }	 
;

attr_list_ne:
|   attr                                  
                        { }	
|   attr TK_COMMA attr_list_ne               
                        { }	
|   error TK_COMMA attr_list_ne              
                        { }	
;
paren_attr_list_ne: 
    TK_LPAREN attr_list_ne TK_RPAREN            
                        { }	
|   TK_LPAREN error TK_RPAREN                   
                        { }	
;
/*** GCC TK_ASM instructions ***/
asmattr:
    /* empty */                        
                        { }
|   TK_VOLATILE  asmattr                  
                        { }
|   TK_CONST asmattr                      
                        { } 
;
asmtemplate: 
    one_string_constant                          
                        { }
|   one_string_constant asmtemplate              
                        { }
;
asmoutputs: 
    /* empty */           
                        { }
|   TK_COLON asmoperands asminputs                       
                        {  }
;
asmoperands:
    /* empty */                       
                        { }
|   asmoperandsne                      
                        { }
;
asmoperandsne:
    asmoperand                         
                        { }
|   asmoperandsne TK_COMMA asmoperand    
                        { }
;
asmoperand:
    string_constant TK_LPAREN expression TK_RPAREN    
                        { }
|   string_constant TK_LPAREN error TK_RPAREN         
                        { }
; 
asminputs: 
    /* empty */               
                        { }
|   TK_COLON asmoperands asmclobber
                        { }                        
;
asmclobber:
    /* empty */                         
                        { }
|   TK_COLON asmcloberlst_ne            
                        { }
;
asmcloberlst_ne:
    one_string_constant 
                        { }                          
|   one_string_constant TK_COMMA asmcloberlst_ne 
                        { }
;
  
%%

