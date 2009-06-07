/* $Id$ */

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
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "misc.h"
#include "transformations.h" // FI: should not be referenced
#include "alias_private.h" // FI: should not be referenced
#include "instrumentation.h" // FI: Wow! Just for int_to_string() which is also defined in prettyprint?

#include "c_parser_private.h"

#include "c_syntax.h"

#define C_ERROR_VERBOSE 1 /* much clearer error messages with bison */

extern int c_lex(void);
extern void c_error(char *);

extern void discard_comments(void);

extern string compilation_unit_name;
extern statement ModuleStatement;

static int CurrentMode = 0; /**< to know the mode of the formal parameter: by value or by reference*/
static bool is_external = TRUE; /**< to know if the variable is declared inside or outside a function, so its scope
				   is the current function or the compilation unit or TOP-LEVEL*/
static int enum_counter = 0; /**< to compute the enumerator value: val(i) = val(i-1) + 1 */
static int abstract_counter = 1; /**< to create temporary entities for abstract types */

extern int loop_counter; /**< Global counter */
extern int derived_counter;

/* The following structures must be stacks because all the related entities are in recursive structures.
   Since there are not stacks with basic types such as integer or logical domain, I used basic_domain
   to avoid creating special stacks for FormalStack, OffsetStack, ... */

extern stack LoopStack;
extern stack SwitchControllerStack;
extern stack SwitchGotoStack;

extern stack StructNameStack; /**< to remember the name of a struct/union and add it to the member prefix name*/

extern stack ContextStack;

extern stack FunctionStack; /**< to know in which function the current formal arguments are declared */
static void PushFunction(entity f)
{
  /*
    string s = local_name_to_scope(entity_name(f));
    entity nf = f;
   */
/* SG: this implementations is completely buggy */
#if 0
  string s = local_name_to_scope(f); // << SG: passing an entity instead of a string
  if(!empty_scope_p(s)) {
    /* Scoping is not used in function naming, static or not */
    string nn = entity_name_without_scope(entity_name(f)); // << SG: passing a string instead of entity

    nf = find_or_create_entity(nn);

    pips_debug(8, "entity \"\%s\" is replaced by entity \"\%s\"\n",
	       entity_name(f), entity_name(nf));

    /* FI: I'm not sure that copy_xxx() takes xxx_undefined as valid
       argument, and I could not find the information in Newgen source
       code. */
    if(!type_undefined_p(entity_type(f)))
      entity_type(nf) = copy_type(entity_type(f));
    if(!value_undefined_p(entity_initial(f)))
      entity_initial(nf) = copy_value(entity_initial(f));
    if(!storage_undefined_p(entity_storage(f)))
      entity_storage(nf) = copy_storage(entity_storage(f));

    /* let's assume that the entity has not been recorded anywhere yet. */
    free_entity(f);
  }
  free(s);
#endif
  stack_push((char *) f, FunctionStack);
}

static void PopFunction()
{
  stack_pop(FunctionStack);
 }

entity GetFunction()
{
  entity f = stack_head(FunctionStack);
  return f;
}

extern stack FormalStack; /**< to know if the entity is a formal parameter or not */
extern stack OffsetStack; /**< to know the offset of the formal argument */

 // FI: I assumed it was the current context; in fact the current
 // context is rather the top of the ContextStack.I tried to maintain
 // as an invariant ycontext==stack_head(ContextStack) But this is not
 // be a good idea as it interferes with cyacc.y use of ycontext and
 // ContextStack
static c_parser_context ycontext = c_parser_context_undefined;

static string expression_comment = string_undefined;
static int expression_line_number = STATEMENT_NUMBER_UNDEFINED;

/* Functions used to manage the block scoping in conjunction with
   ContextStack and ycontext */

string empty_scope() { return strdup("");}

bool empty_scope_p(string s) {return strcmp(s, "")==0;}

/* same kind of testing required for union as well */
bool string_struct_scope_p(string s)
{
  /* Full testing would require a module_name, a block_scope and a struct name */
  /* Just lookup the struct identifier*/
  string ss = strchr(s, MEMBER_SEP_CHAR);
  return ss != NULL;
}

bool string_block_scope_p(string s)
{
  // A block scope string is empty or made of numbers each terminated by BLOCK_SEP_STRING
  char valid[12] = { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', BLOCK_SEP_CHAR, '\0'};
  bool is_block_scope = FALSE;
  string cs = s;
  bool is_number = FALSE;

  pips_debug(10, "Potential block scope string = \"%s\"\n", s);

  if(strspn(s, valid) == strlen(s)) {
    for(cs=s; *cs!='\0'; cs++) {
      if(is_number && isdigit(*cs))
	;
      else if(is_number && *cs==BLOCK_SEP_CHAR)
	is_number = FALSE;
      else if(!is_number && isdigit(*cs))
	is_number = TRUE;
      else if(!is_number && *cs==BLOCK_SEP_CHAR) {
	is_block_scope = FALSE;
	break;
      }
    }
    is_block_scope = !is_number;
  }

  pips_debug(10, "String = \"%s\" is %sa block scope string\n", s, is_block_scope?"":"not ");

  return is_block_scope;
}

/* The scope is moved up the scope tree and a NULL is return when
   there are no more scope to explore. */
string pop_block_scope(string old_scope)
{
  string new_scope = old_scope;
  string last_scope = string_undefined;

  pips_debug(8, "old_scope = \"%s\"\n", old_scope);
  pips_assert("old_scope is a scope", string_block_scope_p(old_scope));

  if(strlen(old_scope)>0) {
    /* get rid of last block separator */
    new_scope[strlen(new_scope)-1] = '\0';
    last_scope = strrchr(new_scope, BLOCK_SEP_CHAR);

    if(last_scope==NULL)
      *new_scope = '\0';
    else
      *(last_scope+1) = '\0';
  }
  else
    new_scope = NULL;

  if(new_scope!=NULL) {
    pips_debug(8, "new_scope = \"%s\"\n", new_scope);
    pips_assert("new_scope is a scope", string_block_scope_p(new_scope));
  }
  else {
    pips_debug(8, "new_scope = NULL\n");
  }

  return new_scope;
}

/* Allocate a new string containing only block scope information */
string scope_to_block_scope(string full_scope)
{
  string l_scope = strrchr(full_scope, BLOCK_SEP_CHAR);
  string f_scope = strchr(full_scope, MODULE_SEP);
  string block_scope = string_undefined;

  pips_debug(8, "full_scope = \"%s\"\n", full_scope);

  if(f_scope==NULL)
    f_scope = full_scope;
  else
    f_scope++;

  if(l_scope==NULL)
    block_scope = strdup("");
  else
    block_scope = gen_strndup0(f_scope, (unsigned) (l_scope-f_scope+1));

  pips_debug(8, "f_scope = \"%s\", l_scope = \"%s\"\n", f_scope, l_scope);
  pips_assert("block_scope is a scope", string_block_scope_p(block_scope));

  return block_scope;
}

c_parser_context CreateDefaultContext()
{
  c_parser_context c = make_c_parser_context(empty_scope(),
					     type_undefined,
					     storage_undefined,
					     NIL,
					     FALSE,
					     FALSE);
  pips_debug(8, "New default context %p\n", c);
  return c;
}

static int C_scope_identifier = -2;

void InitScope()
{
  C_scope_identifier = -1;
}

void EnterScope()
{
  c_parser_context nc = CreateDefaultContext();
  string cs = string_undefined;

  pips_assert("C_scope_identifier has been initialized", C_scope_identifier>-2);

  if(!stack_empty_p(ContextStack)) {
    c_parser_context c = (c_parser_context) stack_head(ContextStack);
    pips_assert("The current context is defined", !c_parser_context_undefined_p(c));
    pips_assert("The current context scope is defined",
		!string_undefined_p(c_parser_context_scope(c))
		&& c_parser_context_scope(c)!=NULL);
    pips_assert("The current context only contains scope information",
		//type_undefined_p(c_parser_context_type(c)) &&
		storage_undefined_p(c_parser_context_storage(c))
		&& ENDP(c_parser_context_qualifiers(c))
		//&& !c_parser_context_typedef(c)
		//&& !c_parser_context_static(c)
		);
    cs = c_parser_context_scope(c);
    pips_assert("scope contains only block scope information", string_block_scope_p(cs));
  }
  else
    cs = "";

  // Add scope information if any
  C_scope_identifier++;
  // A scope is needed right away to distinguish between formal
  // parameters and local variable. See
  // Validation/C_syntax/block_scope01.c, identifier x in function foo
  if(C_scope_identifier>=0) {
    string ns = i2a(C_scope_identifier);

    c_parser_context_scope(nc) = strdup(concatenate(cs, ns, BLOCK_SEP_STRING, NULL));
    free(ns);
  }
  else {
    c_parser_context_scope(nc) = strdup(cs);
  }

  stack_push((char *) nc, ContextStack);
  //ycontext = nc;
  //pips_assert("ycontext is consistant with stack_head(ContextStack)",
  //	      ycontext==stack_head(ContextStack));
  pips_debug(8, "New block scope string: \"%s\" for context %p\n",
	     c_parser_context_scope(nc), nc);
}

string GetScope()
{
  string s = "";

  /* FI: I do not know if it wouldn't be better to initialize the
     ContextStack with a default context before calling the C
     parser */
  if(!stack_empty_p(ContextStack)) {
    c_parser_context c = (c_parser_context) stack_head(ContextStack);

    s = c_parser_context_scope(c);
  }

  return s;
}

void ExitScope()
{
  c_parser_context c = (c_parser_context) stack_head(ContextStack);

  pips_assert("The current context is defined", !c_parser_context_undefined_p(c));
  pips_assert("The current context scope is defined",
	      !string_undefined_p(c_parser_context_scope(c))
	      && c_parser_context_scope(c)!=NULL);
  pips_assert("The current context only contains scope information",
	      //type_undefined_p(c_parser_context_type(c)) &&
	      storage_undefined_p(c_parser_context_storage(c))
	      && ENDP(c_parser_context_qualifiers(c))
	      //&& !c_parser_context_typedef(c)
	      //&& !c_parser_context_static(c)
	      );
  pips_debug(8, "Exiting context scope \"\%s\" in context %p\n",
	     c_parser_context_scope(c), c);
  free_c_parser_context(c);
  (void) stack_pop(ContextStack);
  if(!stack_empty_p(ContextStack)) {
    c_parser_context oc = (c_parser_context) stack_head(ContextStack);
    //pips_assert("ycontext is consistant with stack_head(ContextStack)",
    //		ycontext==stack_head(ContextStack));
    pips_debug(8, "Back to context scope \"\%s\" in context %p\n",
	       c_parser_context_scope(oc), oc);
  }
  else {
    // ycontext = c_parser_context_undefined;
    pips_debug(8, "Back to undefined context scope\n");
  }
}

void PushContext(c_parser_context c)
{
  stack_push((char *) c, ContextStack);
  pips_debug(8, "Context %p with scope \"%s\" is put in stack position %d\n",
	     c, c_parser_context_scope(c), stack_size(ContextStack));
}

void PopContext()
{
  c_parser_context fc = (c_parser_context) stack_head(ContextStack);
  c_parser_context c = c_parser_context_undefined;

  pips_debug(8, "Context %p with scope \"%s\" is popped from stack position %d\n",
	     fc, c_parser_context_scope(fc), stack_size(ContextStack));
  c = (c_parser_context) stack_pop(ContextStack);
  if(stack_empty_p(ContextStack)) {
    pips_debug(8, "context stack is now empty\n");
  }
  else {
    c_parser_context h = (c_parser_context) stack_head(ContextStack);
    pips_debug(8, "Context %p with scope \"%s\" is top of stack at position %d\n",
	       h, c_parser_context_scope(h), stack_size(ContextStack));
  }
}

c_parser_context GetContext()
{

    c_parser_context c = c_parser_context_undefined;

    if(!stack_empty_p(ContextStack))
      c = (c_parser_context) stack_head(ContextStack);
    else
      // Should we return a default context?
      // Not really compatible with a clean memory allocation policy
      pips_internal_error("No current context");

  pips_debug(8, "Context %p is obtained from stack position %d\n",
	     c, stack_size(ContextStack));

  return c;
}

c_parser_context GetContextCopy()
{
  c_parser_context c = (c_parser_context) stack_head(ContextStack);
  c_parser_context cc = copy_c_parser_context(c);
  pips_debug(8, "Context copy %p with scope \"%s\" is obtained from context %p with scope \"%s\" at stack position %d\n",
	     cc, c_parser_context_scope(cc),
	     c, c_parser_context_scope(c),
	     stack_size(ContextStack));
  return cc;
}

%}

/* Bison declarations */

%union {
	cons * liste;
	entity entity;
	expression expression;
	statement statement;
	string string;
	type type;
        parameter parameter;
        int integer;
        qualifier qualifier;
}

%token <string> TK_IDENT
%token <string> TK_CHARCON
%token <string> TK_INTCON
%token <string> TK_FLOATCON
%token <string> TK_NAMED_TYPE

%token <string> TK_STRINGCON
%token <string> TK_WSTRINGCON

%token TK_EOF
%token TK_CHAR TK_INT TK_DOUBLE TK_FLOAT TK_VOID TK_COMPLEX
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
%type <liste> global
%type <liste> attributes attributes_with_asm asmattr
%type <qualifier> attribute
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

%type <liste> type_spec
%type <liste> struct_decl_list

%type <parameter> parameter_decl
%type <entity> enumerator
%type <liste> enum_list
%type <liste> declaration
%type <void> function_def
%type <void> function_def_start
%type <type> type_name
%type <statement> block
%type <liste> local_labels local_label_names
%type <liste> old_parameter_list_ne
%type <liste> old_pardef_list
%type <liste> old_pardef

%type <entity> init_declarator
%type <liste> init_declarator_list
%type <entity> declarator
%type <entity> field_decl
%type <liste> field_decl_list
%type <entity> direct_decl
%type <entity> abs_direct_decl abs_direct_decl_opt
%type <entity> abstract_decl
%type <type> pointer pointer_opt
%type <void> location

%type <string> label
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
%type <liste> my_decl_spec_list
%type <liste> decl_spec_list_opt_no_named
%type <liste> decl_spec_list_opt
%type <entity> old_proto_decl direct_old_proto_decl
%%

interpret: file TK_EOF
                        {YYACCEPT;};
file: globals
                        {
			  /* To handle special case: compilation unit module */
			  list dl = $1;

			  if (TRUE /* dl != NIL*/) { /* A C file with comments only is OK */
			    if(!entity_undefined_p(get_current_module_entity())) {
			      if(!compilation_unit_p(get_current_module_name())) {
				pips_assert("Each variable is declared once", gen_once_p(dl));
				pips_internal_error("Compilation unit rule used for non"
						    " compilation unit %s\n",
						    get_current_module_name());
			      }
			      ifdebug(8) {
				pips_debug(8, "Declaration list for compilation unit %s: ",
					   get_current_module_name());
				print_entities(dl);
				pips_debug(8, "\n");
			      }
			      ModuleStatement = make_statement(entity_empty_label(),
							       STATEMENT_NUMBER_UNDEFINED,
							       STATEMENT_ORDERING_UNDEFINED,
							       string_undefined,
							       make_instruction_block(NIL),
							       dl, NULL, empty_extensions ());
			      if(ENDP(dl)) {
				pips_user_warning("ISO C forbids an empty source file\n");
			      }
			    }
			  }
			}
;

globals:
    /* empty */         { $$ = NIL; }
|   {is_external = TRUE; } global globals
                        {
			  /* Each variable should be declared only
			     once. Type and initial value conflict
			     should have been detected earlier. I
			     would expect $1 and $2 rather than $2 and
			     $3. */
			  if(!compilation_unit_p(get_current_module_name())) {
			    pips_assert("Each variable is declared once", gen_once_p($3));
			  }
			  ifdebug(9) {
			    fprintf(stderr, "New variables $2 (%p) are declared\n", $2);
			    print_entities($2);
			    fprintf(stderr, "\n");
			    fprintf(stderr, "*******Current declarations $3 (%p) are: \n", $3);
			    print_entities($3);
			    fprintf(stderr, "\n");
			  }

			  /* The order of declarations must be
			     preserved: a structure is declared before
			     it is used to declare a variable */
			  /*
			  $$ = NIL;
			  MAP(ENTITY, v, {
			    if(!gen_in_list_p(v, $3))
			      $$ = gen_nconc($$, CONS(ENTITY, v , NIL));}, $2);
			  $$ = gen_nconc($$, $3);
			  */
			  /* Redeclarations are possible in C as long as they are compatible */
			  /* It is assumed that compatibility is checked somewhere else... */
			  $$ = gen_nconc($2, $3);

			  ifdebug(9) {
			    fprintf(stderr, "*******Updated $$ declarations (%p) are: \n", $$);
			    fprintf(stderr, "\n");
			    if(ENDP($$))
			      fprintf(stderr, "Empty list\n");
			    else
			      print_entities($$);
			    fprintf(stderr, "\n");
			  }

			  if(!compilation_unit_p(get_current_module_name())) {
			    pips_assert("Each variable is declared once", gen_once_p($$));
			    ResetCurrentModule();
			  }
			}
|   TK_SEMICOLON globals
                        {
			  pips_assert("Declarations are unique", gen_once_p($2));
			  ifdebug(8) {
			    fprintf(stderr, "*******Current declarations are: \n");
			    print_entities($2);
			    fprintf(stderr, "\n");
			  }
			  $$ = $2;
			}
;

location:
    /* empty */         {}  %prec TK_IDENT

/*** Global Definition ***/
global:
declaration         { discard_C_comment();}
|   function_def        { $$ = NIL; }
|   TK_ASM TK_LPAREN string_constant TK_RPAREN TK_SEMICOLON
                        {
			  CParserError("ASM not implemented\n");
			  $$ = NIL;
			}
|   TK_PRAGMA attr
                        {
			  CParserError("PRAGMA not implemented\n");
			  $$ = NIL;
			}
/* Old-style function prototype. This should be somewhere else, like in
   "declaration". For now we keep it at global scope only because in local
   scope it looks too much like a function call */
|   TK_IDENT TK_LPAREN
                        {
			  entity oe = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			  entity e = oe; //RenameFunctionEntity(oe);
			  pips_debug(2,"Create function %s with old-style function prototype\n",$1);
			  if (storage_undefined_p(entity_storage(e))) {
			    //entity_storage(e) = make_storage_return(e);
			    entity_storage(e) = make_storage_rom();
			  }
			  if (value_undefined_p(entity_initial(e)))
			    entity_initial(e) = make_value(is_value_code,make_code(NIL,strdup(""),make_sequence(NIL),NIL));
			  //pips_assert("e is a module", module_name_p(entity_module_name(e)));
			  PushFunction(e);
			  stack_push((char *) make_basic_logical(TRUE),FormalStack);
			  stack_push((char *) make_basic_int(1),OffsetStack);
			  discard_C_comment();
			}
    old_parameter_list_ne TK_RPAREN old_pardef_list TK_SEMICOLON
                        {
			  entity e = GetFunction();
			  if (type_undefined_p(entity_type(e)))
			    {
			      list paras = MakeParameterList($4,$6,FunctionStack);
			      functional f = make_functional(paras,make_type_unknown());
			      entity_type(e) = make_type_functional(f);
			    }
			  pips_assert("Current function entity is consistent",entity_consistent_p(e));
			  PopFunction();
			  stack_pop(FormalStack);
			  StackPop(OffsetStack);
			  $$ = NIL;
			  discard_C_comment();
			}
/* Old style function prototype, but without any arguments */
|   TK_IDENT TK_LPAREN TK_RPAREN TK_SEMICOLON
                        {
			  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			  pips_debug(2,"Create function %s with old-style prototype, without any argument\n",$1);
			  if (type_undefined_p(entity_type(e)))
			    {
			      functional f = make_functional(NIL,make_type_unknown());
			      entity_type(e) = make_type_functional(f);
			    }
			  if (storage_undefined_p(entity_storage(e))) {
			    // entity_storage(e) = make_storage_return(e);
			    entity_storage(e) = make_storage_rom();
			  }
			  if (value_undefined_p(entity_initial(e)))
			    entity_initial(e) = make_value(is_value_code,make_code(NIL,strdup(""),make_sequence(NIL),NIL));
			  pips_assert("Current function entity is consistent",entity_consistent_p(e));
			  $$ = NIL;
			  discard_C_comment();
			}
/* transformer for a toplevel construct */
|   TK_AT_TRANSFORM TK_LBRACE global TK_RBRACE TK_IDENT /*to*/ TK_LBRACE globals TK_RBRACE
                        {
			  CParserError("CIL AT not implemented\n");
			  $$ = NIL;
			}
/* transformer for an expression */
|   TK_AT_TRANSFORMEXPR TK_LBRACE expression TK_RBRACE TK_IDENT /*to*/ TK_LBRACE expression TK_RBRACE
                        {
			  CParserError("CIL AT not implemented\n");
			  $$ = NIL;
			}
|   location error TK_SEMICOLON
                        {
			  CParserError("Parse error: location error TK_SEMICOLON \n");
			  $$ = NIL;
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
			  $$ = MakeUnaryCall(CreateIntrinsic(UNARY_PLUS_OPERATOR_NAME), $2);
			}
|   TK_MINUS expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(UNARY_MINUS_OPERATOR_NAME), $2);
			}
|   TK_STAR expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(DEREFERENCING_OPERATOR_NAME), $2);
			}
|   TK_AND expression				%prec TK_ADDROF
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(ADDRESS_OF_OPERATOR_NAME), $2);
			}
|   TK_EXCLAM expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(C_NOT_OPERATOR_NAME), $2);
			}
|   TK_TILDE expression
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(BITWISE_NOT_OPERATOR_NAME), $2);
			}
|   TK_PLUS_PLUS expression                    %prec TK_CAST
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(PRE_INCREMENT_OPERATOR_NAME), $2);
			}
|   expression TK_PLUS_PLUS
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(POST_INCREMENT_OPERATOR_NAME), $1);
			}
|   TK_MINUS_MINUS expression                  %prec TK_CAST
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(PRE_DECREMENT_OPERATOR_NAME), $2);
			}
|   expression TK_MINUS_MINUS
		        {
			  $$ = MakeUnaryCall(CreateIntrinsic(POST_DECREMENT_OPERATOR_NAME), $1);
			}
|   expression TK_ARROW id_or_typename
		        {
			  /* Find the struct/union type of the expression
			     then the struct/union member entity and transform it to expression */
			  expression exp = MemberIdentifierToExpression($1,$3);
			  $$ = MakeBinaryCall(CreateIntrinsic(POINT_TO_OPERATOR_NAME),$1,exp);
			}
|   expression TK_DOT id_or_typename
		        {
			  expression exp = MemberIdentifierToExpression($1,$3);
			  $$ = MakeBinaryCall(CreateIntrinsic(FIELD_OPERATOR_NAME),$1,exp);
			}
|   TK_LPAREN block TK_RPAREN
		        {
			  CParserError("GNU extension not implemented\n");
			}
|   paren_comma_expression
		        {
			  /* paren_comma_expression is a list of
			     expressions, maybe reduced to one */
			  if(!string_undefined_p(expression_comment)) {
			    pips_user_warning("comment \"%s\" is lost\n", expression_comment);
			  }
			  expression_comment = pop_current_C_comment();
			  expression_line_number = pop_current_C_line_number();
			  $$ = MakeCommaExpression($1);
			}
|   expression TK_LPAREN arguments TK_RPAREN
			{
			  $$ = MakeFunctionExpression($1,$3);
			}
|   TK_BUILTIN_VA_ARG TK_LPAREN expression TK_COMMA type_name TK_RPAREN
                        {

			  expression e = $3;
			  type t = $5;
			  sizeofexpression e1 = make_sizeofexpression_expression(e);
			  sizeofexpression e2 = make_sizeofexpression_type(t);
			  list l = CONS(SIZEOFEXPRESSION, e1,
					CONS(SIZEOFEXPRESSION, e2, NIL));
			  syntax s = make_syntax_va_arg(l);
			  expression r = make_expression(s, make_normalized_complex());

			  $$ = r;
			  //CParserError("BUILTIN_VA_ARG not implemented\n");
			}
|   expression bracket_comma_expression
			{
			  $$ = MakeArrayExpression($1,$2);
			}
|   expression TK_QUEST opt_expression TK_COLON expression
			{
			  $$ = MakeTernaryCallExpr(CreateIntrinsic(CONDITIONAL_OPERATOR_NAME), $1, $3, $5);
			}
|   expression TK_PLUS expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(PLUS_C_OPERATOR_NAME), $1, $3);
			}
|   expression TK_MINUS expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(MINUS_C_OPERATOR_NAME), $1, $3);
			}
|   expression TK_STAR expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(MULTIPLY_OPERATOR_NAME), $1, $3);
			}
|   expression TK_SLASH expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(DIVIDE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_PERCENT expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_MODULO_OPERATOR_NAME), $1, $3);
			}
|   expression TK_AND_AND expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_AND_OPERATOR_NAME), $1, $3);
			}
|   expression TK_PIPE_PIPE expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_OR_OPERATOR_NAME), $1, $3);
			}
|   expression TK_AND expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(BITWISE_AND_OPERATOR_NAME), $1, $3);
			}
|   expression TK_PIPE expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(BITWISE_OR_OPERATOR_NAME), $1, $3);
			}
|   expression TK_CIRC expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(BITWISE_XOR_OPERATOR_NAME), $1, $3);
			}
|   expression TK_EQ_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_EQUAL_OPERATOR_NAME), $1, $3);
			}
|   expression TK_EXCLAM_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_NON_EQUAL_OPERATOR_NAME), $1, $3);
			}
|   expression TK_INF expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_LESS_THAN_OPERATOR_NAME), $1, $3);
			}
|   expression TK_SUP expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_GREATER_THAN_OPERATOR_NAME), $1, $3);
			}
|   expression TK_INF_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_LESS_OR_EQUAL_OPERATOR_NAME), $1, $3);
			}
|   expression TK_SUP_EQ expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(C_GREATER_OR_EQUAL_OPERATOR_NAME), $1, $3);
			}
|   expression TK_INF_INF expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(LEFT_SHIFT_OPERATOR_NAME), $1, $3);
			}
|   expression TK_SUP_SUP expression
			{
			  $$ = MakeBinaryCall(CreateIntrinsic(RIGHT_SHIFT_OPERATOR_NAME), $1, $3);
			}
|   expression TK_EQ expression
			{
			  expression lhs = $1;
			  expression rhs = $3;
			  /* Check the left hand side expression */
			  if(expression_reference_p(lhs)) {
			    reference r = expression_reference(lhs);
			    entity v = reference_variable(r);
			    type t = ultimate_type(entity_type(v));
			    if(type_functional_p(t)) {
			      pips_user_warning("Ill. left hand side reference to function \"%s\""
                                                " or variable \"%s\" not declared\n",
						entity_user_name(v), entity_user_name(v));
			      CParserError("Ill. left hand side expression");
			    }
			  }
			  (void) simplify_C_expression(rhs);
			  $$ = make_assign_expression(lhs, rhs);
			}
|   expression TK_PLUS_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(PLUS_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_MINUS_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(MINUS_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_STAR_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(MULTIPLY_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_SLASH_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(DIVIDE_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_PERCENT_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(MODULO_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_AND_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(BITWISE_AND_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_PIPE_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(BITWISE_OR_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_CIRC_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(BITWISE_XOR_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_INF_INF_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(LEFT_SHIFT_UPDATE_OPERATOR_NAME), $1, $3);
			}
|   expression TK_SUP_SUP_EQ expression
			{
			  (void) simplify_C_expression($3);
			  $$ = MakeBinaryCall(CreateIntrinsic(RIGHT_SHIFT_UPDATE_OPERATOR_NAME), $1, $3);
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
                        { // Do we know about the size? 2, 4 or 8 bytes?
			  $$ = make_C_constant_entity($1, is_basic_int, 4);
			}
|   TK_FLOATCON
                        {
			  $$ = MakeConstant($1, is_basic_float);
			}
|   TK_CHARCON
                        {
			  $$ = make_C_constant_entity($1, is_basic_int, 1);
			}
|   string_constant
                        {
			  /* The size will be fixed later, hence 0 here. */
			  $$ = make_C_constant_entity($1, is_basic_string, 0);
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
    expression          {
                          expression ie = $1;
			  ifdebug(8) {
			    fprintf(stderr, "Initialization expression: ");
			    print_expression(ie);
			    fprintf(stderr, "\n");
			  }
			  $$ = ie;
                        }
|   TK_LBRACE initializer_list_opt TK_RBRACE
			{
			  /* Deduce the size of an array by its initialization ?*/
			  $$ = MakeBraceExpression($2);
			}

initializer_list:    /* ISO 6.7.8. Allow a trailing COMMA */
    initializer
                        {
			  $$ = CONS(EXPRESSION,$1,NIL);
			}
|   initializer TK_COMMA initializer_list_opt
                        {
			  $$ = CONS(EXPRESSION,$1,$3);
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
                        { $$ = expression_undefined; } /* This should be a null expression,
							  not expression_undefined*/
|   comma_expression
	        	{ $$ = MakeCommaExpression($1); }
;

comma_expression:
    expression
                        {
			  (void) simplify_C_expression($1);
			  $$ = CONS(EXPRESSION,$1,NIL);
			}
|   expression TK_COMMA comma_expression
                        {
			  (void) simplify_C_expression($1);
			  $$ = CONS(EXPRESSION,$1,$3);
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
			  push_current_C_comment();
			  push_current_C_line_number();
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
    TK_LBRACE
                        { EnterScope();}
    local_labels block_attrs declaration_list statement_list TK_RBRACE
                        {
			  $$ = MakeBlock($5,$6);
			  ExitScope();
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
			  //discard_C_comment();
			  $$ = gen_nconc($1,$2);
			}

;
statement_list:
    /* empty */         { $$ = NIL; }
|   statement statement_list
                        {
			  $$ = CONS(STATEMENT,$1,$2);
			}
/*(* GCC accepts a label at the end of a block *)*/
|   label	{ CParserError("gcc not implemented\n"); }
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

label:
   TK_IDENT TK_COLON
                        {
			  // Push the comment associated with the label:
			  push_current_C_comment();
			  $$ = $1;
			}
;

statement:
    TK_SEMICOLON
                    	{
			  /* Null statement in C is represented as continue statement in Fortran*/
			  $$ = make_continue_statement(entity_empty_label());
			}
|   comma_expression TK_SEMICOLON
	        	{
			  if (gen_length($1)==1)
			    $$ = ExpressionToStatement(EXPRESSION(CAR($1)));
			  else
			    $$ = call_to_statement(make_call(CreateIntrinsic(","),$1));
			  if(!string_undefined_p(expression_comment)) {
			    statement_comments($$) = expression_comment;
			    statement_number($$) = expression_line_number;
			    expression_comment = string_undefined;
			    expression_line_number = STATEMENT_NUMBER_UNDEFINED;
			  }
			}
|   block               { }
|   TK_IF paren_comma_expression statement                    %prec TK_IF
                	{
			  $$ = test_to_statement(make_test(MakeCommaExpression($2), $3,
							   make_empty_block_statement()));
			  pips_assert("statement is a test", statement_test_p($$));
                          statement_comments($$) = pop_current_C_comment();
			  statement_number($$) = pop_current_C_line_number();
			}
|   TK_IF paren_comma_expression statement TK_ELSE statement
	                {
			  $$ = test_to_statement(make_test(MakeCommaExpression($2),$3,$5));
			  pips_assert("statement is a test", statement_test_p($$));
                          statement_comments($$) = pop_current_C_comment();
			  statement_number($$) = pop_current_C_line_number();
			}
|   TK_SWITCH
                        {
			  stack_push((char *) make_sequence(NIL),SwitchGotoStack);
			  stack_push((char *) make_basic_int(loop_counter++), LoopStack);
			  /* push_current_C_comment(); */
			}
    paren_comma_expression
                        {
			  stack_push((char *)MakeCommaExpression($3),SwitchControllerStack);
			}
    statement
                        {
			  string sc = pop_current_C_comment();
			  int sn = pop_current_C_line_number();

			  $$ = MakeSwitchStatement($5);
			  $$ = add_comment_and_line_number($$, sc, sn);
			  stack_pop(SwitchGotoStack);
			  stack_pop(SwitchControllerStack);
			  stack_pop(LoopStack);
			}
|   TK_WHILE
                        {
			  stack_push((char *) make_basic_int(loop_counter++), LoopStack);
			  /* push_current_C_comment(); */
			}
    paren_comma_expression statement
	        	{
			  string sc = pop_current_C_comment();
			  int sn = pop_current_C_line_number();
			  pips_assert("While loop body consistent",statement_consistent_p($4));
			  $$ = MakeWhileLoop($3,$4,TRUE);
			  $$ = add_comment_and_line_number($$, sc, sn);
			  stack_pop(LoopStack);
			}
|   TK_DO
                        {
			  stack_push((char *) make_basic_int(loop_counter++), LoopStack);
			  /* push_current_C_comment(); */
			}
    statement TK_WHILE paren_comma_expression TK_SEMICOLON
	        	{
			  $$ = MakeWhileLoop($5,$3,FALSE);
			  /* The line number and comment are related to paren_comma_expression and not to TK_DO */
			  (void) pop_current_C_line_number();
			  (void) pop_current_C_comment();
			  stack_pop(LoopStack);
			}
|   TK_FOR
                        {
			  stack_push((char *) make_basic_int(loop_counter++), LoopStack);
			  /* push_current_C_comment(); */
			}
    TK_LPAREN for_clause opt_expression TK_SEMICOLON opt_expression TK_RPAREN statement
	                {
			  string sc = pop_current_C_comment();
			  int sn = pop_current_C_line_number();
			  expression init = $4;
			  expression cond = $5;
			  expression inc = $7;

			  pips_assert("For loop body consistent",statement_consistent_p($9));

			  if(expression_undefined_p(init))
			    init = make_call_expression(entity_intrinsic(CONTINUE_FUNCTION_NAME),
							NIL);
			  else
			    simplify_C_expression(init);

			  if(expression_undefined_p(cond)) {
			    /* A bool C constant cannot be used
			       because stdbool.h may not be
			       included */
			    /* cond = make_call_expression(MakeConstant(TRUE_OPERATOR_NAME, */
			    /* is_basic_logical), */
			    /* NIL); */
			    cond = int_to_expression(1);
			  }
			  else
			    simplify_C_expression(cond);

			  if(expression_undefined_p(inc))
			    inc = make_call_expression(entity_intrinsic(CONTINUE_FUNCTION_NAME),
							NIL);
			  else
			    simplify_C_expression(inc);

			  /*  The for clause may contain declarations*/
			  $$ = MakeForloop(init, cond, inc,$9);
			  $$ = add_comment_and_line_number($$, sc, sn);
			  stack_pop(LoopStack);
			  pips_assert("For loop consistent",statement_consistent_p($$));
			}
|   label statement
                        {
			  /* Create the statement with label comment in
			     front of it: */
			  $$ = MakeLabeledStatement($1,$2, pop_current_C_comment());
			  ifdebug(8) {
			    pips_debug(8,"Adding label '%s' to statement:\n", $1);
			    print_statement($$);
			  }
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
			  /* $$ =  call_to_statement(make_call(CreateIntrinsic(C_RETURN_FUNCTION_NAME),NIL)); */
                          $$ = make_statement(entity_empty_label(),
			                      get_current_C_line_number(),
			                      STATEMENT_ORDERING_UNDEFINED,
			                      get_current_C_comment(),
			                      call_to_instruction(make_call(CreateIntrinsic(C_RETURN_FUNCTION_NAME),NIL)),
					      NIL, string_undefined, empty_extensions ());
			  statement_consistent_p($$);
			}
|   TK_RETURN comma_expression TK_SEMICOLON
	                {
			  /* $$ =  call_to_statement(make_call(CreateIntrinsic(C_RETURN_FUNCTION_NAME),$2)); */
			  expression res = EXPRESSION(CAR($2));
			  if(expression_reference_p(res)) {
			    reference r = expression_reference(res);
			    entity v = reference_variable(r);
			    type t = ultimate_type(entity_type(v));
			    if(type_functional_p(t)) {
			      pips_user_warning("Ill. returned value: reference to function \"%s\""
                                                " or variable \"%s\" not declared\n",
						entity_user_name(v), entity_user_name(v));
			      CParserError("Ill. return expression");
			    }
			  }
                          $$ = make_statement(entity_empty_label(),
			                      get_current_C_line_number(),
			                      STATEMENT_ORDERING_UNDEFINED,
			                      get_current_C_comment(),
			                      make_instruction(is_instruction_call,
					      make_call(CreateIntrinsic(C_RETURN_FUNCTION_NAME), $2)),
					      NIL, string_undefined, empty_extensions ());
			  statement_consistent_p($$);
			}
|   TK_BREAK TK_SEMICOLON
                        {
			  $$ = MakeBreakStatement(get_current_C_comment());
			}
|   TK_CONTINUE TK_SEMICOLON
                 	{
			  $$ = MakeContinueStatement(get_current_C_comment());
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
    opt_expression TK_SEMICOLON
                        {
			  push_current_C_comment();
			  push_current_C_line_number();
			}
|   declaration
                        {
			  discard_C_comment();
			  CParserError("For clause containing declaration not implemented\n");
			}
;

declaration:                               /* ISO 6.7.*/
    decl_spec_list init_declarator_list TK_SEMICOLON
                        {
			  pips_assert("Declaration list are not redundant", gen_once_p($2));
			  pips_assert("Variable $1 has not been declared before", !gen_in_list_p($1, $2));
			  UpdateEntities($2,ContextStack,FormalStack,FunctionStack,OffsetStack,is_external,TRUE);
			  //stack_pop(ContextStack);
			  PopContext();
			  $$ = gen_nconc($1,$2);
			  /* Remove their type stacks */
			  remove_entity_type_stacks($$);
			  CleanUpEntities($$);
			}
|   decl_spec_list TK_SEMICOLON
                        {
			  //stack_pop(ContextStack);
			  PopContext();
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
			  $$ = CONS(ENTITY,$1,$3);
			}

;
init_declarator:                             /* ISO 6.7 */
    declarator          {
                          /* The default initial value is often zero,
                              but not so for formal parameters */
                          if(value_undefined_p(entity_initial($1)))
			    entity_initial($1) = make_value_unknown();
                        }
|   declarator TK_EQ init_expression
                        {
			  entity v = $1;
			  expression nie = $3;
			  value oiv = entity_initial(v);

			  if(expression_undefined_p(nie)) {
			    /* Do nothing, leave the initial field of entity as it is. */
			    pips_user_warning("Undefined init_expression, why not use value_unknown?\n");
			  }
			  else {

			    /* Put init_expression in the initial value of entity declarator*/
			    if(!value_undefined_p(oiv)){
			      if(value_unknown_p(oiv)) {
				free_value(oiv);
			      }
			      else {
				if(compilation_unit_p(get_current_module_name())) {
				  /* The compilation unit has already
				     been scanned once for
				     declarations. Double definitions
				     are no surprise...*/
				  ;
				}
				else {
				  pips_user_warning("double definition of initial"
						    " value for variable %s\n", entity_name(v));
				  fprintf(stderr, "New initial value expression:\n");
				  print_expression(nie);
				  fprintf(stderr, "Current initial value:\n");
				  if(value_expression_p(oiv)) {
				    print_expression(value_expression(oiv));
				  }
				  else {
				    fprintf(stderr, "Value tag: %d\n", value_tag(entity_initial(v)));
				  }
				  pips_internal_error("Scoping not implemented yet, might be the reason\n");
				}
			      }
			    }
			    entity_initial(v) = make_value_expression(nie);
			  }
			}
;

decl_spec_list:
                       {
			 if(stack_empty_p(ContextStack)) {
                           ycontext = CreateDefaultContext();
			   pips_debug(8, "new default context %p with scope \"%s\"\n", ycontext,
				      c_parser_context_scope(ycontext));
			 }
			 else {
			   /* Copy may be excessive as only the scope needs to be preserved...*/
			   //ycontext = copy_c_parser_context((c_parser_context)stack_head(ContextStack));
			   ycontext = GetContextCopy();
			   /* How can these two problems occur since
			      ycontext is only a copy of the
			      ContextStack's head? Are we in the
			      middle of a stack_push() /stack_pop()?
			      The previous policy was to always
			      allocate a new ycontext, regardless of
			      the stack state */
			   /* FI: a bit afraid of freeing the past type if any */
			   c_parser_context_type(ycontext) = type_undefined;
			   /* A new context is entered: no longer typedef as in
			    "typedef int f(int a)" when we hit "int a"*/
			   c_parser_context_typedef(ycontext) = FALSE;
			   /* FI: sometimes, the scope is erased and lost */
			   //if(strcmp(c_parser_context_scope(ycontext), "TOP-LEVEL:")==0)
			   //  c_parser_context_scope(ycontext) = empty_scope();
			   /* Finally, to avoid problems!*/
			   //c_parse
			   //r_context_scope(ycontext) = empty_scope();

			   /* Only block scope information is inherited */
			   if(!string_block_scope_p(c_parser_context_scope(ycontext))
			      && !string_struct_scope_p(c_parser_context_scope(ycontext))) {
			     pips_debug(8, "Reset modified scope \"%s\"\n",
					c_parser_context_scope(ycontext));
			     c_parser_context_scope(ycontext) = empty_scope();
			   }
			   pips_debug(8, "new context %p with scope \"%s\" copied from context %p (stack size=%d)\n",
				      ycontext,
				      c_parser_context_scope(ycontext),
				      stack_head(ContextStack),
				      stack_size(ContextStack));
                         }
                        }
    my_decl_spec_list { $$ = $2;}
;

my_decl_spec_list:                         /* ISO 6.7 */
                                        /* ISO 6.7.1 */
    TK_TYPEDEF decl_spec_list_opt
                        {
			  /* Add TYPEDEF_PREFIX to entity name prefix and make it a rom storage */
			  c_parser_context_typedef(ycontext) = TRUE;
			  c_parser_context_storage(ycontext) = make_storage_rom();
			  $$ = $2;
			}
|   TK_EXTERN decl_spec_list_opt
                        {
			  /* This can be a variable or a function, whose storage is ram or return  */
			  /* What is the scope in cyacc.y of this scope modification? */
			  pips_debug(8, "Scope of context %p forced to TOP_LEVEL_MODULE_NAME", ycontext);
			  c_parser_context_scope(ycontext) = strdup(concatenate(TOP_LEVEL_MODULE_NAME,
									       MODULE_SEP_STRING,NULL));
			  $$ = $2;
			}
|   TK_STATIC decl_spec_list_opt
                        {
			  c_parser_context_static(ycontext) = TRUE;
			  $$ = $2;
			}
|   TK_AUTO decl_spec_list_opt
                        {
			  /* Make dynamic storage for current entity */
			  $$ = $2;
			}
|   TK_REGISTER decl_spec_list_opt
                        {
			  /* Add to type variable qualifiers */
			  c_parser_context_qualifiers(ycontext) = gen_nconc(c_parser_context_qualifiers(ycontext),
									   CONS(QUALIFIER,make_qualifier_register(),NIL));
			  $$ = $2;
			}
                                        /* ISO 6.7.2 */
|   type_spec decl_spec_list_opt_no_named
                        {
			  $$ = gen_nconc($1,$2);
			}
                                        /* ISO 6.7.4 */
|   TK_INLINE decl_spec_list_opt
                        {
			  pips_user_warning("Keyword \"inline\" ignored\n");
			  $$ = $2;
			}
|   attribute decl_spec_list_opt
                        {
			  c_parser_context_qualifiers(ycontext) = CONS(QUALIFIER,$1,c_parser_context_qualifiers(ycontext));
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
    /* empty */
                        {
			  //stack_push((char *) ycontext, ContextStack);
			  PushContext(ycontext);
			  $$ = NIL;
			} %prec TK_NAMED_TYPE
|   my_decl_spec_list      { }
;

/* (* We add this separate rule to handle the special case when an appearance
    * of NAMED_TYPE should not be considered as part of the specifiers but as
    * part of the declarator. IDENT has higher precedence than NAMED_TYPE  *)
 */
decl_spec_list_opt_no_named:
    /* empty */
                        {
			  //stack_push((char *) ycontext,ContextStack);
			  PushContext(ycontext);
                          $$ = NIL;
			} %prec TK_IDENT
|   my_decl_spec_list      { }
;


type_spec:   /* ISO 6.7.2 */
    TK_VOID
                        {
			  c_parser_context_type(ycontext) = make_type_void();
			  $$ = NIL;
                        }
|   TK_CHAR
                        {
			  c_parser_context_type(ycontext) = make_standard_integer_type(c_parser_context_type(ycontext),
										      DEFAULT_CHARACTER_TYPE_SIZE);
			  $$ = NIL;
			}
|   TK_SHORT
                        {
			  c_parser_context_type(ycontext) = make_standard_integer_type(c_parser_context_type(ycontext),
										      DEFAULT_SHORT_INTEGER_TYPE_SIZE);
			  $$ = NIL;
			}
|   TK_INT
                        {
			  if (c_parser_context_type(ycontext) == type_undefined)
			    {
			      variable v = make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			      c_parser_context_type(ycontext) = make_type_variable(v);
			    }
			  $$ = NIL;
			}
|   TK_COMPLEX
                        {
			  if (c_parser_context_type(ycontext) == type_undefined)
			    {
			      variable v = make_variable(make_basic_complex(DEFAULT_COMPLEX_TYPE_SIZE),NIL,NIL);
			      c_parser_context_type(ycontext) = make_type_variable(v);
			    }
			  else {
			    /* Can be qualified by float, double and long double */
			    type t = c_parser_context_type(ycontext);
			    variable v = type_variable(t);
			    basic b = variable_basic(v);

			    pips_assert("prefix is for type variable",type_variable_p(t));
			    if(basic_float_p(b)) {
			      basic_tag(b) = is_basic_complex;
			      basic_complex(b) = 2*basic_complex(b);
			      if(basic_complex(b)==DEFAULT_COMPLEX_TYPE_SIZE)
				basic_complex(b) += 1;
			    }
			  }
			  $$ = NIL;
			}
|   TK_LONG
                        {
			  c_parser_context_type(ycontext) = make_standard_long_integer_type(c_parser_context_type(ycontext));
			  $$ = NIL;
			}
|   TK_FLOAT
                        {
			  variable v = make_variable(make_basic_float(DEFAULT_REAL_TYPE_SIZE),NIL,NIL);
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = NIL;
			}
|   TK_DOUBLE
                        {
			  variable v = make_variable(make_basic_float(DEFAULT_DOUBLEPRECISION_TYPE_SIZE),NIL,NIL);
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = NIL;
			}
|   TK_SIGNED
                        {
			  /* see the RI document or ri-util.h for explanation */
			  variable v = make_variable(make_basic_int(DEFAULT_SIGNED_TYPE_SIZE*10+
								    DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = NIL;
			}
|   TK_UNSIGNED
                        {
			  variable v = make_variable(make_basic_int(DEFAULT_UNSIGNED_TYPE_SIZE*10+
								    DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = NIL;
			}
|   TK_STRUCT id_or_typename
                        {
			  /* Find the entity associated to the struct, current scope can be [file%][module:][block]*/
			  entity ent = FindOrCreateEntityFromLocalNameAndPrefix($2,STRUCT_PREFIX,is_external);
			  /* Specify the type of the variable that follows this declaration specifier */
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  /* To handle mesa.c (SPEC 2000 benchmark)
			     We have struct HashTable in a file, but do not know its structure and scope,
			     because it is declared in other file  */
			  if (type_undefined_p(entity_type(ent)))
			    entity_type(ent) = make_type_struct(NIL);
			  if (storage_undefined_p(entity_storage(ent)))
			    entity_storage(ent) = make_storage_rom();
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = NIL;
			}
|   TK_STRUCT id_or_typename TK_LBRACE
                        {
			  code c = make_code(NIL,$2,sequence_undefined,NIL);
			  stack_push((char *) c, StructNameStack);
			}
    struct_decl_list TK_RBRACE
                        {
			  /* Create the struct entity */
			  entity ent = MakeDerivedEntity($2,$5,is_external,is_type_struct);
			  /* Specify the type of the variable that follows this declaration specifier*/
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  /* Take from $5 the struct/union entities */
			  list le = TakeDeriveEntities($5);
			  $$ = gen_nconc(le,CONS(ENTITY,ent,NIL));
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  stack_pop(StructNameStack);
			}
|   TK_STRUCT TK_LBRACE
                        {
              string istr = i2a(derived_counter++);
			  code c = make_code(NIL,strdup(concatenate(DUMMY_STRUCT_PREFIX,
								    istr,NULL)),sequence_undefined, NIL);
              free(istr);
			  stack_push((char *) c, StructNameStack);
                        }
    struct_decl_list TK_RBRACE
                        {
			  /* Create the struct entity with unique name s */
			  string s = code_decls_text((code) stack_head(StructNameStack));
			  entity ent = MakeDerivedEntity(s,$4,is_external,is_type_struct);
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  /* Take from $4 the struct/union entities */
			  list le = TakeDeriveEntities($4);
			  $$ = gen_nconc(le,CONS(ENTITY,ent,NIL));
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  stack_pop(StructNameStack);
			}
|   TK_UNION id_or_typename
                        {
			  /* Find the entity associated to the union, current scope can be [file%][module:][block]*/
			  entity ent = FindOrCreateEntityFromLocalNameAndPrefix($2,UNION_PREFIX,is_external);
			  /* Specify the type of the variable that follows this declaration specifier */
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  if (type_undefined_p(entity_type(ent)))
			    entity_type(ent) = make_type_union(NIL);
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = NIL;
			}
|   TK_UNION id_or_typename TK_LBRACE
                        {
			  code c = make_code(NIL,$2,sequence_undefined, NIL);
			  stack_push((char *) c, StructNameStack);
			}
    struct_decl_list TK_RBRACE
                        {
			  entity ent = MakeDerivedEntity($2,$5,is_external,is_type_union);
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  /* Take from $5 the struct/union entities */
			  list le = TakeDeriveEntities($5);
			  $$ = gen_nconc(le,CONS(ENTITY,ent,NIL));
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  stack_pop(StructNameStack);
			}
|   TK_UNION TK_LBRACE
                        {
			  string n = i2a(derived_counter++);
			  code c = make_code(NIL,
					     strdup(concatenate(DUMMY_UNION_PREFIX,n,NULL)),
					     sequence_undefined,
					     NIL);
			  free(n);
			  stack_push((char *) c, StructNameStack);
                        }
    struct_decl_list TK_RBRACE
                        {
			  /* Create the union entity with unique name */
			  string s = code_decls_text((code) stack_head(StructNameStack));
			  entity ent = MakeDerivedEntity(s,$4,is_external,is_type_union);
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  /* Take from $4 the struct/union entities */
			  list le = TakeDeriveEntities($4);
			  $$ = gen_nconc(le,CONS(ENTITY,ent,NIL));
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  stack_pop(StructNameStack);
			}
|   TK_ENUM id_or_typename
                        {
                          /* Find the entity associated to the enum */
			  entity ent = FindOrCreateEntityFromLocalNameAndPrefix($2,ENUM_PREFIX,is_external);
			  /* Specify the type of the variable that follows this declaration specifier */
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);
			  if (type_undefined_p(entity_type(ent)))
			    entity_type(ent) = make_type_enum(NIL);
			  /* FI: What should the initial value be? */
			  if (value_undefined_p(entity_initial(ent)))
			    entity_initial(ent) = make_value_unknown();
			  if(!entity_undefined_p(get_current_module_entity()))
			    AddEntityToDeclarations(ent, get_current_module_entity());
			  else {
			    /* This happens with the old style
			       function declaration at least */
			    /* Oops, we have to assume that the enum
			       is also defined in the compilation
			       unit... else it would be useless. */
			    ;
			  }
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = NIL;
			}
|   TK_ENUM id_or_typename TK_LBRACE enum_list maybecomma TK_RBRACE
                        {
                          /* Create the enum entity */
			  entity ent = MakeDerivedEntity($2,$4,is_external,is_type_enum);
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);

			  InitializeEnumMemberValues($4);
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = CONS(ENTITY,ent,NIL);
			}
|   TK_ENUM TK_LBRACE enum_list maybecomma TK_RBRACE
                        {
			  /* Create the enum entity with unique name */
			  string n = i2a(derived_counter++);
			  string s = strdup(concatenate(DUMMY_ENUM_PREFIX,n,NULL));
			  free(n);
			  entity ent = MakeDerivedEntity(s,$3,is_external,is_type_enum);
			  variable v = make_variable(make_basic_derived(ent),NIL,NIL);

			  InitializeEnumMemberValues($3);
			  c_parser_context_type(ycontext) = make_type_variable(v);
			  $$ = CONS(ENTITY,ent,NIL);
			}
|   TK_NAMED_TYPE
                        {
			  entity ent;
			  ent = FindOrCreateEntityFromLocalNameAndPrefix($1,TYPEDEF_PREFIX,is_external);

			  /* Specify the type of the variable that follows this declaration specifier */
			  if (c_parser_context_typedef(ycontext))
			    {
			      /* typedef T1 T2 => the type of T2 will be that of T1*/
			      pips_debug(8,"typedef T1 T2 where T1 =  %s\n",entity_name(ent));
			      c_parser_context_type(ycontext) = entity_type(ent);
			      $$ = CONS(ENTITY,ent,NIL);
			    }
			  else
			    {
			      /* T1 var => the type of var is basic typedef */
			      variable v = make_variable(make_basic_typedef(ent),NIL,NIL);
			      pips_debug(8,"T1 var where T1 =  %s\n",entity_name(ent));
			      c_parser_context_type(ycontext) = make_type_variable(v);
			      $$ = NIL;
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
			  //c_parser_context ycontext = stack_head(ContextStack);
			  c_parser_context ycontext = GetContext();
			  /* Create the struct member entity with unique name, the name of the
			     struct/union is added to the member name prefix */
              string istr = i2a(derived_counter++);
			  string s = strdup(concatenate("PIPS_MEMBER_",istr,NULL));
			  string derived = code_decls_text((code) stack_head(StructNameStack));
			  entity ent = CreateEntityFromLocalNameAndPrefix(s,strdup(concatenate(derived,
											       MEMBER_SEP_STRING,NULL)),
									  is_external);
			  pips_debug(5,"Current derived name: %s\n",derived);
			  pips_debug(5,"Member name: %s\n",entity_name(ent));
			  entity_storage(ent) = make_storage_rom();
			  entity_type(ent) = c_parser_context_type(ycontext);
              free(s);

			  /* Temporally put the list of struct/union
			     entities defined in $1 to initial value
			     of ent. FI: where is it retrieved? in
			     TakeDeriveEntities()? */
			  entity_initial(ent) = (value) $1;

			  $$ = CONS(ENTITY,ent,$3);

			  //stack_pop(ContextStack);
			  PopContext();
			}
|   decl_spec_list
                        {
			  //c_parser_context ycontext = stack_head(ContextStack);
			  c_parser_context ycontext = GetContext();
			  /* Add struct/union name and MEMBER_SEP_STRING to entity name */
			  string derived = code_decls_text((code) stack_head(StructNameStack));
			  c_parser_context_scope(ycontext) = CreateMemberScope(derived,is_external);
			  c_parser_context_storage(ycontext) = make_storage_rom();
			}
    field_decl_list TK_SEMICOLON struct_decl_list
                        {
			  /* Update the entity in field_decl_list with final type, storage, initial value*/

			  UpdateDerivedEntities($1,$3,ContextStack);

			  /* Create the list of member entities */
			  $$ = gen_nconc($3,$5);
			  //stack_pop(ContextStack);
			  PopContext();

			  /* This code is not good ...
			     I have problem with the global variable ycontext and recursion: ycontext is crushed
			     when this decl_spec_list in struct_decl_list is entered, so the scope and storage
			     of the new context are given to the old context, before it is pushed in the stack.

			     For the moment, I reset the changed values of the context, by hoping that in C,
			     before a STRUCT/UNION declaration, there is no extern, ... */
			  c_parser_context_scope(ycontext) = empty_scope();
			  c_parser_context_storage(ycontext) = storage_undefined;
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
			  $$ = CONS(ENTITY,$1,$3);
			}
;

field_decl: /* (* ISO 6.7.2. Except that we allow unnamed fields. *) */
    declarator
                        {
			}
|   declarator TK_COLON expression
                        {
			  value nv = EvalExpression($3);
			  constant c = value_constant_p(nv)?
			    value_constant(nv) : make_constant_unknown();
			  symbolic s = make_symbolic($3, c);
			  variable v = make_variable(make_basic_bit(s),NIL,NIL);

			  /*pips_assert("Width of bit-field must be a positive constant integer",
			    integer_constant_expression_p($3)); */
			  /* Ignore for this moment if the bit is signed or unsigned */
			  entity_type($1) = make_type_variable(v);
			  $$ = $1;
			}
|   TK_COLON expression
                        {
			  //c_parser_context ycontext = stack_head(ContextStack);
			  c_parser_context ycontext = GetContext();
			  /* Unnamed bit-field : special and unique name */
			  string n = i2a(derived_counter++);
			  string s = strdup(concatenate(DUMMY_MEMBER_PREFIX,n,NULL));
			  entity ent = CreateEntityFromLocalNameAndPrefix(s,c_parser_context_scope(ycontext),is_external);
			  value nv = EvalExpression($2);
			  constant c = value_constant_p(nv)?
			    value_constant(nv) : make_constant_unknown();
			  symbolic se = make_symbolic($2, c);
			  variable v = make_variable(make_basic_bit(se),NIL,NIL);
                          /* pips_assert("Width of bit-field must be a positive constant integer",
                             integer_constant_expression_p($2)); */
			  entity_type(ent) = make_type_variable(v);
              free(n);
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

			  No need to add current struct/union/enum name to the name's scope of the member entity
			  for ENUM, as in the case of STRUCT and UNION */

			  entity ent = CreateEntityFromLocalNameAndPrefix($1,"",is_external);
			  variable v = make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			  type rt = make_type(is_type_variable, v);
			  functional f = make_functional(NIL, rt);

			  entity_storage(ent) = make_storage_rom();
			  entity_type(ent) = make_type_functional(f);
			  /* The information is not yet available, but
			     I need to recognize this entity as
			     symbolic for next rule */
			  entity_initial(ent) = make_value_symbolic(make_symbolic(expression_undefined, make_constant_unknown()));
			  // enum_list is not available yet. Values should be fixed later.
			  /*  entity_initial(ent) = MakeEnumeratorInitialValue(enum_list,enum_counter);*/
			  $$ = ent;
			}
|   TK_IDENT TK_EQ expression
                        {
			  /* Create an entity of is_basic_int, storage rom, initial_value = $3 */
			  /* No, enum member must be functional
			     entity, just like Fortran's parameters */
			  //int i;
			  value vinit = value_undefined;
			  entity ent = CreateEntityFromLocalNameAndPrefix($1,"",is_external);
			  variable v =
			    make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
			  type rt = make_type(is_type_variable, v);
			  functional f = make_functional(NIL, rt);

			  //pips_assert("Enumerated value must be a constant integer",
			  //	      signed_integer_constant_expression_p($3));
			  //i = signed_integer_constant_expression_value($3);
			  vinit = EvalExpression($3);
			  entity_storage(ent) = make_storage_rom();
			  entity_type(ent) = make_type_functional(f);

                          if(value_constant_p(vinit) && constant_int_p(value_constant(vinit))) {
			    entity_initial(ent) =
			      make_value_symbolic(make_symbolic($3, value_constant(vinit)));
			  }
                          else {
			    /* Error or reference to a previous member of the same enum (enum04.c) */
			    /* FI: it might be easier to delay systematically the evaluation */
			    bool is_ok = FALSE;
			    if(expression_call_p($3)) {
			      call c = syntax_call(expression_syntax($3));
			      entity m = call_function(c);

			      if(entity_symbolic_p(m)) {
				is_ok = TRUE;
			      }
			    }
			    if(is_ok)
			      entity_initial(ent) =
				make_value_symbolic(make_symbolic($3, make_constant_unknown()));
			    else {
			      /* Let's try to delay evaluation anyway (enum05.c) */
			      entity_initial(ent) =
				make_value_symbolic(make_symbolic($3, make_constant_unknown()));
			      //pips_internal_error("Constant integer expression not evaluated\n");
			    }
			  }

			  $$ = ent;
			}
;

declarator:  /* (* ISO 6.7.5. Plus Microsoft declarators.*) */
    pointer_opt direct_decl attributes_with_asm
                        {
			  /* Update the type of the direct_decl entity with pointer_opt and attributes*/
			  if (!type_undefined_p($1))
			    UpdatePointerEntity($2,$1,$3);
			  $$ = $2;
			}
;

direct_decl: /* (* ISO 6.7.5 *) */
                                   /* (* We want to be able to redefine named
                                    * types as variable names *) */
    id_or_typename
                        {
			  /* FI: A variable cannot be redeclared
			     within the same scope, but this is not
			     checked yet. */
			  entity e = FindOrCreateCurrentEntity($1,ContextStack,FormalStack,FunctionStack,is_external);
			  /* Initialize the type stack and push the type of found/created entity to the stack.
			     It can be undefined if the entity has not been parsed, or a given type which is
			     used later to check if the declarations are the same for one entity.
			     This stack is put temporarily in the storage of the entity, not a global variable
			     for each declarator to avoid being erased by recursion
			     (FI: this last sentence seems to be wrong) */
			  stack s = get_from_entity_type_stack_table(e);
			  if(stack_undefined_p(s)) {
			    s = stack_make(type_domain,0,0);
			    stack_push((char *) entity_type(e),s);
			    put_to_entity_type_stack_table(e,s);
			  }
			  else {
			    entity cm = get_current_module_entity();
			    /* A function can be redeclared inside itself. see C_syntax/extern.c */
			    if(cm!=e) {
			      extern int yylineno; /* from lexer */
			      /* Dummy parameters can also be redeclared
				 as long as their types are equal */
			      if(dummy_parameter_entity_p(e)) {
				pips_user_warning("Dummy parameter \"%s\" is redefined at line %d (%d)\n",
						  entity_user_name(e),
						  get_current_C_line_number(), yylineno);
				CParserError("Dummy redefinition accepted by gcc but not compatible with ISO standard."
					     " Try to compile with \"gcc -ansi -c\"\n");
			      }
			      else {
				pips_user_warning("Variable \"%s\" is redefined at line %d (%d)\n",
						  entity_user_name(e) /* entity_name(e)*/,
						  get_current_C_line_number(), yylineno);
				CParserError("Variable redefinition not compatible with ISO standard."
					     " Try to compile with \"gcc -ansi -c\"\n");
			      }
			    }
			  }

			  entity_type(e) = type_undefined;
			  discard_C_comment();
			  $$ = e;
			}
|   TK_LPAREN attributes declarator TK_RPAREN
                        {
			  /* Add attributes such as const, restrict, ... to variable's qualifiers */
			  UpdateParenEntity($3,$2);
			  $$ = $3;
			  stack_push((char *) entity_type($$),
			     get_from_entity_type_stack_table($$));
			  // FI: if I rely on the stack, I won't know for a while what
			  // this entity is. And I'd like to make a difference between
			  // a function and a pointer to a function before I declare
			  // dummy arguments. But Nga's design has to be redone:-(.
			  entity_type($$) = type_undefined;
			}
|   direct_decl TK_LBRACKET attributes comma_expression_opt TK_RBRACKET
                        {
			 /* This is the last dimension of an array (i.e a[1][2][3] => [3]).
			    Questions:
			     - What can be attributes ?
			     - Why comma_expression, it can be a list of expressions ?
                             - When the comma_expression is empty (corresponding to the first dimension),
			       the array is of unknown size => can be determined by the intialization ? TO BE DONE*/
			  UpdateArrayEntity($1,$3,$4);
			}
|   direct_decl TK_LBRACKET attributes error TK_RBRACKET
                        {
			  CParserError("Parse error: direct_decl TK_LBRACKET attributes error TK_RBRACKET\n");
			}
|   direct_decl parameter_list_startscope
                        {
			  /* Well, here it can be a function or a pointer to a function */
			  entity e = $1; //RenameFunctionEntity($1);
			  if (value_undefined_p(entity_initial(e))
			      ||value_unknown_p(entity_initial(e)))
			    entity_initial(e) = make_value(is_value_code,make_code(NIL,strdup(""),make_sequence(NIL), NIL));
			  //pips_assert("e is a module", module_name_p(entity_module_name(e)));
			  PushFunction(e);
			}
    rest_par_list TK_RPAREN
                        {
			  entity e = GetFunction();
			  PopFunction();
			  stack_pop(FormalStack);
			  StackPop(OffsetStack);
			  /* Intrinsic functions in C such as printf, fprintf, ... are considered
			     as entities with functional type ???
			     if (!intrinsic_entity_p(e))*/
			  UpdateFunctionEntity(e,$4);
			  $$ = e;
			}
;

parameter_list_startscope:
    TK_LPAREN
                        {
			  stack_push((char *) make_basic_logical(TRUE), FormalStack);
			  stack_push((char *) make_basic_int(1),OffsetStack);
			}
;

rest_par_list:
    /* empty */         { $$ = NIL; }
|   parameter_decl rest_par_list1
                        {
			  $$ = CONS(PARAMETER,$1,$2);
			}
;
rest_par_list1:
    /* empty */         { $$ = NIL; }
|   TK_COMMA
                        {
			  StackPush(OffsetStack);
			}
    TK_ELLIPSIS
                        {
			  /*$$ = CONS(PARAMETER,make_parameter(make_type_varargs(type_undefined),
			    make_mode(CurrentMode,UU), make_dummy_unknown()),NIL); */
			  type at = make_type(is_type_variable,
					      make_variable(make_basic(is_basic_overloaded, UU),
							    NIL, NIL));
			  $$ = CONS(PARAMETER,
				    make_parameter(make_type_varargs(at),
						   make_mode(CurrentMode,UU),
						   make_dummy_unknown()),
				    NIL);
			}
|   TK_COMMA
                        {
			  StackPush(OffsetStack);
			}
    parameter_decl rest_par_list1
                        {
			  $$ = CONS(PARAMETER,$3,$4);
			}
;

parameter_decl: /* (* ISO 6.7.5 *) */
    decl_spec_list declarator
                        {
			  UpdateEntity($2,ContextStack,FormalStack,FunctionStack,OffsetStack,is_external,FALSE);
			  $$ = make_parameter(copy_type(entity_type($2)),
					      make_mode(CurrentMode,UU),
					      make_dummy_identifier($2)); //FI: or should it
			  // be entity_undefined? Are we parsing a compilation unit or a function?
			  /* Set CurentMode where ???? */
			  //stack_pop(ContextStack);
			  PopContext();
			}
|   decl_spec_list abstract_decl
                        {
			  UpdateAbstractEntity($2,ContextStack);
			  $$ = make_parameter(copy_type(entity_type($2)),
					      make_mode(CurrentMode,UU),
					      make_dummy_unknown()); //FI: to be checked
			  RemoveFromExterns($2);
			  free_entity($2);
			  //stack_pop(ContextStack);
			  PopContext();
			}
|   decl_spec_list
                        {
			  c_parser_context ycontext = stack_head(ContextStack);
			  $$ = make_parameter(copy_type(c_parser_context_type(ycontext)),
					      make_mode(CurrentMode,UU),
					      make_dummy_unknown());
			  /* function prototype*/
			  //stack_pop(ContextStack);
			  PopContext();
			}
|   TK_LPAREN parameter_decl TK_RPAREN
                        { $$ = $2; }
;

/* (* Old style prototypes. Like a declarator *) */
old_proto_decl:
    pointer_opt direct_old_proto_decl
                        {
			  if (!type_undefined_p($1))
			    UpdatePointerEntity($2,$1,NIL);
			  $$ = $2;
			}
;

direct_old_proto_decl:
    direct_decl TK_LPAREN
                        {
			  entity e = $1; //RenameFunctionEntity($1);
			  if (value_undefined_p(entity_initial(e)))
			    entity_initial($1) = make_value(is_value_code,make_code(NIL,strdup(""),make_sequence(NIL),NIL));
			  //pips_assert("e is a module", module_name_p(entity_module_name(e)));
			  PushFunction(e);
			  stack_push((char *) make_basic_logical(TRUE),FormalStack);
			  stack_push((char *) make_basic_int(1),OffsetStack);
			}
    old_parameter_list_ne TK_RPAREN old_pardef_list
                        {
			  entity e = GetFunction();
			  list paras = MakeParameterList($4,$6,FunctionStack);
			  PopFunction();
			  stack_pop(FormalStack);
			  StackPop(OffsetStack);
			  (void) UpdateFunctionEntity(e, paras);
			  $$ = e;
			}
|   direct_decl TK_LPAREN TK_RPAREN
                        {
                          (void) UpdateFunctionEntity($1,NIL);
			}
;

old_parameter_list_ne:
    TK_IDENT
                        {
			  $$ = CONS(STRING,$1,NIL);
			}
|   TK_IDENT TK_COMMA old_parameter_list_ne
                        {
			  $$ = CONS(STRING,$1,$3);
			}
;

old_pardef_list:
    /* empty */         { $$ = NIL; }
|   decl_spec_list old_pardef TK_SEMICOLON TK_ELLIPSIS
                        {
			  UpdateEntities($2,ContextStack,FormalStack,FunctionStack,OffsetStack,is_external,FALSE);
			  //stack_pop(ContextStack);
			  PopContext();
			  /* Can we have struct/union definition in $1 ?*/
			  /*$$ = gen_nconc($1,$2);*/
			  $$ = $2;
			}
|   decl_spec_list old_pardef TK_SEMICOLON old_pardef_list
                        {
			  UpdateEntities($2,ContextStack,FormalStack,FunctionStack,OffsetStack,is_external,FALSE);
			  //stack_pop(ContextStack);
			  PopContext();
			  /* Can we have struct/union definition in $1 ?*/
			  /*$$ = gen_nconc($1,gen_nconc($2,$4));*/
			  $$ = gen_nconc($2,$4);
			}
;

old_pardef:
    declarator
                        {
			  $$ = CONS(ENTITY,$1,NIL);
			}
|   declarator TK_COMMA old_pardef
                        {
			  $$ = CONS(ENTITY,$1,$3);
			}
|   error
                        {
			  CParserError("Parse error: error \n");
			}
;

pointer: /* (* ISO 6.7.5 *) */
    TK_STAR attributes pointer_opt
                        {
			  $$ = make_type_variable(make_variable(make_basic_pointer($3),NIL,$2));
			}
;

pointer_opt:
    /* empty */         { $$ = type_undefined;}
|   pointer             { }
;

type_name: /* (* ISO 6.7.6 *) */
    decl_spec_list abstract_decl
                        {
			  UpdateAbstractEntity($2,ContextStack);
			  $$ = copy_type(entity_type($2));
			  RemoveFromExterns($2);
			  free_entity($2);
			  //stack_pop(ContextStack);
			  PopContext();
			}
|   decl_spec_list
                        {
			  c_parser_context ycontext = stack_head(ContextStack);
			  $$ = c_parser_context_type(ycontext);
			  //stack_pop(ContextStack);
			  PopContext();
			}
;

abstract_decl: /* (* ISO 6.7.6. *) */
    pointer_opt abs_direct_decl attributes
                        {
			  /* Update the type of the direct_decl entity with pointer_opt and attributes*/
			  if (!type_undefined_p($1))
			    UpdatePointerEntity($2,$1,$3);
			  $$ = $2;
			}
|   pointer
                        {
			  string n = i2a(abstract_counter++);
			  $$ = FindOrCreateCurrentEntity(strdup(concatenate(DUMMY_ABSTRACT_PREFIX,
									    n,NULL)),
							 ContextStack,
							 FormalStack,
							 FunctionStack,
							 is_external);
			  free(n);
			  UpdatePointerEntity($$,$1,NIL);
			  /* Initialize the type stack and push the type of found/created entity to the stack.
			     It can be undefined if the entity hasnot been parsed, or a given type which is
			     used later to check if the declarations are the same for one entity.
			     This stack is put temporarily in the storage of the entity, not a global variable
			     for each declarator to avoid being erased by recursion */
			  stack s = stack_make(type_domain, 0, 0);
			  //entity_storage($$) = (storage) s;
			  stack_push((char *) entity_type($$),s);
			  put_to_entity_type_stack_table($$, s);
			  /*entity_type($$) = type_undefined;*/
			}
;

abs_direct_decl: /* (* ISO 6.7.6. We do not support optional declarator for
                     * functions. Plus Microsoft attributes. See the
                     * discussion for declarator. *) */
    TK_LPAREN attributes abstract_decl TK_RPAREN
                        {
			  UpdateParenEntity($3,$2);
			  $$ = $3;
			  stack_push((char *) entity_type($$),
				     get_from_entity_type_stack_table($$));
			  entity_type($$) = type_undefined;
			}
|   TK_LPAREN error TK_RPAREN
                        {
			  CParserError("Parse error: TK_LPAREN error TK_RPAREN\n");
			}

|   abs_direct_decl_opt TK_LBRACKET comma_expression_opt TK_RBRACKET
                        {
			  UpdateArrayEntity($1,NIL,$3);
			}
/*(* The next shoudl be abs_direct_decl_opt but we get conflicts *)*/
|   abs_direct_decl_opt parameter_list_startscope
                        {
			  entity e = $1; //RenameFunctionEntity($1);
			  if (value_undefined_p(entity_initial(e)))
			    entity_initial(e) = make_value(is_value_code,make_code(NIL,strdup(""),make_sequence(NIL),NIL));
			  //pips_assert("e is a module", module_name_p(entity_module_name($1)));
			  PushFunction(e);
			}
    rest_par_list TK_RPAREN
                        {
			  entity e = GetFunction();
			  PopFunction();
			  stack_pop(FormalStack);
			  StackPop(OffsetStack);
			  (void) UpdateFunctionEntity(e,$4);
			  $$ = e;
			}
;

abs_direct_decl_opt:
    abs_direct_decl
                        { }
|   /* empty */         {
                          string n = i2a(abstract_counter++);
			  $$ = FindOrCreateCurrentEntity(strdup(concatenate(DUMMY_ABSTRACT_PREFIX,
									    n,NULL)),
							 ContextStack,FormalStack,
							 FunctionStack,
							 is_external);
			  free(n);
			  stack s = stack_make(type_domain,0,0);
			  //entity_storage($$) = (storage) s;
			  stack_push((char *) entity_type($$),s);
			  put_to_entity_type_stack_table($$, s);
			  entity_type($$) = type_undefined;
    }
;

function_def:  /* (* ISO 6.9.1 *) */
    function_def_start
                        {
			  InitializeBlock();
			  is_external = FALSE;
			}
    block
                        {
			  /* Make value_code for current module here */
			  ModuleStatement = $3;
			  pips_assert("Module statement is consistent",statement_consistent_p(ModuleStatement));
			  pips_assert("Module declarations are unique", gen_once_p(statement_declarations(ModuleStatement)));
			  /* Let's delay this? ResetCurrentModule(); */
			  is_external = TRUE;
			}

function_def_start:  /* (* ISO 6.9.1 *) */
    decl_spec_list declarator
                        {
			  UpdateEntity($2,ContextStack,FormalStack,FunctionStack,OffsetStack,is_external, FALSE);
			  //stack_pop(ContextStack);
			  PopContext();
			  pips_debug(2,"Create current module %s\n",entity_user_name($2));
			  MakeCurrentModule($2);
			  clear_C_comment();
			  pips_assert("Module is consistent\n",entity_consistent_p($2));
			}
/* (* Old-style function prototype *) */
|   decl_spec_list old_proto_decl
                        {
			  UpdateEntity($2,ContextStack,FormalStack,FunctionStack,OffsetStack,is_external, FALSE);
			  //stack_pop(ContextStack);
			  PopContext();
			  pips_debug(2,"Create current module %s with old-style prototype\n",entity_user_name($2));
			  MakeCurrentModule($2);
			  clear_C_comment();
			  pips_assert("Module is consistent\n",entity_consistent_p($2));
			}
/* (* New-style function that does not have a return type *) */
|   TK_IDENT parameter_list_startscope
                        {
			  entity oe = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			  entity e = oe; //RenameFunctionEntity(oe);
			  pips_debug(2,"Create current module \"%s\" with no return type\n",
				     entity_name(e));
			  MakeCurrentModule(e);
			  clear_C_comment();
			  //pips_assert("e is a module", module_name_p(entity_module_name(e)));
			  PushFunction(e);
			}
    rest_par_list TK_RPAREN
                        {
			  /* Functional type is unknown or int (by default) or void ?*/
			  functional f = make_functional($4,make_type_unknown());
			  entity e = GetFunction();
			  entity_type(e) = make_type_functional(f);
			  pips_assert("Current module entity is consistent\n",entity_consistent_p(e));
			  PopFunction();
			  stack_pop(FormalStack);
			  StackPop(OffsetStack);
			}
/* (* No return type and old-style parameter list *) */
|   TK_IDENT TK_LPAREN old_parameter_list_ne
                        {
			  entity oe = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			  entity e= oe; //RenameFunctionEntity(oe);
			  pips_debug(2,"Create current module %s with no return type + old-style parameter list\n",$1);
			  MakeCurrentModule(e);
			  clear_C_comment();
			  //pips_assert("e is a module", module_name_p(entity_module_name(e)));
			  PushFunction(e);
			  stack_push((char *) make_basic_logical(TRUE),FormalStack);
			  stack_push((char *) make_basic_int(1),OffsetStack);
			}
    TK_RPAREN old_pardef_list
                        {
			  list paras = MakeParameterList($3,$6,FunctionStack);
			  functional f = make_functional(paras,make_type_unknown());
			  entity e = GetFunction();
			  entity_type(e) = make_type_functional(f);
			  pips_assert("Current module entity is consistent\n",entity_consistent_p(e));
			  PopFunction();
			  stack_pop(FormalStack);
			  StackPop(OffsetStack);
			}
/* (* No return type and no parameters *) */
|   TK_IDENT TK_LPAREN TK_RPAREN
                        {
			  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,$1);
			  /* Functional type is unknown or int (by default) or void ?*/
			  functional f = make_functional(NIL,make_type_unknown());
			  entity_type(e) = make_type_functional(f);
			  pips_debug(2,"Create current module %s with no return type and no parameters\n",$1);
			  MakeCurrentModule(e);
			  clear_C_comment();
			  pips_assert("Current module entity is consistent\n",entity_consistent_p(e));
			}
;

/*** GCC attributes ***/
attributes:
    /* empty */
                        { $$ = NIL; }
|   attribute attributes
                        { $$ = CONS(QUALIFIER,$1,$2); }
;

/* (* In some contexts we can have an inline assembly to specify the name to
    * be used for a global. We treat this as a name attribute *) */
attributes_with_asm:
    /* empty */
                        { $$ = NIL; }
|   attribute attributes_with_asm
                        { $$ = CONS(QUALIFIER,$1,$2); }
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
			  $$ = make_qualifier_const();
			}
|   TK_RESTRICT
                        {
			  $$ = make_qualifier_restrict();
			}
|   TK_VOLATILE
                        {
			  $$ = make_qualifier_volatile();
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

