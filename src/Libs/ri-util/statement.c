/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
 /*
    Function for statement, and its subtypes:
     - instruction

    Lei ZHOU         12 Sep. 1991
    Francois IRIGOIN
  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"

#include "text.h"
#include "text-util.h"
#include "ri.h"

#include "ri-util.h"

#include "properties.h"
#include "effects-simple.h"
#include "preprocessor.h"
#include "control.h"


/** @defgroup statement_util Methods dealing with statements
    @{
*/

/******************************************************* EMPTY STATEMENT */
/* detects a statement with no special effect...
 */

/* Define the static buffer size */
#define STATIC_BUFFER_SZ 100


/** @defgroup statement_predicate Statement predicate methods
    @{
*/

static bool cannot_be_empty(bool *statement_is_empty)
{
    gen_recurse_stop(NULL);
    return *statement_is_empty = false;;
}

static bool call_filter(call c,bool *statement_is_empty)
{
    entity e = call_function(c);
    if (ENTITY_CONTINUE_P(e) || ENTITY_RETURN_P(e))
        return false;
    else
        return cannot_be_empty(statement_is_empty);
}

bool 
empty_code_p(statement s)
{
    if ((!s) || statement_undefined_p(s)) 
        return true;

    bool statement_is_empty = true;
    gen_context_recurse(s,&statement_is_empty, call_domain, call_filter, gen_null);

    pips_debug(3, "returning %d\n", statement_is_empty);

    return statement_is_empty;
}

bool empty_code_list_p(list l)
{
    MAP(STATEMENT, s, if (!empty_code_p(s)) return FALSE, l);
    return true;
}


bool
empty_comments_p(string s)
{
  /* Could be replaced by a macro. See macro empty_comments */
  pips_assert("comments cannot be NULL", s!=NULL);
  return (s == NULL || string_undefined_p(s) || strcmp(s,"")==0);
}
bool
comments_equal_p(string c1, string c2)
{
    return 
        ( empty_comments_p(c1) && empty_comments_p(c2)) ||
        (!empty_comments_p(c1) && !empty_comments_p(c2) && same_string_p(c1,c2));
}

/** @defgroup statements_p Predicates on statements

    @{
*/


/* Test if a statement is an assignment. */
bool
assignment_statement_p(statement s) {
  instruction i = statement_instruction(s);

  return instruction_assign_p(i);
}


/* Test if a statement is a C or Fortran "return" */
bool return_statement_p(statement s) {
  instruction i = statement_instruction(s);
  return return_instruction_p(i);
}

/* Test if a statement is a Fortran "return" */
bool fortran_return_statement_p(statement s) {
  instruction i = statement_instruction(s);
  return fortran_return_instruction_p(i);
}

/* Test if a statement is a C "return" */
bool C_return_statement_p(statement s) {
  instruction i = statement_instruction(s);
  return C_return_instruction_p(i);
}


/* Test if a statement is a CONTINUE, that is the FORTRAN nop, the ";" in
   C or the "pass" in Python... according to the language.
*/
bool continue_statement_p(statement s) {
  instruction i = statement_instruction(s);

  return instruction_continue_p(i);
}

bool declaration_statement_p(statement s) {
  instruction i = statement_instruction(s);

  return instruction_continue_p(i) && !ENDP(statement_declarations(s));
}

/* Check that all statements contained in statement list sl are a
   continue statements. */
bool continue_statements_p(list sl)
{
  bool continue_p = TRUE;

  FOREACH(STATEMENT, s, sl) {
    if(!continue_statement_p(s)) {
      continue_p = FALSE;
      break;
    }
  }

  return continue_p;
}


/* Test if a statement is a Fortran STOP.
*/
bool stop_statement_p(statement s)
{
  instruction i = statement_instruction(s);

  return instruction_stop_p(i);
}


/* Test if a statement is a Fortran FORMAT.
*/
bool format_statement_p(statement s)
{
  instruction i = statement_instruction(s);

  return instruction_format_p(i);
}


bool write_statement_p(statement s)
{
  instruction i = statement_instruction(s);

  return (native_instruction_p(i, WRITE_FUNCTION_NAME));
}

bool statement_less_p(statement st1, statement st2)
{
  int o1 = statement_ordering( st1 ) ;
  int o2 = statement_ordering( st2 ) ;

  if (ORDERING_NUMBER( o1 ) != ORDERING_NUMBER( o2 )) {
    fprintf(stderr, "cannot compare %td (%d,%d) and %td (%d,%d)\n",
	    statement_number(st1),
	    ORDERING_NUMBER( o1 ), ORDERING_STATEMENT( o1 ),
	    statement_number(st2),
	    ORDERING_NUMBER( o2 ), ORDERING_STATEMENT( o2 ));

    abort();
  }

  return( ORDERING_STATEMENT(o1) < ORDERING_STATEMENT(o2)) ;
}

bool statement_possible_less_p(statement st1, statement st2)
{
  int o1 = statement_ordering( st1 ) ;
  int o2 = statement_ordering( st2 ) ;

  if (ORDERING_NUMBER( o1 ) != ORDERING_NUMBER( o2 )) {
    return(TRUE);
  }
  else
    return( ORDERING_STATEMENT(o1) < ORDERING_STATEMENT(o2));
}

/* Statement classes induced from instruction type
 *
 * Naming conflict between English, block_statement_p(), and NewGen convention
 * isntruction_loop_p()
 */

/* See also macro statement_block_p() */
bool block_statement_p(statement s)
{
  instruction i = statement_instruction(s);
  bool r = instruction_sequence_p(i);

  return r;
}

bool statement_test_p(statement s)
{
  return(instruction_test_p(statement_instruction(s)));
}


bool statement_loop_p(statement s)
{
  return(instruction_loop_p(statement_instruction(s)));
}

bool statement_whileloop_p(statement s)
{
  return(instruction_whileloop_p(statement_instruction(s)));
}

bool statement_forloop_p(statement s)
{
  return(instruction_forloop_p(statement_instruction(s)));
}

bool unstructured_statement_p(statement s)
{
  return(instruction_unstructured_p(statement_instruction(s)));
}

/* Test if a statement is empty. */
bool empty_statement_p(statement st)
{
  instruction i;

  return(entity_empty_label_p(statement_label(st)) &&
	 instruction_block_p(i=statement_instruction(st)) &&
	 ENDP(instruction_block(i)) &&
	 ENDP(statement_declarations(st)));
}


bool unlabelled_statement_p(statement st)
{
  return(entity_empty_label_p(statement_label(st)));
}

bool nop_statement_p(statement s)
{
  /* NOP statements are useful to fill in empty test branches.
   * The definition of NOP in PIPS has changed over the years.
   * A NOP statement is an empty block. Like blocks, it cannot be
   * labelled nor commented. It has no statement number because it
   * is invisible to users.
   *
   * Dangling comments, like labels, are attached to CONTINUE statements.
   *
   * Note 1: blocks are now called "sequences"
   * Note 2: see also empty_statement_p()
   */
  bool nop = FALSE;
  instruction i = statement_instruction(s);

  if(instruction_block_p(i) && ENDP(instruction_block(i))) {
    pips_assert("No label!", entity_empty_label_p(statement_label(s)));
    pips_assert("No comments", empty_comments_p(statement_comments(s)));
    pips_assert("No statement number",
		statement_number(s) == STATEMENT_NUMBER_UNDEFINED);
    nop = TRUE;
  }

  return nop;
}

/* Return true if the statement is an empty instruction block without
   label or a continue without label or a recursive combination of
   above.

   FI: I add a check on declarations. With their initializations, they
   have side effects. See C_syntax/block01.c.

   FI: same issue with CONTINUE.

   If there is an extension on it (such as a pragma) return false since
   this statement may be useful/
 */
bool empty_statement_or_labelless_continue_p(statement st)
{
  instruction i;

  if (!entity_empty_label_p(statement_label(st))
      || !empty_extensions_p(statement_extensions(st)))
    return FALSE;

  if (continue_statement_p(st))
    return ENDP(statement_declarations(st));

  i = statement_instruction(st);
  if (instruction_block_p(i) && ENDP(statement_declarations(st))) {
    MAP(STATEMENT, s,
	{
	  if (!empty_statement_or_labelless_continue_p(s))
	    /* Well there is at least one possibly usefull thing... */
	    return FALSE;
	},
	instruction_block(i));
    return TRUE;
  }
  return FALSE;
}


/* Return true if the statement is an empty instruction block or a
   continue or a recursive combination of above. */
bool empty_statement_or_continue_p(statement st)
{
  instruction i;

  if (continue_statement_p(st))
    return TRUE;
  i = statement_instruction(st);
  if (instruction_block_p(i)) {
    MAP(STATEMENT, s,
	{
	  if (!empty_statement_or_continue_p(s))
	    /* Well there is at least one possibly usefull thing... */
	    return FALSE;
	},
	instruction_block(i));
    return TRUE;
  }
  return FALSE;
}


/* Return true if the statement is an empty instruction block or a
   continue without comments or without LABEL or without declarations
   or a recursive combination of above. */
bool empty_statement_or_continue_without_comment_p(statement st)
{
  instruction i;
  string the_comments = statement_comments(st);

  /* The very last condition should be sufficient */
  if (!empty_comments_p(the_comments))
    return FALSE;

  if (!entity_empty_label_p(statement_label(st)))
    return FALSE;
  if (continue_statement_p(st) && ENDP(statement_declarations(st)))
    return TRUE;

  i = statement_instruction(st);
  if (instruction_block_p(i)) {
    MAP(STATEMENT, s,
	{
	  if (!empty_statement_or_continue_without_comment_p(s))
	    return FALSE;
	},
	instruction_block(i));
    /* Everything in the block are commentless continue or empty
       statements: */
    return TRUE;
  }
  /* Everything else useful: */
  return FALSE;
}


bool statement_call_p(statement s)
{
  return(instruction_call_p(statement_instruction(s)));
}


bool check_io_statement_p(statement s)
{
  bool check_io = FALSE;
  instruction i = statement_instruction(s);

  if(instruction_test_p(i)) {
    syntax c = expression_syntax(test_condition(instruction_test(i)));

    if(syntax_reference_p(c)) {
      check_io = io_entity_p(reference_variable(syntax_reference(c)));
    }
  }

  return check_io;
}

/** @} */

/* functions to generate statements */

/** Build a statement from a give instruction

    There is also a macro instruction_to_statement() that does the same job.
*/
statement
make_stmt_of_instr(instruction instr) {
  return(make_statement(entity_empty_label(),
			STATEMENT_NUMBER_UNDEFINED,
			STATEMENT_ORDERING_UNDEFINED,
			empty_comments,
			instr,NIL,NULL,
			empty_extensions ()));
}


instruction
make_assign_instruction(expression l,
                        expression r)
{
   call c = call_undefined;
   instruction i = instruction_undefined;

   pips_assert("make_assign_statement",
               syntax_reference_p(expression_syntax(l)));
   c = make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME),
                 CONS(EXPRESSION, l, CONS(EXPRESSION, r, NIL)));
   i = make_instruction(is_instruction_call, c);

   return i;
}


statement make_assign_statement(expression l,
				expression r)
{
    return make_stmt_of_instr(make_assign_instruction(l, r));
}


/** @defgroup block_statement_constructors Block/sequence statement constructors
    @{
 */

/** Build an instruction block from a list of statements
 */
instruction make_instruction_block(list statements) {
  return make_instruction_sequence(make_sequence(statements));
}


/** Make a block statement from a list of statement
 */
statement make_block_statement(list body) {
  statement b;

  b = make_stmt_of_instr(make_instruction_block(body));
  return b;
}


/** Build an empty statement (block/sequence)*/
statement make_empty_block_statement() {
  statement b = make_block_statement(NIL);
  return b;
}


/** Build an empty statement with declaration list, declaration text
    and comment. Shouldn't we also update the extensions field? */
statement make_empty_statement_with_declarations_and_comments(list d,
							      string dt,
							      string c) {
  statement s = make_block_statement(NIL);
  statement_declarations(s)=d;
  statement_decls_text(s) = dt;
  statement_comments(s) = c;
  return s;
}


/** Build a statement sequence from a statement list

    @return :
    - statement_undefined if the statement list is NIL
    - the statement of a list with one statement, after discarding the list
    - a statement block with the statement list else.
*/
statement
make_statement_from_statement_list(list l) {
  switch (gen_length(l)) {
  case 0:
    return statement_undefined;
  case 1:
    {
      statement stat = STATEMENT(CAR(l));
      gen_free_list(l);
      return stat;
    }
  default:
    return make_block_statement(l);
  }
}


/** Build a statement sequence from a statement NULL-terminated varargs list

    It is called with
    make_statement_from_statement_varargs_list(s1, s2,..., sn, NULL)

    @return a statement block with the statement list
*/
statement
make_statement_from_statement_varargs_list(statement s, ...) {
  va_list args;
  /* The statement list */
  list sl = NIL;

  /* Analyze in args the variadic arguments that may be after s: */
  va_start(args, s);
  /* Since a variadic function in C must have at least 1 non variadic
     argument (here the s), just skew the varargs analysis: */
  for(;;) {
    if (s == NULL)
      /* No more statement */
      break;
    /* Use an O(n2) algorithm that would be enough to concatenate few
       statements */
    sl = gen_nconc(sl, CONS(STATEMENT, s, NIL));
    /* Get the next argument: */
    s = va_arg(args, statement);
  }
  /* Release the variadic analyzis: */
  va_end(args);

  /* Build a statement block from a normal list: */
  return make_block_statement(sl);
}


/** Build a statement block from a statement if not already a statement
    block.

    @return the new block statement or the old statement
 */
statement
make_block_with_stmt_if_not_already(statement stmt) {
  if (block_statement_p(stmt))
    /* It is already a block statement */
    return stmt;
  else
    return make_block_statement(CONS(STATEMENT, stmt, NIL));
}

/** @} */


statement make_return_statement(entity module)
{
    char *module_name = entity_local_name(module);
    string name = concatenate( module_name, MODULE_SEP_STRING, LABEL_PREFIX,
		       RETURN_LABEL_NAME,NULL);
    entity l = gen_find_tabulated(name, entity_domain);
    if (entity_undefined_p(l)) l = make_label(strdup(name));
    return make_call_statement(RETURN_FUNCTION_NAME, NIL, l, empty_comments);
}

/*****************************************************************************

  Make a Fortran io statement : PRINT *,"message" following this format :

  (keyword,lci,lio)

  keyword  (READ, WRITE, PRINT...).

  lci is a list of 'control specifications'. its has the following format:
        ("UNIT=", 6, "FMT=", "*", "RECL=", 80, "ERR=", 20)
         default : * = LIST_DIRECTED_FORMAT_NAME

  lio is the list of expressions to write or references to read.
     ("IOLIST=", exp, ...)

*****************************************************************************/
statement make_print_statement(string message)
{
  expression fmt = MakeCharacterConstantExpression(LIST_DIRECTED_FORMAT_NAME);
  expression e1 = MakeCharacterConstantExpression("IOLIST=");
  expression e2 = MakeCharacterConstantExpression(message);
  list args = CONS(EXPRESSION,e1,CONS(EXPRESSION,e2,NIL));
  /*TK_PRINT = 301*/
  /* This function has been moved from alias_check.c and uses a
     function still in the syntax library, hence the 301 argument... */
  extern instruction MakeSimpleIoInst2(int /*keyword*/, expression /*f*/, list /*io_list*/);
  instruction ins = MakeSimpleIoInst2(301,fmt,args);
  return instruction_to_statement(ins);
}

statement make_C_print_statement(string message)
{
  statement s = statement_undefined;
  list args = NIL;
  /* MakeConstant() generates a Fortran constant... Does not seem
     to matter for strings... */
  expression eb =
    make_call_expression(MakeConstant(message,is_basic_string),NIL);
  entity lun = global_name_to_entity(TOP_LEVEL_MODULE_NAME,
				     STDERR_NAME);
  expression ea = expression_undefined;

  if(entity_undefined_p(lun)) {
    /* stderr has not yet been encountered by the parser... Should
       it be defined by bootstrap.c or by the C parser no matter
       what? */
    lun = make_stderr_variable();
  }
  ea = entity_to_expression(lun);

  args = CONS(EXPRESSION,ea, CONS(EXPRESSION,eb,NIL));
  s = make_call_statement(FPRINTF_FUNCTION_NAME,
			   args,
			   entity_undefined,
			   empty_comments);
  return s;
}

/* Generate a print of a constant character string on stderr for C or
   on stdout for Fortran.

   This is not clever as the format of the message is language
   dependent: simple quotes are used as delimiters in Fortran and
   double quotes in C. Should message be language independent and the
   delimiters added in this function instead? I did not do it to avoid
   yet another strdup() when strdup is used to generate "message", but
   this is questionable.

   This is not clever as this function could easily be generalized
   with a vararg to generate more general print statements.
*/
statement make_any_print_statement(string message)
{
  entity m = get_current_module_entity();
  statement s = statement_undefined;

  if(fortran_module_p(m))
    s = make_print_statement(message);
  else
    s = make_C_print_statement(message);

  return s;
}

/* This function returns a Fortran stop statement with an error message
 */
statement make_stop_statement(string message)
{
  list args=NIL;
  expression e;

  e = make_call_expression(MakeConstant(message,is_basic_string),NIL);

  args = CONS(EXPRESSION,e,NIL);

  return make_call_statement(STOP_FUNCTION_NAME, args, entity_undefined, empty_comments);

}

/* This function returns a statement ending with a C exit statement. A
   "fprintf(stderr, errmess);" is generated before "exit(n);" if
   errmess is not empty and a sequence statement ending wih exit() is
   returned.
 */
statement make_exit_statement(int n, string errmess)
{
  statement s = statement_undefined;
  statement s1 = statement_undefined;
  expression e1 = int_to_expression(n);
  list args1 = CONS(EXPRESSION,e1,NIL);

  s1 = make_call_statement(EXIT_FUNCTION_NAME,
			  args1,
			  entity_undefined,
			  empty_comments);

  if(strlen(errmess)>0) {
    statement s2 = make_C_print_statement(errmess);
    /* There must be a nicer vararg
       function... make_statement_from_statement_varargs_list(s2, s1, NULL) */
    s = make_block_statement(CONS(STATEMENT,s2, CONS(STATEMENT,s1,NIL)));
  }
  else
    s = s1;

  return s;

}


/* adds a RETURN statement to *ps if necessary
 */
void
insure_return_as_last_statement(
    entity module,
    statement *ps)
{
    statement last = find_last_statement(*ps);
    if (statement_undefined_p(last) || !return_statement_p(last))
    {
	statement ret = make_return_statement(module);
	if (statement_block_p(*ps))
	{
	    sequence_statements(instruction_sequence(
		statement_instruction(*ps))) =
		gen_nconc(sequence_statements(instruction_sequence(
		    statement_instruction(*ps))),
			  CONS(STATEMENT, ret, NIL));
	}
	else
	{
	    *ps = make_block_statement(CONS(STATEMENT, *ps,
				       CONS(STATEMENT, ret, NIL)));
	}
    }
}


statement make_continue_statement(entity l)
{
    return make_call_statement(CONTINUE_FUNCTION_NAME, NIL, l,
			       empty_comments);
}


/* Make a simple continue statement to be used as a NOP or ";" in C

   @return the statement
*/
statement make_plain_continue_statement()
{
    return make_continue_statement(entity_empty_label());
}


/* To preserve declaration lines and comments, declaration statements
   are used. */
statement make_declaration_statement(entity v, int sn, string cs)
{
  statement ds = make_call_statement(CONTINUE_FUNCTION_NAME,
				     NIL,
				     entity_empty_label(),
				     cs);
  statement_declarations(ds) = CONS(ENTITY, v, NIL);
  statement_number(ds) = sn;

  return ds;
}

/* Build a while loop statement.

   If \a before is true, build a
   while (\a condition)
     \a s;

   else build a
   do
     \a s;
   while (\a condition);

  @param line_number is used to specify a source-line number to the statement
*/
statement make_whileloop_statement(expression condition,
				   statement s,
				   int line_number,
				   bool before) {
  whileloop w;
  statement smt;

  w  = make_whileloop(condition, s, entity_empty_label(),
		      before ? make_evaluation_before() : make_evaluation_after());
  smt = make_statement(entity_empty_label(),
		       line_number,
		       STATEMENT_ORDERING_UNDEFINED,
		       //pop_current_C_comment(),
		       string_undefined,
		       make_instruction_whileloop(w),
		       NIL, string_undefined,
		       empty_extensions ());
  return smt;
}

/* This function is limited to intrinsics calls... A full function
   name or an entity could be passed as first argument. */
statement make_call_statement(function_name, args, l, c)
string function_name;
list args;
entity l; /* label, default entity_undefined */
string c; /* comments, default empty_comments (was: "" (was: string_undefined)) */
{
  entity called_function;
  statement cs;

  called_function = entity_intrinsic(function_name);

  l = (l==entity_undefined)? entity_empty_label() : l;
  cs = make_statement(l,
		      STATEMENT_NUMBER_UNDEFINED,
		      STATEMENT_ORDERING_UNDEFINED,
		      c,
		      make_instruction(is_instruction_call,
				       make_call(called_function,args)),
		      NIL,
		      NULL,
		      empty_extensions ());

  ifdebug(8) {
    pips_debug(8, "cs is call to %s\n", function_name);
    safe_print_statement(cs);
  }

  return cs;
}


/** Build a statement from a given expression
 */
statement make_expression_statement(expression e)
{
  instruction i = make_instruction_expression(e);
  statement s = instruction_to_statement(i);

  return s;
}


/* Direct accesses to second level fields */

loop
statement_loop(s)
statement s;
{
    pips_assert("statement_loop", statement_loop_p(s));

    return(instruction_loop(statement_instruction(s)));
}

test
statement_test(s)
statement s;
{
    pips_assert("statement_test", statement_test_p(s));

    return(instruction_test(statement_instruction(s)));
}

call
statement_call(s)
statement s;
{
    pips_assert("statement_call", statement_call_p(s));

    return(instruction_call(statement_instruction(s)));
}

list
statement_block(s)
statement s;
{
    pips_assert("statement_block", statement_block_p(s));

    return(instruction_block(statement_instruction(s)));
}



bool assignment_block_or_statement_p(statement s)
{
    instruction i = statement_instruction(s);

    switch(instruction_tag(i)) {
    case is_instruction_block:
	return assignment_block_p(statement_instruction(s));
	break;
    case is_instruction_test:
	break;
    case is_instruction_loop:
	break;
    case is_instruction_whileloop:
	break;
    case is_instruction_goto:
	pips_error("assignment_block_p", "unexpected GO TO\n");
    case is_instruction_call:
	return assignment_statement_p(s);
    case is_instruction_unstructured:
	break;
    default: pips_error("assignment_block_or_statement_p",
			"ill. instruction tag %d\n", instruction_tag(i));
    }
    return FALSE;
}


void
print_statement_set(fd, r)
FILE *fd;
set r;
{
    fprintf(fd, "Set contains statements");

    SET_MAP(s, {
	fprintf(fd, " %02td", statement_number((statement) s));
    }, r);

    fprintf(fd, "\n");
}


/** Print a statement on stdout

    Print the statement according to the current PRETTYPRINT_C_CODE
    property

    See text_named_module() for improvements.
*/
void print_statement(statement s)
{
  debug_on("TEXT_DEBUG_LEVEL");
  set_alternate_return_set();
  reset_label_counter();
  push_current_module_statement(s);
  bool previous_is_fortran_p = get_prettyprint_is_fortran();
  /* Prettyprint in the correct language: */
  set_prettyprint_is_fortran_p(!get_bool_property("PRETTYPRINT_C_CODE"));
  text txt = text_statement(entity_undefined, 0, s, NIL);
  print_text(stderr, txt);
  free_text(txt);
  /* Put back the previous prettyprint language: */
  set_prettyprint_is_fortran_p(previous_is_fortran_p);
  pop_current_module_statement();
  reset_alternate_return_set();
  debug_off();
}


void print_statements(list sl)
{
  FOREACH(STATEMENT, s, sl) {
    print_statement(s);
  }
}


void print_statement_of_module(statement s, string mn)
{
  if(entity_undefined_p(get_current_module_entity())) {
    entity m = local_name_to_top_level_entity(mn);
    set_current_module_entity(m);
    reset_label_counter();
    print_statement(s);
    reset_current_module_entity();
  }
  else
    print_statement(s);
}

text statement_to_text(statement s)
{
  text t = text_undefined;

  debug_on("PRETTYPRINT_DEBUG_LEVEL");
  set_alternate_return_set();
  reset_label_counter();
  t = text_statement(entity_undefined, 0, s, NIL);
  reset_alternate_return_set();
  debug_off();

  return t;
}

void safe_print_statement(statement s)
{
  if(statement_undefined_p(s)) {
    fprintf(stderr, "Statement undefined: %s\n",
	    statement_identification(s));
  }
  else if(continue_statement_p(s)
     && entity_return_label_p(statement_label(s))) {
    /* The return label only can be associated to a RETURN call,
       however the controlizer does not follow this consistency
       rule. */
    fprintf(stderr, "%s\n", statement_identification(s));
  }
  else
    print_statement(s);
}

void print_parallel_statement(statement s)
{
  string cstyle = strdup(get_string_property(PRETTYPRINT_PARALLEL));
  set_string_property(PRETTYPRINT_PARALLEL, "doall");
  print_statement(s);
  set_string_property(PRETTYPRINT_PARALLEL, cstyle);
  free(cstyle);
}


/* Mapping from statement number to statement
 *
 * Warning: STATEMENT_NUMBER_UNDEFINED may be equal to HASH_ENTRY_FREE_FOR_PUT
 * because integer keys are not well supported by the newgen hash package.
 *
 * Two statements in the same piece of code may have the same number because:
 *  - they were generated by the parser to simulate a unique user statement
 *    (e.g. a computed GO TO)
 *  - they were generated by a program transformation from the same user
 *    statement (e.g. DO statements by a loop distribution)
 *
 * A statement may be not numbered because:
 *  - the parser assumed that it did not matter (e.g. a GO TO transformed into
 *    an arc by the controlizer); but the parser is not really consistent...
 *  - a program generation phase has no way to generate a realistic statement
 *    number from a user statement number (e.g. code for banks in WP65)
 */

/* static variable temporarily required to use gen_recurse() */
static hash_table number_to_statement = hash_table_undefined;
/* To keep track of duplicate numbers */
static set duplicate_numbers = set_undefined;

static void update_number_to_statement(statement s)
{
    if(statement_number(s)!=STATEMENT_NUMBER_UNDEFINED) {
	if(hash_get(number_to_statement, (char *) statement_number(s))
	   != HASH_UNDEFINED_VALUE) {
	    duplicate_numbers = set_add_element(duplicate_numbers,
						duplicate_numbers,
						(char *) statement_number(s));
	}
	else {
	    hash_put(number_to_statement, (char *) statement_number(s), (char *) s);
	}
    }
}

statement
apply_number_to_statement(hash_table nts, _int n)
{
    /* This function used to be inline in prettyprinting functions for user views.
     * It was assumed that all statements produced by the parser had a defined
     * statement number. In order to keep a nice statement numbering scheme,
     * GO TO statements are not (always) numbered. So n hasa to be tested.
     */

    statement s = statement_undefined;

    if(n!=STATEMENT_NUMBER_UNDEFINED) {
	s = (statement) hash_get(nts, (char *) n);
	if (s == (statement) HASH_UNDEFINED_VALUE) {
	    s = statement_undefined;
	}
    }

    return s;
}

hash_table 
build_number_to_statement(nts, s)
hash_table nts;
statement s;
{
    pips_assert("build_number_to_statement", nts!=hash_table_undefined
		&& !statement_undefined_p(s) && set_undefined_p(duplicate_numbers));

    number_to_statement = nts;
    duplicate_numbers = set_make(set_int);

    gen_recurse(s, statement_domain, gen_true, update_number_to_statement);

    /* Eliminate duplicate and hence meaningless numbers */
    SET_MAP(n, {
	hash_del(number_to_statement, n);
    }, duplicate_numbers);

    /* nts is updated by side effect on number_to_statement */
    number_to_statement = hash_table_undefined;
    set_free(duplicate_numbers);
    duplicate_numbers = set_undefined;
    return nts;
}

void print_number_to_statement(nts)
hash_table nts;
{
    HASH_MAP(number, stmt, {
	fprintf(stderr,"%td\t", (_int) number);
	print_statement((statement) stmt);
    }, nts);
}

hash_table allocate_number_to_statement()
{
    hash_table nts = hash_table_undefined;

    /* let's assume that 50 statements is a good approximation of a
     * module size.
     */
    nts = hash_table_make(hash_int, 50);

    return nts;
}


/** Get rid of all labels in controlized code before duplication

    All labels have become useless and they cannot be freely duplicated.

    One caveat: FORMAT statements!
*/
statement
clear_labels(statement s) {
  gen_recurse(s, statement_domain, gen_true, clear_label);
  clean_up_sequences(s);
  return s;
}


void
clear_label(s)
statement s;
{
    if(format_statement_p(s)) {
	user_error("clear_label", "Cannot clear label for FORMAT %s!\n",
		   label_local_name(statement_label(s)));
    }

    statement_label(s) = entity_empty_label();

    if(instruction_loop_p(statement_instruction(s))) 
	loop_label(instruction_loop(statement_instruction(s))) = 
	    entity_empty_label();
}


statement
st_make_nice_test(condition, ltrue, lfalse)
expression condition;
list ltrue,lfalse;
{
    statement
	stattrue = make_statement_from_statement_list(ltrue),
	statfalse = make_statement_from_statement_list(lfalse);
    bool
	notrue=(stattrue==statement_undefined),
	nofalse=(statfalse==statement_undefined);
    
    if ((notrue) && (nofalse)) 
	return(make_continue_statement(entity_undefined)); 

    if (nofalse) 
    {
	return
	    (make_stmt_of_instr
	     (make_instruction
	      (is_instruction_test,
	       make_test(condition,
			 stattrue,
			 make_continue_statement(entity_undefined)))));
    }

    if (notrue) 
    {
	expression
	    newcond = MakeUnaryCall(entity_intrinsic(NOT_OPERATOR_NAME),
				    condition);

	return
	    (make_stmt_of_instr
	     (make_instruction
	      (is_instruction_test,
	       make_test(newcond,
			 statfalse,
			 make_continue_statement(entity_undefined)))));
    }

    return(make_stmt_of_instr(make_instruction(is_instruction_test,
					       make_test(condition,
							 stattrue,
							 statfalse))));
}


/* statement makeloopbody(l, s_old) 
 * make a statement for a loop body, using the fields of a previously existing statement
 *
 * Preserving the labels may be sometimes a good thing (hyperplane or
 * tiling transformation, outermostloop) or a bad thing for innermost
 * loops, sometimes replicated loops
 *
 * FI: the name of this function is not well chosen.
 */
statement makeloopbody(loop l, statement s_old, bool inner_p)
{
    statement state_l;
    instruction instr_l;
    statement l_body;

    instr_l = make_instruction(is_instruction_loop,l);
    state_l = make_statement(inner_p? entity_empty_label() : statement_label(s_old),
			     statement_number(s_old),
			     statement_ordering(s_old),
			     statement_comments(s_old),
			     instr_l,NIL,NULL,
			     statement_extensions(s_old));
    l_body = make_statement(entity_empty_label(),
			    STATEMENT_NUMBER_UNDEFINED,
			    STATEMENT_ORDERING_UNDEFINED,
			    empty_comments,
			    make_instruction_block(CONS(STATEMENT,state_l,NIL)),
			    NIL,NULL,
			    empty_extensions ());

    return(l_body);
}


/* Does work neither with undefined statements nor with defined
   statements with undefined instructions. Returns a statement
   identity, its number and the breakdown of its ordering, as well as
   information about the instruction. */

string external_statement_identification(statement s)
{
    static char buffer[STATIC_BUFFER_SZ];
    instruction i = statement_instruction(s);
    string instrstring = instruction_identification(i);
    int so = statement_ordering(s);
    entity called = entity_undefined;
    int nb_char = 0;

    if(same_string_p(instrstring, "CALL")) {
	called = call_function(instruction_call(i));
    }

    nb_char = snprintf(buffer, STATIC_BUFFER_SZ, "%td (%d, %d): %s %s\n",
	    statement_number(s),
	    ORDERING_NUMBER(so),
	    ORDERING_STATEMENT(so),
	    instrstring,
	    entity_undefined_p(called)? "" : module_local_name(called));

    pips_assert ("checking static buffer overflow", nb_char < STATIC_BUFFER_SZ);

    return buffer;
}

/* Like external_statement_identification(), but with internal
   information, the hexadecimal address of the statement */
string statement_identification(statement s)
{
    static char buffer[STATIC_BUFFER_SZ];
    instruction i = statement_instruction(s);
    string instrstring = instruction_identification(i);
    int so = statement_ordering(s);
    entity called = entity_undefined;
    int nb_char = 0;

    if(same_string_p(instrstring, "CALL")) {
	called = call_function(instruction_call(i));
    }

    nb_char = snprintf(buffer, STATIC_BUFFER_SZ, "%td (%d, %d) at %p: %s %s\n",
		 statement_number(s),
		 ORDERING_NUMBER(so),
		 ORDERING_STATEMENT(so),
		 s,
		 instrstring,
		 entity_undefined_p(called)? "" : module_local_name(called));

    pips_assert ("static buffer overflow, increase the buffer size", nb_char < STATIC_BUFFER_SZ);

    return buffer;
}

string 
safe_statement_identification(statement s)
{
  if(statement_undefined_p(s)) {
    return "undefined statement\n";
  }
  else if(instruction_undefined_p(statement_instruction(s))) {
    return "statement with undefined intruction\n";
  }
  else
    return statement_identification(s);
}



/* Return true if the statement has an empty statement: */
bool
statement_with_empty_comment_p(statement s)
{
    string the_comments = statement_comments(s);
    return empty_comments_p(the_comments);
}


static bool 
gather_all_comments_of_a_statement_filter(statement s, string * all_comments)
{
    string the_comments = statement_comments(s);
    if (!empty_comments_p(the_comments)) {
        string old = *all_comments;
        *all_comments = strdup(
                old==NULL?
                the_comments:
                concatenate(old, the_comments, NULL));
        free(old);
    }
    return true;
}


/* Gather all the comments recursively found in the given statement
   and return them in a strduped string (NULL if no comment found).

   Do not forget to free the string returned later when no longer
   used. */
string gather_all_comments_of_a_statement(statement s)
{
    string comments = NULL;
    gen_context_recurse(s, &comments, statement_domain,
		gather_all_comments_of_a_statement_filter, gen_null);
    return comments?comments:empty_comments;
}


/* Find the first non-empty comment of a statement, if any: */
string find_first_statement_comment(statement s)
{
    instruction i = statement_instruction(s);
    if (instruction_sequence_p(i)) {
	MAP(STATEMENT, st, {
	    string comment = find_first_statement_comment(st);
	    if (!empty_comments_p(comment))
		/* We've found it! */
		return comment;
	}, sequence_statements(instruction_sequence(statement_instruction(s))));
	/* No comment found: */
	return empty_comments;
    }
    else
	/* Ok, plain statement: */
	return statement_comments(s);
}

/* Find the first comment of a statement, if any.
 *
 * Unfortunately empty_comments may be used to decorate a statement
 * although this makes the statement !gen_defined_p(). empty_comments
 * is also used as a return value to signal that non statement
 * legally carrying a comment has been found.
 *
 * The whole comment business should be cleaned up.
 */
string find_first_comment(statement s)
{
    instruction i = statement_instruction(s);
    if (instruction_sequence_p(i)) {
	MAP(STATEMENT, st, {
	    string comment = find_first_statement_comment(st);
	    /* let's hope the parser generates an empty string as
	       comment rather than empty_comments which is defined as
	       empty_string */
	    if (comment!=empty_comments)
		/* We've found it! */
		return comment;
	}, sequence_statements(instruction_sequence(statement_instruction(s))));
	/* No comment found: */
	return empty_comments;
    }
    else
	/* comment carrying statement: */
	return statement_comments(s);
}


/* Put a comment on a statement in a safe way. That is it find the
   first non-block statement to attach it or insert a CONTINUE and put
   the statement on it. You should free the old one...

   The comment should have been malloc()'ed before.

   Return TRUE on success, FALSE else. */
bool
try_to_put_a_comment_on_a_statement(statement s,
				    string the_comments)
{
    instruction i = statement_instruction(s);
    if (instruction_sequence_p(i)) {
	/* We are not allowed to put a comment on a sequence, find the
           first non sequence statement if any: */
	MAP(STATEMENT, st,
	    {
		if (try_to_put_a_comment_on_a_statement(st, the_comments))
		    /* Ok, we succeed to put the comment: */
		    return TRUE;
	    },
	    sequence_statements(instruction_sequence(i)));
	/* Hmm, no good statement found to attach a comment: */
	return FALSE;
    }
    else {
	/* Ok, it is a plain statement, we can put a comment on it: */
	statement_comments(s) = the_comments;
	return TRUE;
    }
}


/* Similar to try_to_put_a_comment_on_a_statement() but insert a
   CONTINUE to put the comment on it if there is only empty
   sequence(s)

   The comment should have been malloc()'ed before.
*/
void
put_a_comment_on_a_statement(statement s,
			     string the_comments)
{
    if (empty_comments_p(the_comments))
	/* Nothing to add... */
	return;

    if (!try_to_put_a_comment_on_a_statement(s, the_comments)) {
	/* It failed because it is an empty sequence. So add a
           CONTINUE to attach the statement on it: */
	sequence seq = instruction_sequence(statement_instruction(s));
	statement s_cont = make_continue_statement(entity_empty_label());
	statement_comments(s_cont) = the_comments;
	sequence_statements(seq) = CONS(STATEMENT,
					s_cont,
					sequence_statements(seq));
    }
}


/* Append a comment string (if non empty) to the comments of a
   statement, if the c.

   @param the_comments is strdup'ed in this function.
*/
void
append_comments_to_statement(statement s,
			     string the_comments)
{
    string old;

    if (empty_comments_p(the_comments))
	/* Nothing to add... */
	return;

    old = find_first_statement_comment(s);
    if (empty_comments_p(old))
	/* There is no comment yet: */
	put_a_comment_on_a_statement(s, strdup(the_comments));
    else {
	put_a_comment_on_a_statement(s, strdup(concatenate(old, the_comments, NULL)));
	free(old);
    }
}


/* Insert a comment string (if non empty) at the beginning of the
   comments of a statement.

   @param the_comments is strdup'ed in this function.
*/
void
insert_comments_to_statement(statement s,
			     string the_comments)
{
    string old;

    if (empty_comments_p(the_comments))
	/* Nothing to add... */
	return;

    old  = find_first_comment(s);
    if (empty_comments_p(old))
	/* There are no comments yet: */
	put_a_comment_on_a_statement(s, strdup(the_comments));
    else {
	put_a_comment_on_a_statement(s, strdup(concatenate(the_comments, old, NULL)));
	/* Courageous: you have to be sure that the comment returned
	   by find_first_comment() belongs to the statement which is
	   going to be used by put_a_comment_on_a_statement() knowing
	   that both can be different from s if s is a sequence. */
	free(old);
    }
}


/* as name indicate, a comment is added.
 */
#define ERROR_PREFIX "!ERROR: "
#define BUFSIZE 1024

void add_one_line_of_comment(statement s, string format, ...)
{
  char buffer[BUFSIZE];
  int error;
  va_list some_arguments;

  va_start(some_arguments, format);
  error = vsnprintf(buffer, BUFSIZE, format, some_arguments);
  if (error<0) {
      pips_internal_error("buffer too small");
  }
  va_end(some_arguments);

  if (s)
  {
      if (string_undefined_p(statement_comments(s)) || statement_comments(s)==NULL)
      {
	  statement_comments(s) = strdup(concatenate(ERROR_PREFIX, buffer, "\n", NULL));
      }
      else
      {
	  string newcom =
	      strdup(concatenate(statement_comments(s), ERROR_PREFIX, buffer, "\n", NULL));
	  free(statement_comments(s));
	  statement_comments(s) = newcom;
      }
  }
  else
  {
      fprintf(stderr, ERROR_PREFIX "%s\n", buffer);
  }
}


/* Since block cannot carry comments nor line numbers, they must be
   moved to an internal continue statement.

   A prettier version could be to make a new block containing the
   continue and then the old block. Comments might be better located.
 */
statement add_comment_and_line_number(statement s, string sc, int sn)
{
  string osc = statement_comments(s);
  statement ns = s;

  if(osc!=NULL && !string_undefined_p(osc)) {
    free(osc);
  }

  if(!statement_block_p(s)) {
    statement_comments(s) = sc;
    statement_number(s) = sn;
  }
  else if(!string_undefined_p(sc)) {
    /* A continue statement must be inserted as first block statement*/
    statement nops = make_continue_statement(entity_undefined);
    list sss = sequence_statements(instruction_sequence(statement_instruction(s)));
    statement_comments(nops) = sc;
    statement_number(nops) = sn;
    sequence_statements(instruction_sequence(statement_instruction(s))) =
      CONS(STATEMENT, nops, sss);
  }
  return ns;
}


/* Since blocks are not represented in Fortran, they cannot
 * carry a label. A continue could be added by the prettyprinter
 * but it was decided not to support this facility.
 * For debugging, it's better to have a transparent prettyprinter
 * and no ghost statement with no ordering and no number.
 *
 * Insert a CONTINUE and move the label, comment and statement number
 * if any from the statement to the CONTINUE, if label there is.
 */
void
fix_sequence_statement_attributes(statement s)
{
    pips_assert("Should be an instruction block...",
		instruction_block_p(statement_instruction(s)));

    if (unlabelled_statement_p(s) && empty_comments_p(statement_comments(s))) {
	/* The statement block has no label and no comment: just do
           nothing. */
	statement_number(s) = STATEMENT_NUMBER_UNDEFINED;
    }
    else {
	/* There are some informations we need to keep: just add a
	   CONTINUE to keep them: */
	list instructions;
	statement continue_s;
	string label_name =
	    entity_local_name(statement_label(s)) + strlen(LABEL_PREFIX);

	instructions = instruction_block(statement_instruction(s));

	if (strcmp(label_name, RETURN_LABEL_NAME) == 0)
	    /* This the label of a RETURN, do not forward it: */
	    continue_s = make_continue_statement(entity_empty_label());
	else
	    continue_s = make_continue_statement(statement_label(s));

	statement_label(s) = entity_empty_label();
	statement_comments(continue_s) = statement_comments(s);
	statement_comments(s) = empty_comments;
	statement_number(continue_s) = statement_number(s);
	statement_number(s) = STATEMENT_NUMBER_UNDEFINED;

	instructions = CONS(STATEMENT, continue_s, instructions);
	instruction_block(statement_instruction(s)) = instructions;
    }
}


/* Apply fix_sequence_statement_attributes() on the statement only if
   it really a sequence. */
void
fix_statement_attributes_if_sequence(statement s)
{
    instruction i = statement_instruction(s);
    if (!instruction_undefined_p(i)) {
	if (instruction_sequence_p(i))
	    fix_sequence_statement_attributes(s);
    }
}


/* See if statement s is labelled and can be reached by a GO TO */
entity
statement_to_label(statement s)
{
  entity l = statement_label(s);

  if(format_statement_p(s)) {
    /* You cannot jump onto a non-executable statement */
    l = entity_empty_label();
  }
  else if(entity_empty_label_p(l)) {
    instruction i = statement_instruction(s);

    switch(instruction_tag(i)) {

    case is_instruction_sequence:
      /* The initial statement may be an unlabelled CONTINUE... */
      MAPL(cs, {
	statement stmt = STATEMENT(CAR(cs));
	l = statement_to_label(stmt);
	if(!entity_empty_label_p(l)||!continue_statement_p(stmt))
	  break;
      }, sequence_statements(instruction_sequence(i)));
      break;
    case is_instruction_unstructured:
      l = statement_to_label(control_statement(unstructured_control(instruction_unstructured(i))));
      break;
    case is_instruction_call:
    case is_instruction_test:
    case is_instruction_loop:
    case is_instruction_whileloop:
    case is_instruction_goto:
    case is_instruction_forloop:
    case is_instruction_expression:
      break;
    default:
      pips_error("statement_to_label", "Ill. tag %d for instruction",
		  instruction_tag(i));
    }
  }

  debug(8, "statement_to_label", "stmt %s, pointed by = %s\n",
	statement_identification(s), entity_local_name(l));

  return l;
}


/* Add a label to a statement. If the statement cannot accept a label (it
   is a sequence or it has already a label...), add a CONTINUE in front of
   the statement to carry it.

   @return \param s with the new \param label if it was possible to add it
   directly, or a new statement sequence with the new \param label added
   to a CONTINUE followed by the old statement \param s.

   \param label is the label to add to the statement

   \param s is the statement to be labelled

   \param labeled_statement is a pointer to a statement. It is initialized
   by the function to the statement that really get the label on. It is
   useful when we need to track it, for exemple for "goto" generation in
   the parser.

   The caller is responsible to *avoid* garbage collecting on \param s if a
   new statements are allocated in this fonction since it is included in
   the return statement.
 */
statement
add_label_to_statement(entity label,
		       statement s,
		       statement *labeled_statement) {
  /* Test with statement_to_label to deal with a label inside an
     instruction block: */
  entity old_label = statement_to_label(s);
  if (old_label == entity_undefined
      || (!instruction_sequence_p(statement_instruction(s))
	  && unlabelled_statement_p(s))) {
    statement_label(s) = label;
    *labeled_statement = s;
    return s;
  }
  else {
    /* Add a continue as a label landing pad since a statement can not
       have more than 1 label and a sequence cannot have a label: */
    statement c = make_continue_statement(label);
    *labeled_statement = c;
    list stmts = gen_statement_cons(s, NIL);
    stmts = gen_statement_cons(c, stmts);
    statement seq = make_block_statement(stmts);
    return seq;
  }
}


/* Returns FALSE is no syntactic control path exits s (i.e. even if TRUE is returned
 * there might be no control path). Subroutines and
 * functions are assumed to always return to keep the analysis intraprocedural.
 * See the continuation library for more advanced precondition-based analyses.
 *
 * TRUE is a safe default value.
 *
 * The function name is misleading: a RETURN statement does not return...
 * It should be called "statement_does_continue()"
 */
bool
statement_does_return(statement s)
{
    bool returns = TRUE;
    instruction i = statement_instruction(s);
    test t = test_undefined;

    switch(instruction_tag(i)) {
    case is_instruction_sequence:
	MAPL(sts,
	     {
		 statement st = STATEMENT(CAR(sts)) ;
		 if (!statement_does_return(st)) {
		     returns = FALSE;
		     break;
		 }
	     },
		 sequence_statements(instruction_sequence(i)));

	break;
    case is_instruction_unstructured:
	returns = unstructured_does_return(instruction_unstructured(i));
	break;
    case is_instruction_call:
	/* the third condition is due to a bug/feature of unspaghettify */
	returns = !stop_statement_p(s) && !return_statement_p(s) &&
	    !(continue_statement_p(s) && entity_return_label_p(statement_label(s)));
	break;
    case is_instruction_test:
	t = instruction_test(i);
	returns = statement_does_return(test_true(t)) ||
	    statement_does_return(test_false(t));
	break;
    case is_instruction_loop:
    case is_instruction_whileloop:
	/* No precise answer, unless you can prove the loop executes at
	 * least one iteration.
	 */
	returns = TRUE;
	break;
    case is_instruction_goto:
	/* returns = statement_does_return(instruction_goto(i)); */
	returns = FALSE;
	break;
    case is_instruction_forloop:
      break;
    case is_instruction_expression:
      break;
    default:
	pips_error("statement_does_return", "Ill. tag %d for instruction",
		   instruction_tag(i));
    }

    debug(8, "statement_does_return", "stmt %s, does return= %d\n",
	  statement_identification(s), returns);

    return returns;
}

bool
unstructured_does_return(unstructured u)
{
  bool returns = FALSE;
  control entry = unstructured_control(u);
  control exit = unstructured_exit(u);
  list nodes = NIL;

  FORWARD_CONTROL_MAP(c, {
      returns = returns || (c==exit);
  }, entry, nodes);
  gen_free_list(nodes);

  return returns;
}


/* Insert some statements before or after a given statement.

   It should not be used with parsed code containing "goto", i.e. before being
   controllized, because "goto" points to a given statement and here we
   change the statement that could be pointed to...
*/
void insert_or_append_a_statement_list(statement target,
				       list s_list,
				       bool after_p)
{
  if (statement_block_p(target)) {
    sequence seq = instruction_sequence(statement_instruction(target));
    if (after_p) // append
	sequence_statements(seq) = gen_nconc(sequence_statements(seq), s_list);
    else // insert
	sequence_statements(seq) = gen_nconc(s_list, sequence_statements(seq));
  }
  else {
    statement new_statement = make_stmt_of_instr(statement_instruction(target));
    /* Create the new statement sequence with s_list first: */
    if (after_p) // append
      statement_instruction(target) =
	make_instruction_block(CONS(STATEMENT,
				    new_statement,
				    s_list));
    else // insert
      statement_instruction(target) =
	make_instruction_block(gen_nconc(s_list, CONS(STATEMENT,
						      new_statement,
						      NIL)));
    statement_label(new_statement) = statement_label(target);
    statement_label(target) = entity_empty_label();
    statement_number(new_statement) = statement_number(target);
    statement_number(target) = STATEMENT_NUMBER_UNDEFINED;
    statement_ordering(target) = STATEMENT_ORDERING_UNDEFINED;
    statement_comments(new_statement) = statement_comments(target);
    statement_comments(target) = empty_comments;
    statement_extensions(new_statement) = statement_extensions(target);
    statement_extensions(target) = empty_extensions ();
  }
}

void insert_a_statement_list_in_a_statement(statement target,
					    list s_list)
{
  insert_or_append_a_statement_list(target, s_list, FALSE);
}

void append_a_statement_list_to_a_statement(statement target,
					    list s_list)
{
  insert_or_append_a_statement_list(target, s_list, TRUE);
}

/* Insert one single statement before or after a target statement.

   It should not be used with parsed code containing "goto", i.e. before being
   controllized, because "goto" points to a given statement and here we
   change the statement that could be pointed to...
*/
void insert_or_append_a_statement(statement target,
				  statement new,
				  bool after_p)
{
  list s_list = CONS(STATEMENT, new, NIL);
  insert_or_append_a_statement_list(target, s_list, after_p);
}

void insert_a_statement(statement target,
			statement new)
{
  insert_or_append_a_statement(target, new, FALSE);
}

void append_a_statement(statement target,
			statement new)
{
  insert_or_append_a_statement(target, new, TRUE);
}


void
gather_and_remove_all_format_statements_rewrite(statement s,list *all_formats)
{
    instruction i = statement_instruction(s);
    if (instruction_format_p(i)) {
        /* Put the instruction with the statement attributes in
           new_format. */
        statement new_format = make_stmt_of_instr(i);
        statement_label(new_format) = statement_label(s);
        statement_number(new_format) = statement_number(s);
        statement_comments(new_format) = statement_comments(s);
        statement_extensions(new_format) = statement_extensions(s);
        /* Replace the old FORMAT with a NOP: */
        statement_instruction(s) = make_instruction_block(NIL);
        statement_label(s) = entity_empty_label();
        statement_number(s) = STATEMENT_NUMBER_UNDEFINED;
        statement_ordering(s) = STATEMENT_ORDERING_UNDEFINED;
        statement_comments(s) = empty_comments;
        statement_extensions(s) = empty_extensions();

        *all_formats = CONS(STATEMENT, new_format,*all_formats);
    }
}


/* Used to keep aside the FORMAT before many code transformation that
   could remove them either. It just return a list of all the FORMAT
   statement and replace them with NOP. */
list
gather_and_remove_all_format_statements(statement s)
{
    list all_formats = NIL;

    gen_context_recurse(s,&all_formats, statement_domain,
		gen_true, gather_and_remove_all_format_statements_rewrite);

    return all_formats = gen_nreverse(all_formats);
}


/* Transfer all the FORMATs at the very beginning of a module: */
void
put_formats_at_module_beginning(statement s)
{
    /* Pick up all the FORMATs of the module: */
    list formats = gather_and_remove_all_format_statements(s);
    ifdebug (1)
	pips_assert("Incorrect statements...", statement_consistent_p(s));
    /* And put them at the very beginning of the module: */
    insert_a_statement_list_in_a_statement(s, formats);
    ifdebug (1)
	pips_assert("Incorrect statements...", statement_consistent_p(s));
}


/* Transfer all the FORMATs at the very end of a module: */
void
put_formats_at_module_end(statement s)
{
    /* Pick up all the FORMATs of the module: */
    list formats = gather_and_remove_all_format_statements(s);
    ifdebug (1)
	pips_assert("Incorrect statements...", statement_consistent_p(s));
    /* And put them at the very beginning of the module: */
    append_a_statement_list_to_a_statement(s, formats);
    ifdebug (1)
	pips_assert("Incorrect statements...", statement_consistent_p(s));
}


bool
figure_out_if_it_is_a_format(instruction i, bool *format_inside_statement_has_been_found)
{
    if (instruction_format_p(i)) {
        *format_inside_statement_has_been_found = true;
        /* Useless to go on further: */
        gen_recurse_stop(NULL);
        return false;
    }
    return true;
}


/* Return TRUE only if there is a FORMAT inside the statement: */
bool
format_inside_statement_p(statement s)
{
    bool format_inside_statement_has_been_found = false;

    gen_context_recurse(s,&format_inside_statement_has_been_found,
            instruction_domain,	figure_out_if_it_is_a_format, gen_null);

    return format_inside_statement_has_been_found;
}


/* Number of comment line *directly* attached to a statement.
 * Should be zero for sequences.
 */
int
statement_to_comment_length(statement stmt)
{
    int length = 0;
    string cmt = statement_comments(stmt);

    if(string_undefined_p(cmt)) {
	/* Is it allowed? Should be the empty string... */
	length = 0;
    }
    else {
	char c;
	while((c=*cmt++)!= '\0')
	    if(c=='\n')
		length++;
    }

    return length;
}


/* Poor attempt at associating physical line numbers to statement.
 *
 * Lines used for declarations are counted elsewhere: see
 * module_to_declaration_length()
 *
 * It is assumed that each statement fits on one physical line,
 * excepts sequences which require 0 line. Non statement such as
 * ENDIF, ELSE, ENDDO,... must be taken care of too.
 */

static int current_line = -1;
static persistant_statement_to_int stmt_to_line = persistant_statement_to_int_undefined;

static bool
down_counter(statement s)
{
    instruction i = statement_instruction(s);

    current_line += statement_to_comment_length(s);

    /* Is it printable on one line? */
    if(instruction_sequence_p(i)) {
	current_line += 0;
    }
    else if(empty_statement_or_labelless_continue_p(s)) {
	/* Must be an unlabelled CONTINUE */
	current_line += 0;
    }
    else {
	current_line += 1;
    }

    extend_persistant_statement_to_int(stmt_to_line, s, current_line);

    return TRUE;
}

static void
up_counter(statement s)
{
    instruction i = statement_instruction(s);

    if(instruction_loop_p(i)) {
	loop l = instruction_loop(i);

	if(entity_empty_label_p(loop_label(l))) {
	    /* There must be an ENDDO here */
	    current_line += 1;
	}
    }
    else if(instruction_test_p(i)
	    && statement_number(s) != STATEMENT_NUMBER_UNDEFINED) {
	/* There must be an ENDIF here */
	/* I doubt it's a real good way to detect synthetic IFs */
	    current_line += 1;
    }
    /* What could be done for ELSE and ELSEIF? */
    /* An else stack should be built. Use
     * DEFINE_LOCAL_STACK(name, type)
     * DEFINE_LOCAL_STACK(else_stack, statement)
     * Note that it should be pushed and poped in the down_counter() function
     */
}

persistant_statement_to_int
statement_to_line_number(statement s)
{
    persistant_statement_to_int s_to_l = persistant_statement_to_int_undefined;

    stmt_to_line = make_persistant_statement_to_int();

    current_line = 0;

    gen_multi_recurse(s, statement_domain, down_counter, up_counter, NULL);

    s_to_l = stmt_to_line;
    stmt_to_line = persistant_statement_to_int_undefined;
    return s_to_l;
}

/* insert statement s1 before or after statement s
 *
 * If statement s is a sequence, simply insert s1 at the begining
 * or at the end of the sequence s.
 *
 * If not, create a new statement s2 with s's fields and update
 * s as a sequence with no comments and undefined number and ordering.
 * The sequence is either "s1;s2" if "before" is TRUE or "s2;s1" else.
 *
 *
 * ATTENTION !!! : this version is not for unstructured case
 *
 */

void insert_statement(statement s,
		      statement s1,
		      bool before)
{
  list ls;
  instruction i = statement_instruction(s);
  if (instruction_sequence_p(i))
    {
      ls = instruction_block(i);
      if (before)
      {
          if(!ENDP(ls) && declaration_statement_p(STATEMENT(CAR(ls))))
          {
              while( !ENDP(CDR(ls)) && declaration_statement_p(STATEMENT(CAR(CDR(ls))))) POP(ls);
              CDR(ls)=CONS(STATEMENT,s1,CDR(ls));
              ls=instruction_block(i);
          }
          else
              ls = CONS(STATEMENT,s1,ls);
    }
      else
	ls = gen_nconc(ls,CONS(STATEMENT,s1,NIL));
      instruction_block(i) = ls;
    }
  else
    {
      statement s2 = copy_statement(s);
      /* SG: s still holds the label
       * we would like to move the label to s2 and to clear s
       * however this would lead to incoherency*/
      statement_label(s)=entity_empty_label();
      if (before)
	ls = CONS(STATEMENT,s1,CONS(STATEMENT,s2,NIL));
      else
	ls = CONS(STATEMENT,s2,CONS(STATEMENT,s1,NIL));

      statement_instruction(s)=instruction_undefined;/* SG: this is important*/
      update_statement_instruction(s, make_instruction_sequence(make_sequence(ls)));
    }
}

#define PIPS_DECLARATION_COMMENT "PIPS generated variable\n"
static string
default_generated_variable_commenter(__attribute__((unused))entity e)
{
    return strdup(PIPS_DECLARATION_COMMENT);
}

/* commenters are function used to add comments to pips-created variables
 * they are handled as a limited size stack
 * all commenters are supposed to return allocated data
 */
#define MAX_COMMENTERS 8 

typedef string (*generated_variable_commenter)(entity);
static generated_variable_commenter generated_variable_commenters[MAX_COMMENTERS] = {
    [0]=default_generated_variable_commenter /* c99 inside :) */
};
static size_t nb_commenters=1;

void push_generated_variable_commenter(string (*commenter)(entity))
{
    pips_assert("not exceeding stack commenters stack limited size\n",nb_commenters<MAX_COMMENTERS);
    generated_variable_commenters[nb_commenters++]=commenter;
}
void pop_generated_variable_commenter()
{
    pips_assert("not removing default commenter",nb_commenters!=1);
    --nb_commenters;
}
string generated_variable_comment(entity e)
{
    string tmp = generated_variable_commenters[nb_commenters-1](e);
    string out;
    asprintf(&out,"%s%s",c_module_p(get_current_module_entity())?"//":"C ",tmp);
    free(tmp);
    return out;
}



/* Declarations are not only lists of entities, but also statement to
   carry the line number, comments,... For the time begin, a
   declaration statement is a continue statement. */
statement add_declaration_statement(statement s, entity e)
{
    if(statement_block_p(s)) {
        list sl = statement_block(s); //statement list
        list cl = list_undefined; // current statement list
        list pl = NIL; // previous statement list
        list nsl = list_undefined; // new statement list
        string comment = generated_variable_comment(e);
        statement ds = make_declaration_statement(e,
                STATEMENT_NUMBER_UNDEFINED,
                comment);

        /* Look for the last declaration: it is pointed to by pl */
        for(cl=sl; !ENDP(cl); POP(cl)) {
            statement cs = STATEMENT(CAR(cl));
            if(declaration_statement_p(cs)) {
                pl = cl;
            }
            else {
                break;
            }
        }

        /* Do we have previous declarations to skip? */
        if(!ENDP(pl)) {
            /* SG: if CAR(pl) has same comment and same type as ds, merge them */
            statement spl = STATEMENT(CAR(pl));
            if( comments_equal_p(statement_comments(spl),comment) &&
                    basic_equal_strict_p(entity_basic(e),entity_basic(ENTITY(CAR(statement_declarations(spl))))))
            {
                free_statement(ds);
                statement_declarations(spl)=gen_nconc(statement_declarations(spl),CONS(ENTITY,e,NIL));
                nsl=sl;
            }
            /* SG: otherwise, insert ds */
            else
            {
                CDR(pl) = NIL; // Truncate sl
                nsl = gen_nconc(sl, CONS(STATEMENT, ds, cl));
            }
        }
        else { // pl == NIL
            /* The new declaration is inserted before sl*/
            pips_assert("The above loop was entered at most once", sl==cl);
            nsl = CONS(STATEMENT, ds, cl);
        }

        instruction_block(statement_instruction(s)) = nsl;
    }
    else
    {
        pips_internal_error("can only add declarations to statement blocks\n");
    }

    ifdebug(8) {
        pips_debug(8, "Statement after declaration insertion:\n");
        print_statement(s);
    }

    return s;
}

/* Replace the instruction in statement s by instruction i.
 *
 * Free the old instruction.
 *
 * If the new instruction is a sequence,
 * add a CONTINUE to carry the label and/or the comments. The
 * statement number and ordering are set to undefined.
 *
 * Else if the new instruction is not a sequence, the statement number
 * and the ordering are also set to undefined because the connexion
 * with the parsed line is lost and the ordering most likely obsolete.
 * If the parsed statement number is still meaningfull, the caller must
 * take care of it.
 *
 * If the initial instruction has been re-used, do not forget
 * to dereference it before calling this function:
 *
 * statement_instruction(s) = instruction_undefined;
 *
 * Be careful with the label and the comments too: they may have
 * been reused.
 *
 *
 * ATTENTION !!! :  this version is not for unstructured case
 *
 *
 */

statement update_statement_instruction(statement s,instruction i)
{
    /* reset numbering and ordering */
    statement_number(s) = STATEMENT_NUMBER_UNDEFINED;
    statement_ordering(s) = STATEMENT_ORDERING_UNDEFINED;

    /* try hard to keep comments and label when relevant */
    if (instruction_sequence_p(i) && (
                !statement_with_empty_comment_p(s) ||
                (!unlabelled_statement_p(s) && !entity_return_label_p(statement_label(s))) /*SG:note the special returnèlabel case */
                ) )
    {
        statement cs = make_continue_statement(statement_label(s));
        statement_comments(cs)= statement_comments(s);
        statement_comments(s) = empty_comments;
        statement_label(s)=entity_empty_label();

        /* add the CONTINUE statement before the sequence instruction i */
        list seq = make_statement_list(cs,instruction_to_statement(i));

        free_instruction(statement_instruction(s));
        statement_instruction(s) =  make_instruction_sequence(make_sequence(seq));
    }

    else
    {
        if(entity_return_label_p(statement_label(s)) && !return_statement_p(s))
            statement_label(s)=entity_empty_label();
        free_instruction(statement_instruction(s));
        statement_instruction(s) = i;
        /* SG: if the old statement had declarations, they are removed
         * maybe we should regenerate the new one if any to keep global coherency ?*/
        gen_free_list(statement_declarations(s));
        statement_declarations(s)=NIL;
    }
    return s;
}

/* Assume that statement rs appears in statement as and replaced it
   by a statement list. */
void statement_replace_with_statement_list(statement as, statement rs, list sl)
{
  ifdebug(8) {
    pips_debug(8, "Ancestor statement before substitution:\n");
    print_statement(as);
  }
  if(statement_block_p(as)) {
    list asl = statement_block(as);
    instruction asi = statement_instruction(as);
    if(asl==NULL) {
      pips_internal_error("The statement is not part of its ancestor");
    }
    gen_substitute_chunk_by_list(&asl, (void *) rs, sl);
    instruction_block(asi) = asl;
  }
  else {
    pips_internal_error("not implemented yet");
  }
  ifdebug(8) {
    pips_debug(8, "Ancestor statement after substitution:\n");
    print_statement(as);
  }
}

static bool find_implicit_goto(statement s, list * tl)
{
  bool found = FALSE;

  if(statement_call_p(s)) {
    call c = instruction_call(statement_instruction(s));
    entity f = call_function(c);

    if(ENTITY_WRITE_P(f) || ENTITY_READ_P(f) || ENTITY_OPEN_P(f) || ENTITY_CLOSE_P(f)) {
      list ce = list_undefined;
    
      for(ce = call_arguments(c); !ENDP(ce); ce = CDR(CDR(ce))) {
	expression e = EXPRESSION(CAR(ce));
	entity f1 = call_function(syntax_call(expression_syntax(e)));
      
	pips_assert("expression e must be a call", expression_call_p(e));
      
	if (strcmp(entity_local_name(f1), "ERR=") == 0 || 
	    strcmp(entity_local_name(f1), "END=") == 0) {
	  expression ne = EXPRESSION(CAR(CDR(ce))); /* Next Expression */
	  entity l = call_function(syntax_call(expression_syntax(ne)));
      
	  pips_assert("expression ne must be a call", expression_call_p(ne));
	  pips_assert("l is a label", entity_label_p(l));
	  
	  *tl = CONS(ENTITY, l, *tl);
	}
      }
    }
    /* No need to go down in call statements */
    found = TRUE;
  }
  
  return !found;
}

/* Look for labels appearing in END= or ERR= IO clauses and allocate a
   label list. */
list statement_to_implicit_target_labels(statement s)
{
  list ll = NIL; /* Label List */
   
  gen_context_recurse(s, (void *) &ll,statement_domain, find_implicit_goto, gen_null);
  
  return ll;
}

/* Make sure that s and all its substatements are defined */

static bool undefined_statement_found_p(statement s, bool * p_undefined_p)
{
  if(!(* p_undefined_p)) {
    ifdebug(9) {
      fprintf(stderr, "----\n");
      /* The recursive descent in s is fatal when there is a problem. */
      /* print_statement(s); */
      if(statement_undefined_p(s)) {
	/* You probably want a breakpoint here... */
	abort();
      }
      else {
	pips_debug(9,"checking statement %03td (%td,%td) with address %p:\n",
		   statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
		   ORDERING_STATEMENT(statement_ordering(s)), s);
      }
    }
    * p_undefined_p = statement_undefined_p(s);
  }
  return !(* p_undefined_p);
}

bool all_statements_defined_p(statement s)
{
  bool undefined_p = FALSE;

  gen_context_recurse(s, (void *) &undefined_p ,statement_domain,
		      undefined_statement_found_p, gen_null);
  return !undefined_p;
}


static bool add_statement_declarations(statement s, list *statement_to_all_included_declarations)
{
    /* naive version with O(n^2) complexity)*/
    if(declaration_statement_p(s))
    {
        FOREACH(ENTITY,e,statement_declarations(s))
            *statement_to_all_included_declarations=gen_once(e,*statement_to_all_included_declarations);
    }
  return true;
}

/* Get a list of all variables declared recursively within a statement */
list statement_to_declarations(statement s)
{
  list statement_to_all_included_declarations = NIL;

  gen_context_recurse(s, &statement_to_all_included_declarations,
		      statement_domain, add_statement_declarations, gen_null);

  return statement_to_all_included_declarations;
}

/* Get a list of all variables declared recursively within an instruction */
list instruction_to_declarations(instruction i)
{
  list statement_to_all_included_declarations = NIL;

  gen_context_recurse(i,&statement_to_all_included_declarations,
		      statement_domain, add_statement_declarations, gen_null);

  return statement_to_all_included_declarations;
}

/* Returns the declarations contained in a list of statement. */
list statements_to_declarations(list sl)
{
  instruction i = make_instruction_block(sl);
  list dl = instruction_to_declarations(i);
  /* The declaration list is reversed by
     instruction_to_declarations(). */
  dl = gen_nreverse(dl);
  return dl;
}

/************************************************************ STAT VARIABLES */

// local struct
typedef struct {
  list lents;
  set sents;
} entities_t;

static bool add_stat_referenced_entities(reference r, entities_t * vars)
{
  // a function may be referenced, eg in a function pointer assignment.
  entity var = reference_variable(r);
  if (!set_belong_p(vars->sents, var))
  {
    vars->lents = CONS(entity, var, vars->lents);
    set_add_element(vars->sents, vars->sents, var);
  }
  return true;
}

static bool add_loop_index_entity(loop l, entities_t * vars)
{
  entity idx = loop_index(l);
  if (!set_belong_p(vars->sents, idx))
  {
    vars->lents = CONS(entity, idx, vars->lents);
    set_add_element(vars->sents, vars->sents, idx);
  }
  return true;
}

static bool add_ref_entities_in_init(statement s, entities_t * vars)
{
  FOREACH(entity, var, statement_declarations(s))
  {
    value init = entity_initial(var);
    if (value_expression_p(init))
      // only references down there
      gen_context_recurse(value_expression(init), vars,
	  reference_domain, add_stat_referenced_entities, gen_null);
  }
  return true;
}

/* Get a list of all variables referenced recursively within a statement:
 * - as reference in expressions in the code
 * - as loop indexes, which may not be used anywhere else
 * - as references in initilializations
 */
list statement_to_referenced_entities(statement s)
{
  entities_t vars;
  vars.lents = NIL;
  vars.sents = set_make(set_pointer);

  gen_context_multi_recurse
    (s, &vars,
     reference_domain, add_stat_referenced_entities, gen_null,
     loop_domain, add_loop_index_entity, gen_null,
     statement_domain, add_ref_entities_in_init, gen_null,
     NULL);

  set_free(vars.sents), vars.sents = NULL;
  return gen_nreverse(vars.lents);
}

/***************************************************** USER FUNCTIONS CALLED */

static bool add_stat_called_user_entities(call c, entities_t * funcs)
{
  /* FI: I do not know what should be done for pointers to function,
     assuming they can be called. */
  entity f = call_function(c);
  value fv = entity_initial(f);

  if(!set_belong_p(funcs->sents, f) && value_code_p(fv))
  {
    funcs->lents = CONS(entity, f, funcs->lents);
    set_add_element(funcs->sents, funcs->sents, f);
  }
  return true;
}

static bool add_stat_called_in_inits(statement s, entities_t * funcs)
{
  FOREACH(entity, var, statement_declarations(s))
  {
    value init = entity_initial(var);
    if (value_expression_p(init))
      gen_context_recurse(value_expression(init), funcs,
		  call_domain, add_stat_called_user_entities, gen_null);
  }
  return true;
}

/* Get a list of all user function called recursively within a statement:
 * - in the code
 * - in initialisations
 */
list statement_to_called_user_entities(statement s)
{
  entities_t funcs;
  funcs.lents = NIL;
  funcs.sents = set_make(set_pointer);

  gen_context_multi_recurse
     (s, &funcs,
      call_domain, add_stat_called_user_entities, gen_null,
      statement_domain, add_stat_called_in_inits, gen_null,
      NULL);

  set_free(funcs.sents), funcs.sents = NULL;
  return gen_nreverse(funcs.lents);
}

/* Return first reference found */
static reference first_reference_to_v = reference_undefined;
static entity variable_searched = entity_undefined;

static bool first_reference_to_v_p(reference r)
{
  bool result= TRUE;
  if (reference_undefined_p(first_reference_to_v)) {
    entity rv = reference_variable(r);
    if (rv == variable_searched) {
      first_reference_to_v = r;
      result = FALSE;
    }
  }
  else
    result = FALSE;

  return result;
}

reference find_reference_to_variable(statement s, entity v) 
{
  reference r = reference_undefined;
  first_reference_to_v = reference_undefined;
  variable_searched = v;
  gen_recurse(s, reference_domain, first_reference_to_v_p, gen_null);
  r = copy_reference(first_reference_to_v);
  first_reference_to_v = reference_undefined;
  variable_searched = entity_undefined;
  return r;
}

static int reference_count = -1;

/* Count static references */
static bool count_static_references_to_v_p(reference r)
{
  bool result= TRUE;
    entity rv = reference_variable(r);
    if (rv == variable_searched) {
      reference_count++;
    }
  return result;
}

int count_static_references_to_variable(statement s, entity v) 
{
  reference_count = 0;
  variable_searched = v;
  gen_recurse(s, reference_domain, count_static_references_to_v_p, gen_null);
  variable_searched = entity_undefined;
  return reference_count;
}

/* Estimate count of dynamic references */
static int    loop_depth;

static bool count_references_to_v_p(reference r)
{
  bool result= TRUE;
  entity rv = reference_variable(r);
  if (rv == variable_searched) {
    /* 10: arbitrary value for references nested in at least one loop */
    reference_count += (loop_depth > 0 ? 10 : 1 );
  }
  return result;
}

/* This function checks reference to proper elements, not slices */
static bool count_element_references_to_v_p(reference r)
{
  bool result= TRUE;
  entity rv = reference_variable(r);
  if (rv == variable_searched) {
    list inds = reference_indices(r);
    size_t d = type_depth(ultimate_type(entity_type(rv)));
    if (gen_length(inds) == d) {
      /* 10: arbitrary value for references nested in at least one loop */
      reference_count += (loop_depth > 0 ? 10 : 1 );
    }
  }
  return result;
}

static bool count_loop_in(loop __attribute__ ((unused)) l)
{
  loop_depth++;
  return TRUE;
}

static void count_loop_out(loop __attribute__ ((unused)) l)
{
  loop_depth--;
}

int count_references_to_variable(statement s, entity v)
{
  reference_count = 0;
  loop_depth = 0;
  variable_searched = v;
  gen_multi_recurse(s, loop_domain, count_loop_in, count_loop_out,
		    reference_domain, count_references_to_v_p, gen_null, NULL);
  variable_searched = entity_undefined;
  return reference_count;
}
int count_references_to_variable_element(statement s, entity v)
{
  reference_count = 0;
  loop_depth = 0;
  variable_searched = v;
  gen_multi_recurse(s, loop_domain, count_loop_in, count_loop_out,
		    reference_domain, count_element_references_to_v_p, gen_null, NULL);
  variable_searched = entity_undefined;
  return reference_count;
}
/**
 * @name get_statement_depth and its auxilary functions
 * @{ */

static bool is_substatement = false;

bool statement_substatement_walker(statement some, statement s)
{
	if( !is_substatement)
		is_substatement = (some==s);
	return  !is_substatement;
}

/**
 * @brief search a statement inside a statement
 *
 * @param s searched statement
 * @param root where to start searching from
 *
 * @return true if found
 */
bool statement_substatement_p(statement s, statement root)
{
  is_substatement= false;
  //printf("searching::::::::::::::\n");
  //print_statement(s);
  //printf("inside::::::::::::\n");
  //print_statement(root);
  gen_context_recurse(root,s,statement_domain,statement_substatement_walker,gen_null);
  //if(is_substatement) printf(":::::::::found !\n");
  //else printf("::::::::not found !\n");
  return is_substatement;
}

/**
 * @brief computes the block-depth of a statement
 * usefull to generate entity declared at particular block level
 *
 * @param s statement we compute the depth of
 * @param root outer statement containing s
 *
 * @return positive integer
 */
int get_statement_depth(statement s, statement root)
{
  if( s == root )
    return 0;
  else {
    instruction i = statement_instruction(root);
    switch(instruction_tag(i))
      {
      case is_instruction_sequence:
	{
	  FOREACH(STATEMENT,stmt,instruction_block(i))
	    {
	      if(statement_substatement_p(s,stmt))
		return 1+get_statement_depth(s,stmt);
	    }
	  pips_internal_error("you should never reach this point");
	  return -1;
	}
      case is_instruction_test:
	return
	  statement_substatement_p(s,test_true(instruction_test(i))) ?
	  get_statement_depth(s,test_true(instruction_test(i))):
	get_statement_depth(s,test_false(instruction_test(i)));
      case is_instruction_loop:
	return get_statement_depth(s,loop_body(instruction_loop(i)));
      case is_instruction_whileloop:
	return get_statement_depth(s,whileloop_body(instruction_whileloop(i)));
      case is_instruction_forloop:
	return get_statement_depth(s,forloop_body(instruction_forloop(i)));
      case is_instruction_unstructured:
	pips_internal_error("not implemented for unstructured");
	return -1;
      default:
	pips_internal_error("you should never reach this point");
	return -1;
      };
  }

}

/**  @} */

/**
 * @name declarations updater
 * @{ */

/**
 * helper looking in a reference for referenced entities
 *
 * @param r reference to check
 * @param re set to fill
 */
static void statement_clean_declarations_reference_walker(reference r, set re)
{
  entity e = reference_variable(r);
  if( !entity_constant_p(e) && ! intrinsic_entity_p(e) )
    set_add_element(re,re,e);
}

/**
 * helper looking in a call for referenced entities
 *
 * @param c call to check
 * @param re set to fill
 */
static void statement_clean_declarations_call_walker(call c, set re)
{
  entity e = call_function(c);
  if( !entity_constant_p(e) && ! intrinsic_entity_p(e) )
    set_add_element(re,re,e);
}

/**
 * helper looking in a loop for referenced entities
 *
 * @param l loop to check
 * @param re set to fill
 */
static void statement_clean_declarations_loop_walker(loop l, set re)
{
  entity e = loop_index(l);
  if( !entity_constant_p(e) && ! intrinsic_entity_p(e) )
    set_add_element(re,re,e);
}


/**
 * helper looking in a list for referenced entities
 *
 * @param l list to check
 * @param re set to fill
 */
static
void statement_clean_declarations_list_walker(list l, set re)
{
  FOREACH(ENTITY,e,l)
    if( !entity_constant_p(e) && ! intrinsic_entity_p(e) )
      set_add_element(re,re,e);
}

/**
 * helper looking in a ram for referenced entities
 *
 * @param r ram to check
 * @param re set to fill
 */
static void statement_clean_declarations_ram_walker(ram r, set re)
{
  statement_clean_declarations_list_walker(ram_shared(r),re);
}

/**
 * helper looking in an area for referenced entities
 *
 * @param a area to check
 * @param re set to fill
 */
static void statement_clean_declarations_area_walker(area a, set re)
{
    statement_clean_declarations_list_walker(area_layout(a),re);
}

/**
 * helper diving into an entity to find referenced entities
 *
 * @param e entity to dive into
 * @param re set to fill
 *
 */
void entity_get_referenced_entities(entity e, set re)
{
  /*if(entity_variable_p(e))*/ {
    gen_context_multi_recurse(entity_type(e),re,
			      reference_domain,gen_true,statement_clean_declarations_reference_walker,
			      call_domain,gen_true,statement_clean_declarations_call_walker,
			      NULL
			      );
    /* SG: I am unsure wether it is valid or not to find an entity with undefined initial ... */
    if( !value_undefined_p(entity_initial(e) ) ) {
      gen_context_multi_recurse(entity_initial(e),re,
				call_domain,gen_true,statement_clean_declarations_call_walker,
				reference_domain,gen_true,statement_clean_declarations_reference_walker,
				area_domain,gen_true,statement_clean_declarations_area_walker,
				ram_domain,gen_true,statement_clean_declarations_ram_walker,
				NULL);
    }
  }
}

/**
 * helper iterating over statement declaration to find referenced entities
 *
 * @param s statement to check
 * @param re set to fill
 */
static void statement_clean_declarations_statement_walker(statement s, set re)
{
  FOREACH(ENTITY,e,statement_declarations(s))
    entity_get_referenced_entities(e,re);
}


/**
 * retrieves the set of entites used in elem
 * beware that this entites may be formal parameters, functions etc
 * so please filter this set depending on your need
 *
 * @param elem  element to check (any gen_recursifiable type is allowded)
 *
 * @return set of referenced entities
 */
set get_referenced_entities(void* elem)
{
  set referenced_entities = set_make(set_pointer);

  /* if s is an entity it self, add it */
  if(INSTANCE_OF(entity,(gen_chunkp)elem))
      set_add_element(referenced_entities,referenced_entities,elem);

  /* gather entities from s*/
  gen_context_multi_recurse(elem,referenced_entities,
			    loop_domain,gen_true,statement_clean_declarations_loop_walker,
			    reference_domain,gen_true,statement_clean_declarations_reference_walker,
			    call_domain,gen_true,statement_clean_declarations_call_walker,
			    statement_domain,gen_true,statement_clean_declarations_statement_walker,
			    ram_domain,gen_true,statement_clean_declarations_ram_walker,
			    NULL);

  /* gather all entities referenced by referenced entities */
  set other_referenced_entities = set_make(set_pointer);
  SET_FOREACH(entity,e,referenced_entities)
    {
      entity_get_referenced_entities(e,other_referenced_entities);
    }

  /* merge results */
  set_union(referenced_entities,other_referenced_entities,referenced_entities);
  set_free(other_referenced_entities);

  return referenced_entities;
}

/**
 * remove useless entities from declarations
 * an entity is flagged useless when no reference is found in stmt
 * and when it is not used by an entity found in stmt
 *
 * @param declarations list of entity to purge
 * @param stmt statement where entities are used
 *
 */
static void statement_clean_declarations_helper(list declarations, statement stmt)
{
    set referenced_entities = get_referenced_entities(stmt);
    list decl_cpy = gen_copy_seq(declarations);

    /* look for entity that are used in the statement
     * SG: we need to work on  a copy of the declarations because of
     * the RemoveLocalEntityFromDeclarations
     */
    FOREACH(ENTITY,e,decl_cpy)
    {
        /* area and parameters are always used, so are referenced entities */
        if( formal_parameter_p(e) || entity_area_p(e) || set_belong_p(referenced_entities,e) /*|| storage_return_p(entity_storage(e))*/);
        else
        {
            /* entities whose declaration has a side effect are always used too */
            bool has_side_effects_p = false;
            value v = entity_initial(e);
            if( value_expression_p(v) )
            {
                list effects = expression_to_proper_effects(value_expression(v));
                FOREACH(EFFECT, eff, effects)
                {
                    if( action_write_p(effect_action(eff)) ) has_side_effects_p = true;
                }
                gen_full_free_list(effects);
            }

            /* do not keep the declaration, and remove it from any declaration_statement */
            if( !has_side_effects_p ) {
                RemoveLocalEntityFromDeclarations(e,get_current_module_entity(),stmt);
            }
        }
    }

    gen_free_list(decl_cpy);
    set_free(referenced_entities);
}

/**
 * check if all entities used in s and module are declared in module
 * does not work as well as expected on c module because it does not fill the statement declaration
 * @param module module to check
 * @param s statement where reference can be found
 */
static void entity_generate_missing_declarations(entity module, statement s)
{
  /* gather referenced entities */
  set referenced_entities = get_referenced_entities(s);
  set ref_tmp = set_make(set_pointer);
  /* gather all entities referenced by referenced entities */
  SET_FOREACH(entity,e0,referenced_entities) {
    entity_get_referenced_entities(e0,ref_tmp);
  }

  referenced_entities=set_union(referenced_entities,ref_tmp,referenced_entities);
  set_free(ref_tmp);

  /* fill the declarations with missing entities (ohhhhh a nice 0(n²) algorithm*/
  list new = NIL;
  SET_FOREACH(entity,e1,referenced_entities) {
    if(gen_chunk_undefined_p(gen_find_eq(e1,entity_declarations(module))))
      new=CONS(ENTITY,e1,new);
  }

  set_free(referenced_entities);
  sort_list_of_entities(new);
  entity_declarations(module)=gen_nconc(new,entity_declarations(module));
}


/**
 * remove all the entity declared in s but never referenced
 * it's a lower version of use-def-elim !
 *
 * @param s statement to check
 */
void statement_clean_declarations(statement s)
{
    if(statement_block_p(s)) {
        statement_clean_declarations_helper( statement_declarations(s),s);
    }
}

/**
 * remove all entities declared in module but never used in s
 *
 * @param module module to check
 * @param s statement where entites may be used
 */
void entity_clean_declarations(entity module,statement s)
{
    entity curr = get_current_module_entity();
    if( ! same_entity_p(curr,module)) {
        reset_current_module_entity();
        set_current_module_entity(module);
    }
    else
        curr=entity_undefined;

    statement_clean_declarations_helper(entity_declarations(module),s);
    if(fortran_module_p(module)) /* to keep backward compatibility with hpfc*/
        entity_generate_missing_declarations(module,s);

    if(!entity_undefined_p(curr)){
        reset_current_module_entity();
        set_current_module_entity(curr);
    }

}

/**  @} */




/**
 * @name statement finders
 * find statements with particular constraints
 * @{ */

/**
 * structure used by find_statements_with_label_walker
 */
struct fswl {
    list l; ///< list of statement matching condition
    entity key; ///< used for the condition
};

/**
 * helper to find statement with a particular label
 * as label should be unique, the function stops once a statement is found
 *
 * @param s statement to inspect
 * @param p struct containing the list to fill and the label to search
 *
 * @return
 */
static bool find_statements_with_label_walker(statement s, struct fswl *p)
{
  if( same_entity_p(statement_label(s),p->key) ||
      (statement_loop_p(s)&& same_entity_p(loop_label(statement_loop(s)),p->key)) )
    {
      p->l=CONS(STATEMENT,s,p->l);
      gen_recurse_stop(NULL);
    }
  return true;
}

/**
 * find a statement in s with entity label
 *
 * @param s statement to search into
 * @param label label of the searched statement
 *
 * @return list containing a unique element
 * a list is returned for coherence with the other find_satements functions
 */
list find_statements_with_label(statement s, entity label)
{
  struct fswl p = {  NIL, label };
  gen_context_recurse(s,&p,statement_domain,find_statements_with_label_walker,gen_null);
  return p.l;
}

static bool find_statements_interactively_walker(statement s, list *l)
{
  string answer = string_undefined;
  do {
    while( string_undefined_p(answer) || empty_string_p(answer)  )
      {
	user_log("Do you want to pick the following statement ?\n"
		 "*********************************************\n");
	print_statement(s);

	answer = user_request(
			      "*********************************************\n"
			      "[y/n] ?"
			      );
	if( !answer ) pips_user_error("you did not answer !\n");
      }
    if( answer[0]!='y' && answer[0]!='n' )
      {
	pips_user_warning("answer by 'y' or 'n' !\n");
	free(answer);
	answer=string_undefined;
      }
  } while(string_undefined_p(answer));
  bool pick = answer[0]=='y';
  if(pick) {
    *l=CONS(STATEMENT,s,*l);
    return false;
  }
  else if( !ENDP(*l) )
    gen_recurse_stop(NULL);
  return true;
}

/**
 * prompt the user to select contiguous statement in s
 *
 * @param s statement to search into
 *
 * @return list of selected statement
 */
list find_statements_interactively(statement s)
{
  list l =NIL;
  gen_context_recurse(s,&l,statement_domain,find_statements_interactively_walker,gen_null);
  return gen_nreverse(l);
}

/**
 * used to pass parameters to find_statements_with_comment_walker
 */
struct fswp {
    list l;
    string begin;
};

extension statement_with_pragma_p(statement s, string seed)
{
    list exs = extensions_extension(statement_extensions(s));
    FOREACH(EXTENSION,ex,exs)
    {
        pragma pr = extension_pragma(ex);
        if(pragma_string_p(pr) && strstr(pragma_string(pr),seed))
        {
            return ex;
        }
    }
    return NULL;
}

static bool find_statements_with_pragma_walker(statement s, struct fswp *p)
{
    if(statement_with_pragma_p(s,p->begin))
    {
        p->l=CONS(STATEMENT,s,p->l);
        gen_recurse_stop(NULL);
    }
    return true;
}


list find_statements_with_pragma(statement s, string begin)
{
  struct fswp p = { NIL, begin };
  gen_context_recurse(s,&p,statement_domain,find_statements_with_pragma_walker,gen_null);
  return gen_nreverse(p.l);
}

static bool look_for_user_call(call c, bool * user_call_p)
{
  entity f = call_function(c);
  bool go_on_p = TRUE;

  if(value_code_p(entity_initial(f))) {
    * user_call_p = TRUE;
    go_on_p = FALSE;
  }
  return go_on_p;
}

/* Check if a statement s contains a call to a user-defined
   function. This may be useful because PIPS does not contain a
   control effect analysis and because a user-defined function can
   hide a control effect such as exit(), abort() or STOP. */
bool statement_contains_user_call_p(statement s)
{
  bool user_call_p = FALSE;
  gen_context_recurse(s, &user_call_p, call_domain, look_for_user_call, gen_null);
  return user_call_p;
}

static bool look_for_control_effects(call c, bool * control_effect_p)
{
  entity f = call_function(c);
  bool go_on_p = TRUE;
  value fv = entity_initial(f);

  if(value_code_p(fv)) {
    * control_effect_p = TRUE;
    go_on_p = FALSE;
  }
  else if(value_intrinsic_p(fv)) {
    if(ENTITY_EXIT_SYSTEM_P(f)
       || ENTITY_ABORT_SYSTEM_P(f)
       || ENTITY_C_RETURN_P(f)
       || ENTITY_RETURN_P(f)
       || ENTITY_ASSERT_SYSTEM_P(f)
       || ENTITY_ASSERT_FAIL_SYSTEM_P(f)) {
      * control_effect_p = TRUE;
      go_on_p = FALSE;
    }
  }
  return go_on_p;
}

/* Check if a statement s contains a call to a user-defined function
   or to an intrinsic with control effects. This may be useful because
   PIPS does not contain a control effect analysis and because a
   user-defined function can hide a control effect such as exit(),
   abort() or STOP. */
bool statement_may_have_control_effects_p(statement s)
{
  bool control_effect_p = FALSE;

  /* Preserve branch targets, without checking if they are useful or
     not because it can be done by another pass */
  control_effect_p = !entity_empty_label_p(statement_label(s));

  if(!control_effect_p) {
    /* These statements may hide a non-terminating loop. I assume that
       do loops always terminate. They also always have a memory
       write effect for the index, which may not be true for the
       other kinds of loops. Unstructured could be tested to see if
       they have a syntactical control cycle or not. */
    control_effect_p = statement_whileloop_p(s) || statement_forloop_p(s)
      || statement_unstructured_p(s);
    if(!control_effect_p)
      gen_context_recurse(s, &control_effect_p, call_domain, look_for_control_effects, gen_null);
  }

  return control_effect_p;
}

/* Make (a bit more) sure that s is gen_defined_p in spite of poor
   decision for empty fields and that strdup can be used on the string
   fields. */
statement normalize_statement(statement s)
{
  if(string_undefined_p(statement_decls_text(s))
     || statement_decls_text(s)==NULL)
    statement_decls_text(s) = strdup("");
  if(empty_comments_p(statement_comments(s))
    || statement_comments(s)==NULL)
	      statement_comments(s) = strdup("");
  return s;
}
/**  @} */

struct sb {
    statement s;
    bool res;
};

static bool statement_in_statement_walker(statement st, struct sb* sb)
{
    ifdebug(7) {
        pips_debug(7,"considering statement:\n");
        print_statement(st);
    }
    if(sb->s==st)
    {
        sb->res=true; gen_recurse_stop(0);
    }
    return true;
}

bool statement_in_statement_p(statement s, statement st)
{
    struct sb sb = { s,false };
    gen_context_recurse(st,&sb,statement_domain, statement_in_statement_walker,gen_null);
    return sb.res;
}

bool statement_in_statements_p(statement s, list l)
{
    FOREACH(STATEMENT,st,l)
        if(statement_in_statement_p(s,st)) return true;
    return false;
}

/* That's all folks */

/** @} */
