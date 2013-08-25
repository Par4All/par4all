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
#include "control.h"
#include "syntax.h"


/** @defgroup statement_util Methods dealing with statements
    @{
*/

/******************************************************* EMPTY STATEMENT */
/* detects a statement with no special effect...
 */


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
    MAP(STATEMENT, s, if (!empty_code_p(s)) return false, l);
    return true;
}


bool
empty_comments_p(const char* s)
{
  /* Could be replaced by a macro. See macro empty_comments */
  pips_assert("comments cannot be NULL", s!=NULL);
  return (string_undefined_p(s) || strcmp(s,"")==0);
}


bool
comments_equal_p(string c1, string c2)
{
    return
        ( empty_comments_p(c1) && empty_comments_p(c2)) ||
        (!empty_comments_p(c1) && !empty_comments_p(c2) && same_string_p(c1,c2));
}


/* Return true if the statement has an empty statement: */
bool
statement_with_empty_comment_p(statement s)
{
    string the_comments = statement_comments(s);
    return empty_comments_p(the_comments);
}


/* Test if a statement is an assignment. */
bool
assignment_statement_p(statement s) {
  instruction i = statement_instruction(s);

  return instruction_assign_p(i);
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
	pips_internal_error("unexpected GO TO");
    case is_instruction_call:
	return assignment_statement_p(s);
    case is_instruction_unstructured:
	break;
    default: pips_internal_error("ill. instruction tag %d", instruction_tag(i));
    }
    return false;
}


/* Test if a statement is a C or Fortran "return" */
bool return_statement_p(statement s) {
  instruction i = statement_instruction(s);
  return return_instruction_p(i);
}

bool exit_statement_p(statement s) {
  instruction i = statement_instruction(s);
  return exit_instruction_p(i);
}

bool abort_statement_p(statement s) {
  instruction i = statement_instruction(s);
  return abort_instruction_p(i);
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

bool forloop_statement_p(statement s) {
  instruction i = statement_instruction(s);

  return instruction_forloop_p(i);
}

/* Had to be optimized according to Beatrice Creusillet. We assume
   that false is returned most of the time. String operations are
   avoided (almost) as much as possible.

   For the time being a declaration statement is a call to continue
   with non-empty declarations.

   FI: This could be fixed by adding + declaration in the instruction domainn
*/
bool declaration_statement_p(statement s) {
  bool declaration_p = false;

  /* Initial implementation. It would have been better to check
     !ENDP() first. */
  //return instruction_continue_p(i) &&
  //!ENDP(statement_declarations(s));

  if(!ENDP(statement_declarations(s))) {
    instruction i = statement_instruction(s);
    if(instruction_call_p(i)) {
      call c = instruction_call(i);
      entity f = call_function(c);
      declaration_p = entity_continue_p(f);
    }
  }

  return declaration_p;
}

/* Check that all statements contained in statement list sl are a
   continue statements. */
bool continue_statements_p(list sl)
{
  bool continue_p = true;

  FOREACH(STATEMENT, s, sl) {
    if(!continue_statement_p(s)) {
      continue_p = false;
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
    return(true);
  }
  else
    return( ORDERING_STATEMENT(o1) < ORDERING_STATEMENT(o2));
}


/* @defgroup statement_2nd_access_test  Direct test of the instruction
   type of statement

   With these methods you can directly test the type of the instruction
   inside a statement

   @{
*/

/* Statement classes induced from instruction type
 *
 * Naming conflict between English, block_statement_p(), and NewGen convention
 * isntruction_loop_p()
 */

/* See also macro statement_block_p() */
bool statement_sequence_p(statement s)
{
  instruction i = statement_instruction(s);
  bool r = instruction_sequence_p(i);

  return r;
}

bool statement_test_p(statement s)
{
  return instruction_test_p(statement_instruction(s));
}


bool statement_loop_p(statement s)
{
  return instruction_loop_p(statement_instruction(s));
}

bool statement_whileloop_p(statement s)
{
  return instruction_whileloop_p(statement_instruction(s));
}

bool statement_goto_p(statement s)
{
  return instruction_goto_p(statement_instruction(s));
}

bool statement_call_p(statement s)
{
  return instruction_call_p(statement_instruction(s));
}

bool statement_unstructured_p(statement s)
{
  return instruction_unstructured_p(statement_instruction(s));
}

bool statement_forloop_p(statement s)
{
  return instruction_forloop_p(statement_instruction(s));
}

bool statement_multitest_p(statement s)
{
  return instruction_multitest_p(statement_instruction(s));
}

bool statement_expression_p(statement s)
{
  return instruction_expression_p(statement_instruction(s));
}


/* Test if a statement is empty. */
bool empty_statement_p(statement st)
{
  instruction i;

  return(unlabelled_statement_p(st) &&
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
  bool nop = false;
  instruction i = statement_instruction(s);

  if(instruction_block_p(i) && ENDP(instruction_block(i))) {
    pips_assert("No label!",  unlabelled_statement_p(s));
    pips_assert("No comments", empty_comments_p(statement_comments(s)));
    pips_assert("No statement number",
		statement_number(s) == STATEMENT_NUMBER_UNDEFINED);
    nop = true;
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

  if (!unlabelled_statement_p(st)
      || !empty_extensions_p(statement_extensions(st)))
    return false;

  if (continue_statement_p(st))
    return ENDP(statement_declarations(st));

  i = statement_instruction(st);
  if (instruction_block_p(i) && ENDP(statement_declarations(st))) {
    MAP(STATEMENT, s,
	{
	  if (!empty_statement_or_labelless_continue_p(s))
	    /* Well there is at least one possibly usefull thing... */
	    return false;
	},
	instruction_block(i));
    return true;
  }
  return false;
}


/* Return true if the statement is an empty instruction block or a
   continue or a recursive combination of above. */
bool empty_statement_or_continue_p(statement st)
{
  instruction i;

  if (continue_statement_p(st))
    return true;
  i = statement_instruction(st);
  if (instruction_block_p(i)) {
    FOREACH(STATEMENT, s,instruction_block(i))
	{
	  if (!empty_statement_or_continue_p(s))
	    /* Well there is at least one possibly usefull thing... */
	    return false;
	}
    return true;
  }
  return false;
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
    return false;

  if (!unlabelled_statement_p(st))
    return false;
  if (continue_statement_p(st) && ENDP(statement_declarations(st)))
    return true;

  i = statement_instruction(st);
  if (instruction_block_p(i)) {
    MAP(STATEMENT, s,
	{
	  if (!empty_statement_or_continue_without_comment_p(s))
	    return false;
	},
	instruction_block(i));
    /* Everything in the block are commentless continue or empty
       statements: */
    return true;
  }
  /* Everything else useful: */
  return false;
}


bool check_io_statement_p(statement s)
{
  bool check_io = false;
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


/* Duplicate statement comments

   @param comment is the comment to duplicate

   @return the duplicated comment. If it was an allocated string, it is
   strdup()ed
*/
string
comments_dup(string comment) {
  if (comment == NULL || string_undefined_p(comment))
    /* Not allocated, just the same: */
    return comment;
  else
    return strdup(comment);
}


/* Duplicate statement decls_text

   @param dt is the decls_text to duplicate

   @return the duplicated decls_text. If it was an allocated string, it is
   strdup()ed
*/
string
decls_text_dup(string dt) {
  if (dt == NULL)
    /* Not allocated, just the same: */
    return dt;
  else
    return strdup(dt);
}


statement make_assign_statement(expression l,
				expression r)
{
    return instruction_to_statement(make_assign_instruction(l, r));
}


/** @defgroup block_statement_constructors Block/sequence statement constructors
    @{
 */

/** Build a statement from a give instruction
*/
statement
instruction_to_statement(instruction instr) {
  return(make_statement(entity_empty_label(),
			STATEMENT_NUMBER_UNDEFINED,
			STATEMENT_ORDERING_UNDEFINED,
			empty_comments,
			instr,NIL,NULL,
			empty_extensions(), make_synchronization_none()));
}


/** Make a block statement from a list of statement
 */
statement make_block_statement(list body) {
  statement b;

  b = instruction_to_statement(make_instruction_block(body));
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


/* Move all the attributes from one statement to another one.

   Do not copy the instruction

   @param from is the statement we pick the attributes from

   @param to is the statement we copy the attributes into
*/
void move_statement_attributes(statement from,
			       statement to) {
  statement_label(to) = statement_label(from);
  statement_number(to) = statement_number(from);
  statement_ordering(to) = statement_ordering(from);
  statement_comments(to) = statement_comments(from);
  //statement_instruction(to) = statement_instruction(from);
  statement_declarations(to) = statement_declarations(from);
  statement_decls_text(to) = statement_decls_text(from);
  statement_extensions(to) = statement_extensions(from);
}


/* Copy all the attributes from one statement to another one.

   Do not copy the instruction

   @param from is the statement we pick the attributes from

   @param to is the statement we copy the attributes into
*/
void copy_statement_attributes(statement from,
			       statement to) {
  statement_label(to) = statement_label(from);
  statement_number(to) = statement_number(from);
  statement_ordering(to) = statement_ordering(from);
  statement_comments(to) = comments_dup(statement_comments(from));
  //statement_instruction(to) = statement_instruction(from);
  statement_declarations(to) = gen_copy_seq(statement_declarations(from));
  statement_decls_text(to) = decls_text_dup(statement_decls_text(from));
  statement_extensions(to) = copy_extensions(statement_extensions(from));
}


/** Build a statement sequence from a statement list

    @param[in] l is the statement list

    @return the statement if there was only 1 sattement in the list or
    build a block from the statement list. So, if the list was NIL, this
    makes an empty statement block.
*/
statement
make_statement_from_statement_list_or_empty_block(list l) {
  if (gen_length(l) == 1) {
    statement stat = STATEMENT(CAR(l));
    gen_free_list(l);
    return stat;
  }
  else
    return make_block_statement(l);
}


/** Build a statement sequence from a statement list

    @return :
    - statement_undefined if the statement list is NIL
    - the statement of a list with one statement, after discarding the list
    - a statement block with the statement list else.
*/
statement
make_statement_from_statement_list(list l) {
  if (l == NIL)
    return statement_undefined;
  else
    return make_statement_from_statement_list_or_empty_block(l);
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
  if (statement_sequence_p(stmt))
    /* It is already a block statement */
    return stmt;
  else
    return make_block_statement(CONS(STATEMENT, stmt, NIL));
}

/** @} */


statement make_return_statement(entity module)
{
    const char *module_name = entity_local_name(module);
    entity l = FindEntity(module_name, LABEL_PREFIX RETURN_LABEL_NAME);
    if (entity_undefined_p(l)) l = make_label(module_name, LABEL_PREFIX RETURN_LABEL_NAME);
    return make_call_statement(c_module_p(module)?C_RETURN_FUNCTION_NAME:RETURN_FUNCTION_NAME, NIL, l, empty_comments);
}

/*****************************************************************************

  Make a Fortran io statement : PRINT *,"message" following this format :
BEL_PREFIX
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
  entity lun = FindEntity(TOP_LEVEL_MODULE_NAME,
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


/* Make a declaration(s) statement

   To preserve declaration lines and comments, declaration statements are
   used

   @param l list of variables to add in the declaration

   @param sn is the statement number to use

   @param cs is the comment string to associate with the declaration
*/
statement make_declarations_statement(list l, int sn, string cs)
{
  statement ds = make_plain_continue_statement();
  statement_declarations(ds) = l;
  statement_number(ds) = sn;
  statement_comments(ds) = cs;
  return ds;
}

/* Make *one* declaration statement.
 */
statement make_declaration_statement(entity v, int sn, string cs)
{
  return make_declarations_statement(CONS(ENTITY, v, NIL), sn, cs);
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
		       empty_extensions (), make_synchronization_none());
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
		      empty_extensions (), make_synchronization_none());

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


/* @defgroup statement_2nd_access_method  Direct statement accessors to
   second level fields

   With these methods you can access directly to the inside of the
   instructions

   @{
*/

/* Get the sequence of a statement sequence */
sequence
statement_sequence(statement s)
{
  pips_assert("statement_sequence", statement_sequence_p(s));

  return instruction_sequence(statement_instruction(s));
}


/* Get the list of block statements of a statement sequence */
list
statement_block(statement s)
{
  pips_assert("statement_sequence", statement_sequence_p(s));

  return sequence_statements(statement_sequence(s));
}


/* Get the test of a statement */
test
statement_test(statement s)
{
  pips_assert("statement_test", statement_test_p(s));

  return instruction_test(statement_instruction(s));
}

/* returns the effective true branch of a test by skipping a possible
   sequence of one element. OK, this should be performed
   recursively... */
statement effective_test_true(test t)
{
  statement ets = test_true(t);

  if(statement_block_p(ets)) {
    list sl = statement_block(ets);
    if(gen_length(sl)==1) {
      ets = STATEMENT(CAR(sl));
    }
  }

  return ets;
}

/* Get the loop of a statement */
loop
statement_loop(statement s)
{
  pips_assert("statement_loop", statement_loop_p(s));

  return instruction_loop(statement_instruction(s));
}


/* Get the whileloop of a statement */
whileloop statement_whileloop(statement s)
{
  pips_assert("statement_whileloop", statement_whileloop_p(s));

  return instruction_whileloop(statement_instruction(s));
}


/* Get the goto of a statement

   @return the statement pointed to by the "goto"
*/
statement
statement_goto(statement s)
{
  pips_assert("statement_goto", statement_goto_p(s));

  return instruction_goto(statement_instruction(s));
}


/* Get the call of a statement */
call
statement_call(statement s)
{
  pips_assert("statement_call", statement_call_p(s));

  return instruction_call(statement_instruction(s));
}


/* Get the unstructured of a statement */
unstructured
statement_unstructured(statement s)
{
  pips_assert("statement_unstructured", statement_unstructured_p(s));

  return instruction_unstructured(statement_instruction(s));
}


/* Get the forloop of a statement */
forloop
statement_forloop(statement s)
{
  pips_assert("statement s is a forloop", statement_forloop_p(s));

  return instruction_forloop(statement_instruction(s));
}


/* Get the multitest of a statement */
multitest
statement_multitest(statement s)
{
  pips_assert("statement_multitest", statement_multitest_p(s));

  return instruction_multitest(statement_instruction(s));
}


/* Get the expression of a statement */
expression
statement_expression(statement s)
{
  pips_assert("statement_expression", statement_expression_p(s));

  return instruction_expression(statement_instruction(s));
}

/* @} */


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

    Print the statement according to the current PRETTYPRINT_LANGUAGE
    property

    See text_named_module() for improvements.
*/
void print_statement(statement s)
{
  if(statement_undefined_p(s))
    fprintf(stderr, "Undefined statement\n");
  // For debugging with gdb, dynamic type checking
  else if(statement_domain_number(s)!=statement_domain)
    (void) fprintf(stderr,"Arg. \"s\"is not a statement.\n");
  else {
    debug_on("TEXT_DEBUG_LEVEL");
    set_alternate_return_set();
    reset_label_counter();
    push_current_module_statement(s);
    text txt = text_statement(entity_undefined, 0, s, NIL);
    print_text(stderr, txt);
    free_text(txt);
    pop_current_module_statement();
    reset_alternate_return_set();
    debug_off();
  }
}


void print_statements(list sl)
{
  FOREACH(STATEMENT, s, sl) {
    print_statement(s);
  }
}


void print_statement_of_module(statement s, const char* mn)
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
    fprintf(stderr, "Statement undefined\n");
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
	    (instruction_to_statement
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
	    (instruction_to_statement
	     (make_instruction
	      (is_instruction_test,
	       make_test(newcond,
			 statfalse,
			 make_continue_statement(entity_undefined)))));
    }

    return(instruction_to_statement(make_instruction(is_instruction_test,
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
			     statement_extensions(s_old), make_synchronization_none());
    l_body = make_statement(entity_empty_label(),
			    STATEMENT_NUMBER_UNDEFINED,
			    STATEMENT_ORDERING_UNDEFINED,
			    empty_comments,
			    make_instruction_block(CONS(STATEMENT,state_l,NIL)),
			    NIL,NULL,
			    empty_extensions (), make_synchronization_none());

    return(l_body);
}


/* Does work neither with undefined statements nor with defined
   statements with undefined instructions. Returns a statement
   identity, its number and the breakdown of its ordering, as well as
   information about the instruction. */

string external_statement_identification(statement s)
{
  string buffer;
  instruction i = statement_instruction(s);
  string instrstring = instruction_identification(i);
  int so = statement_ordering(s);
  entity called = entity_undefined;

  if(same_string_p(instrstring, "CALL")) {
    called = call_function(instruction_call(i));
  }

  int n = asprintf(&buffer, "%td (%d, %d): %s %s\n",
                   statement_number(s),
                   ORDERING_NUMBER(so),
                   ORDERING_STATEMENT(so),
                   instrstring,
                   entity_undefined_p(called)? "" : module_local_name(called));

  pips_assert("asprintf ok", n!=-1);

  return buffer;
}

/* Like external_statement_identification(), but with internal
   information, the hexadecimal address of the statement

   Allocate a new string.
 */
string statement_identification(statement s)
{
  char * buffer;
  instruction i = statement_instruction(s);
  string instrstring = instruction_identification(i);
  int so = statement_ordering(s);
  entity called = entity_undefined;

  if(same_string_p(instrstring, "CALL")) {
    called = call_function(instruction_call(i));
  }

  int n = asprintf(&buffer, "%td (%d, %d) at %p: %s %s\n",
                   statement_number(s),
                   ORDERING_NUMBER(so),
                   ORDERING_STATEMENT(so),
                   s,
                   instrstring,
                   entity_undefined_p(called)? "" : module_local_name(called));

  pips_assert("asprintf ok", n!=-1);

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


/* Find the first non-empty comment of a statement, if any
 * returns a pointer to the comment if found,
 * pointer to a "string_undefined" otherwise */
char** find_first_statement_comment(statement s)
{
    static char * an_empty_comment = empty_comments;
    if (statement_block_p(s)) {
        FOREACH(STATEMENT, st, statement_block(s)){
            char** comment = find_first_statement_comment(st);
            if (!empty_comments_p(*comment))
                /* We've found it! */
                return comment;
        }
        /* No comment found: */
        return &an_empty_comment;
    }
    else
        /* Ok, plain statement: */
        return &statement_comments(s);
}

/* Find the first comment of a statement, if any.
 * returns a pointer to the comment if found, NULL otherwise
 *
 * Unfortunately empty_comments may be used to decorate a statement
 * although this makes the statement !gen_defined_p(). empty_comments
 * is also used as a return value to signal that non statement
 * legally carrying a comment has been found.
 *
 * The whole comment business should be cleaned up.
 */
static
char** find_first_comment(statement s)
{
    static char * an_empty_comment = empty_comments;
    if (statement_block_p(s)) {
        FOREACH(STATEMENT, st, statement_block(s)){
            char** comment = find_first_statement_comment(st);
            /* let's hope the parser generates an empty string as
               comment rather than empty_comments which is defined as
               empty_string */
            if (*comment!=empty_comments)
                /* We've found it! */
                return comment;
        }
        /* No comment found: */
        return &an_empty_comment;
    }
    else
        /* comment carrying statement: */
        return &statement_comments(s);
}


/* Put a comment on a statement in a safe way. That is it find the
   first non-block statement to attach it or insert a CONTINUE and put
   the statement on it. You should free the old one...

   The comment should have been malloc()'ed before.

   Return true on success, false else. */
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
		    return true;
	    },
	    sequence_statements(instruction_sequence(i)));
	/* Hmm, no good statement found to attach a comment: */
	return false;
    }
    else {
	/* Ok, it is a plain statement, we can put a comment on it: */
	statement_comments(s) = the_comments;
	return true;
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


    if (empty_comments_p(the_comments))
	/* Nothing to add... */
	return;

    char **old = find_first_statement_comment(s);
    if (empty_comments_p(*old))
        /* There is no comment yet: */
        put_a_comment_on_a_statement(s, strdup(the_comments));
    else {
        char * new = strdup(concatenate(*old, the_comments, NULL));
        free(*old);
        *old=empty_comments;
        put_a_comment_on_a_statement(s, new);
    }
}


/* Insert a comment string (if non empty) at the beginning of the
   comments of a statement.

   @param the_comments is strdup'ed in this function.
*/
void insert_comments_to_statement(statement s,
                                  const char* the_comments) {
  if (empty_comments_p(the_comments))
    /* Nothing to add... */
    return;

  char **old  = find_first_comment(s);
  if ( empty_comments_p(*old) ) {
    /* There are no comments yet: */
    put_a_comment_on_a_statement(s, strdup(the_comments));
  } else {
      char * new = strdup(concatenate(the_comments, *old, NULL));
      free(*old);
      *old=empty_comments;
      put_a_comment_on_a_statement(s, new);
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
      /* FI: why don't you try to keep them on the first statement,
	 just in case it has empty slots? */
	/* There are some informations we need to keep: just add a
	   CONTINUE to keep them: */
	list instructions;
	statement continue_s;
	const char* label_name =
	    entity_local_name(statement_label(s)) + sizeof(LABEL_PREFIX) -1;

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

	/* FI: at least in for loop to while loop conversion, the
	   extensions is moved down to the while loop. I'm not sure
	   this is the best source code location to fix the
	   problem. I do not know if the extensions have been reused
	   directly and so do not need to be freed here or not. In
	   fact, it is copied below and could be freed. */
	// FI: This leads to an inconsistency in gen_recurse() when the
	// while loop is reached, but only when the new_controlizer is
	// used; I cannot spend more time on this right now
	// See: transform_a_for_loop_into_a_while_loop(). it might be
	// safer to operate at the statement level with a test for
	// forloop statements rather than use ancestors
	extensions ext = statement_extensions(s);
	free_extensions(ext);
	statement_extensions(s) = make_extensions(NIL);

	instructions = CONS(STATEMENT, continue_s, instructions);
	instruction_block(statement_instruction(s)) = instructions;

	//pips_assert("The transformed statement s is consistent",
	//	    statement_consistent_p(s));
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
      pips_internal_error("Ill. tag %d for instruction",
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


/* Returns false is no syntactic control path exits s (i.e. even if true is returned
 * there might be no control path). Subroutines and
 * functions are assumed to always return to keep the analysis intraprocedural.
 * See the continuation library for more advanced precondition-based analyses.
 *
 * true is a safe default value.
 *
 * The function name is misleading: a RETURN statement does not return...
 * It should be called "statement_does_continue()"
 */
bool
statement_does_return(statement s)
{
    bool returns = true;
    instruction i = statement_instruction(s);
    test t = test_undefined;

    switch(instruction_tag(i)) {
    case is_instruction_sequence:
	MAPL(sts,
	     {
		 statement st = STATEMENT(CAR(sts)) ;
		 if (!statement_does_return(st)) {
		     returns = false;
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
	returns = true;
	break;
    case is_instruction_goto:
	/* returns = statement_does_return(instruction_goto(i)); */
	returns = false;
	break;
    case is_instruction_forloop:
      break;
    case is_instruction_expression:
      break;
    default:
	pips_internal_error("Ill. tag %d for instruction",
		   instruction_tag(i));
    }

    debug(8, "statement_does_return", "stmt %s, does return= %d\n",
	  statement_identification(s), returns);

    return returns;
}

bool
unstructured_does_return(unstructured u)
{
  bool returns = false;
  control entry = unstructured_control(u);
  control exit = unstructured_exit(u);
  list nodes = NIL;

  FORWARD_CONTROL_MAP(c, {
      returns = returns || (c==exit);
  }, entry, nodes);
  gen_free_list(nodes);

  return returns;
}

void
gather_and_remove_all_format_statements_rewrite(statement s,list *all_formats)
{
    instruction i = statement_instruction(s);
    if (instruction_format_p(i)) {
        /* Put the instruction with the statement attributes in
           new_format. */
        statement new_format = instruction_to_statement(i);
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
    formats=gen_nreverse(formats);
    FOREACH(STATEMENT,f,formats)
        insert_statement(s,f,true);
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
    FOREACH(STATEMENT,f,formats)
        insert_statement(s,f,false);
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


/* Return true only if there is a FORMAT inside the statement:

   @addtogroup statement_predicate
*/
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

    return true;
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
 * If statement s is a sequence, simply insert s1 at the beginning
 * or at the end of the sequence s.
 *
 * If not, create a new statement s2 with s's fields and update
 * s as a sequence with no comments and undefined number and ordering.
 * The sequence is either "s1;s2" if "before" is true or "s2;s1" else.
 *
 *
 * ATTENTION !!! : this version is not for unstructured case and this
 * is not asserted.
 *
 * s cannot be a declaration statement, because the insertion scheme
 * used would modify its scope. Also s1 cannot be a declaration if s
 * is not a sequence. And if s is a sequence
 * add_declaration_statement() should be used instead to insert the
 * new declaration at the best possible place.
 */
static void generic_insert_statement(statement s,
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
      statement s2 = make_statement(
              statement_label(s),
              statement_number(s),
              statement_ordering(s),
              statement_comments(s),
              statement_instruction(s),
              statement_declarations(s),
              statement_decls_text(s),
              statement_extensions(s), make_synchronization_none()
              );
      /* no duplication */
      statement_label(s)=entity_empty_label();
      statement_number(s)=STATEMENT_NUMBER_UNDEFINED;
      statement_ordering(s)=STATEMENT_ORDERING_UNDEFINED;
      statement_comments(s)=string_undefined;
      statement_instruction(s)=instruction_undefined;
      statement_declarations(s)=NIL;
      statement_decls_text(s)=NULL;
      statement_extensions(s) = empty_extensions();

      if (before)
	ls = CONS(STATEMENT,s1,CONS(STATEMENT,s2,NIL));
      else
	ls = CONS(STATEMENT,s2,CONS(STATEMENT,s1,NIL));

      update_statement_instruction(s, make_instruction_sequence(make_sequence(ls)));
    }
}

/* This is the normal entry point. See previous function for comments. */
void insert_statement(statement s, statement s1, bool before)
{
  pips_assert("Neither s nor s1 are declaration statements",
	      !declaration_statement_p(s)
	      && !declaration_statement_p(s1));
  generic_insert_statement(s, s1, before);
}

/* Break the IR consistency or, at the very least, do not insert new
   declarations at the usual place, i.e. at the end of the already
   existing declarations. */
void insert_statement_no_matter_what(statement s, statement s1, bool before)
{
  generic_insert_statement(s, s1, before);
}

void append_statement_to_block_statement(statement b, statement s)
{
  pips_assert("b is a block statement", statement_block_p(b));
  list sl = statement_block(b);
  sequence_statements(statement_sequence(b)) =
    gen_nconc(sl, CONS(STATEMENT, s, NIL));
}

// there should be a space at the beginning of the string
#define PIPS_DECLARATION_COMMENT "PIPS generated variable\n"
static string
default_generated_variable_commenter(__attribute__((unused))entity e)
{
  return strdup(PIPS_DECLARATION_COMMENT);
}

// hmmm... why not use the "stack" type, instead of betting?
#define MAX_COMMENTERS 8

/* commenters are function used to add comments to pips-created variables
 * they are handled as a limited size stack
 * all commenters are supposed to return allocated data
 */
typedef string (*generated_variable_commenter)(entity);
static generated_variable_commenter
generated_variable_commenters[MAX_COMMENTERS] = {
  [0]=default_generated_variable_commenter /* c99 inside :) */
};
static size_t nb_commenters=1;

void push_generated_variable_commenter(string (*commenter)(entity))
{
  pips_assert("not exceeding stack commenters stack limited size\n",
              nb_commenters<MAX_COMMENTERS);
  generated_variable_commenters[nb_commenters++]=commenter;
}

void pop_generated_variable_commenter()
{
    pips_assert("not removing default commenter",nb_commenters!=1);
    --nb_commenters;
}

string generated_variable_comment(entity e)
{
    return generated_variable_commenters[nb_commenters-1](e);;
}



/* Add a new declaration statement
 *
 * Declarations are not only lists of entities at the block level and
 * in the symbol table, but also declaration statement to
 * carry the line number, comments, and the declarations local to the
 * declaration statement. This function is low-level and does not
 * maintain the consistency of the PIPS internal representation.
 *
 * @param s Statement s must be a sequence. A new declaration statement is
 * generated to declare e. This statement is inserted after the
 * existing declaration statements in s, or, if no declaration statement
 * is present, at the beginning of the sequence in s.
 *
 * @param e Variable to declare.
 * The declaration of e at s level is not checked. The existence of a
 * previous declaration statement for e is not checked either. The
 * caller is responsible for the consistency management between the
 * declaration statements in s and the list of declaration at s level.
 *
 * @return statement s modified by side-effects.
 *
 * For the time being, a declaration statement is a continue
 * statement.
 * Beware that the possible dependencies between the new declaration
 * and existing declarations are not checked.
 */
static statement generic_add_declaration_statement(statement s, entity e, bool before_p)
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

	if (before_p)
	  cl = sl;
	else
	  {
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
	  }

        /* Do we have previous declarations to skip? */
        if(!ENDP(pl)) {
            /* SG: if CAR(pl) has same comment and same type as ds, merge them */
            statement spl = STATEMENT(CAR(pl));
            entity ecar = ENTITY(CAR(statement_declarations(spl)));
            if( comments_equal_p(statement_comments(spl),comment) &&
                    !basic_undefined_p(entity_basic(e)) && !basic_undefined_p(entity_basic(ecar)) &&
                    basic_equal_p(entity_basic(e),entity_basic(ecar)) &&
                    qualifiers_equal_p(entity_qualifiers(e),entity_qualifiers(ecar))
              )
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
	    if (before_p && !ENDP(sl) && declaration_statement_p(STATEMENT(CAR(sl))))
	      {
		statement ssl = STATEMENT(CAR(sl));
		entity ecar = ENTITY(CAR(statement_declarations(ssl)));
		if( comments_equal_p(statement_comments(ssl),comment) &&
                    !basic_undefined_p(entity_basic(e)) && !basic_undefined_p(entity_basic(ecar)) &&
                    basic_equal_p(entity_basic(e),entity_basic(ecar)) &&
                    qualifiers_equal_p(entity_qualifiers(e),entity_qualifiers(ecar))
		    )
		{
		  free_statement(ds);
		  statement_declarations(ssl)= CONS(ENTITY,e,statement_declarations(ssl));
		  nsl=sl;
		}
		else
		  nsl = CONS(STATEMENT, ds, sl);
	      }
	    else
	      nsl = CONS(STATEMENT, ds, sl);
        }

        instruction_block(statement_instruction(s)) = nsl;
    }
    else
    {
        pips_internal_error("can only add declarations to statement blocks");
    }

    ifdebug(8) {
        pips_debug(8, "Statement after declaration insertion:\n");
        print_statement(s);
    }

    return s;
}

statement add_declaration_statement(statement s, entity e)
{
  return generic_add_declaration_statement(s, e, false);
}

statement add_declaration_statement_at_beginning(statement s, entity e)
{
  return generic_add_declaration_statement(s, e, true);
}


/* s is assumed to be a block statement and its declarations field is
   assumed to be correct, but not necessarily the declaration
   statements within the block s. This function checks that no variable is
   declared by a declaration statement within the block but not
   declared at the block level. Also, if a variable is declared at the
   block level but not by a declaration statement, a new declaration
   statement is added. */
void fix_block_statement_declarations(statement s)
{
  list bdvl = gen_copy_seq(statement_declarations(s)); // block declarations

  pips_assert("s is a block statement", statement_block_p(s));
  pips_assert("No multiple declarations in declarations of s", gen_once_p(bdvl));

  if(!ENDP(bdvl)) {
    list sl = statement_block(s);
    list edvl = statements_to_direct_declarations(sl); // effective declarations
    list mdvl = gen_copy_seq(statement_declarations(s)); // block declarations

    // List of variables declared within the block but not at the
    // block level
    gen_list_and_not(&mdvl, edvl); // missing declarations

    // Add the missing declaration statements.
    // Might be better to add only one statement for all missing variables...
    FOREACH(ENTITY, v, mdvl) {
      add_declaration_statement(s, v);
    }
    gen_free_list(mdvl);

    // List of variables declared within the block but not at the
    // block level
    gen_list_and_not(&edvl, bdvl);

    pips_assert("edvl-bdvl=empty set", ENDP(edvl));
    gen_free_list(edvl);
  }
  gen_free_list(bdvl);
}

/* Declarations are not only lists of entities, but also statement to
   carry the line number, comments,... For the time begin, a
   declaration statement is a continue statement. */
statement remove_declaration_statement(statement s, entity e)
{
    if(statement_block_p(s)) {
        list sl = statement_block(s); //statement list
        list cl = list_undefined; // current statement list

        /* Look for the last declaration: it is pointed to by pl */
        for(cl=sl; !ENDP(cl); POP(cl)) {
            statement cs = STATEMENT(CAR(cl));
            if(declaration_statement_p(cs)) {
              // We don't use FOREACH because we aim to delete an element while iterating
              list prev=NULL;
              for(list current = statement_declarations(cs);
                  !ENDP(current);
                  POP(current)) {
                if(e == ENTITY(CAR(current))) {
                  pips_debug(0,"Removing %s\n",entity_name(e));
                  // For replacing it, we have first to remove the old one from the list
                  if(prev == NULL) {
                    statement_declarations(cs) = CDR(current);
                  } else {
                    CDR(prev) = CDR(current);
                  }
                  free(current);
                  break;
                }
                prev=current; // Save current iterator, because we might remove the next one
              }
            }
        }
    }
    else
    {
        pips_internal_error("can only add declarations to statement blocks");
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
                (!unlabelled_statement_p(s) && !entity_return_label_p(statement_label(s))) /*SG:note the special return-label case */
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
        free_instruction(statement_instruction(s));
        statement_instruction(s) = i;
        /* SG: if the old statement had declarations, they are removed
         * maybe we should regenerate the new one if any to keep global coherency ?*/
        gen_free_list(statement_declarations(s));
        statement_declarations(s)=NIL;
        free_extensions(statement_extensions(s));
        statement_extensions(s)=empty_extensions();
        if(entity_return_label_p(statement_label(s)) && !return_statement_p(s))
            statement_label(s)=entity_empty_label();
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
  bool found = false;

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
    found = true;
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

static bool collect_labels(statement s, list * pll)
{
  entity l = statement_label(s);
  if(!entity_empty_label_p(l))
    *pll = CONS(ENTITY, l, *pll);
  return true;
}


/* Look for non-empty labels appearing directly or indirectly and allocate a
   label list.

   The code is assumed correct. Usage of labels, for instance in
   Fortran IOs, are not checked. Only the statement label field is checked.
 */
list statement_to_labels(statement s)
{
  list ll = NIL; /* Label List */

  gen_context_recurse(s, (void *) &ll, statement_domain,
		      collect_labels, gen_null);

  /* To have labels in order of appearance */
  ll = gen_nreverse(ll);

  return ll;
}


/* Make sure that s and all its substatements are defined

   @addtogroup statement_predicate
*/
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


/*
   @addtogroup statement_predicate
*/
bool all_statements_defined_p(statement s)
{
  bool undefined_p = false;

  gen_context_recurse(s, (void *) &undefined_p ,statement_domain,
		      undefined_statement_found_p, gen_null);
  return !undefined_p;
}


typedef struct {
  list statement_to_all_included_declarations;
  set cache;
} add_statement_declarations_t;

/* Add the declarations of a statement to a list if not already here

   This function is indeed to be used by statement_to_declarations() and
   instruction_to_declarations() but not by its own.

   @return true (to go on diving in a gen_recurse())
*/
static bool add_statement_declarations(statement s,  add_statement_declarations_t* ctxt)
{
    if(declaration_statement_p(s))
    {
        FOREACH(ENTITY,e,statement_declarations(s))
            if(!set_belong_p(ctxt->cache,e)) {
                ctxt->statement_to_all_included_declarations=CONS(ENTITY,e,ctxt->statement_to_all_included_declarations);
                set_add_element(ctxt->cache,ctxt->cache,e);
            }
        return false;/* no declaration in a declaration!*/
    }
  return true;
}

/* Get a list of all variables declared recursively within a statement.
 * It works for any newgen type, not only statements.
 * Warning: the list must be freed !
 */
list statement_to_declarations(void* s)
{
  add_statement_declarations_t ctxt = { NIL, set_make(set_pointer) };
  gen_context_multi_recurse(s, &ctxt,
          call_domain,gen_false,gen_null,
          statement_domain, add_statement_declarations, gen_null,
          NULL);
  set_free(ctxt.cache);
  return ctxt.statement_to_all_included_declarations;
}

/* Returns the declarations contained in a list of statement. */
list statements_to_declarations(list sl)
{
    list  tail = NIL;    //< avoid a costly gen_nconc
    list  head = NIL;  //< by managing the tail by end
    FOREACH(STATEMENT,st,sl) {
        list s2d = statement_to_declarations(st);
        if(ENDP(head)) head=tail=s2d;
        else CDR(tail)=s2d;
        tail = gen_last(tail);
    }
    return head;
}

/* Get a list of all variables declared recursively within an instruction */
list instruction_to_declarations(instruction i)
{
  return statement_to_declarations(i);
}



static list internal_statement_to_direct_declarations(statement st);

static list unstructured_to_direct_declarations(unstructured u)
{
  list blocks = NIL; // list of statements
  control c_in = unstructured_control(u); // entry point of u
  list dl = NIL; // declaration list

  CONTROL_MAP(c,
	      {
		statement s = control_statement(c);
		list sdl = internal_statement_to_direct_declarations(s);
		dl = gen_nconc(dl, sdl);
	      },
	      c_in,
	      blocks);

  gen_free_list(blocks);
  return dl;
}

/* No recursive descent */
static list internal_statement_to_direct_declarations(statement st)
{
  list sdl = NIL;

  if(declaration_statement_p(st)) {
    sdl = gen_copy_seq(statement_declarations(st));
  }
  else if(statement_unstructured_p(st)) {
    unstructured u = statement_unstructured(st);
    sdl = unstructured_to_direct_declarations(u);
  }
  return sdl;
}

/* Returns the declarations contained directly in the declaration
   statements of a list of statements.

   @param sl
   List of statements

   @return a newly allocated list of entities appearing in the statement
   declarations of the list. No recursive descent in loops or tests or
   sequences because only variables of the current scope are
   returned. Recursive descent in unstructured statements because
   their statements are in the same scope as the statements in list sl
*/
list statements_to_direct_declarations(list sl)
{
  list  dl = NIL;
  FOREACH(STATEMENT,st,sl) {
    list sdl = internal_statement_to_direct_declarations(st);
    dl = gen_nconc(dl, sdl);
  }

  return dl;
}

/* Returns the declarations contained directly in a statement s

   @param s
   any kind of statement

   @return a newly allocated list of entities appearing statement s.
   If s is a sequence, look for all declaration statements in the
   statement list of s.

   No recursive descent in loops or tests or sequences because only
   variables of the current scope are returned. Recursive descent in
   unstructured statements because their statements are in the same
   scope as the statements in list sl
*/
list statement_to_direct_declarations(statement s)
{
  list dl = NIL;
  if(statement_block_p(s)) {
    list sl = statement_block(s);
    dl = statements_to_direct_declarations(sl);
  }
  else
    dl = internal_statement_to_direct_declarations(s);
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
  bool result= true;
  if (reference_undefined_p(first_reference_to_v)) {
    entity rv = reference_variable(r);
    if (rv == variable_searched) {
      first_reference_to_v = r;
      result = false;
    }
  }
  else
    result = false;

  return result;
}


static bool declarations_first_reference_to_v_p(statement st)
{
  bool result = true;
  if (reference_undefined_p(first_reference_to_v))
    {
      if (declaration_statement_p(st))
	{
	  FOREACH(ENTITY, decl, statement_declarations(st))
	    {
	      value init_val = entity_initial(decl);
	      if (! value_undefined_p(init_val))
		{
		  gen_recurse(init_val,  reference_domain, first_reference_to_v_p, gen_null);
		  if (!reference_undefined_p(first_reference_to_v))
		    {
		      result = false;
		      break;
		    }
		}
	    }
	}
    }
  else
    result = false;

  return result;
}

reference find_reference_to_variable(statement s, entity v)
{
  reference r = reference_undefined;
  first_reference_to_v = reference_undefined;
  variable_searched = v;
  gen_multi_recurse(s,
		    statement_domain, declarations_first_reference_to_v_p, gen_null,
		    reference_domain, first_reference_to_v_p, gen_null,
		    NULL);
  r = copy_reference(first_reference_to_v);
  first_reference_to_v = reference_undefined;
  variable_searched = entity_undefined;
  return r;
}

static int reference_count = -1;

/* Count static references */
static bool count_static_references_to_v_p(reference r)
{
  bool result= true;
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
  bool result= true;
  entity rv = reference_variable(r);
  if (rv == variable_searched) {
    /* 10: arbitrary value for references nested in at least one loop */
    reference_count += (loop_depth > 0 ? 10 : 1 );
  }
  return result;
}

static bool declarations_count_references_to_v_p(statement st)
{
  if (declaration_statement_p(st))
    {
      FOREACH(ENTITY, decl, statement_declarations(st))
	{
	  value init_val = entity_initial(decl);
	  if (! value_undefined_p(init_val))
	    {
	      gen_recurse(init_val,  reference_domain, count_references_to_v_p, gen_null);
	    }
	}
    }
  return true;
}


/* This function checks reference to proper elements, not slices */
static bool count_element_references_to_v_p(reference r)
{
  bool result= true;
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

static bool declarations_count_element_references_to_v_p(statement st)
{
  if (declaration_statement_p(st))
    {
      FOREACH(ENTITY, decl, statement_declarations(st))
	{
	  value init_val = entity_initial(decl);
	  if (! value_undefined_p(init_val))
	    {
	      gen_recurse(init_val,  reference_domain, count_element_references_to_v_p, gen_null);
	    }
	}
    }
  return true;
}

static bool count_loop_in(loop __attribute__ ((unused)) l)
{
  loop_depth++;
  return true;
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
		    statement_domain, declarations_count_references_to_v_p, gen_null,
		    reference_domain, count_references_to_v_p, gen_null, NULL);
  variable_searched = entity_undefined;
  return reference_count;
}
int count_references_to_variable_element(statement s, entity v)
{
  reference_count = 0;
  loop_depth = 0;
  variable_searched = v;
  gen_multi_recurse(s, statement_domain, declarations_count_element_references_to_v_p, gen_null,
		    loop_domain, count_loop_in, count_loop_out,
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
 * NOT INTENDED to generate entity name declared at particular block level :
 * The block scope depends on the number of different blocks at the same depth !
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





/**
 * @name statement finders
 * find statements with particular constraints
 * @{ */

/**
 * structure used by find_statements_with_label_walker
 */
struct fswl {
    statement st; ///< statement matching condition
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
      p->st=s;
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
 * @return statement found or statement_undefined
 */
statement find_statement_from_label(statement s, entity label)
{
  struct fswl p = {  statement_undefined , label };
  gen_context_recurse(s,&p,statement_domain,find_statements_with_label_walker,gen_null);
  return p.st;
}
statement find_statement_from_label_name(statement s, const char *module_name ,const char * label_name)
{
    entity label = find_label_entity(module_name,label_name);
    if(entity_undefined_p(label))
        return statement_undefined;
    else
        return find_statement_from_label(s,label);
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
    const char*begin;
};


/*
   @addtogroup statement_predicate
*/

/* Test if a statement has some pragma

   @param s is the statement to test

   @return true if a statement has a pragma
*/
bool statement_with_pragma_p(statement s) {
  list exs = extensions_extension(statement_extensions(s));
  FOREACH(EXTENSION, ex, exs) {
    if(extension_pragma_p(ex)) {
      return true;
    }
  }
  return false;
}


/* Get the extension of a statement with pragma begining with a prefix

   @param s is the statement to work on

   @param begin is the prefix a pragma has to begin with to be selected

   @return the first extension matching the pragma
*/
extension get_extension_from_statement_with_pragma(statement s, const char* seed)
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
    if (get_extension_from_statement_with_pragma(s,p->begin))
    {
        p->l=CONS(STATEMENT,s,p->l);
        gen_recurse_stop(NULL);
    }
    return true;
}


/* Get a list of statements with pragma begining with a prefix

   @param s is the statement to start to recurse

   @param begin is the prefix a pragma has to begin with to be selected

   @return a list of statement
*/
list find_statements_with_pragma(statement s, const char* begin)
{
  struct fswp p = { NIL, begin };
  gen_context_recurse(s,&p,statement_domain,find_statements_with_pragma_walker,gen_null);
  return gen_nreverse(p.l);
}


static bool look_for_user_call(call c, bool * user_call_p)
{
  entity f = call_function(c);
  bool go_on_p = true;

  if(value_code_p(entity_initial(f))) {
    * user_call_p = true;
    go_on_p = false;
  }
  return go_on_p;
}

/* Check if a statement s contains a call to a user-defined
   function. This may be useful because PIPS does not contain a
   control effect analysis and because a user-defined function can
   hide a control effect such as exit(), abort() or STOP.

   @addtogroup statement_predicate
*/
bool statement_contains_user_call_p(statement s)
{
  bool user_call_p = false;
  gen_context_recurse(s, &user_call_p, call_domain, look_for_user_call, gen_null);
  return user_call_p;
}

static bool look_for_control_effects(call c, bool * control_effect_p)
{
  entity f = call_function(c);
  bool go_on_p = true;
  value fv = entity_initial(f);

  if(value_code_p(fv)) {
    * control_effect_p = true;
    go_on_p = false;
  }
  else if(value_intrinsic_p(fv)) {
    if(ENTITY_EXIT_SYSTEM_P(f)
       || ENTITY_ABORT_SYSTEM_P(f)
       || ENTITY_C_RETURN_P(f)
       || ENTITY_RETURN_P(f)
       || ENTITY_ASSERT_SYSTEM_P(f)
       || ENTITY_ASSERT_FAIL_SYSTEM_P(f)) {
      * control_effect_p = true;
      go_on_p = false;
    }
  }
  return go_on_p;
}

/* Check if a statement s contains a call to a user-defined function
   or to an intrinsic with control effects. This may be useful because
   PIPS does not contain a control effect analysis and because a
   user-defined function can hide a control effect such as exit(),
   abort() or STOP.

   @addtogroup statement_predicate
*/
bool statement_may_have_control_effects_p(statement s)
{
  bool control_effect_p = false;

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


static bool look_for_exiting_intrinsic_calls(call c, bool * control_effect_p)
{
  entity f = call_function(c);
  bool go_on_p = true;
  value fv = entity_initial(f);

  if(value_intrinsic_p(fv)) {
    if(ENTITY_EXIT_SYSTEM_P(f)
       || ENTITY_ABORT_SYSTEM_P(f)
       || ENTITY_C_RETURN_P(f)
       || ENTITY_RETURN_P(f)
       || ENTITY_ASSERT_SYSTEM_P(f)
       || ENTITY_ASSERT_FAIL_SYSTEM_P(f)) {
      * control_effect_p = true;
      go_on_p = false;
    }
  }
  return go_on_p;
}

bool statement_may_contain_exiting_intrinsic_call_p(statement s)
{
  bool control_effect_p = false;

  if(!control_effect_p)
      gen_context_recurse(s, &control_effect_p, call_domain, look_for_exiting_intrinsic_calls, gen_null);

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

/* Look if a statement is another one

   @param s is the potentially inside statement

   @param st is the outside statement

   @return true is @p s is inside @p st

   @addtogroup statement_predicate
*/
bool statement_in_statement_p(statement s, statement st)
{
    struct sb sb = { s,false };
    gen_context_recurse(st,&sb,statement_domain, statement_in_statement_walker,gen_null);
    return sb.res;
}


/* Look if at least one statement of a list of statements is in another
   one

   @param l is the list of statements

   @param st is the outside statement

   @return true is at least one statement of @p l is inside @p st

   @addtogroup statement_predicate
*/
bool statement_in_statements_p(statement s, list l)
{
    FOREACH(STATEMENT,st,l)
        if(statement_in_statement_p(s,st)) return true;
    return false;
}

/* A simplified version of find_last_statement() located in
 * prettyprint.c and designed to be used within the prettyprinter
 */
statement last_statement(statement s)
{
    statement last = statement_undefined;

    pips_assert("statement is defined", !statement_undefined_p(s));

    if(statement_sequence_p(s)) {
	list ls = instruction_block(statement_instruction(s));

	last = (ENDP(ls)? statement_undefined :
		last_statement(STATEMENT(CAR(gen_last(ls)))));
    }
    else if(statement_unstructured_p(s)) {
	unstructured u = statement_unstructured(s);
	list trail = unstructured_to_trail(u);

	last = control_statement(CONTROL(CAR(trail)));

	gen_free_list(trail);
    }
    else if(statement_call_p(s)) {
	/* Hopefully it is a return statement.
	 * Since the semantics of STOP is ignored by the parser, a
	 * final STOp should be followed by a RETURN.
	 */
	last = s;
    }
    else if(statement_goto_p(s))
      last = s;
    else if(statement_expression_p(s))
      last = s;
    else if(statement_loop_p(s))
      last = s;
    else if(statement_whileloop_p(s))
      last = s;
    else if(statement_forloop_p(s))
      last = s;

    return last;
}

/* That's all folks */

/** @} */

/* purge a statement from its extensions */
void statement_remove_extensions(statement s) {
    extensions ext = statement_extensions(s);
    gen_full_free_list(extensions_extension(ext));
    extensions_extension(ext)=NIL;
}
/**
 * @brief remove the label of a statement if the statement is not
 * unstructured. labels on fortran loops and Fortran return are also
 * preserved
 *
 * @param s statement considered
 */
void statement_remove_useless_label(statement s)
{
    instruction i = statement_instruction(s);
    if(!instruction_unstructured_p(i) &&
            c_module_p(get_current_module_entity())
      ) {
        // under pyps, do not remove loop label like this */
        if(!(get_bool_property("PYPS") && instruction_loop_p(i))) {
            if( !entity_empty_label_p( statement_label(s)) && !fortran_return_statement_p(s) ) {
                /* SG should free_entity ? */
                statement_label(s)=entity_empty_label();

                /* OK but guarded by previous test */
                if( instruction_loop_p(i) )
                    loop_label(instruction_loop(i))=entity_empty_label();
                if( instruction_whileloop_p(i) )
                    whileloop_label(instruction_whileloop(i))=entity_empty_label();
            }
        }
    }
}


/* return true if s is enclosed in stmt*/
bool belong_to_statement(statement stmt, statement s, bool found_p)
{
  if(!found_p){
    if(statement_ordering(s) == statement_ordering(stmt))
      return true;
    else
      {
	instruction inst = statement_instruction(stmt);
	switch(instruction_tag(inst))  {
	case is_instruction_block: {
	  bool bs = found_p;
	  MAPL( stmt_ptr,
		{
		    statement local_stmt = STATEMENT(CAR( stmt_ptr ));
		    bs = bs || belong_to_statement(local_stmt, s, found_p);
		},
		instruction_block( inst ) );
	  return bs;
	}
	case is_instruction_test :{
	  test t = instruction_test(inst);
	  bool bt = found_p || belong_to_statement(test_true(t), s, found_p);
	  return bt || belong_to_statement(test_false(t), s, found_p);
	  break;
	}
	case is_instruction_loop : {
	  loop l = statement_loop(stmt);
	  statement body = loop_body(l);
	  return found_p || belong_to_statement(body, s, found_p);
	  break;
	}
	case is_instruction_forloop :{
	  forloop l = statement_forloop(stmt);
	  statement body = forloop_body(l);
	  return found_p || belong_to_statement(body, s, found_p);
	  break;
	}
	case is_instruction_whileloop : {
	  whileloop l = statement_whileloop(stmt);
	  statement body = whileloop_body(l);
	  return found_p || belong_to_statement(body, s, found_p);
	  break;
	}
	case is_instruction_call:
	  return found_p || false;
	  break;
	default:
	  break;
	}
      }
  }
  else return true;
  return found_p;
}

