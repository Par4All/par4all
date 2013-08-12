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
/* Handling of C return statements. Unlike Fortran return statements,
   C return statements carry the value returned by functions. In some
   sense, they have a continuation since they return to the caller,
   but there is no local continuation. So C return cannot be simply
   replaced by goto statements. This only preserves the control
   semantics. An additional variable, the return value, must be
   declared and used to collect information about the different values
   that are returned. This translation can be removed when there is
   only one return at the end of the function body. Note that pass
   restructure_control may be useful for procedure with several
   returns because unspaghettify, called by the controlizer, is not
   strong enough.
 *
 * Francois Irigoin
 */

#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "properties.h"

#include "c_syntax.h"

//static entity C_end_label = entity_undefined;
//static char *C_end_label_local_name = RETURN_LABEL_NAME;

entity Generate_C_ReturnLabel(entity m)
{
  entity l = make_label(entity_module_name(m), LABEL_PREFIX RETURN_LABEL_NAME);
  return l;
}

/* Generate a unique call statement to RETURN per module */

/* Saved to be reused each time a return must be converted into a
   goto b ythe parser */
static statement C_return_statement = statement_undefined;
/* Saved to optimize the internal representation instead of relying
   on the parser and/or on control_restructure: is it possible to
   optimize? what is the last expression list returned? Which
   statement becomes redundant? */
static int number_of_return_statements = -1;
static list last_returned_value = list_undefined;
static statement last_return_value_assignment = statement_undefined;
/* It is saved because it is hard to retrieve from within
   actual_c_parser() once the aprsing is
   over. get_current_module_entity() seems to return an undefined
   entity */
static entity return_value_entity = entity_undefined;
/* get_current_module_entity() is reset too early by the parser */
static entity return_current_module = entity_undefined;

statement Generate_C_ReturnStatement()
{
  instruction i = instruction_undefined;
  entity m = get_current_module_entity();
  entity l = Generate_C_ReturnLabel(m);
  statement s = statement_undefined;
  type mt = entity_type(m);
  functional f = type_functional(mt);
  type r = functional_result(f);

  if(type_void_p(r))
    i = make_call_instruction(entity_intrinsic(C_RETURN_FUNCTION_NAME), NIL);
  else {
    entity rv = function_to_return_value(m);
    expression arg = entity_to_expression(rv);
    return_value_entity = rv;
    return_current_module = m;
    i = make_call_instruction(entity_intrinsic(C_RETURN_FUNCTION_NAME),
			      CONS(EXPRESSION, arg, NIL));
  }

  // FI: I'd like to add a return label to this special statement...
  s = instruction_to_statement(i);
  statement_label(s) = l;
  return s;
}

/* The return statement must be reset when it is used by the parser to
 * add the return statement to the function body or when a parser
 * error is encountered.
 */
void Reset_C_ReturnStatement()
{
  C_return_statement = statement_undefined;
  number_of_return_statements = -1;
  last_returned_value = list_undefined;
  last_return_value_assignment = statement_undefined;
  return_value_entity = entity_undefined;
  return_current_module = entity_undefined;
  //c_end_label = entity_undefined;
}

/* This function is used to generate all goto's towards the unique
 * return used to C replace return statement and to insert this unique
 * return at the end of the current function's body.
 */
statement Get_C_ReturnStatement()
{
  if(statement_undefined_p(C_return_statement)) {
    pips_assert("No return statement yet\n",  number_of_return_statements==-1);
    number_of_return_statements = 0;
    C_return_statement = Generate_C_ReturnStatement();
  }
  number_of_return_statements++;
  return C_return_statement;
}

/* This function creates a goto instruction to label end_label. This is
 * done to eliminate return statements.
 *
 * Note: I was afraid the mouse trap would not work to analyze
 * multiple procedures but there is no problem. I guess that MakeGotoInst()
 * generates the proper label entity regardless of end_label. FI.
 */

/* Generates the internal representation of a C return statement. If
 * e is an undefined expression, a goto returnlabel is
 * generated. Else, e is assigned to the return value of the current
 * function, a goto is generated and both are returned within a
 * block.
 */
statement C_MakeReturnStatement(list el, int ln, string c)
{
  instruction inst = instruction_undefined;
  instruction ainst = instruction_undefined;
  instruction ginst = instruction_undefined;
  statement s = statement_undefined;

  last_returned_value = el;

  if(!ENDP(el)) {
    /* Assign the expression to the return value of the current
       function */
    entity f = get_current_module_entity();
    entity rv = function_to_return_value(f);
    if(ENDP(CDR(el))) {
      expression e = EXPRESSION(CAR(el));
      ainst = make_assign_instruction(entity_to_expression(rv), e);
    }
    else {
      pips_internal_error("This case is not implemented yet.");
    }
  }

  ginst = make_instruction_goto(Get_C_ReturnStatement());

  if(instruction_undefined_p(ainst)) {
    inst = ginst;
    s = make_statement(entity_empty_label(),
		       ln,
		       STATEMENT_ORDERING_UNDEFINED,
		       c,
		       inst,
		       NIL,
		       string_undefined,
		       empty_extensions(), make_synchronization_none());
  }
  else {
    statement as = make_statement(entity_empty_label(),
		     ln,
		     STATEMENT_ORDERING_UNDEFINED,
		     c,
		     ainst,
		     NIL,
		     string_undefined,
		     empty_extensions(), make_synchronization_none());
    statement gs = instruction_to_statement(ginst);
    list sl = CONS(STATEMENT, as, CONS(STATEMENT, gs, NIL));
    last_return_value_assignment = as;
    inst = make_instruction_block(sl);
    s = instruction_to_statement(inst);
  }

  pips_assert("inst is consistent", instruction_consistent_p(inst));
  pips_assert("s is consistent", statement_consistent_p(s));


  return s;
}

int GetReturnNumber()
{
  return number_of_return_statements;
}

/* When return statements have been encountered, each of them has
 * been replaced by a goto to a unique return statement. This unique
 * return statement may have to be added to the function body.
 */
void FixCReturnStatements(statement ms)
{
  if(get_bool_property("C_PARSER_RETURN_SUBSTITUTION")) {
    /* How many return statements have been encountered? */
    int nrs = GetReturnNumber();
    int replace_p = false;
    if(nrs==-1 || nrs==0)
      ; /* nothing to be done */
    else if(nrs==1) {
      /* If the return statement is the last statement of the module
	 statement, the goto and the assignment can be replaced by a
	 call to return. Otherwise, a return statement with the
	 proper label must be added at the end of the module statement
	 ms */
      //statement ls = find_last_statement(ms);
      statement ls = last_statement(ms);
      if(!statement_undefined_p(ls)) {
	instruction li = statement_instruction(ls);
	if(instruction_goto_p(li)) {
	  statement ts = instruction_goto(li);
	  if(ts==C_return_statement) {
	    //instruction lrvai =
	    //  statement_instruction(last_return_value_assignment);
	    /* The goto instruction and, possibly, the return value
	       assignment can be removed: just remove the label? */
	    statement_instruction(ls) = call_to_instruction
	      (make_call(CreateIntrinsic(C_RETURN_FUNCTION_NAME),
			 last_returned_value));
	    if(!statement_undefined_p(last_return_value_assignment))
	      statement_instruction(last_return_value_assignment) =
		make_continue_instruction();
	    free_instruction(li);
	    //free_instruction(lrvai); contains the expression resued above!
	    replace_p = false;
	  }
	  else
	    replace_p = true;
	}
	else
	  replace_p = true;
      }
      else
	replace_p = true;
    }
    else if(nrs>1)
      replace_p = true;
    else
      pips_internal_error("The number of return statements has"
			  " not been initialized");

    if(replace_p) {
      /* Do not forget to declare the return variable... */
      if(!entity_undefined_p(return_value_entity)) {
	pips_assert("ms is a block", statement_block_p(ms));
	set_current_module_entity(return_current_module);
	AddLocalEntityToDeclarations(return_value_entity,
				     return_current_module,
				     ms);
	reset_current_module_entity();
      }
      insert_statement(ms, C_return_statement, false);
    }
  }
  Reset_C_ReturnStatement();
}
