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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/* Handling of RETURN statements and substitution of alternate returns.
 *
 * Most of the code deals with alternate returns. Functions GenerateReturn()
 * and MakeReturn() are sufficient to process regular RETURN statements.
 *
 * Francois Irigoin
 */

#include <stdio.h>

#include "genC.h"
#include "parser_private.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "properties.h"

#include "syntax.h"

/* Should alternate returns be substituted or not ?
 *
 * If alternate returns are substituted, the code declarations should be
 * regenerated unless hide_rc_p is true and some PIPS run-time Fortran
 * functions provided. The corresponding property should be checked.  */
static bool substitute_rc_p = false;
static bool substitute_stop_p = false;
static bool hide_rc_p = false;
#define GET_RC_PREFIX "GET_"
#define SET_RC_PREFIX "SET_"

void
SubstituteAlternateReturns(const char* option)
{
    substitute_rc_p = (strcmp(option, "RC")==0) || (strcmp(option, "HRC")==0) ;
    hide_rc_p = (strcmp(option, "HRC")==0) ;
    substitute_stop_p = (strcmp(option, "STOP")==0);

    if(!(substitute_rc_p || substitute_stop_p || strcmp(option, "NO")==0)) {
	user_log("Unknown option \"%s\" for property "
		 "PARSER_SUBSTITUTE_ALTERNATE_RETURNS.\n"
		 "Three options are available for alternate return handling: "
		 "\"NO\", \"RC\" and \"STOP\"\n", option);
	ParserError("SubstituteAlternateReturns", "Illegal property value");
    }

    if((substitute_rc_p || substitute_stop_p)
       && !get_bool_property("PRETTYPRINT_ALL_DECLARATIONS"))
	user_warning("SubstituteAlternateReturns",
		     "Module declarations should be regenerated."
		     " Set property PRETTYPRINT_ALL_DECLARATIONS.\n");
}

bool
SubstituteAlternateReturnsP()
{
    return substitute_rc_p;
}


/* Variable used to carry return code replacing the alternate returns
 *
 * This variable may be either a formal parameter if the module uses
 * alternate returns, or a dynamic variable if it uses such a module.
 * When both conditions are true, the return code is a formal parameter.
 */
static entity return_code_variable = entity_undefined;

entity GetReturnCodeVariable()
{
  const char* rc_name = get_string_property("PARSER_RETURN_CODE_VARIABLE");
  /* Cannot be asserted because the return_code_variable may either be a
   * formal parameter if the current module uses multiple return, or
   * a dynamic variable if it does not and if it calls a subroutine
   * using alternate returns
   */
  /*
    pips_assert("entity return_code_variable is undefined",
    entity_undefined_p(return_code_variable));
  */
  return_code_variable = FindEntity(get_current_module_name(), rc_name);
  if(entity_undefined_p(return_code_variable)) {
    return_code_variable = FindOrCreateEntity(get_current_module_name(), rc_name);
  }

  return return_code_variable;
}

static entity GetFullyDefinedReturnCodeVariable()
{
  entity rc = GetReturnCodeVariable();

  /* Type, storage and initial value may have been set up earlier in
     MakeFormalParameter(). */
  if(type_undefined_p(entity_type(rc))) {
    /* We must be dealing with the actual variable, not with the formal
       variable. */
    const char* module_name = get_current_module_name();
    entity f = local_name_to_top_level_entity(module_name);
    entity a = FindEntity(module_name, DYNAMIC_AREA_LOCAL_NAME);

    entity_type(rc) = MakeTypeVariable(make_basic(is_basic_int, (void *) 4), NIL);

    entity_storage(rc) =
      make_storage(is_storage_ram,
		   make_ram(f, a,
			    add_variable_to_area(a, rc),
			    NIL));

    entity_initial(rc) = make_value_unknown();
  }

  pips_assert("rc is defined", !entity_undefined_p(rc));

  return rc;
}

bool
ReturnCodeVariableP(entity rcv)
{
    return rcv == return_code_variable;
}

void
ResetReturnCodeVariable()
{
    return_code_variable = entity_undefined;
}


/* Remember if the current module uses alternate returns. If yes,
 * the RETURN statements must include an assignment to the return code
 * variable.
 */
static bool current_module_uses_alternate_returns = false;
/* The current number of alternate returns is used to process a module
   declaration */
static int current_number_of_alternate_returns = -1;

bool uses_alternate_return_p()
{
    return current_module_uses_alternate_returns;
}

void uses_alternate_return(bool use)
{
  if(use && strcmp(get_string_property("PARSER_SUBSTITUTE_ALTERNATE_RETURNS"), "NO")==0) {
    pips_user_warning
      ("Lines %d-%d: Alternate return not processed with current option \"%s\". "
       "Formal label * ignored.\n"
       "See property PARSER_SUBSTITUTE_ALTERNATE_RETURNS for other options\n",
       line_b_I, line_e_I, get_string_property("PARSER_SUBSTITUTE_ALTERNATE_RETURNS"));
    ParserError("uses_alternate_return", "Alternate returns prohibited by user\n");
  }

  current_number_of_alternate_returns++;

  current_module_uses_alternate_returns = use;
}

void set_current_number_of_alternate_returns()
{
  current_number_of_alternate_returns = 0;
}

void reset_current_number_of_alternate_returns()
{
  current_number_of_alternate_returns = -1;
}

int get_current_number_of_alternate_returns()
{
  return current_number_of_alternate_returns;
}


/* Update the formal and actual parameter lists by adding the return code
 * variable as last argument.
 *
 * To avoid an explicit check in gram.y which is large enough, the additions
 * are conditional to the alternate return substitution.
 */
list add_formal_return_code(list fpl)
{
    list new_fpl = fpl;

    if(uses_alternate_return_p() && !hide_rc_p && substitute_rc_p) {
	entity frc = GetReturnCodeVariable();

	/* Type, storage and initial value are set up later in MakeFormalParameter() */
	new_fpl = gen_nconc(fpl, CONS(ENTITY, frc, NIL));
    }
    return new_fpl;
}

list add_actual_return_code(list apl)
{
  list new_apl = apl;

  if(substitute_rc_p && !hide_rc_p && !ENDP(get_alternate_returns())) {
    entity frc = GetFullyDefinedReturnCodeVariable();

    new_apl = gen_nconc(apl, CONS(EXPRESSION, entity_to_expression(frc), NIL));
  }

  return new_apl;
}

/* Keep track of the labels used as actual arguments for alternate returns
 * and generate the tests to check the return code.
 */

static list alternate_returns = list_undefined;

void
add_alternate_return(string label_name)
{
    entity l = entity_undefined;

    if(substitute_rc_p) {
	l = MakeLabel(label_name);
	alternate_returns = arguments_add_entity(alternate_returns, l);
    }
    else {
	pips_user_warning("Lines %d-%d: Alternate return towards label %s not supported. "
			  "Actual label argument internally substituted by a character string.\n",
			  line_b_I, line_e_I, label_name);
    }
}

list
get_alternate_returns()
{
    return alternate_returns;
}

void
set_alternate_returns()
{
    pips_assert("alternate return list is undefined", list_undefined_p(alternate_returns));
    alternate_returns = NIL;
    current_number_of_alternate_returns = 0;
}

void
reset_alternate_returns()
{
    pips_assert("alternate return list is defined", !list_undefined_p(alternate_returns));
    gen_free_list(alternate_returns);
    alternate_returns = list_undefined;
    current_number_of_alternate_returns = -1;
}

/* ParserError() cannot guess if it has been performed or not, because it
   is reinitialized before and after each call statement. If the error
   occurs within a call, alternate returns must be reset. Else they should
   not be reset.*/
void soft_reset_alternate_returns()
{
  if(!list_undefined_p(alternate_returns)) {
    reset_alternate_returns();
  }
}

static statement make_get_rc_statement(expression rc_ref)
{
  statement s_get = statement_undefined;
  instruction i_get = instruction_undefined;
  string get_rc_name = strdup(concatenate(GET_RC_PREFIX,
					  get_string_property("PARSER_RETURN_CODE_VARIABLE"),
					  NULL));
  entity get_rc = FindEntity(TOP_LEVEL_MODULE_NAME, get_rc_name);

  if(entity_undefined_p(get_rc)) {
    get_rc = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, get_rc_name);
    /* get_rc takes an argument and returns void */
    entity_type(get_rc) =
      make_type(is_type_functional,
		make_functional(CONS(PARAMETER,
				     make_parameter(make_type(is_type_variable,
							      make_variable(make_basic(is_basic_int, (value) 4),
									    NIL,NIL)),
						    make_mode(is_mode_reference, UU),
						    make_dummy_unknown()),
				     NIL),
				make_type(is_type_void, UU)));
    /*
    entity_type(get_rc) =
      make_type(is_type_functional,
		make_functional(NIL,
				make_type(is_type_variable,
					  make_variable(make_basic(is_basic_int, (value) 4),
							NIL,NIL))));
    */
    entity_storage(get_rc) = make_storage(is_storage_rom, UU);
    entity_initial(get_rc) = make_value(is_value_code, code_undefined);
    update_called_modules(get_rc);
  }

  pips_assert("Function get_rc is defined", !entity_undefined_p(get_rc));

  i_get = make_instruction(is_instruction_call,
			   make_call(get_rc, CONS(EXPRESSION, rc_ref, NIL)));
  s_get = instruction_to_statement(i_get);
  statement_number(s_get) = get_statement_number();

  return s_get;
}


instruction generate_return_code_checks(list labels)
{
    instruction i = instruction_undefined;
    list lln = NIL;
    entity rcv = GetFullyDefinedReturnCodeVariable();
    expression ercv = expression_undefined;

    pips_assert("The label list is not empty", !ENDP(labels));



    ercv = entity_to_expression(rcv);

    FOREACH(ENTITY, l, labels) {
        lln = CONS(STRING, (char*)label_local_name(l), lln);
    }

    i = MakeComputedGotoInst(lln, ercv);

    /* The reset is controlled from gram.y, as is the set */
    /* reset_alternate_returns(); */
    gen_free_list(lln);

    instruction_consistent_p(i);

    if(hide_rc_p) {
      statement s_init_rcv = make_get_rc_statement(entity_to_expression(rcv));

      ifdebug(2) {
	pips_debug(2, "Additional statement generated for hide_rc_p:\n");
	print_statement(s_init_rcv);
	}
      pips_assert("i is a sequence", instruction_block_p(i));
      instruction_block(i) = CONS(STATEMENT, s_init_rcv, instruction_block(i));
      instruction_consistent_p(i);
    }

    return i;
}


/* This function creates a goto instruction to label end_label. This is
 * done to eliminate return statements.
 *
 * Note: I was afraid the mouse trap would not work to analyze
 * multiple procedures but there is no problem. I guess that MakeGotoInst()
 * generates the proper label entity regardless of end_label. FI.
 */

LOCAL entity end_label = entity_undefined;
LOCAL char *end_label_local_name = RETURN_LABEL_NAME;

static entity set_rc_function()
{
  string set_rc_name = strdup(concatenate(SET_RC_PREFIX,
					  get_string_property("PARSER_RETURN_CODE_VARIABLE"),
					  NULL));
  entity set_rc = FindEntity(TOP_LEVEL_MODULE_NAME, set_rc_name);

  if(entity_undefined_p(set_rc)) {
    set_rc = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, set_rc_name);
    /* set_rc takes no argument and returns a scalar int */
    entity_type(set_rc) =
      make_type(is_type_functional,
		make_functional(CONS(PARAMETER,
				     make_parameter(make_type(is_type_variable,
							      make_variable(make_basic(is_basic_int, (value) 4),
									    NIL,NIL)),
						    make_mode(is_mode_reference, UU),
						    make_dummy_unknown()),
				     NIL),
				make_type(is_type_void, UU)));
    entity_storage(set_rc) = make_storage(is_storage_rom, UU);
    entity_initial(set_rc) = make_value(is_value_code, code_undefined);
    update_called_modules(set_rc);
  }

  pips_assert("Function set_rc is defined", !entity_undefined_p(set_rc));

  return set_rc;
}

/* The return code may be directly assigned or indirectly through a PIPS
   run-time function call.*/
static statement make_set_rc_statement(expression e)
{
  expression rc = expression_undefined_p(e)? int_to_expression(0) : e;
  statement src = statement_undefined;

  if(hide_rc_p) {
    instruction isrc = make_instruction(is_instruction_call,
					make_call(set_rc_function(),
						  CONS(EXPRESSION, rc, NIL)));

    src = instruction_to_statement(isrc);
  }
  else {
    src = make_assign_statement(entity_to_expression(GetReturnCodeVariable()), rc);
  }

  ifdebug(2) {
    pips_debug(2, "Statement generated: ");
    print_statement(src);
  }

  return src;
}

instruction MakeReturn(expression e)
{
  instruction inst = instruction_undefined;

  if(!expression_undefined_p(e) && !substitute_rc_p && !substitute_stop_p) {
    user_error("MakeReturn",
	       "Lines %d-%d: Alternate return not supported. "
	       "Standard return generated\n",
	       line_b_I,line_e_I);
  }

  if (end_label == entity_undefined) {
    end_label = MakeLabel(end_label_local_name);
  }

  if(substitute_rc_p && uses_alternate_return_p()) {
    /* Assign e to the return code variable, but be sure not to count
     * this assignment as a user instruction. Wrap if with the Go To
     * in a block and return the block instruction.
     *
     * See how code is synthesized for computed goto's...
     */
    statement src = make_set_rc_statement(e);
    statement jmp = instruction_to_statement(MakeGotoInst(end_label_local_name));

    statement_number(src) = get_statement_number();
    statement_number(jmp) = get_statement_number();
    //    (void) get_next_statement_number();
    inst = make_instruction_block(CONS(STATEMENT, src, CONS(STATEMENT, jmp, NIL)));
    instruction_consistent_p(inst);
  }
  else if(!expression_undefined_p(e) && substitute_stop_p && uses_alternate_return_p()) {
    /* Let's try to provide more useful information to the user */
    /* inst = MakeZeroOrOneArgCallInst("STOP", e); */
    if(expression_call_p(e)) {
      const char* mn = get_current_module_name();
      const char* sn = entity_local_name(call_function(syntax_call(expression_syntax(e))));

      inst = MakeZeroOrOneArgCallInst
	("STOP",
	 MakeCharacterConstantExpression(strdup(concatenate("\"", sn, " in ", mn, "\"", NULL))));
    }
    else {
      pips_internal_error("unexpected argument type for RETURN");
    }
  }
  else {
    inst = MakeGotoInst(end_label_local_name);
  }

  return inst;
}

/* Generate a unique call to RETURN per module */
void GenerateReturn()
{
    instruction inst = instruction_undefined;
    /* statement c = MakeStatement(l, make_continue_instruction()); */


    if(substitute_rc_p && uses_alternate_return_p()) {
	entity l = MakeLabel(strdup(end_label_local_name));
	expression rc = int_to_expression(0);
	statement src = make_set_rc_statement(rc);
	statement jmp = statement_undefined;
	    (MakeZeroOrOneArgCallInst("RETURN", expression_undefined));

	statement_number(src) = get_statement_number();
	jmp = MakeStatement(l, MakeZeroOrOneArgCallInst("RETURN", expression_undefined));
	/*
	statement_number(jmp) = get_statement_number();
	(void) get_next_statement_number();
	*/
	inst = make_instruction_block(CONS(STATEMENT, src, CONS(STATEMENT, jmp, NIL)));
	instruction_consistent_p(inst);
    }
    else {
	strcpy(lab_I, end_label_local_name);
	inst = MakeZeroOrOneArgCallInst("RETURN", expression_undefined);
    }

    LinkInstToCurrentBlock(inst, true);
}
