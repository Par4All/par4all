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
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "properties.h"

#include "syntax.h"

/* Should alternate returns be substituted or not ?
 *
 * If alternate returns are substituted, the code declarations should
 * be regenerated. The corresponding property should be checked.
 */
static bool substitute_p = FALSE;

void
SubstituteAlternateReturns(bool do_it)
{
    substitute_p = do_it;

    if(do_it && !get_bool_property("PRETTYPRINT_ALL_DECLARATIONS"))
	user_warning("SubstituteAlternateReturns",
		     "Module declarations should be regenerated."
		     " Set property PRETTYPRINT_ALL_DECLARATIONS.\n");
}

bool
SubstituteAlternateReturnsP()
{
    return substitute_p;
}


/* Variable used to carry return code replacing the alternate returns
 *
 * This variable may be either a formal parameter if the module uses
 * alternate returns, or a dynamic variable if it uses such a module.
 * When both conditions are true, the return code is a formal parameter.
 */
static entity return_code_variable = entity_undefined;

entity
GetReturnCodeVariable()
{
    string rc_name = get_string_property("PARSER_RETURN_CODE_VARIABLE");
    /* Cannot be asserted because the return_code_variable may either be a
     * formal parameter if the current module uses multiple return, or
     * a dynamic variable if it does not and if it calls a subroutine
     * using alternate returns
     */
    /*
    pips_assert("entity return_code_variable is undefined",
		entity_undefined_p(return_code_variable));
		*/
    return_code_variable = global_name_to_entity(get_current_module_name(), rc_name);
    if(entity_undefined_p(return_code_variable)) {
	return_code_variable = FindOrCreateEntity(get_current_module_name(), rc_name);
    }

    return return_code_variable;
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
static bool current_module_uses_alternate_returns = FALSE;

bool
uses_alternate_return_p()
{
    return current_module_uses_alternate_returns;
}

void
uses_alternate_return(bool use)
{
     current_module_uses_alternate_returns = use;
}

/* Update the formal and actual parameter lists by adding the return code
 * variable as last argument.
 *
 * To avoid an explicit check in gram.y which is large enough, the additions
 * are conditional to the alternate return substitution.
 */
list
add_formal_return_code(list fpl)
{
    list new_fpl = fpl;

    if(substitute_p && uses_alternate_return_p()) {
	entity frc = GetReturnCodeVariable();

	/* Type, storage and initial value are set up later in MakeFormalParameter() */
	new_fpl = gen_nconc(fpl, CONS(ENTITY, frc, NIL));
    }
    return new_fpl;
}

list
add_actual_return_code(list apl)
{
    list new_apl = apl;

    if(substitute_p && !ENDP(get_alternate_returns())) {
	entity frc = GetReturnCodeVariable();
	expression frcr = entity_to_expression(frc);

	/* Type, storage and initial value may have been set up earlier in MakeFormalParameter() */
	if(type_undefined_p(entity_type(frc))) {
	    string module_name = get_current_module_name();
	    entity f = local_name_to_top_level_entity(module_name);
	    entity a = global_name_to_entity(module_name, DYNAMIC_AREA_LOCAL_NAME); 

	    entity_type(frc) = MakeTypeVariable(make_basic(is_basic_int, 4), NIL);

	    entity_storage(frc) = 
		make_storage(is_storage_ram,
			     make_ram(f, a,
				      add_variable_to_area(a, frc),
				      NIL));

	    entity_initial(frc) = MakeValueUnknown();
	}
	new_apl = gen_nconc(apl, CONS(EXPRESSION,frcr , NIL));
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

    if(substitute_p) {
	l = MakeLabel(label_name);
	alternate_returns = arguments_add_entity(alternate_returns, l);
    }
    else {
	pips_user_warning("Lines %d-%d: Alternate return towards label %s not supported. "
			  "Actual label argument ignored.\n", line_b_I, line_e_I, label_name);
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
}

void
reset_alternate_returns()
{
    pips_assert("alternate return list is defined", !list_undefined_p(alternate_returns));
    gen_free_list(alternate_returns);
    alternate_returns = list_undefined;
}

instruction
generate_return_code_checks(list labels)
{
    instruction i = instruction_undefined;
    list lln = NIL;
    entity rcv = GetReturnCodeVariable();

    pips_assert("The label list is not empty", !ENDP(labels));

    MAP(ENTITY, l, {
	lln = CONS(STRING, label_local_name(l), lln);
    }, labels);

    i = MakeComputedGotoInst(lln, rcv);

    /* The reset is controlled from gram.y, as is the set */
    /* reset_alternate_returns(); */
    gen_free_list(lln);

    gen_consistent_p(i);

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

instruction MakeReturn(expression e)
{
    instruction inst = instruction_undefined;

    if(!expression_undefined_p(e) && !substitute_p) {
	user_warning("MakeReturn", 
		     "Lines %d-%d: Alternate return not supported. "
		     "Standard return generated\n",
		     line_b_I,line_e_I);
    }

    if (end_label == entity_undefined) {
	end_label = MakeLabel(end_label_local_name);
    }

    if(substitute_p && uses_alternate_return_p()) {
	/* Assign e to the return code variable, but be sure not to count
	 * this assignment as a user instruction. Wrap if with the Go To
	 * in a block and return the block instruction.
	 *
	 * See how code is synthesized for computed goto's...
	 */
	expression rc = expression_undefined_p(e)? int_to_expression(0) : e;
	statement src = make_assign_statement(entity_to_expression(GetReturnCodeVariable()), rc);
	statement jmp = instruction_to_statement(MakeGotoInst(end_label_local_name));

	statement_number(src) = look_at_next_statement_number();
	statement_number(jmp) = look_at_next_statement_number();
	(void) get_next_statement_number();
	inst = make_instruction_block(CONS(STATEMENT, src, CONS(STATEMENT, jmp, NIL)));
	gen_consistent_p(inst);
    }
    else {
	inst = MakeGotoInst(end_label_local_name);
    }

    return inst;
}

/* Generate a unique call to RETURN per module */
void 
GenerateReturn()
{
    instruction inst = instruction_undefined;
    /* statement c = MakeStatement(l, make_continue_instruction()); */


    if(substitute_p && uses_alternate_return_p()) {
	entity l = MakeLabel(strdup(end_label_local_name));
	expression rc = int_to_expression(0);
	statement src = make_assign_statement(entity_to_expression(GetReturnCodeVariable()), rc);
	statement jmp = statement_undefined;
	    (MakeZeroOrOneArgCallInst("RETURN", expression_undefined));

	statement_number(src) = look_at_next_statement_number();
	jmp = MakeStatement(l, MakeZeroOrOneArgCallInst("RETURN", expression_undefined));
	/*
	statement_number(jmp) = look_at_next_statement_number();
	(void) get_next_statement_number();
	*/
	inst = make_instruction_block(CONS(STATEMENT, src, CONS(STATEMENT, jmp, NIL)));
	gen_consistent_p(inst);
    }
    else {
	strcpy(lab_I, end_label_local_name);
	inst = MakeZeroOrOneArgCallInst("RETURN", expression_undefined);
    }

    LinkInstToCurrentBlock(inst, TRUE);
}
