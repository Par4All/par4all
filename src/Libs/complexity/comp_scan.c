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
/*
 * scan the Abstract Syntax Tree of a program to count operations
 */


#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "text-util.h"
#include "misc.h"
#include "matrice.h"
#include "semantics.h"
#include "effects-simple.h"
#include "effects-generic.h"
#include "transformer.h"
#include "complexity.h"

hash_table hash_callee_to_complexity = hash_table_undefined;
hash_table hash_complexity_parameters = hash_table_undefined;

/* declares the static variable complexity_map and defines its access functions */
DEFINE_CURRENT_MAPPING(complexity, complexity)


/* bool complexities(module_name)
 *   Entry point called by the pips makefile
 *
 *   "hash_callee_to_complexity" contains a list of summary_complexity
 *   of callees,
 *   Non-recursive: callees complexities are supposed to be computed
 *   when arriving here.
 *   
 *   "hash_complexity_parameters" contains the list of variables that we
 *   don't want to evaluate; so they will appear in the polynomial.
 *   All other variables (except for the loop indices) are evaluated
 *   as the scan proceeds, thanks to semantic analysis. Those which
 *   can't be evaluated are replaced by the pseudo-variable UNKNOWN_VARIABLE,
 *   which will be given an arbitrary value at the end of the evaluation.
 */
bool uniform_complexities(module_name)
char *module_name;
{
    bool success = true;

    //set_string_property("COMPLEXITY_COST_TABLE", "all_1");
    success = any_complexities(module_name);

    return success;
}

bool fp_complexities(module_name)
char *module_name;
{
    bool success = true;

    //set_string_property("COMPLEXITY_COST_TABLE", "fp_1");
    success = any_complexities(module_name);

    return success;
}

bool any_complexities(module_name)
char *module_name;
{
    transformer precond = transformer_undefined;
    list effects_list = NIL;
    entity module_entity;
    statement module_stat;

    trace_on("complexities %s", (char *)module_name);
    debug_on("COMPLEXITY_DEBUG_LEVEL");

    /* we may need to print preconditions for debugging purposes */
    set_precondition_map( (statement_mapping)
	db_get_memory_resource(DBR_PRECONDITIONS, module_name, true));
    set_cumulated_rw_effects((statement_effects)
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
    set_current_module_entity(module_name_to_entity(module_name));
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, true ) );

    module_entity = get_current_module_entity();
    module_stat = get_current_module_statement();

    module_to_value_mappings(module_entity);

    precond = load_statement_precondition(module_stat);
    effects_list = load_cumulated_rw_effects_list(module_stat);

    init_cost_table();

    if (get_bool_property("COMPLEXITY_PRINT_COST_TABLE"))
	fprint_cost_table(stdout);

    set_complexity_map( MAKE_STATEMENT_MAPPING() );

    hash_callee_to_complexity = fetch_callees_complexities(module_name);
    hash_complexity_parameters = fetch_complexity_parameters(module_name);

    add_formal_parameters_to_hash_table(module_entity,
					hash_complexity_parameters);

    add_common_variables_to_hash_table(module_entity,
				       hash_complexity_parameters);

    (void)statement_to_complexity(module_stat, precond, effects_list);

    remove_common_variables_from_hash_table(module_entity,
					    hash_complexity_parameters);

    remove_formal_parameters_from_hash_table(module_entity, 
					     hash_complexity_parameters);
    debug_off();
    trace_off();

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr,"\n");
	fprintf(stderr,"Before  ------- COMPLEXITIES; module %s\n",module_name);
    }

    DB_PUT_MEMORY_RESOURCE(DBR_COMPLEXITIES,
			   strdup(module_name),
			   (char *) get_complexity_map() );
    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr,"After   ------- COMPLEXITIES; module %s\n",module_name);
    }

    hash_callee_to_complexity =
	free_callees_complexities(hash_callee_to_complexity);
    reset_precondition_map();
    reset_cumulated_rw_effects();
    reset_complexity_map();
    reset_current_module_entity();
    reset_current_module_statement();
    free_value_mappings();

    return true;
}

bool summary_complexity(module_name)
char *module_name;
{
  complexity summary_comp = make_zero_complexity();// complexity_undefined;
  complexity summary_comp_dup = make_zero_complexity();// complexity_undefined;

    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, true ) );

    set_complexity_map( (statement_mapping)
		       db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));
    summary_comp = load_statement_complexity( get_current_module_statement() );

    trace_on("summary complexity %s", module_name);
    debug_on("COMPLEXITY_DEBUG_LEVEL");

    if ( summary_comp == COMPLEXITY_NOT_FOUND )
	pips_internal_error("No summary complexity!");
    else {
	/* summary_comp_dup = complexity_dup(summary_comp); */
	summary_comp_dup = copy_complexity(summary_comp);
    }

    pips_assert("summary_comp is conssten",
		complexity_consistent_p(summary_comp));
    pips_assert("summary_comp_dup is consistent",
		complexity_consistent_p(summary_comp_dup));

    trace_off();

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr,"\n");
	fprintf(stderr,"Before ======== SUMMARY ; module %s\n",module_name);
    }

    debug_off();

    DB_PUT_MEMORY_RESOURCE(DBR_SUMMARY_COMPLEXITY,
			   strdup(module_name),
			   (char *) summary_comp_dup);

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr,"After  ======== SUMMARY ; module %s\n",module_name);
    }

    reset_complexity_map();
    reset_current_module_statement();

    return true;
}

/* starting point of Abstract Syntax Tree */
complexity statement_to_complexity(statement stat,
				   transformer precon __attribute__ ((__unused__)),
				   list eff_list __attribute__ ((__unused__)))
{
    instruction instr = statement_instruction(stat);
    int so = statement_ordering(stat);
    transformer precond = load_statement_precondition(stat);
    list effects_list = load_cumulated_rw_effects_list(stat);
    complexity comp = make_zero_complexity();

    trace_on("statement %s, ordering (%d %d)",
	     module_local_name(statement_label(stat)),
	     ORDERING_NUMBER(so), ORDERING_STATEMENT(so));
    /*
    if ( perfectly_nested_loop_p(stat) )
	fprintf(stderr, "PERFECTLY nested loop, ordering %d\n",
		statement_ordering(perfectly_nested_loop_to_body(stat)));
    else
	fprintf(stderr, "NOT a perfectly nested loop\n");
*/
    if (instr != instruction_undefined)
	comp = instruction_to_complexity(instr, precond, effects_list);
    else {
	pips_internal_error("instruction undefined");
    }

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr,"complexity for statement (%td,%td) at %p\n",
		(statement_ordering(stat) >> 16),
		(statement_ordering(stat) & 0xffff), comp);
	complexity_fprint(stderr, comp, true, true);
    }

    /* Check and store the complexity in the statement_mapping */
    complexity_check_and_warn("statement_to_complexity", comp);
    pips_assert("comp is consistent", complexity_consistent_p(comp));

    /* SET_STATEMENT_MAPPING(get_complexity_map() , stat, (char *) comp); */
    store_statement_complexity(stat, comp);

    trace_off();
    return(comp);
}

/* The only element available of the statement */
complexity instruction_to_complexity(instr, precond, effects_list)
instruction instr;
transformer precond;
list effects_list;
{
  complexity comp = make_zero_complexity();// complexity_undefined;

    trace_on("instruction");

    switch (instruction_tag(instr)) {
    case is_instruction_block:
	comp = block_to_complexity(instruction_block(instr), precond, effects_list);
	break;
    case is_instruction_test:
	comp = test_to_complexity(instruction_test(instr), precond, effects_list);
	break;
    case is_instruction_loop:
	comp = loop_to_complexity(instruction_loop(instr), precond, effects_list);
	break;
    case is_instruction_whileloop:
        comp= whileloop_to_complexity(instruction_whileloop(instr), precond, effects_list);
	break;
    case is_instruction_goto:
	comp = goto_to_complexity(instruction_goto(instr), precond, effects_list);
	break;
    case is_instruction_call: {
	basic callbasic = MAKE_INT_BASIC;          /* dummy parameter */
	comp = call_to_complexity(instruction_call(instr),
				  &callbasic, precond, effects_list);
	break;
    }
    case is_instruction_unstructured:
	comp = unstructured_to_complexity(instruction_unstructured(instr),
					  precond, effects_list);
	break;
    case is_instruction_expression: { // copied from is_instruction_call
	basic callbasic = MAKE_INT_BASIC;          /* dummy parameter */
	comp = expression_to_complexity(instruction_expression(instr),
					&callbasic,
					precond, effects_list);
	break;
    }
    default:
	pips_internal_error("instruction tag %d isn't in 14->19.",
		   (int) instruction_tag(instr));
    }
    complexity_check_and_warn("instruction_to_complexity", comp);

    trace_off();
    return(comp);
}

/* 1st element of instruction */
/* Modification:
 *  - To postpone the variable evaluation, we reverse the block.
 *    Attention: gen_nreverse destroys the original list.  LZ  4 Dec. 92
 *
 *  - Secondly, we check the variables in complexity. If a certain variable
 *    is must_be_written variable ( obtained from cumulated effects ),
 *    we evaluate it.  LZ 5 Dec. 92
 *
 *  - When the variable is must_be_written, the precondition is no longer
 *    available with this effect. So we need to use the next statement effect
 *    with current precondition to evaluate the variable. LZ 9 Dec. 92
 *
 *  - Francois suggested not to create new list, ie, reverse block.
 *    Hence we don't use MAPL any more here. Instead, gen_nthcdr is used.
 *    16 Dec 92
 */
complexity block_to_complexity(list block, // list of statements
			       transformer precond,
			       list effects_list)
{
  complexity comp = make_zero_complexity();
  int block_length = gen_length( block );
  int i;

  trace_on("block");

  if (get_bool_property("COMPLEXITY_EARLY_EVALUATION")) {
    FOREACH(STATEMENT, stat, block) {
      complexity ctemp = statement_to_complexity(stat, precond, effects_list);
      complexity_add(&comp, ctemp);
    }
  }
  else { // default property setting
    for ( i = block_length; i > 0 ; i-- ) {
      statement stat = STATEMENT(CAR(gen_nthcdr( i-1, block )));
      statement up_stat;
      transformer prec = load_statement_precondition(stat);
      complexity ctemp = statement_to_complexity(stat,
						 prec, effects_list);

      list cumu_list = load_cumulated_rw_effects_list(stat);
      //extern int default_is_inferior_pvarval(Pvecteur *, Pvecteur *);

      Pbase pb = VECTEUR_NUL;

      if ( i > 1 ) {
	up_stat = STATEMENT(CAR(gen_nthcdr( i-2, block )));
	cumu_list = load_cumulated_rw_effects_list(up_stat);
      }
      complexity_add(&comp, ctemp);

      pb = vect_dup(polynome_used_var(complexity_polynome(comp),
				      default_is_inferior_pvarval));

      for ( ; !VECTEUR_NUL_P(pb); pb = pb->succ) {
	bool mustbewritten;
	entity v = (entity) (pb->var);
	//char *var = variable_local_name(pb->var);

	//mustbewritten = is_must_be_written_var(cumu_list, var);
	mustbewritten = effects_write_variable_p(cumu_list, v);

	if ( mustbewritten ) {
	  complexity csubst;

	  csubst = evaluate_var_to_complexity((entity)pb->var,
					      prec,
					      cumu_list, 1);
	  comp = complexity_var_subst(comp, pb->var, csubst);
	}
      }
    }
  }

  if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
    (void) complexity_consistent_p(comp);
    fprintf(stderr, "block comp is at %p and comp value is ", comp);
    complexity_fprint(stderr, comp, false, true);
  }
  complexity_check_and_warn("block_to_complexity", comp);

  trace_off();
  return(comp);
}

/* 2nd element of instruction */
complexity test_to_complexity(test_instr, precond, effects_list)
test test_instr;
transformer precond;
list effects_list;
{
    complexity comp, ctrue, cfalse, ctemp;
    basic testbasic;

    trace_on("test");

    ctrue = statement_to_complexity(test_true(test_instr), precond, effects_list);
    cfalse = statement_to_complexity(test_false(test_instr), precond, effects_list);
    comp = expression_to_complexity(test_condition(test_instr),
				    &testbasic, precond, effects_list);

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "\n");
	fprintf(stderr, "YYY  test true  complexity: ");
	complexity_fprint(stderr, ctrue, false, true);
	fprintf(stderr, "YYY  test false complexity: ");
	complexity_fprint(stderr, cfalse, false, true);
	fprintf(stderr, "YYY  test cond. complexity: ");
	complexity_fprint(stderr, comp, false, true);
    }

    ctemp = complexity_dup(ctrue);
    complexity_add(&ctemp, cfalse);
    complexity_scalar_mult(&ctemp, 0.5);
    complexity_add(&comp, ctemp);
    complexity_rm(&ctemp);

    if ( !complexity_zero_p(comp) )
	ifcount_halfhalf(complexity_ifcount(comp)) ++;

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "YYY  test total complexity: ");
	complexity_fprint(stderr, comp, true, true);
	fprintf(stderr, "\n");
    }
    complexity_check_and_warn("test_to_complexity", comp);    

    trace_off();
    return(comp);
}

/* 3rd element of instruction */
complexity loop_to_complexity(loop_instr, precond, effects_list)
loop loop_instr;
transformer precond;
list effects_list;
{
    entity ll = loop_label (loop_instr);
    entity index = loop_index(loop_instr);
    range rng = loop_range(loop_instr);
    statement s = loop_body(loop_instr);
    execution ex = loop_execution(loop_instr);
    complexity comp, cbody, crange, clower, cupper, cincr, cioh, cboh;
    string sl,su,si;
    /* FI: Lei chose to allocate the UL and UU entities in the current
       module... Maybe, we are ready fo some dynamic profiling... */
    /* string mod_name = entity_module_name(ll); */
    basic ibioh = MAKE_INT_BASIC; 
    basic ibboh = MAKE_INT_BASIC;

    trace_on("loop %s label %s",entity_name(index),
	                        entity_name(loop_label(loop_instr)));

    if ( empty_global_label_p(entity_name(ll)) ) {
	/* if the statement were still reachable, we could try to use the
	 * statement nunber...
	 */
        sl=strdup("UL_");
        su=strdup("UU_");
        si=strdup("UU_");
    }
    else {
        /*  In order to get rid of at-sign, add 1 , LZ 010492 */
        asprintf(&sl,"UL_%s",entity_local_name(loop_label(loop_instr))+1);
        asprintf(&su,"UU_%s",entity_local_name(loop_label(loop_instr))+1);
        asprintf(&si,"UI_%s",entity_local_name(loop_label(loop_instr))+1);
    }

    /* tell callees that they mustn't try to evaluate the loop index */

    hash_put(hash_complexity_parameters, (char *)strdup(module_local_name(index)), 
	     HASH_LOOP_INDEX);

    crange = range_to_complexity(rng, precond, effects_list);
    cbody = statement_to_complexity(s, precond, effects_list);

    cioh = make_constant_complexity((float)
				    intrinsic_cost(LOOP_INIT_OVERHEAD, &ibioh));
    cboh = make_constant_complexity((float)
				    intrinsic_cost(LOOP_BRANCH_OVERHEAD, &ibboh));
    /* cioh cboh are derived from overhead file "overhead" LZ, 280993 */

    complexity_polynome_add(&cbody, complexity_polynome(cboh));

    if ( execution_parallel_p(ex) ) {
	user_warning("loop_to_complexity", "DOALL not implemented yet\n");
    }

    clower = expression_to_complexity_polynome(range_lower(rng),
				    precond, effects_list, 
				    KEEP_SYMBOLS, MINIMUM_VALUE);
    if ( complexity_unknown_p(clower) ) {
        /*
           clower = make_single_var_complexity(1.0,
           (Variable)FindOrCreateEntity(mod_name, sl));
           */
        Variable var = (Variable)make_new_scalar_variable_with_prefix(sl, get_current_module_entity(),MakeBasic(is_basic_int));
        AddEntityToCurrentModule((entity)var);
        clower = make_single_var_complexity(1.0,var);
    }

    cupper = expression_to_complexity_polynome(range_upper(rng),
				    precond, effects_list, 
				    KEEP_SYMBOLS, MAXIMUM_VALUE);
    if ( complexity_unknown_p(cupper) ) {
        /*
           cupper = make_single_var_complexity(1.0,
           (Variable)FindOrCreateEntity(mod_name, su));
           */
        Variable var = (Variable)make_new_scalar_variable_with_prefix(su, get_current_module_entity(),MakeBasic(is_basic_int));
        AddEntityToCurrentModule((entity)var);
        cupper = make_single_var_complexity(1.0,var);
    }

    cincr  = expression_to_complexity_polynome(range_increment(rng),
				    precond, effects_list, 
				    KEEP_SYMBOLS, MINIMUM_VALUE);

    if ( complexity_constant_p(cincr) ) {
	int incr = complexity_TCST(cincr);
	if (incr == 0)
	    user_error("loop_to_complexity", "null increment\n");
	else if ( incr < 0 ) {
	    complexity cswap;

	    complexity_scalar_mult(&cincr, -1.0);
	    if ( incr != -1.0 ) {
		complexity_div(&clower, cincr);
		complexity_div(&cupper, cincr);
	    }
	    complexity_scalar_mult(&cincr, -1.0/incr);
	    cswap = clower;
	    clower = cupper;
	    cupper = cswap;
	}
	else if ( incr != 1) {
	    complexity_div(&clower, cincr);
	    complexity_div(&cupper, cincr);
	    complexity_scalar_mult(&cincr, 1.0/incr);
	}
	else if ( complexity_constant_p(clower) || complexity_constant_p(cupper) ) {
	    ;
	}
    }

    if ( complexity_constant_p(clower) && complexity_constant_p(cupper)) {
	float lower = complexity_TCST(clower);
	float upper = complexity_TCST(cupper);
	if (lower>upper) {
	    /* zero iteration for sure */
	    comp = make_zero_complexity();
	}
	else {
	    /* at least one iteration */
	    comp = complexity_sigma(cbody, (Variable)index, clower, cupper);
	}
    }
    else {
	/* maybe some iteration */
	comp = complexity_sigma(cbody, (Variable)index, clower, cupper);
	/* an intermediate test based on preconditions would give better
	   result with affine loop bounds */
    }

    /*
    if ( !complexity_constant_p(cincr) 
	|| complexity_constant_p(clower) || complexity_constant_p(cupper) ) {
	complexity_div(&comp, cincr);
    }
    */

    if(!complexity_constant_p(cincr)) {
        if(complexity_is_monomial_p(cincr) && complexity_degree(cincr)==1) {
            complexity_div(&comp, cincr);
        }
        else {
            free_complexity(cincr);
            Variable var = (Variable)make_new_scalar_variable_with_prefix(si, get_current_module_entity(),MakeBasic(is_basic_int));
            AddEntityToCurrentModule((entity)var);
            cincr = make_single_var_complexity(1.0,var);
        }
    }

    if ( complexity_constant_p(comp) && complexity_TCST(comp) < 0 )
	comp = make_zero_complexity();

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "\n");
	fprintf(stderr, "@@@  body  complexity: ");
	complexity_fprint(stderr, cbody, false, true);
	fprintf(stderr, "@@@  range complexity: ");
	complexity_fprint(stderr, crange, false, true);
	fprintf(stderr, "@@@  lower     value : ");
	complexity_fprint(stderr, clower, false, true);
	fprintf(stderr, "@@@  upper     value : ");
	complexity_fprint(stderr, cupper, false, true);
	fprintf(stderr, "@@@  increment value : ");
	complexity_fprint(stderr, cincr, false, true);
	fprintf(stderr, "@@@  sigma complexity: ");
	complexity_fprint(stderr, comp, false, true);
    }
/*
    if ( !complexity_constant_p(cincr) ) {
	complexity_div(&comp, cincr);
    }
    else {
	float incr = complexity_TCST(cincr);
	if (incr == 0)
	    user_error("loop_to_complexity", "null increment\n");

	if ( complexity_constant_p(clower) && complexity_constant_p(cupper) ) {
	    float lower = complexity_TCST(clower);
	    float upper = complexity_TCST(cupper);
	    int times=0;
	    float curf;

	    for ( curf = lower; curf <= upper; curf += incr )
		times++;

	    complexity_scalar_mult(&comp, 1./(float)times);
	    if ( complexity_zero_p(comp) )
		comp = make_constant_complexity(1.0);
	}
	else {
	    complexity_scalar_mult(&comp, 1.0/incr);
	}
    }
*/
    complexity_polynome_add(&comp, complexity_polynome(crange));
    complexity_polynome_add(&comp, complexity_polynome(cioh));
    complexity_stats_add(&comp, cincr);
    rangecount_guessed(complexity_rangecount(comp)) ++;

    complexity_rm(&crange);
    complexity_rm(&cioh);
    complexity_rm(&cboh);
    complexity_rm(&clower);
    complexity_rm(&cupper);
    complexity_rm(&cincr);

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "@@@  loop total compl: ");
	complexity_fprint(stderr, comp, true, true);
	fprintf(stderr, "\n");
    }

    hash_del(hash_complexity_parameters, (char *)(module_local_name(index)));
    complexity_check_and_warn("loop_to_complexity", comp);    

    trace_off();
    free(sl);free(su);free(si);
    return(comp);
}

/* 4th element of instruction */
complexity goto_to_complexity(statement st __attribute__ ((__unused__)),
			      transformer precond __attribute__ ((__unused__)),
			      list effects_list __attribute__ ((__unused__)))
{
    pips_internal_error("A GOTO is remaining.");
    return(make_zero_complexity());
}

/* 5th element of instruction */
complexity whileloop_to_complexity(whileloop while_instr, transformer precond, list effects_list)
{
    basic b;
    expression exp1, exp2,exp3;
    expression exp = whileloop_condition(while_instr);
    complexity range ;
    complexity cbody = statement_to_complexity(whileloop_body(while_instr), precond, effects_list),
      ctest = expression_to_complexity(exp, &b,precond, effects_list);
    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
        fprintf(stderr, "\n");
        fprintf(stderr, "YYY  body  complexity: ");
        complexity_fprint(stderr, cbody, false, true);
        fprintf(stderr, "YYY  test complexity: ");
        complexity_fprint(stderr, ctest, false, true);
    }
    
    syntax		s 	 = expression_syntax( exp );
    entity		fun  	 = call_function(syntax_call( s ));
    if (ENTITY_LESS_THAN_P( fun )) {
      exp1 = EXPRESSION(CAR( call_arguments(syntax_call(expression_syntax( exp )))));
      exp2 = EXPRESSION(CAR(CDR( call_arguments(syntax_call(expression_syntax( exp ))))));
      exp3 = make_op_exp( MINUS_OPERATOR_NAME,exp1,exp2);
      range = expression_to_complexity_polynome(exp3,
						precond, effects_list, 
						KEEP_SYMBOLS, MINIMUM_VALUE);
    }
    else
      {
	range = make_complexity_unknown(UNKNOWN_RANGE_NAME);
      }
    complexity_add(&cbody,ctest);
    complexity_mult(&cbody,range);
    
    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
      fprintf(stderr, "YYY  while total complexity: ");
        complexity_fprint(stderr, cbody, true, true);
        fprintf(stderr, "\n");
    }
    complexity_check_and_warn(__FUNCTION__, cbody);

    return cbody;
}

/* 3rd element of syntax */
complexity call_to_complexity(call_instr, pbasic, precond, effects_list)
call call_instr;
basic *pbasic;
transformer precond;
list effects_list;
{
    complexity comp = make_zero_complexity();
    entity f = call_function(call_instr);
    list args= call_arguments(call_instr);
    type   t = entity_type(f);
    value  v = entity_initial(f);

    const char *name = module_local_name(f);

    trace_on("call %s", entity_name(f));

    pips_assert("call_to_complexity",
		type_statement_p(t)||type_functional_p(t)||type_void_p(t));

    switch (value_tag(v)) {
    case is_value_code: {
	complexity compcallee;

	/* The only case that hash_callee_to_complexity is used */
	compcallee = (complexity) hash_get(hash_callee_to_complexity, 
					   (char *) f);
	if ( compcallee == COMPLEXITY_NOT_FOUND )
	    user_error("call_to_complexity", "unknown complexity\n");
	/* transform formal params into real ones */
	comp = replace_formal_parameters_by_real_ones(compcallee, 
						      f, args, 
						      precond, 
						      effects_list);
	break;
    }
    case is_value_symbolic:
    case is_value_constant:
	if ( !type_statement_p(t) )            /* if not format case */
	    *pbasic = entity_basic(f);
	comp = make_zero_complexity();
	break;
    case is_value_intrinsic:
	comp = arguments_to_complexity(args, pbasic, precond, effects_list);
	complexity_float_add(&comp, (float) intrinsic_cost(name, pbasic));
	break;
    case is_value_unknown:
	user_error("call_to_complexity", "unknown value");
    default:
	pips_internal_error("value_tag is %d not in 37->41", (int)value_tag(v));
    }

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "call '%s': ", name);
	complexity_fprint(stderr, comp, false, true);
	fprintf(stderr, "call comp is at %p\n", comp);
    }
    complexity_check_and_warn("call_to_complexity", comp);

    trace_off();
    return(comp);
}

/* 2nd element of call */
/* arguments_to_complexity
 * Return the sum of the complexities of the list of expressions passed.
 * Return also in *pbasic the basic of the "biggest" argument
 * (int/float/double/...)
 */
complexity arguments_to_complexity(exprlist, pbasic, precond, effects_list)
list exprlist;
basic *pbasic;
transformer precond;
list effects_list;
{
    complexity comp = make_zero_complexity();
    basic abasic = MAKE_INT_BASIC;

    trace_on("arguments");

    *pbasic = MAKE_INT_BASIC;
    MAPL (pa, {
	expression e = EXPRESSION(CAR(pa));
	complexity ctmp = expression_to_complexity(e, &abasic, precond, effects_list);
	complexity_add(&comp, ctmp);
	if (is_inferior_basic(*pbasic, abasic)) {
	    free_basic(*pbasic);
	    *pbasic = simple_basic_dup(abasic);
	}
    },exprlist);

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "argument comp is at %p and value is ", comp);
	complexity_fprint(stderr, comp, false, true);
    }
    complexity_check_and_warn("arguments_to_complexity", comp);

    trace_off();
    return(comp);
}

/* 2nd element of call        --arguments */
/* 2nd element of reference   --indices   */
complexity expression_to_complexity(expr, pbasic, precond, effects_list)
expression expr;
basic *pbasic;
transformer precond;
list effects_list;
{
    syntax s = expression_syntax(expr);
    complexity comp = make_zero_complexity();// complexity_undefined;

    trace_on("expression");

    if ( s != syntax_undefined )
	comp = syntax_to_complexity(s, pbasic, precond, effects_list);
    else
	pips_internal_error("syntax undefined");

    complexity_check_and_warn("expression_to_complexity", comp);

    trace_off();
    return(comp);
}

complexity subscript_to_complexity(subscript sub,
        basic * pbasic, transformer precond, list effects_list) {

    complexity cl = expression_to_complexity(
            subscript_array(sub),
            pbasic,
            precond,
            effects_list);
    complexity cr = indices_to_complexity(
            subscript_indices(sub),
            pbasic,
            precond,
            effects_list);
    complexity_add(&cl,cr);
    complexity_rm(&cr);
    complexity_float_add(&cl,1.f);/*SG:let us assume . cost 1 */
    return cl;

}

/* the only available element of expression */
complexity syntax_to_complexity(s, pbasic, precond, effects_list)
syntax s;
basic *pbasic;
transformer precond;
list effects_list;
{
  complexity comp = make_zero_complexity();// complexity_undefined;

  trace_on("syntax");

  switch (syntax_tag(s)) {
  case is_syntax_reference:
    comp = reference_to_complexity(syntax_reference(s), pbasic,
				   precond, effects_list);
    break;
  case is_syntax_range:
    comp = range_to_complexity(syntax_range(s), precond, effects_list);
    break;
  case is_syntax_call:
    comp = call_to_complexity(syntax_call(s), pbasic, precond, effects_list);
    break;
  case is_syntax_cast: {
    cast c = syntax_cast(s);
    /* A cost for casting could be added, although casting is mostly
       used for typing issue. However, (float) 2 has a cost.  */
	/* TypeCast operation added by Molka Becher to complexity_cost_tables 
	   for handling the cost of cast when it is necessary, 24 March 2011 */
    basic ib = MAKE_INT_BASIC;
    comp = expression_to_complexity(cast_expression(c), pbasic, precond, effects_list);
    complexity c1 = make_constant_complexity((float)intrinsic_cost(TYPE_CAST_COST, &ib));
    complexity_add(&comp, c1);
    complexity_rm(&c1);
    break;
  }
  case is_syntax_sizeofexpression: {
    sizeofexpression se = syntax_sizeofexpression(s);
    if(sizeofexpression_expression_p(se))
      comp = expression_to_complexity(sizeofexpression_expression(se),
				      pbasic, precond, effects_list);
    else {
      /* Compiler generated constant: equivalent to a constant load --> 0*/
      comp = make_zero_complexity();
    }
    break;
  }
  case is_syntax_subscript:
    comp=subscript_to_complexity(syntax_subscript(s),pbasic,precond,effects_list);
    break;
  case is_syntax_application:
    pips_internal_error("Not implemented yet");
    break;
  case is_syntax_va_arg:
    pips_internal_error("Not implemented yet");
    break;
  default:
    pips_internal_error("syntax_tag %d, not in %d->%d", (int)syntax_tag(s),
	       is_syntax_reference, is_syntax_va_arg);
  }

  complexity_check_and_warn("syntax_to_complexity", comp);

  trace_off();
  return(comp);
}

/* 1st element of syntax */
/* reference_to_complexity:
 *   return the complexity of the computing of the indices
 *   plus the complexity of one memory read
 *   Also return basic type of variable in *pbasic.
 */
complexity reference_to_complexity(ref, pbasic, precond, effects_list)
reference ref;
basic *pbasic;
transformer precond;
list effects_list;
{
    entity var = reference_variable(ref);
    list ind = reference_indices(ref);

    const char* name = module_local_name(var);
    basic b = variable_basic(type_variable(entity_basic_concrete_type(var)));
    basic ib = MAKE_INT_BASIC;    /* indices basic */
    complexity comp, ci, ca;      /* ci=compindexation, ca=compaccess */

    trace_on("reference %s", entity_name(var));

    if ( basic_int_p(b)     || basic_float_p(b)   ||
	 basic_complex_p(b) || basic_logical_p(b) )
	*pbasic = simple_basic_dup(b);
    else if (basic_string_p(b))
	*pbasic = MAKE_STRING_BASIC;
    else if (basic_derived_p(b) || basic_pointer_p(b)){
	*pbasic = MAKE_ADDRESS_BASIC;
    }
    else {
      user_warning("reference_to_complexity",
		     "basic_tag %d, not in 1->9\n", (int)basic_tag(b));
    }

    comp = indices_to_complexity(ind, &ib, precond, effects_list);

    /* ci=compindexation counts multi-dimension arrays indexation costs */

    switch ( gen_length(ind) ) {
    case 0:
	ci = make_zero_complexity();
	break;
    case 1:
	ci = make_constant_complexity((float)
				      intrinsic_cost(ONE_INDEX_NAME, &ib));
	break;
    case 2:
	ci = make_constant_complexity((float)
				      intrinsic_cost(TWO_INDEX_NAME, &ib));
	break;
    case 3:
	ci = make_constant_complexity((float)
				      intrinsic_cost(THREE_INDEX_NAME, &ib));
	break;
    case 4:
	ci = make_constant_complexity((float)
				      intrinsic_cost(FOUR_INDEX_NAME, &ib));
	break;
    case 5:
	ci = make_constant_complexity((float)
				      intrinsic_cost(FIVE_INDEX_NAME, &ib));
	break;
    case 6:
	ci = make_constant_complexity((float)
				      intrinsic_cost(SIX_INDEX_NAME, &ib));
	break;
    default:
	ci = make_constant_complexity((float)
				      intrinsic_cost(SEVEN_INDEX_NAME, &ib));
    }

    if(entity_register_p(var))
      ca = make_zero_complexity();
    else
      ca = make_constant_complexity((float)
				    intrinsic_cost(MEMORY_READ_NAME,
						   &ib));

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "\n");
	fprintf(stderr, ">>>  ref. %s argus   complexity: ", name);
	complexity_fprint(stderr, comp, false, true);
	fprintf(stderr, ">>>  ref. %s access  complexity: ", name);
	complexity_fprint(stderr, ca, false, true);
	fprintf(stderr, ">>>  ref. %s indices complexity: ", name);
	complexity_fprint(stderr, ci, false, true);
    }

    complexity_add(&comp, ca);
    complexity_add(&comp, ci);
    complexity_rm(&ca);
    complexity_rm(&ci);

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, ">>>  ref. %s total   complexity: ", name);
	complexity_fprint(stderr, comp, true, true);
	fprintf(stderr,"\n");
    }

    complexity_check_and_warn("reference_to_complexity", comp);    
    
    trace_off();
    return(comp);
}

/* 2nd element of reference */
/* exactly the same with arguments_to_complexity
 * I add this function in order to make the structure clearer
 * Apr. 15,91
 */
complexity indices_to_complexity(exprlist, pbasic, precond, effects_list)
list exprlist;
basic *pbasic;
transformer precond;
list effects_list;
{
    complexity comp = make_zero_complexity();
    basic ibasic = MAKE_INT_BASIC;

    trace_on("indices");

    *pbasic = MAKE_INT_BASIC;

    MAPL (pa, {
	expression e = EXPRESSION(CAR(pa));
	complexity ctmp = expression_to_complexity(e, &ibasic, precond, effects_list);
	complexity_add(&comp, ctmp);
	complexity_rm(&ctmp);
	
	if (is_inferior_basic(*pbasic, ibasic)) {
	    free_basic(*pbasic);
	    *pbasic = simple_basic_dup(ibasic);
	}
    },exprlist);

    complexity_check_and_warn("indices_to_complexity", comp);    
    
    trace_off();
    return(comp);
}

/* 2nd element of syntax */
complexity range_to_complexity(rng, precond, effects_list)
range rng;
transformer precond;
list effects_list;
{
    complexity compl, compu, compi;
    expression lower = range_lower(rng);
    expression upper = range_upper(rng);
    expression incr  = range_increment(rng);
    basic rngbasic;

    trace_on("range");

    if (!expression_undefined_p(lower))
	compl = expression_to_complexity(lower, &rngbasic, precond,
					 effects_list);
    else 
	pips_internal_error("lower undefined");

    if (!expression_undefined_p(upper))
	compu = expression_to_complexity(upper, &rngbasic, precond,
					 effects_list);
    else 
	pips_internal_error("upper undefined");

    if (!expression_undefined_p(incr))
	compi = expression_to_complexity(incr, &rngbasic, precond,
					 effects_list);
    else 
	pips_internal_error("increment undefined");

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "\n");
	fprintf(stderr, "---  range lower complexity: ");
	complexity_fprint(stderr, compl, false, true);
	fprintf(stderr, "---  range upper complexity: ");
	complexity_fprint(stderr, compu, false, true);
	fprintf(stderr, "---  range incr  complexity: ");
	complexity_fprint(stderr, compi, false, true);
    }

    complexity_add(&compl, compu);
    complexity_add(&compl, compi);
    complexity_rm(&compu);
    complexity_rm(&compi);

    if (get_bool_property("COMPLEXITY_INTERMEDIATES")) {
	fprintf(stderr, "---  range total complexity: ");
	complexity_fprint(stderr, compl, true, true);
	fprintf(stderr, "\n");
    }
    complexity_check_and_warn("range_to_complexity", compl);

    trace_off();
    return(compl);
}






