/* 
 *
 * This phase is used for PHRASE project. 
 *
 *
 * NB: The PHRASE project is an attempt to automatically (or semi-automatically)
 * transform high-level language for partial evaluation in reconfigurable logic 
 * (such as FPGAs or DataPaths). 
 *
 * This pass tries to generate finite state machine from arbitrary code 
 * by applying rules numeroting branches of a dependance graph and using 
 * it as state variable for the finite state machine. 
 *
 * alias fsm_generation 'FSM Generation'
 *
 * fsm_generation        > MODULE.code
 *       < PROGRAM.entities
 *       < MODULE.dg
 *       < MODULE.code
 *
 */ 

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "text-util.h"

#include "dg.h"
#include "transformations.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "ricedg.h"
#include "semantics.h"
#include "control.h"

#include "fsm_generation.h"

static graph dependence_graph;

/**
 * DEBUG FUNCTION: return a string representing the type of the
 * statement (SEQUENCE, CALL, etc...)
 */
static string statement_type_as_string (statement stat)
{
  instruction i = statement_instruction(stat);
  switch (instruction_tag(i)) {
  case is_instruction_test: {
    return strdup("TEST");   
    break;
  }
  case is_instruction_sequence: {
    return strdup("SEQUENCE");   
    break;
  }
  case is_instruction_loop: {
    return strdup("LOOP");   
    break;
  }
  case is_instruction_whileloop: {
    return strdup("WHILELOOP");   
    break;
  }
  case is_instruction_forloop: {
    return strdup("FORLOOP");   
    break;
  }
  case is_instruction_call: {
    return strdup("CALL");   
    break;
  }
  case is_instruction_unstructured: {
    return strdup("UNSTRUCTURED");   
    break;
  }
  case is_instruction_goto: {
    return strdup("GOTO");   
    break;
  }
  default:
    return strdup("UNDEFINED");   
    break;
  }
}

/**
 * This function build and return an expression given
 * an entity an_entity
 */
expression make_expression_from_entity(entity an_entity)
{
  return make_entity_expression(an_entity, NIL);
  
  /* return(make_expression(make_syntax(is_syntax_call, 
     make_call(state_variable, NIL)),
     normalized_undefined)); */
}

expression make_expression_with_state_variable(entity state_variable,
					       int value,
					       string intrinsic_name)
					       
{
  return MakeBinaryCall (entity_intrinsic(intrinsic_name),
			 make_expression_from_entity (state_variable),
			 int_expr (value));
}

/**
 * This function creates (and add declaration) state variable if this
 * variable doesn't exist. This variable will be used in Finite State
 * Machine code built during this phase. Additionaly, this function return
 * the created or found variable.
 */
static entity create_state_variable (string module_name) 
{
  entity module;
  entity new_state_variable;
  bool declaration_need_to_be_added = FALSE;

  module = local_name_to_top_level_entity(module_name);

  /* Assert that entity represent a value code */
  pips_assert("it is a code", value_code_p(entity_initial(module)));

  if ((gen_find_tabulated(concatenate(module_name, 
				      MODULE_SEP_STRING, 
				      STATE_VARIABLE_NAME, 
				      NULL),
			  entity_domain)) == entity_undefined) {
    declaration_need_to_be_added = TRUE;
  } 

  new_state_variable = find_or_create_scalar_entity (STATE_VARIABLE_NAME,
						     module_name,
						     is_basic_int);

  /* new_state_variable =  make_scalar_integer_entity ("state",
     module_name); */
  
  if (declaration_need_to_be_added) {
    add_variable_declaration_to_module(module, new_state_variable);
  }

  /* c = value_code(entity_initial(module));
     code_declarations(c) = CONS (ENTITY, 
     gen_once (new_state_variable, code_declarations(c)); */
  //gen_once (new_state_variable, statement_declarations(stat));
  
  return new_state_variable;
}

/**
 * This function build and return a statement representing the
 * initial assigment of the state_variable, given the UNSTRUCTURED
 * statement stat.
 */
static statement make_state_variable_assignement_statement (statement stat, entity state_variable, int assignement_value) 
{
  statement returned_statement;
  instruction new_instruction;
  call assignment_call;

  assignment_call = make_call (entity_intrinsic(ASSIGN_OPERATOR_NAME),
			       CONS(EXPRESSION, 
				    make_expression_from_entity(state_variable), 
				    CONS(EXPRESSION, int_expr (assignement_value), NIL)));
  
  new_instruction 
    = make_instruction(is_instruction_call,
		       assignment_call);
  
  returned_statement = make_statement(statement_label(stat),
				      statement_number(stat),
				      statement_ordering(stat),
				      empty_comments,
				      new_instruction,
				      NIL,NULL);  

  return returned_statement;
}

/**
 * Return the state variable value corresponding to the entry
 * in a unstructured statement
 */
static int entry_state_variable_value_for_unstructured (statement stat) 
{
  unstructured u;

  pips_assert("Statement is UNSTRUCTURED in FSM_GENERATION", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_unstructured);

  u = instruction_unstructured(statement_instruction(stat));

  return statement_ordering(control_statement(unstructured_entry(u)));
}

/**
 * Return the state variable value corresponding to the exit
 * in a unstructured statement
 * NB: always return 0
 */
static int exit_state_variable_value_for_unstructured (statement stat) 
{
  pips_assert("Statement is UNSTRUCTURED in FSM_GENERATION", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_unstructured);

  return 0;
}

/**
 * This function build and return a statement representing the
 * initial assigment of the state_variable, given the UNSTRUCTURED
 * statement stat.
 */
static statement make_reset_state_variable_statement (statement stat, entity state_variable) 
{
  return make_state_variable_assignement_statement 
    (stat, 
     state_variable, 
     entry_state_variable_value_for_unstructured(stat));
}

/**
 * This function build a transition statement (a TEST statement)
 * corresponding to the current control current_node and the
 * root_statement root_statement. This TEST statement takes a condition on
 * the state_variable having the value matching the statement ordering
 * value, and the control statement for the test_true value. The
 * test_false value is set with a continue statement, before to be
 * eventually replaced in next control node by a new statement.
 */
static statement make_transition_statement(control current_node,
					   statement root_statement, 
					   entity state_variable)
{
  statement returned_statement = NULL;
  statement transition_statement = NULL;
  statement stat = control_statement (current_node);
  instruction test_instruction;
  instruction transition_instruction;
  sequence transition_sequence;
  test new_test;
  expression test_condition;
  int successors_nb;

  pips_debug(2,"\n\nTRANSITION: Module statement: ###############################\n");
  ifdebug(2) {
    print_statement(stat);
  }
  pips_debug(2,"domain number = %d\n", statement_domain_number(stat));
  pips_debug(2,"entity = UNDEFINED\n");
  pips_debug(2,"statement number = %d\n", statement_number(stat));
  pips_debug(2,"statement ordering = %d\n", statement_ordering(stat));
  if (statement_with_empty_comment_p(stat)) {
    pips_debug(2,"statement comments = EMPTY\n");
  }
  else {
    pips_debug(2,"statement comments = %s\n", statement_comments(stat));
  }
  pips_debug(2,"statement instruction = %s\n", statement_type_as_string(stat));

  pips_debug(2,"\npredecessors = %d\n", gen_length(control_predecessors(current_node)));
  pips_debug(2,"successors = %d\n", gen_length(control_successors(current_node)));

  test_condition 
    = make_expression_with_state_variable (state_variable,
					   statement_ordering(stat),
					   EQUIV_OPERATOR_NAME);

  successors_nb = gen_length(control_successors(current_node));

  if ((successors_nb == 0) || (successors_nb == 1)) {
    /* This is the exit node, or a non-test statement */
    int next_value;
    statement state_variable_assignement;
    
    if (successors_nb == 0) {
      /* This is the exit node, just generate exit code for state_variable */
      next_value = exit_state_variable_value_for_unstructured (root_statement);
    }
    else { /* successors_nb == 1 */
      /* This is a "normal" node, ie not a TEST statement, just add 
	 assignement for state_variable with new value */
      next_value 
	= statement_ordering
	(control_statement
	 (CONTROL(gen_nth(0,control_successors(current_node)))));
    }
    
    state_variable_assignement
      = make_state_variable_assignement_statement 
      (stat, state_variable, next_value);
    
    transition_sequence 
      = make_sequence (CONS(STATEMENT, 
			    stat, 
			    CONS(STATEMENT, state_variable_assignement, NIL)));

    transition_instruction 
      = make_instruction(is_instruction_sequence,
			 transition_sequence);
  
    transition_statement = make_statement(entity_empty_label(),
					  statement_number(stat),
					  statement_ordering(stat),
					  empty_comments,
					  transition_instruction,NIL,NULL);
  }
  else if (successors_nb == 2) {
    /* This is a "test" node, ie with a TEST statement, just add 
       assignement for state_variable with new value after each
       statement in TEST*/
    int value_if_true = statement_ordering
      (control_statement
       (CONTROL(gen_nth(0,control_successors(current_node)))));
    int value_if_false = statement_ordering
      (control_statement
       (CONTROL(gen_nth(1,control_successors(current_node)))));
    statement transition_statement_if_true;
    statement transition_statement_if_false;
    sequence transition_sequence_if_true;
    sequence transition_sequence_if_false;
    instruction transition_instruction_if_true;
    instruction transition_instruction_if_false;
    statement state_variable_assignement_if_true;
    statement state_variable_assignement_if_false;
    statement old_statement_if_true;
    statement old_statement_if_false;
    test current_test;
    
    pips_assert("Statement with 2 successors is a TEST in FSM_GENERATION", 
		instruction_tag(statement_instruction(stat)) 
		== is_instruction_test);
    
    current_test = instruction_test (statement_instruction(stat));

    // Begin computing for the TRUE statement

    old_statement_if_true = test_true(current_test);
    
    state_variable_assignement_if_true
      = make_state_variable_assignement_statement 
      (stat, state_variable, value_if_true);
    
    transition_sequence_if_true
      = make_sequence (CONS(STATEMENT, 
			    old_statement_if_true, 
			    CONS(STATEMENT, state_variable_assignement_if_true, NIL)));

    transition_instruction_if_true 
      = make_instruction(is_instruction_sequence,
			 transition_sequence_if_true);
  
    transition_statement_if_true 
      = make_statement
      (entity_empty_label(),
       statement_number(stat),
       statement_ordering(stat),
       empty_comments,
       transition_instruction_if_true,NIL,NULL);

    test_true(current_test) = transition_statement_if_true;

    // Begin computing for the FALSE statement

    old_statement_if_false = test_false(current_test);
    
    state_variable_assignement_if_false
      = make_state_variable_assignement_statement 
      (stat, state_variable, value_if_false);
    
    transition_sequence_if_false
      = make_sequence (CONS(STATEMENT, 
			    old_statement_if_false, 
			    CONS(STATEMENT, state_variable_assignement_if_false, NIL)));

    transition_instruction_if_false 
      = make_instruction(is_instruction_sequence,
			 transition_sequence_if_false);
  
    transition_statement_if_false 
      = make_statement
      (entity_empty_label(),
       statement_number(stat),
       statement_ordering(stat),
       empty_comments,
       transition_instruction_if_false,NIL,NULL);

    test_false(current_test) = transition_statement_if_false;

    transition_statement = stat;

  }
  else {
    pips_assert("I should NOT be there :-)", 2+2 != 4); /* :-) */
  }

  new_test = make_test (test_condition, transition_statement, 
			make_continue_statement(entity_undefined));
  
  test_instruction = make_instruction (is_instruction_test,new_test);

  returned_statement = make_statement (statement_label(root_statement),
				       statement_number(root_statement),
				       statement_ordering(root_statement),
				       empty_comments,
				       test_instruction,NIL,NULL);

  return returned_statement;

}

/**
 * This function build and return a statement representing the
 * transitions computation in the FSM, given the UNSTRUCTURED
 * statement stat.
 */
static statement make_fsm_transitions_statement (statement stat, 
						 entity state_variable)
{
  statement returned_statement = NULL;
  statement current_statement = NULL;
  unstructured nodes_graph;
  list blocs = NIL ;

  pips_assert("Statement is UNSTRUCTURED in FSM_GENERATION", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_unstructured);

  nodes_graph = instruction_unstructured(statement_instruction(stat));
  
  /*gen_recurse(unstructured_entry(nodes_graph), control_domain,
    transitions_filter, transitions_statements);*/
  CONTROL_MAP (current_control, { 
    statement transition_statement;
    transition_statement = make_transition_statement (current_control, 
						      stat, 
						      state_variable);
    if (returned_statement == NULL) {
      returned_statement = transition_statement;
      current_statement = returned_statement;
    }
    else {
      instruction i = statement_instruction(current_statement);
      test t;
      pips_assert("Statement is TEST in FSM_GENERATION transitions", 
		  instruction_tag(i) == is_instruction_test);
      t = instruction_test(i);
      test_false (t) = transition_statement;
      current_statement = transition_statement;
    }
  }, unstructured_entry(nodes_graph), blocs);

  pips_debug(2,"blocs count = %d\n", gen_length(blocs));
  
  return returned_statement;
}

/**
 * This function build and return a statement representing the
 * FSM code equivalent to the given unstructured statement stat.
 */
static statement make_fsm_from_statement(statement stat, entity state_variable)
{
  statement returned_statement;
  statement loop_statement;
  whileloop new_whileloop;
  expression loop_condition;
  statement loop_body;
  entity loop_entity = NULL;
  evaluation loop_evaluation = NULL;
  instruction loop_instruction;
  instruction sequence_instruction;
  sequence new_sequence;
  
  /* Assert that given stat is UNSTRUCTURED */
  pips_assert("Statement is UNSTRUCTURED in FSM_GENERATION", 
	      instruction_tag(statement_instruction(stat)) 
	      == is_instruction_unstructured);
  
  /* Create loop condition: state variable is not equal to exit value */
  loop_condition 
    = make_expression_with_state_variable 
    (state_variable,
     exit_state_variable_value_for_unstructured(stat),
     NON_EQUIV_OPERATOR_NAME);
  
  /* Evaluation is done BEFORE to enter the loop */
  loop_evaluation = make_evaluation_before();

  /* No label for loop */
  loop_entity = entity_empty_label();

  /* Computes the statement representing the transitions */
  loop_body = make_fsm_transitions_statement (stat,
					      state_variable);

  /* Build the loop */
  new_whileloop = make_whileloop(loop_condition, 
				 loop_body, 
				 loop_entity, 
				 loop_evaluation);

  loop_instruction = make_instruction(is_instruction_whileloop,new_whileloop);

  loop_statement = make_statement(statement_label(stat),
				  statement_number(stat),
				  statement_ordering(stat),
				  empty_comments,
				  loop_instruction,NIL,NULL);
  
  
  new_sequence 
    = make_sequence (CONS(STATEMENT, 
			  make_reset_state_variable_statement(stat, 
							      state_variable), 
			  CONS(STATEMENT, loop_statement, NIL)));
    
  sequence_instruction 
    = make_instruction(is_instruction_sequence,
		       new_sequence);
  
  returned_statement = make_statement(statement_label(stat),
				      statement_number(stat),
				      statement_ordering(stat),
				      empty_comments,
				      sequence_instruction,NIL,NULL);
  /*statement_instruction(loop_body) 
    = make_instruction_block(CONS(STATEMENT,returned_statement,NIL));
  */
  return returned_statement;
}

/* 
 * This function is recursively called during FSMization. It takes
 * the statement to fsmize stat as parameter, while module_name is 
 * the name of the module where FSMization is applied.
 */
static statement fsmize_statement (statement stat, 
				   entity state_variable,
				   string module_name)
{
  statement returned_statement = NULL;
  instruction i = statement_instruction(stat);

  pips_debug(2,"\n\nTEST: Module statement: =====================================\n");
  ifdebug(2) {
    print_statement(stat);
  }
  pips_debug(2,"domain number = %d\n", statement_domain_number(stat));
  pips_debug(2,"entity = UNDEFINED\n");
  pips_debug(2,"statement number = %d\n", statement_number(stat));
  pips_debug(2,"statement ordering = %d\n", statement_ordering(stat));
  if (statement_with_empty_comment_p(stat)) {
    pips_debug(2,"statement comments = EMPTY\n");
  }
  else {
    pips_debug(2,"statement comments = %s\n", statement_comments(stat));
  }
  pips_debug(2,"statement instruction = %s\n", statement_type_as_string(stat));
  switch (instruction_tag(i)) {
  case is_instruction_test: 
    {
    // Declare the test data structure which will be used
    test current_test = instruction_test(i);
    statement true_statement, new_true_statement;
    statement false_statement, new_false_statement;

    pips_debug(2, "TEST\n");   

    // Compute new statement for true statement, and replace
    // the old one by the new one
    true_statement = test_true (current_test);
    new_true_statement = fsmize_statement(true_statement, state_variable, module_name);
    if (new_true_statement != NULL) {
      test_true (current_test) = new_true_statement;
    }

    // Do the same for the false statement
    false_statement = test_false (current_test);
    new_false_statement = fsmize_statement(false_statement, state_variable, module_name);
    if (new_false_statement != NULL) {
      test_false (current_test) = new_false_statement;
    }
    
    break;
  }
  case is_instruction_sequence: 
    {
    sequence seq = instruction_sequence(i);
    pips_debug(2, "SEQUENCE\n");   
    MAP(STATEMENT, current_stat,
    {
      statement new_stat = fsmize_statement(current_stat, state_variable, module_name);
      if (new_stat != NULL) {
	gen_list_patch (sequence_statements(seq), current_stat, new_stat);
      }
    }, sequence_statements(seq));
    break;
  }
  case is_instruction_loop: {
    pips_debug(2, "LOOP\n");   
    break;
  }
  case is_instruction_whileloop: {
    pips_debug(2, "WHILELOOP\n");   
    break;
  }
  case is_instruction_forloop: {
    pips_debug(2, "FORLOOP\n");   
    break;
  }
  case is_instruction_call: {
    pips_debug(2, "CALL\n");   
    break;
  }
  case is_instruction_unstructured: {
    statement new_statement = make_fsm_from_statement (stat, state_variable);
    pips_debug(2, "Displaying statement\n");   
    print_statement (new_statement);
    pips_debug(2, "UNSTRUCTURED\n");  
    returned_statement = new_statement;
    break;
  }
  case is_instruction_goto: {
    pips_debug(2, "GOTO\n");   
    break;
  }
  default:
    pips_debug(2, "UNDEFINED\n");   
    break;
  }

  return returned_statement;
}

/*********************************************************
 * Phase main
 *********************************************************/

bool fsm_generation(string module_name)
{
  entity module;
  entity state_variable;

   /* get the resources */
  statement stat = (statement) db_get_memory_resource(DBR_CODE, 
						      module_name, 
						      TRUE);

  module = local_name_to_top_level_entity(module_name);
  
  set_current_module_statement(stat);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  dependence_graph = 
    (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);
  
  debug_on("FSM_GENERATION_DEBUG_LEVEL");
  /* Now do the job */
  
  /* gen_recurse(stat, statement_domain,
     test_fsm_filter, test_fsm); */
  state_variable = create_state_variable(module_name);

  fsmize_statement(stat, state_variable, module_name);
  
  pips_assert("Statement is consistent after FSM_GENERATION", 
	       statement_consistent_p(stat));
  
  /* Reorder the module, because new statements have been added */  
  module_reorder(stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, 
			 compute_callees(stat));
  
  /* update/release resources */
  reset_current_module_statement();
  reset_current_module_entity();
  
  debug_off();
  
  return TRUE;
}
