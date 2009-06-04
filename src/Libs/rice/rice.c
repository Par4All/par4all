/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/* Remi Triolet
 *
 * Modifications:
 *
 * - Bruno Baron:
 *
 * - Francois Irigoin: I do not understand why regions of statements are
 *   implemented as set of statements instead of set of statement orderings
 *   since the dependence graph refers to statement orderings.
 *
 * - Francois Irigoin: use a copy of the code statement to generate
 *   the parallel code, instead of using side effects on the sequential
 *   code statement. This works because the dependence graph uses
 *   persistent pointers towards statement, statement_ordeging, and because
 *   the references are the same in the sequential code and in its copy.
 *   So references are still accessible, although it may be useless for
 *   parallelization.
 */

#include "local.h"

/* the dependence graph of the current loop nest */
graph dg = graph_undefined;

/* to know if do loop parallelization must be done */
bool rice_distribute_only;

/* This is an horrendous hack. ENCLOSING should be passed as an
   argument, but I don't have the courage to make the changes . */

int enclosing = 0 ;


void rice_unstructured(u,l,codegen_fun)
unstructured u ;
int l ;
statement (*codegen_fun)(statement, graph, set, int, bool);
{
    cons *blocs = NIL ;

    CONTROL_MAP(c, {
	statement st = rice_statement(control_statement(c),l,codegen_fun);
	control_statement(c) = st;
    }, unstructured_control(u), blocs);

    gen_free_list( blocs );
}

statement rice_statement(statement stat, 
			 int l, 
			 statement (*codegen_fun)(statement, graph, set, int, bool))
{
    instruction istat = statement_instruction(stat);
    statement new_stat = stat; // Most statements are modified by side effects

    switch (instruction_tag(istat)) {
      case is_instruction_block: {
	  MAPL(pc, {
	      statement st = STATEMENT(CAR(pc));
	      STATEMENT_(CAR(pc)) = rice_statement(st,l,codegen_fun);
	  }, instruction_block(istat));
	  break;
      }
      case is_instruction_test: {
	  test obj_test = instruction_test(istat);
	  test_true(obj_test) = rice_statement(test_true(obj_test),l,codegen_fun);
	  test_false(obj_test) = rice_statement(test_false(obj_test),l,codegen_fun);
	  break;
      }
      case is_instruction_loop: {
	  new_stat = rice_loop(stat,l,codegen_fun);
	  ifdebug(7){
	      if(statement_loop_p(new_stat))
	          pips_debug(7, "regenerated loop is %s:",
			  execution_sequential_p(loop_execution(instruction_loop(statement_instruction(new_stat))))?
			  "sequential" : "parallel");
	      else
	          pips_debug(7, "regenerated code:");
	      if (statement_consistent_p(new_stat))
		  fprintf(stderr, " consistent\n");
	      print_parallel_statement(new_stat);
	  }
	  break;
      }
      case is_instruction_whileloop:
      case is_instruction_forloop:
      case is_instruction_goto: 
      case is_instruction_call: 
	break;
      case is_instruction_unstructured: {
	  unstructured obj_unstructured = instruction_unstructured(istat);
	  rice_unstructured(obj_unstructured,l,codegen_fun);
	  break;
      }
      default:
	pips_internal_error("default case reached\n");
    }

    return(new_stat);
}


/* Eventually parallelize a do-loop with à la Rice algorithm.

   @return a statement with the same loop (not parallelized), a statement
   with a parallelized do-loop or an empty statement (the loop body was
   empty so it is trivially parallelized in this way).
*/
statement rice_loop(statement stat,
		    int l,
		    statement (*codegen_fun)(statement, graph, set, int, bool)
		    )
{
  statement nstat;
  instruction istat = statement_instruction(stat);
  set region;
  statement b = statement_undefined;

  ifdebug(1) {
    debug(1, "rice_loop", "original nest of loops:\n\n");
    print_statement(stat);
  }

  if ((region = distributable_loop(stat)) == set_undefined) {
    int so = statement_ordering(stat);
    user_warning("rice_loop",
		 "Cannot apply Allen & Kennedy's algorithm on "
		 "loop %s with index %s at Statement %d (%d, %d)"
		 " because it contains either tests or goto statements"
		 " which prohibit loop distribution. You could activate the"
		 " coarse_grain_parallelization rule.\n",
		 label_local_name(loop_label(instruction_loop(istat))),
		 entity_local_name(loop_index(instruction_loop(istat))),
		 statement_number(stat),
		 ORDERING_NUMBER(so), ORDERING_STATEMENT(so));

    enclosing++ ;
    loop_body(instruction_loop(istat)) =
      rice_statement(loop_body(instruction_loop(istat)),l+1,codegen_fun);
    enclosing-- ;
    return(stat);
  }

  ifdebug(2) {
    debug(2, "rice_loop", "applied on region:\n");
    print_statement_set(stderr, region);
  }

  set_enclosing_loops_map( loops_mapping_of_statement(stat) );
  nstat = codegen_fun(stat, dg, region, l, TRUE);

  ifdebug(7){
    pips_debug(7, "consistency checking for CodeGenerate output: ");
    if (statement_consistent_p(nstat))
      fprintf(stderr," gen consistent\n");
  }

  if (nstat == statement_undefined )
    /* The code generation did not generate anything, probably because the
       loop body was empty ("CONTINUE"/";", "{ }"), so no loop is
       generated: */
    nstat = make_empty_statement();

  /* FI: I'd rather not return a block when a unique loop statement has to
   * be wrapped.
   */
  pips_assert("block or loop",
	      instruction_block_p(statement_instruction(nstat)) ||
	      instruction_loop_p(statement_instruction(nstat))) ;
  statement_label(nstat) = entity_empty_label();
  statement_number(nstat) =
    (statement_block_p(nstat)? STATEMENT_NUMBER_UNDEFINED :
     statement_number(stat));
  statement_ordering(nstat) = statement_ordering(stat);
  statement_comments(nstat) = statement_comments(stat);
  /* Do not forget to move forbidden information associated with
     block: */
  fix_sequence_statement_attributes_if_sequence(nstat);
  ifdebug(1) {
    fprintf(stderr, "final nest of loops:\n\n");
    print_parallel_statement(nstat);
  }

  /* StatementToContext should be freed here. but it is not easy. */
  set_free(region);

  clean_enclosing_loops();

  return nstat;
}

/*
 * RICE_DEBUG_LEVEL (properly?) included, FC 23/09/93
 */
bool
do_it(
    string mod_name,
    bool distribute_p,
    string what,
    statement (*codegen_fun)(statement, graph, set, int, bool)
    )
{
    statement mod_stat = statement_undefined;
    /* In spite of its name, the new statement "mod_parallel_stat"
     * may be sequential if distribute_p is true
     */
    statement mod_parallel_stat = statement_undefined;

    set_current_module_statement( (statement)
				  db_get_memory_resource(DBR_CODE, 
							 mod_name, 
							 TRUE) );
    mod_stat = get_current_module_statement();

    debug_on("RICE_DEBUG_LEVEL");

    print_parallelization_statistics(mod_name, "ante", mod_stat);

    ifdebug(7)
    {
	pips_debug(7, "\nTesting NewGen consistency for initial code %s:\n",
		mod_name);
	if (statement_consistent_p((statement)mod_stat))
	    fprintf(stderr," NewGen consistent statement\n");
    }

    ifdebug(1) {
	debug(1, "do_it", "original sequential code:\n\n");
	print_statement(mod_stat);
    }

    mod_parallel_stat = copy_statement(mod_stat);

    ifdebug(7)
    {
	debug(7, "do_it",
	      "\nTesting NewGen consistency for copy code %s:",
		mod_name);
	if (statement_consistent_p((statement)mod_parallel_stat))
	    fprintf(stderr," NewGen consistent statement copy\n");
    }

    ifdebug(1) {
	debug(1, "do_it", "copy of sequential code:\n\n");
	print_statement(mod_stat);
    }

    if (graph_undefined_p(dg)) {
	dg = (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);
    }
    else {
	pips_error("do_it", "dg should be undefined\n");
    }

    /* Make sure the dependence graph points towards the code copy */
    if(ordering_to_statement_initialized_p())
	reset_ordering_to_statement();
    set_ordering_to_statement(mod_parallel_stat);

    rice_distribute_only = distribute_p ;
    enclosing = 0;
    // rice_statement works by side effects, most of the times, but
    // not for loops
    mod_parallel_stat = rice_statement(mod_parallel_stat,1,codegen_fun);

    /* Regenerate statement_ordering for the parallel code */
    reset_ordering_to_statement();
    module_body_reorder(mod_parallel_stat);

    ifdebug(7)
    {
	pips_debug(7, "\nparallelized code for module %s:",mod_name);
	if (statement_consistent_p(mod_parallel_stat))
	    fprintf(stderr," gen consistent\n");
	print_parallel_statement(mod_parallel_stat);
    }

    debug_off();

    /* FI: This may be parallel or sequential code */
    DB_PUT_MEMORY_RESOURCE(what, mod_name, (char*) mod_parallel_stat);

    print_parallelization_statistics(mod_name, "post", mod_parallel_stat);

    dg = graph_undefined;
    reset_current_module_statement();
    return TRUE;
}


/****************************************************** PIPSMAKE INTERFACE */

bool distributer(string mod_name)
{  
    bool success;
    entity module = local_name_to_top_level_entity(mod_name);

    set_current_module_entity(module);

    debug_on("RICE_DEBUG_LEVEL");

    success = do_it( mod_name, TRUE, DBR_CODE, &CodeGenerate ) ;

    debug_off();  
    reset_current_module_entity();

    return success;
}

static bool rice(string mod_name)
{ 
    bool success = TRUE;
    entity module = local_name_to_top_level_entity(mod_name);
    set_current_module_entity(module);

    success = do_it( mod_name, FALSE, DBR_PARALLELIZED_CODE, &CodeGenerate);

    reset_current_module_entity();
    return success;
}

bool rice_all_dependence(string mod_name)
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", TRUE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", FALSE ) ;
    return rice( mod_name ) ;
}

bool rice_data_dependence(string mod_name)
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", TRUE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", TRUE ) ;
    pips_user_warning("This phase is designed for experimental purposes only. The generated code is most likely to be illegal, especially if a privatization phase was performed before.\n");
    return rice( mod_name ) ;
}

bool rice_cray(string mod_name)
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", FALSE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", FALSE ) ;
    return rice( mod_name ) ;
}
