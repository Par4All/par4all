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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ri.h"
#include "ri-util.h"
#include "graph.h"
#include "dg.h"
#include "database.h"

#include "misc.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "control.h"

#include "constants.h"
#include "properties.h"
#include "resources.h"

#include "chains.h"
#include "ricedg.h"
#include "rice.h"

/* the dependence graph of the current loop nest */
graph dg = graph_undefined;

/* to know if do loop parallelization must be done */
bool rice_distribute_only;

/* This is an horrendous hack. ENCLOSING should be passed as an
   argument, but I don't have the courage to make the changes . */

int enclosing = 0 ;


void rice_unstructured(u,l)
unstructured u ;
int l ;
{
    cons *blocs = NIL ;

    CONTROL_MAP(c, {
	statement st = rice_statement(control_statement(c),l);
	control_statement(c) = st;
    }, unstructured_control(u), blocs);

    gen_free_list( blocs );
}

statement rice_statement(stat,l)
statement stat;
int l ;
{
    instruction istat = statement_instruction(stat);

    switch (instruction_tag(istat)) {
      case is_instruction_block: {
	  MAPL(pc, {
	      statement st = STATEMENT(CAR(pc));
	      STATEMENT(CAR(pc)) = rice_statement(st,l);
	  }, instruction_block(istat));
	  break;
      }
      case is_instruction_test: {
	  test obj_test = instruction_test(istat);
	  test_true(obj_test) = rice_statement(test_true(obj_test),l);
	  test_false(obj_test) = rice_statement(test_false(obj_test),l);
	  break;
      }
      case is_instruction_loop: {
	  stat = rice_loop(stat,l);
	  ifdebug(7){
	      fprintf(stderr, "\nparalized loop :");
	      if (gen_consistent_p((statement)stat))
		  fprintf(stderr," gen consistent ");
	  }
	  break;
      }
      case is_instruction_goto: 
      case is_instruction_call: 
	break;
      case is_instruction_unstructured: {
	  unstructured obj_unstructured = instruction_unstructured(istat);
	  rice_unstructured(obj_unstructured,l);
	  break;
      }
      default:
	fprintf(stderr, "[rice_statement] case default reached\n");
	abort();
    }

    return(stat);
}

statement 
rice_loop(statement stat, int l)
{
    statement nstat;
    instruction istat = statement_instruction(stat);
    set region;

    ifdebug(1) {
	debug(1, "rice_loop", "original nest of loops:\n\n");
	print_statement(stat);
    }

    if ((region = distributable_loop(stat)) == set_undefined) {
	int so = statement_ordering(stat);
	user_warning("rice_loop", 
		     "Cannot apply Allen & Kennedy's algorithm on "
		     "Loop %d at Statement %d (%d, %d)\n",
		     label_local_name(loop_index(instruction_loop(istat))),
		     statement_number(stat),
		     ORDERING_NUMBER(so), ORDERING_STATEMENT(so));

	enclosing++ ;
	loop_body(instruction_loop(istat)) = 
	    rice_statement(loop_body(instruction_loop(istat)),l+1);
	enclosing-- ;
	return(stat);
    }

    ifdebug(2) {
	debug(2, "rice_loop", "applied on region:\n");
	print_statement_set(stderr, region);
    }

    set_enclosing_loops_map( loops_mapping_of_statement(stat) );

    nstat = CodeGenerate(dg, region, l, TRUE);
    ifdebug(7){
	pips_debug(7, "consistency checking for CodeGenerate output: ");
	if (gen_consistent_p((statement)nstat))
	    fprintf(stderr," gen consistent\n");
    }
    pips_assert( "nstat is defined", nstat != statement_undefined ) ;
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
	print_statement(nstat);
    }

    /* StatementToContext should be freed here. but it is not easy. */
    set_free(region);

    STATEMENT_MAPPING_MAP(ignore, val, {
	gen_free_list( (list) val ) ;
    }, get_enclosing_loops_map()) ;

    clean_enclosing_loops();

    return nstat;
}

static void
print_statistics(string module, string msg, statement s)
{
    if (get_bool_property("PARALLELIZATION_STATISTICS"))
    {
	fprintf(stderr, "%s %s parallelization statistics", module, msg);
	print_number_of_loop_statistics(stderr, "", s);
    }
}

/*
 * RICE_DEBUG_LEVEL (properly?) included, FC 23/09/93
 */
static bool
do_it(
    string mod_name,
    bool distribute_p,
    string what)
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

    print_statistics(mod_name, "ante", mod_stat);

    ifdebug(7)
    {
	fprintf(stderr, "\nTesting NewGen consistency for initial code %s:\n",
		mod_name);
	if (gen_consistent_p((statement)mod_stat))
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
	if (gen_consistent_p((statement)mod_parallel_stat))
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
    initialize_ordering_to_statement(mod_parallel_stat);

    rice_distribute_only = distribute_p ;
    enclosing = 0;
    rice_statement(mod_parallel_stat,1);   

    /* Regenerate statement_ordering for the parallel code */
    reset_ordering_to_statement();
    module_body_reorder(mod_parallel_stat);

    ifdebug(7)
    {
	fprintf(stderr, "\nparallelized code %s:",mod_name);
	if (gen_consistent_p((statement)mod_parallel_stat))
	    fprintf(stderr," gen consistent ");
    }

    debug_off();

    /* FI: This may be parallel or sequential code */
    DB_PUT_MEMORY_RESOURCE(what, strdup(mod_name), (char*) mod_parallel_stat);

    print_statistics(mod_name, "post", 
		     mod_parallel_stat);

    dg = graph_undefined;
    reset_current_module_statement();

    return TRUE;
}


/****************************************************** PIPSMAKE INTERFACE */

bool 
distributer(string mod_name)
{  
    bool success;
    entity module = local_name_to_top_level_entity(mod_name);

    set_current_module_entity(module);

    debug_on("RICE_DEBUG_LEVEL");

    success = do_it( mod_name, TRUE, DBR_CODE ) ;

    debug_off();  
    reset_current_module_entity();

    return success;
}

static bool 
rice(string mod_name)
{ 
    bool success = TRUE;
    entity module = local_name_to_top_level_entity(mod_name);
    set_current_module_entity(module);

    success = do_it( mod_name, FALSE, DBR_PARALLELIZED_CODE);

    reset_current_module_entity();
    return success;
}

bool rice_all_dependence(mod_name)
char *mod_name;
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", TRUE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", FALSE ) ;
    return rice( mod_name ) ;
}

bool rice_data_dependence(mod_name)
char *mod_name;
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", TRUE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", TRUE ) ;
    return rice( mod_name ) ;
}

bool rice_cray(mod_name)
char *mod_name;
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", FALSE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", FALSE ) ;
    return rice( mod_name ) ;
}
