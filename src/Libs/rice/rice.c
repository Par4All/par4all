#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <values.h>

#include "genC.h"
#include "mapping.h"

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
graph dg;

/* to know if do loop parallelization must be done */
bool rice_distribute_only;

int Nbrdoall = 0;

/* This is an horrendous hack. ENCLOSING should be passed as an
   argument, but I don't have the courage to make the changes . */

int enclosing = 0 ;

void distributer(mod_name)
char *mod_name;
{  
    entity
	module = local_name_to_top_level_entity(mod_name);

    set_current_module_entity(module);

    debug_on("RICE_DEBUG_LEVEL");

    do_it( mod_name, TRUE, DBR_CODE ) ;

    debug_off();  
    reset_current_module_entity();
}

void rice_all_dependence(mod_name)
char *mod_name;
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", TRUE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", FALSE ) ;
    rice( mod_name ) ;
}

void rice_data_dependence(mod_name)
char *mod_name;
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", TRUE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", TRUE ) ;
    rice( mod_name ) ;
}

void rice_cray(mod_name)
char *mod_name;
{
    set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", FALSE ) ;
    set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", FALSE ) ;
    rice( mod_name ) ;
}

void rice(mod_name)
char *mod_name;
{ 
    entity
	module = local_name_to_top_level_entity(mod_name);

    set_current_module_entity(module);

    do_it( mod_name, FALSE, DBR_PARALLELIZED_CODE);
    ifdebug(1)
    {
	fprintf(stderr,"\nThe number of DOALLs :\n");
	fprintf(stderr," Nbrdoall=%d",Nbrdoall);
    }

    reset_current_module_entity();
}

/*
 * RICE_DEBUG_LEVEL (properly?) included, FC 23/09/93
 */
void do_it(mod_name, distribute_p, what ) 
char *mod_name ;
bool distribute_p ;
char *what ;
{
    statement mod_stat;
    instruction mod_inst;

    set_current_module_statement( (statement)
				  db_get_memory_resource(DBR_CODE, 
							 mod_name, 
							 FALSE) );
    mod_stat = get_current_module_statement();

    ifdebug(7)
    {
	fprintf(stderr, "\nTesting NewGen consistency for initial code %s:\n",
		mod_name);
	if (gen_consistent_p((statement)mod_stat))
	    fprintf(stderr," NewGen consistent statement\n");
	else 
	{    fprintf(stderr," NewGen inconsistent statement\n");
	    abort();
	}
    }

    debug_off();
    /* FI, BB: hack*/
    /*(void) db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, FALSE);*/
    mod_inst = statement_instruction(mod_stat);
    pips_assert( "do_it", instruction_unstructured_p(mod_inst)) ;
    dg = (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

    debug_on("RICE_DEBUG_LEVEL");
    rice_distribute_only = distribute_p ;
    enclosing = 0;
    rice_statement(mod_stat,1);   
    module_body_reorder(mod_stat);

    debug_off();
    ifdebug(7)
    {
	fprintf(stderr, "\nparalized code %s:",mod_name);
	if (gen_consistent_p((statement)mod_stat))
	    fprintf(stderr," gen consistent ");
	else {
	    fprintf(stderr,"\nFalse NewGen statement");
	    exit(1);
	}
    }
    DB_PUT_MEMORY_RESOURCE(what, strdup(mod_name), (char*) mod_stat);

    /* FI: hack to make hash tables consistent with the code;
     * their values are good but the pointers to statements are wrong;
     * the false introduction of a new version of the code forces a
     * recomputation of all of them (18 May 1993)
     */
    reset_current_module_statement();
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, mod_name, TRUE) );
    mod_stat = get_current_module_statement();

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), (char*) mod_stat);

    reset_current_module_statement();
}

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
	  if (get_debug_level() >=7){
	      fprintf(stderr, "\nparalized loop :");
	      if (gen_consistent_p((statement)stat))
		  fprintf(stderr," gen consistent ");
	      else {
		  fprintf(stderr,"\nFalse NewGen statement");
		  exit(1);
	      }
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

statement rice_loop(stat,l)
statement stat;
int l ;
{
    statement nstat;
    instruction istat = statement_instruction(stat);
    set region;

    if (get_debug_level() >= 1) {
	fprintf(stderr, "\n original nest of loops:\n\n");
	print_text(stderr, text_statement(entity_undefined, 0, stat));
    }

    if ((region = distributable_loop(stat)) == set_undefined) {
	user_warning("rice_loop", 
		     "can't apply kennedy's algorithm on this loop");

	enclosing++ ;
	loop_body(instruction_loop(istat)) = 
	    rice_statement(loop_body(instruction_loop(istat)),l+1);
	enclosing-- ;
	return(stat);
    }

    if (get_debug_level() >= 2) {
	fprintf(stderr, "[rice_loop] applied on region:\n");
	print_statement_set(stderr, region);
    }

    set_enclosing_loops_map( loops_mapping_of_statement(stat) );

    nstat = CodeGenerate(dg, region, l, TRUE);
    if (get_debug_level() >=7){
	fprintf(stderr, "\nCodeGenerate : ");
	if (gen_consistent_p((statement)nstat))
	    fprintf(stderr," gen consistent\n");
	else {
	    fprintf(stderr,"\nFalse NewGen statement");
	    exit(1);
	}
    }
    pips_assert( "rice_loop", nstat != statement_undefined ) ;
    pips_assert("rice_loop", 
		instruction_block_p(statement_instruction(nstat))) ;
    statement_label(nstat) = entity_empty_label();
    statement_number(nstat) = statement_number(stat);
    statement_ordering(nstat) = statement_ordering(stat);
    statement_comments(nstat) = statement_comments(stat);

    if (get_debug_level() >= 1) {
	fprintf(stderr, "final nest of loops:\n\n");
	print_text(stderr, text_statement(entity_undefined, 0, nstat));
    }

    /* StatementToContext should be freed here. but it is not easy. */
    set_free(region);

    STATEMENT_MAPPING_MAP(ignore, val, {
	gen_free_list( (list) val ) ;
    }, get_enclosing_loops_map()) ;

    reset_enclosing_loops_map();

    return(nstat);
}
