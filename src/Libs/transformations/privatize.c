/* -- privatize.c 

   This algorithm introduces local definitions into loops that are
   kennedizable.

 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "misc.h"

#include "ri.h"
#include "database.h"

#include "ri-util.h"
#include "graph.h"
#include "dg.h"
#include "control.h"
#include "chains.h"
#include "pipsdbm.h"

#include "resources.h"

#include "effects-generic.h"
#include "effects-simple.h"

/* instantiation of the dependence graph */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

/* PRIVATIZABLE checks whether the entity E is privatizable. */

static bool privatizable(entity e)
{
    storage s = entity_storage( e ) ;

    return( entity_scalar_p( e ) && 
	    storage_ram_p( s ) &&
	    dynamic_area_p( ram_section( storage_ram( s )))) ;
}

/* SCAN_STATEMENT gathers the list of enclosing LOOPS of statement S. 
   Moreover, the locals of loops are initialized to all possible private
   entities. */

static void scan_unstructured(unstructured u, list loops) ;

static void scan_statement(statement s, list loops)
{
    instruction i = statement_instruction(s);

    if (get_enclosing_loops_map() == hash_table_undefined) {
	set_enclosing_loops_map( MAKE_STATEMENT_MAPPING() );
    }
    store_statement_enclosing_loops(s, loops);

    switch(instruction_tag(i)) {
    case is_instruction_block:
	MAPL(ps, {scan_statement(STATEMENT(CAR(ps)), loops);},
	     instruction_block(i));
	break ;
    case is_instruction_loop: {
	loop l = instruction_loop(i);
	statement b = loop_body(l);
	list new_loops = 
		gen_nconc(gen_copy_seq(loops), CONS(STATEMENT, s, NIL)) ;
	list locals = NIL ;

	MAPL( fs, { 
	    effect f = EFFECT( CAR( fs )) ;
	    entity e = effect_entity( f ) ;

	    if( action_write_p( effect_action( f )) &&
	        privatizable( e ) && 
	        gen_find_eq( e, locals ) == entity_undefined ) {
		locals = CONS( ENTITY, e, locals ) ;
	    }
	}, load_cumulated_rw_effects_list( b )) ;
	loop_locals( l ) = CONS( ENTITY, loop_index( l ), locals ) ;
	scan_statement( b, new_loops ) ;
	hash_del(get_enclosing_loops_map(), (char *) s) ;
	store_statement_enclosing_loops(s, new_loops);
	break;
    }
    case is_instruction_test: {
	test t = instruction_test( i ) ;

	scan_statement( test_true( t ), loops ) ;
	scan_statement( test_false( t ), loops ) ;	
	break ;
    }
    case is_instruction_unstructured: 
	scan_unstructured( instruction_unstructured( i ), loops ) ;
	break ;
    case is_instruction_call:
    case is_instruction_goto:
	break ;
    default:
	pips_error("scan_statement", "unexpected tag %d\n", instruction_tag(i));
    }
}

static void scan_unstructured(unstructured u, list loops)
{
    list blocs = NIL ;

    CONTROL_MAP( c, {scan_statement( control_statement( c ), loops );},
		 unstructured_control( u ), blocs ) ;
    gen_free_list( blocs ) ;
}

/* LOOP_PREFIX returns the common list prefix of lists L1 and L2. */

static list loop_prefix(list l1, list l2)
{
    statement st ;

    if( ENDP( l1 )) {
	return( NIL ) ;
    }
    else if( ENDP( l2 )) {
	return( NIL ) ;
    }
    else if( (st=STATEMENT( CAR( l1 ))) == STATEMENT( CAR( l2 ))) {
	return( CONS( STATEMENT, st, loop_prefix( CDR( l1 ), CDR( l2 )))) ;
    }
    else {
	return( NIL ) ;
    }
}

/* UPDATE_LOCALS removes the entity E from the locals of loops in LS that
   are not in common with the PREFIX. */

static void 
update_locals(list prefix, list ls, entity e)
{
  debug(1, "update_locals", "Begin\n");

    if( ENDP( prefix )) {
      if(!ENDP(ls)) {
	ifdebug(1) {
	    debug(1, "update_locals", "Removing %s", entity_name( e )) ;
	    fprintf( stderr, " from locals of " ) ;
	    MAPL( sts, {statement st = STATEMENT( CAR( sts )) ;
			
			fprintf( stderr, "%d ", statement_number( st )) ;},
		 ls ) ;
	    fprintf( stderr, "\n" ) ;
	}
	MAPL( sts, {statement st = STATEMENT( CAR( sts )) ;
		    instruction i = statement_instruction( st ) ;

		    pips_assert( "remove_local", instruction_loop_p( i )) ;
		    gen_remove( &loop_locals( instruction_loop( i )), e );},
	      ls ) ;
      }
      else {
	debug(1, "update_locals", "ls is empty, end of recursion\n");
      }
    }
    else {
	pips_assert( "update_locals", 
		     STATEMENT( CAR( prefix )) == STATEMENT( CAR( ls ))) ;

	debug(1, "update_locals", "Recurse on common prefix\n");

	update_locals( CDR( prefix ), CDR( ls ), e ) ;
    }

  debug(1, "update_locals", "End\n");
}

/* expression_implied_do_index_p
   return true if the given entity is the index of an implied do
   contained in the given expression. --DB
*/
static bool expression_implied_do_index_p(expression exp,entity e)
{
  bool li=FALSE;
  bool dep=FALSE;
  if (expression_implied_do_p(exp))
    {
      list args = call_arguments(syntax_call(expression_syntax(exp)));
      expression arg1 = EXPRESSION(CAR(args)); /* loop index */
      expression arg2 = EXPRESSION(CAR(CDR(args))); /* loop range */
      entity index = reference_variable(syntax_reference(expression_syntax(arg1)));
      range r = syntax_range(expression_syntax(arg2));
      list range_effects;

      debug(5,"expression_implied_do_index_p","begin\n");
      debug(7,"expression_implied_do_index_p",
	    "%s implied do index ? index: %s\n",
		entity_name(e),entity_name(index));

      range_effects = proper_effects_of_range(r);

      MAP(EFFECT, eff, 
	  {
	    if (reference_variable(effect_reference(eff)) == e &&
		action_read_p(effect_action(eff))) 
	      {
			
		debug(7, "expression_implied_do_index_p", 
			      "index read in range expressions\n");
		dep=TRUE;
			
	      }
	   }, range_effects);
      free_effects(make_effects(range_effects));
  
      if (!dep)
	{
	  if (same_entity_p(e,index)) li=TRUE;
	  else 
	    MAP(EXPRESSION,expr,{
	      syntax s = expression_syntax(expr);
	      if(syntax_call_p(s))
		{
		  debug(5,"expression_implied_do_index_p","Nested implied do\n");
		  if (expression_implied_do_index_p(expr,e))
		    li=TRUE;
		}
	    },CDR(CDR(args)));
	}
  debug(5,"expression_implied_do_index_p","end\n");
}
  return li;
}

/* is_implied_do_index
   returns true if the given entity is the index of one of the
   implied do loops of the given instruction. --DB
*/
bool is_implied_do_index(entity e, instruction ins)
{
  bool li = FALSE;

  debug(5,"is_implied_do_index","entity name: %s ", entity_name( e )) ;

  if (instruction_call_p(ins))
    MAP(EXPRESSION,exp,{
      if (expression_implied_do_index_p(exp,e)) li=TRUE;
    },call_arguments( instruction_call( ins ) ));

  ifdebug(5)
    fprintf(stderr, "%s\n", bool_to_string(li));

  return li;
}

/* TRY_PRIVATIZE knows that the effect F on entity E is performed in the
   statement ST of the vertex V of the dependency graph. Arrays are not 
   privatized. */

static void 
try_privatize(vertex v, statement st, effect f, entity e)
{
    cons *ls ;

    if( !entity_scalar_p( e )) {
	return ;
    }

    ls = load_statement_enclosing_loops(st);

    ifdebug(1) {
	if(statement_loop_p(st)) {
	    debug(1, "try_privatize", "Trying to privatize %s in loop statement %d with local(s) ",
		  entity_local_name( e ), statement_number( st )) ;
	    print_arguments(loop_locals(statement_loop(st)));
	}
	else {
	    debug(1, "try_privatize", "Trying to privatize %s in statement %d\n",
	    entity_local_name( e ), statement_number( st )) ;
	}
    }

    MAPL( succs, {
	successor succ = SUCCESSOR( CAR( succs )) ;
	vertex succ_v = successor_vertex( succ ) ;
	dg_vertex_label succ_l = 
	    (dg_vertex_label)vertex_vertex_label( succ_v ) ;
	dg_arc_label arc_l = 
	    (dg_arc_label)successor_arc_label( succ ) ;
	statement succ_st = 
	    ordering_to_statement(dg_vertex_label_statement(succ_l));
	instruction succ_i = statement_instruction( succ_st ) ;
	cons *succ_ls = load_statement_enclosing_loops( succ_st ) ;
		  
	if( v == succ_v ) {
	    continue ;
	}
	MAPL( cs, {
	    conflict c = CONFLICT( CAR( cs )) ;
	    effect sc = conflict_source( c ) ;
	    effect sk = conflict_sink( c ) ;
	    cons *prefix ;
			     
	    if(!entity_conflict_p( e, effect_entity( sc )) ||
	       !entity_conflict_p( e, effect_entity( sk )) ||
	       action_write_p( effect_action( sk))) {
		continue ;
	    }
	    /* PC dependance and the sink is a loop index */
	    if(action_read_p( effect_action( sk )) &&
	       (instruction_loop_p( succ_i) ||
	       is_implied_do_index( e, succ_i))) {
		continue ;
	    }
	    debug(5,"try_privatize","Conflict for %s between statements %d and %d\n",
		  entity_local_name(e), statement_number(st), statement_number(succ_st));
	    debug(5,"try_privatize","remove %s from locals in enclosing loops\n",
		  entity_local_name(e));
	    prefix = loop_prefix( ls, succ_ls ) ;
	    update_locals( prefix, ls, e ) ;
	    update_locals( prefix, succ_ls, e ) ;
	    gen_free_list( prefix ) ;
	},
	     dg_arc_label_conflicts( arc_l )) ;
    },
	 vertex_successors( v )) ;

    debug(1, "try_privatize", "End\n");
}

/* PRIVATIZE_DG looks for definition of entities that are locals to the loops
   in the dependency graph G for the control graph U. */

bool privatize_module(char *mod_name)
{
    entity module;
    statement mod_stat;
    instruction mod_inst;
    graph mod_graph;

    set_current_module_entity( local_name_to_top_level_entity(mod_name) );
    module = get_current_module_entity();

    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, mod_name, TRUE) );
    mod_stat = get_current_module_statement();

    initialize_ordering_to_statement(mod_stat);
    
    mod_inst = statement_instruction(mod_stat);

    /*
    if (! instruction_unstructured_p(mod_inst))
	pips_error("privatize_module", "unstructured expected\n");
	*/

    set_proper_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, TRUE));

    set_cumulated_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, TRUE) );

    mod_graph = (graph)
	db_get_memory_resource(DBR_CHAINS, mod_name, TRUE);

    debug_on("PRIVATIZE_DEBUG_LEVEL");

    /* Build maximal lists of private variables in loop locals */
    /* scan_unstructured(instruction_unstructured(mod_inst), NIL); */
    scan_statement(mod_stat, NIL);

    /* remove non private variables from locals */
    MAPL( vs, {
	vertex v = VERTEX( CAR( vs )) ;
	dg_vertex_label vl = (dg_vertex_label) vertex_vertex_label( v ) ;
	statement st = 
	    ordering_to_statement(dg_vertex_label_statement(vl));
	       
	MAPL( fs, {
	    effect f = EFFECT( CAR( fs )) ;
	    entity e = effect_entity( f ) ;

	    if( action_write_p( effect_action( f ))) {
		try_privatize( v, st, f, e ) ;
	    }
	}, load_proper_rw_effects_list( st )) ;
    }, graph_vertices( mod_graph )) ;

    /* sort locals
     */
    sort_all_loop_locals(mod_stat);

    debug_off();
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);

    reset_current_module_entity();
    reset_current_module_statement();
    reset_proper_rw_effects();
    reset_cumulated_rw_effects();
    clean_enclosing_loops();

    return TRUE;
}








