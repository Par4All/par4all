/* transformationss package 
 *
 * coarse_grain_parallelization.c :  Beatrice Creusillet, october 1996
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the functions parallelizing loops from the 
 * regions of their bodies. This enables the parallelization of loops
 * containing if constructs and unstructured parts of code. But it may
 * be less aggressive that Allen and Kennedy on other loop nests, because
 * the regions of the loop bodies may be imprecise.
 *
 */

#include <stdio.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "syntax.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "transformer.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"

/* includes pour system generateur */
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "ricedg.h"

/*********************************************************************************/
/* USEFUL VARIABLES AND ACCESS FUNCTIONS                                         */
/*********************************************************************************/
/* statement stack */
DEFINE_LOCAL_STACK(current_stmt, statement)

/* FIFO list of integers, to keep track of parallel loops */
static list l_parallel_loops = NIL;

/* This list is handled in normal order to allow FIFO (!= stack)*/
static void
set_l_parallel_loops()
{
    pips_assert("l_parallel_loops should be NIL", ENDP(l_parallel_loops));
    l_parallel_loops = NIL;
}

static void
reset_l_parallel_loops()
{
    pips_assert("l_parallel_loops should be NIL", ENDP(l_parallel_loops));
    l_parallel_loops = NIL;
}


static void
l_parallel_loops_push(int parallel_p)
{
    l_parallel_loops = gen_nconc(l_parallel_loops,CONS(INT, parallel_p, NIL));
}

static void
l_parallel_loops_pop()
{
    list l_tmp = l_parallel_loops;
    
    POP(l_parallel_loops);
    CDR(l_tmp) = NIL;
    gen_free_list(l_tmp);
}

static int 
l_parallel_loops_head()
{
    return(INT(CAR(l_parallel_loops)));
}

#define PARALLEL_LOOP 1
#define SEQUENTIAL_LOOP 0

/*********************************************************************************/

/*********************************************************************************/
/* COARSE GRAIN PARALLELIZATION                                                  */
/*********************************************************************************/

static bool
whole_loop_parallelize(loop l)
{
    int this_loop_tag = l_parallel_loops_head();

    if (this_loop_tag == PARALLEL_LOOP)
    {
	execution_tag(loop_execution(l)) = is_execution_parallel; 
    }
    l_parallel_loops_pop();
    return(TRUE);
}

static bool 
whole_loop_dependence_test(loop l)
{
    statement inner_stat = loop_body(l);
    statement loop_stat = current_stmt_head();
    list l_enclosing_loops = CONS(STATEMENT, loop_stat, NIL);
    list l_reg = load_statement_inv_regions(inner_stat);
    list l_conflicts = NIL; 
    list l_true_conflicts = NIL;
    list l_tmp;

    pips_debug(1,"begin\n");

    /* FIRST, BUILDS LIST OF CONFLICTS */       
    pips_debug(1,"building conflicts\n");
    ifdebug(2)
    {
	fprintf(stderr, "original invariant regions:\n");
	print_regions(l_reg);
    }
 
    MAP(EFFECT, reg,
	{
	    entity e = region_entity(reg);

	    if (region_write_p(reg))
	    {
		conflict conf = conflict_undefined;

		/* add a write-write conflict to the list */
		conf = make_conflict(reg, reg, cone_undefined);
		l_conflicts = gen_nconc(l_conflicts, CONS(CONFLICT, conf, NIL));

		/* search for a write-read/read-write conflict */
		MAP(EFFECT, reg2,
		    {
			if (same_entity_p(e,region_entity(reg2))
			    && region_read_p(reg2))
			{
			    /* add a write-read conflict */
			    conf = make_conflict(reg, reg2, cone_undefined);
			    l_conflicts =
				gen_nconc(l_conflicts, CONS(CONFLICT, conf, NIL));
			    /* there is at most one read region for entity e */
			    break;
			}
		    },
		    l_reg);
	    }
	},
	l_reg);

    /* THEN, TESTS CONFLICTS */
    pips_debug(1,"testing conflicts\n");
    /* We want to test for write/read and read/write dependences at the same
     * time. */
    Finds2s1 = TRUE;
    for(l_tmp = l_conflicts; !ENDP(l_tmp); POP(l_tmp))
    {
	conflict conf = CONFLICT(CAR(l_tmp));
	effect reg1 = conflict_source(conf);
	effect reg2 = conflict_sink(conf);
	list levels = NIL;
	list levelsop = NIL;
	Ptsg gs = SG_UNDEFINED;
	Ptsg gsop = SG_UNDEFINED;
	
	ifdebug(2)
	{
	    fprintf(stderr, "testing conflict from:\n");
	    print_region(reg1);
	    fprintf(stderr, "\tto:\n");	    
	    print_region(reg2);    
	}

	/* Use the function TestCoupleOfReferences from ricedg */
	/* We only consider one loop at a time, disconnected from 
         * the other enclosing and inner loops. Thus l_enclosing_loops
         * only contains the current loop statement.
         * The list of loop variants is empty, because we use loop invariant
         * regions (they have been composed by the loop transformer).
	 */
	levels = TestCoupleOfReferences(l_enclosing_loops, region_system(reg1), 
					inner_stat, reg1, region_reference(reg1),
					l_enclosing_loops, region_system(reg2),
					inner_stat, reg2, region_reference(reg2),
					NIL, &gs, &levelsop, &gsop);	
	ifdebug(2)
	{
	    fprintf(stderr, "result:\n");
	    if (ENDP(levels) && ENDP(levelsop))
		fprintf(stderr, "\tno dependence\n");
	    if (!ENDP(levels))
	    {		
		list tmp_levels = levels;
		fprintf(stderr, "\tdependence at levels: ");	    
		for(;!ENDP(tmp_levels); POP(tmp_levels))
		{
		    fprintf(stderr, " %d", INT(CAR(tmp_levels)));
		}
		fprintf(stderr, "\n");	    		
		if (!SG_UNDEFINED_P(gs)) 
		{	
		    Psysteme sc = SC_UNDEFINED;
		    fprintf(stderr, "\tdependence cone:\n");
		    sg_fprint_as_dense(stderr, gs, gs->base);
		    sc = sg_to_sc_chernikova(gs);
		    fprintf(stderr,"\tcorresponding linear system:\n");
		    sc_fprint(stderr,sc,entity_local_name);
		    sc_rm(sc);
		}	
	    }
	    if (!ENDP(levelsop))
	    {		
		list tmp_levels = levelsop;
		fprintf(stderr, "\topposite dependence at levels: ");	    
		for(;!ENDP(tmp_levels); POP(tmp_levels))
		{
		    fprintf(stderr, " %d", INT(CAR(tmp_levels)));
		}
		fprintf(stderr, "\n");	    		
		if (!SG_UNDEFINED_P(gsop)) 
		{	
		    Psysteme sc = SC_UNDEFINED;
		    fprintf(stderr, "\tdependence cone:\n");
		    sg_fprint_as_dense(stderr, gsop, gsop->base);
		    sc = sg_to_sc_chernikova(gsop);
		    fprintf(stderr,"\tcorresponding linear system:\n");
		    sc_fprint(stderr,sc,entity_local_name);
		    sc_rm(sc);
		}	
	    }
	}
	/* If the dependence cannot be disproved, add it to the list
         * of assumed dependences.
	 */
	if (!ENDP(levels) || !ENDP(levelsop))
	{
	    l_true_conflicts = gen_nconc(l_true_conflicts, 
					 CONS(CONFLICT, conf, NIL));
	}
	gen_free_list(levels);
	gen_free_list(levelsop);	
	if (!SG_UNDEFINED_P(gs)) sg_rm(gs);
	if (!SG_UNDEFINED_P(gsop)) sg_rm(gsop);
	
    }

    /* IS THE LOOP PARALLEL ? */
    if (ENDP(l_true_conflicts))
    {
	/* YES! */
	pips_debug(1, "PARALLEL LOOP\n");
	l_parallel_loops_push(PARALLEL_LOOP);
	/* execution_tag(loop_execution(l)) = is_execution_parallel; */
    }
    else
    {
	/* NO! */
	pips_debug(1, "SEQUENTIAL LOOP\n");
	l_parallel_loops_push(SEQUENTIAL_LOOP);
    }

    /* FINALLY, FREE CONFLICTS */
    pips_debug(1,"freeing conflicts\n");
    MAP(CONFLICT, c,
	{
	    conflict_source(c) = effect_undefined;
	    conflict_sink(c) = effect_undefined;
	    gen_free(c);
	},
	l_conflicts);
    gen_free_list(l_conflicts);
    gen_free_list(l_true_conflicts);
    gen_free_list(l_enclosing_loops);

    pips_debug(1,"end\n");

    return(TRUE);
}

static bool
stmt_inward(statement s)
{
    pips_debug(3, "entering statement %03d\n", statement_number(s));
    current_stmt_push(s);
    pips_debug(3, "end\n");
    return(TRUE);
}

static void 
stmt_outward(statement s)
{
    pips_debug(3, "leaving statement %03d\n", statement_number(s));    
    current_stmt_pop();
    pips_debug(3, "end\n");    
}

static void
coarse_grain_loop_parallelization(statement module_stat,
				  statement module_parallelized_stat)
{
    make_current_stmt_stack();
    set_l_parallel_loops();
    pips_debug(1,"begin\n");
    
    gen_multi_recurse(module_stat,
		      statement_domain, stmt_inward, stmt_outward, 
		      loop_domain, whole_loop_dependence_test, gen_null, 
		      NULL);   

    gen_multi_recurse(module_parallelized_stat,
		      loop_domain, whole_loop_parallelize, gen_null,
		      NULL);   
    
    pips_debug(1,"end\n");
    free_current_stmt_stack();
    reset_l_parallel_loops();
}

bool 
coarse_grain_parallelization(string module_name)
{
    statement module_stat, module_parallelized_stat;
    entity module;

    /* Get the code of the module. */
    set_current_module_entity( local_name_to_top_level_entity(module_name) );
    module = get_current_module_entity();
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();
    set_cumulated_rw_effects((statement_effects)
	   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
    module_to_value_mappings(module);
 
    /* Invariant  read/write regions */
    set_invariant_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_INV_REGIONS, module_name, TRUE));

    debug_on("COARSE_GRAIN_PARALLELIZATION_DEBUG_LEVEL");

    module_parallelized_stat = copy_statement(module_stat);
    coarse_grain_loop_parallelization(module_stat, module_parallelized_stat);
 
    debug_off();    

    DB_PUT_MEMORY_RESOURCE(DBR_PARALLELIZED_CODE,
			   strdup(module_name),
			   (char*) module_parallelized_stat);
	
    
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_invariant_rw_effects();
    free_value_mappings();
    return(TRUE);
}


