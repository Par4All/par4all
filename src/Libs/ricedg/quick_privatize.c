/*********************************************************************************/
/* QUICK PRIVATIZATION                                                           */
/*********************************************************************************/

#include "local.h"
#include "transformations.h"

static bool quick_privatize_loop(statement /*stat*/, list /*successors*/);
static bool quick_privatize_statement_pair(statement /*s1*/, statement /*s2*/, 
					   list /*conflicts*/);

void quick_privatize_graph(dep_graph)
graph dep_graph;
{

    /* we analyze arcs exiting from loop statements */
    MAP(VERTEX, v1, 
    {
	statement s1 = vertex_to_statement(v1);
	list successors = vertex_successors(v1);
	
	if (statement_loop_p(s1)) 
	{
	    loop l = statement_loop(s1);
	    list locals = loop_locals(l);
	    entity ind = loop_index(l);
	    
	    if (gen_find_eq(ind, locals) == entity_undefined) 
	    {
		if (quick_privatize_loop(s1, successors)) 
		{
		    debug(1, "quick_privatize_graph", 
			  "Index for loop %d privatized\n",
			  statement_number(s1));
		    
		    loop_locals(l) = CONS(ENTITY, ind, locals);
		}
		else 
		{
		    debug(1, "quick_privatize_graph", 
			  "could not privatize loop %d\n", statement_number(s1));
		}
	    }
	}
    },
	graph_vertices(dep_graph) );
}



static bool quick_privatize_loop(stat, successors)
statement stat;
list successors;
{
    debug(3, "quick_privatize_loop", "arcs from %d\n", statement_number(stat));

    MAP(SUCCESSOR, su,
    {
	dg_arc_label dal =  (dg_arc_label) successor_arc_label(su);
	statement st = vertex_to_statement(successor_vertex(su));
	
	debug(3, "quick_privatize_loop", "arcs to %d\n", statement_number(st));
	
	if (! quick_privatize_statement_pair(stat, st, 
					     dg_arc_label_conflicts(dal)))
	    return(FALSE);
    },
	successors );

    return(TRUE);
}



static bool quick_privatize_statement_pair(s1, s2, conflicts)
statement s1, s2;
list conflicts;
{
    loop l1 = statement_loop(s1);
    entity ind1 = loop_index(l1);

    MAP(CONFLICT, c,
    {
	effect f1 = conflict_source(c);
	reference r1 = effect_reference(f1);
	entity e1 = reference_variable(r1);

	effect f2 = conflict_sink(c);
	reference r2 = effect_reference(f2);
	entity e2 = reference_variable(r2);

	debug(2, "quick_privatize_statement_pair", 
	      "conflict between %s & %s\n", entity_name(e1), entity_name(e2));

	/* equivalence or conflict not created by loop index. I give up ! */
	if (e1 != ind1)  
	    continue; 
	
	if (action_write_p(effect_action(f1)) && 
	    action_read_p(effect_action(f2))) 
	{
	    /* we must know where this read effect come from. if it
	       comes from the loop body, the arc may be ignored. */

	    list loops = load_statement_enclosing_loops(s2);

	    if (gen_find_eq(s1, loops) == entity_undefined) 
	    {
		loop l2;
		entity ind2;
		list range_effects;

		debug(3, "quick_privatize_statement_pair", 
		      "the arc goes outside the loop body.\n");

		if ( is_implied_do_index(e1,statement_instruction(s2)))
		  {
		    debug(3,"quick_privatize_statement_pair","s2 is an implied loop\n");
		    return(TRUE);
		  }
		if (! statement_loop_p(s2)) 
		{
		    debug(3, "quick_privatize_statement_pair", "s2 not a loop\n"); 
		    return(FALSE);
		}

		/* s2 is a loop. if there are no read effet in the range
		   part, ignore this conflict. */
		l2 = statement_loop(s2);
		ind2 = loop_index(l2);
		range_effects = proper_effects_of_range(loop_range(l2));

		MAP(EFFECT, e, 
		{
		    if (reference_variable(effect_reference(e)) == ind2 &&
			action_read_p(effect_action(e))) 
		    {
			
			debug(3, "quick_privatize_statement_pair", 
			      "index read in range expressions\n"); 
			
			free_effects(make_effects(range_effects));
			return(FALSE);
		    }
		}, range_effects);
		free_effects(make_effects(range_effects));
	    }
	}
    },
	conflicts );
    
    return(TRUE);
}

