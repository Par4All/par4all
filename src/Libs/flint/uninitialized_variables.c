/*
   Verify the initializations of the variables of a module.

   Ronan.Keryell@cri.ensmp.fr

   */

#include <stdio.h> 
#include <stdlib.h>
#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "misc.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "graph.h"
#include "dg.h"
/* Instantiation of the dependence graph: */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
/* Just to be able to use ricedg.h: */
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
/* */
#include "ricedg.h"
#include "semantics.h"
#include "transformations.h"
#include "flint.h"


static hash_table flint_statement_def_use_variables;
static bool flint_no_uninitialized_variable_ouput_yet;

/* Build the tables relating a statement with the variables it uses
   that has been initialized elsewhere.

   This function is heavily inspired by
   build_statement_to_statement_dependence_mapping() in the use-def
   elimination transformation: */
static void
flint_initialize_statement_def_use_variables(graph dependence_graph)
{
    flint_statement_def_use_variables = hash_table_make(hash_pointer, 0);
   
    MAP(VERTEX,
	a_vertex,
	{
	    statement s1 = vertex_to_statement(a_vertex);

	    pips_debug(7, "\tSuccessor list: %p for statement ordering %d\n", 
		       vertex_successors(a_vertex),
		       dg_vertex_label_statement(vertex_vertex_label(a_vertex)));
	    MAP(SUCCESSOR, a_successor,
		{
		    vertex v2 = successor_vertex(a_successor);
		    statement s2 = vertex_to_statement(v2);
		    dg_arc_label an_arc_label = successor_arc_label(a_successor);
		    pips_debug(7, "\t%p --> %p with conflicts\n", s1, s2);
		    /* Try to find at least one of the use-def chains between
		       s and a successor: */
		    MAP(CONFLICT, a_conflict,
			{
			    statement use;
			    statement def;
			    cell a_use_cell;
                        
			    ifdebug(7) 
			    {
				fprintf(stderr, "\t\tfrom ");
				print_words(stderr, words_effect(conflict_source(a_conflict)));
				fprintf(stderr, " to ");
				print_words(stderr, words_effect(conflict_sink(a_conflict)));
				fprintf(stderr, "\n");
			    }
                    
			    /* Something is useful for the current
			       statement if it writes something that
			       is used in the current statement: */
			    if (action_read_p(effect_action(conflict_source(a_conflict)))
				&& action_write_p(effect_action(conflict_sink(a_conflict)))) {
				use = s1;
				a_use_cell =
				    effect_cell(conflict_source(a_conflict));
				def = s2;
			    }
			    else if (action_write_p(effect_action(conflict_source(a_conflict)))
				     && action_read_p(effect_action(conflict_sink(a_conflict)))) {
				def = s1;
				a_use_cell =
				    effect_cell(conflict_sink(a_conflict));
				use = s2;
			    }
			    else
				/* The dependance is not a use-def
				   one, look forward... */
				continue;
                        
			    {
				/* Mark that we will visit the node
				   that defined a source for this
				   statement, if not already visited: */
				entity a_variable;
				set def_use_variables;
				
				/* Get the variable entity involved in
                                   the dependence: */
				if (cell_preference_p(a_use_cell))
				    a_variable = reference_variable(preference_reference(cell_preference(a_use_cell)));
				else
				    a_variable = reference_variable(cell_reference(a_use_cell));
    
				def_use_variables = (set) hash_get(flint_statement_def_use_variables, (char *) use);
                                       
				if (def_use_variables == (set) HASH_UNDEFINED_VALUE) {
				    /* It is the first dependence we
				       found for use. Create the set: */
				    def_use_variables = set_make(set_pointer);
				    hash_put(flint_statement_def_use_variables,
					     (char *) use,
					     (char *) def_use_variables);
				}

				/* Mark the fact that s2 create something
				   useful for s1: */
				set_add_element(def_use_variables,
						def_use_variables,
						(char *)  a_variable);

				pips_debug(6, "\tUse: statement %p (%#x). Def: statement %p (%#x), variable \"%s\".\n",
					   use, statement_ordering(use),
					   def, statement_ordering(def),
					   entity_minimal_name(a_variable));
			    }
                        
			    /* One use-def is enough for this variable
			       couple: */
			    break;
			},
                        dg_arc_label_conflicts(an_arc_label));
		},
		vertex_successors(a_vertex));

	},
	graph_vertices(dependence_graph));
}


/* Remove the flint_statement_def_use_variables data structure: */
static void
flint_free_statement_def_use_variables()
{
    HASH_MAP(a_statement, def_use_variables,
	     {
		 set_free((set) def_use_variables);
	     },
	     flint_statement_def_use_variables);
    hash_table_free(flint_statement_def_use_variables);
    flint_statement_def_use_variables = NULL;
}

    
/* Return true if a_variable is not initialized elsewhere: */
static bool
flint_variable_uninitialize_elsewhere(statement s,
				      entity a_variable)
{
    set def_use_variables = (set) hash_get(flint_statement_def_use_variables,
					   (char *) s);
    if (def_use_variables == (set) HASH_UNDEFINED_VALUE)
	/* There is no variable for this statement with previously
           initialized value. Just return TRUE: */
	return TRUE;

    if (set_belong_p(def_use_variables, (char *) a_variable))
	/* OK, looks like the variable have been previously initialized: */
	return FALSE;

    return TRUE;
}


/* Warn about uninitialized variables in this statement: */
static bool
flint_check_uninitialized_variables_in_statement(statement s)
{
    list effects_list = load_proper_rw_effects_list(s);

    bool something_said_about_this_statement = FALSE;

    /* It appears that effects are packed by statement number or
       ordering. I assume that to factorize the prettyprint: */
    MAP(EFFECT, an_effect,
	{
	    reference a_reference = effect_reference(an_effect);
	    entity a_variable = reference_variable(a_reference);
	    if (action_read_p(effect_action(an_effect))
		&& flint_variable_uninitialize_elsewhere(s, a_variable)) {
		if (flint_no_uninitialized_variable_ouput_yet) {
		    /* Nothing has been output yet: add a banner: */
		    raw_flint_message(FALSE,
				      "\n\nNon-initialized variables:"
				      "\n------------------------\n");
		    flint_no_uninitialized_variable_ouput_yet = FALSE;
		}
		if (!something_said_about_this_statement) {
		    raw_flint_message(FALSE,
				      "In statement number %d (%d.%d):\n",
				      statement_number(s),
				      ORDERING_NUMBER(statement_ordering(s)),
				      ORDERING_STATEMENT(statement_ordering(s)));
		    something_said_about_this_statement = TRUE;
		}
		raw_flint_message(TRUE,
				  "\t\"%s\" used but not initialized.\n",
				  entity_minimal_name(a_variable));
	    }
	},
	effects_list);

    /* Go on recursion... */
    return TRUE;
}


/* Warn about conservatively uninitialized variables in the module: */
void
flint_uninitialized_variables(graph dependence_graph,
			      statement module_stat)
{
    flint_initialize_statement_def_use_variables(dependence_graph);
    flint_no_uninitialized_variable_ouput_yet = TRUE;
    
    gen_recurse(module_stat,
		statement_domain,
		flint_check_uninitialized_variables_in_statement,
		gen_null);
    
    if (!flint_no_uninitialized_variable_ouput_yet)
	raw_flint_message(FALSE,
			  "\n");
    
    flint_free_statement_def_use_variables();
}
