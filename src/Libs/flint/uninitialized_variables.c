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
   Verify the initializations of the variables of a module.

   Ronan.Keryell@cri.ensmp.fr

   */

#include "local.h"

static hash_table flint_statement_def_use_variables;
static bool flint_no_uninitialized_variable_ouput_yet;

/* Build the tables relating a statement with the variables it uses
   that have been initialized elsewhere.

   This function is heavily inspired by
   build_statement_to_statement_dependence_mapping() in the use-def
   elimination transformation: */
static void
flint_initialize_statement_def_use_variables(graph dependence_graph)
{
  flint_statement_def_use_variables = hash_table_make(hash_pointer, 0);

  FOREACH(VERTEX, a_vertex, graph_vertices(dependence_graph))	{
    statement s1 = vertex_to_statement(a_vertex);

    pips_debug(7, "\tSuccessor list: %p for statement ordering %td\n",
	       vertex_successors(a_vertex),
	       dg_vertex_label_statement(vertex_vertex_label(a_vertex)));
    FOREACH(SUCCESSOR, a_successor, vertex_successors(a_vertex)) {
      vertex v2 = successor_vertex(a_successor);
      statement s2 = vertex_to_statement(v2);
      dg_arc_label an_arc_label = successor_arc_label(a_successor);
      pips_debug(7, "\t%p --> %p with conflicts\n", s1, s2);
      /* Try to find at least one of the use-def chains between
	 s and a successor: */
      FOREACH(CONFLICT, a_conflict,
	      dg_arc_label_conflicts(an_arc_label)) {
	statement use;
	statement def;
	effect src_eff = conflict_source(a_conflict);
	effect sink_eff = conflict_sink(a_conflict);

	ifdebug(7) {
	  fprintf(stderr, "\t\tfrom ");
	  print_words(stderr, words_effect(src_eff));
	  fprintf(stderr, " to ");
	  print_words(stderr, words_effect(sink_eff));
	  fprintf(stderr, "\n");
	}

	/* Something is useful for the current statement if it writes
	   something that is used in the current statement: */
	if (action_write_p(effect_action(src_eff))
	    && action_read_p(effect_action(sink_eff))) {
	  def = s1;
	  use = s2;
	}
	else
	  /* The dependance is not a use-def one, look forward... */
	  /* FI: it looks more like a def-use... */
	  continue;

	{
	  /* Mark that we will visit the node that defined a source
	     for this statement, if not already visited: */
	  /* FI: in Fortran, with static aliasing, this is not really
	     safe. a_variable is not uniquely defined */
	  entity a_variable = reference_variable(effect_any_reference(src_eff));
	  set def_use_variables;

	  def_use_variables = (set) hash_get(flint_statement_def_use_variables, (char *) use);

	  if (def_use_variables == (set) HASH_UNDEFINED_VALUE) {
	    /* It is the first dependence found for use. Create the
	       set. */
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

	  pips_debug(6, "\tUse: statement %p (%#tx). Def: statement %p (%#tx), variable \"%s\".\n",
		     use, (_uint) statement_ordering(use),
		     def, (_uint) statement_ordering(def),
		     entity_minimal_name(a_variable));
	}

	/* One use-def is enough for this variable
	   couple: */
	break;
      }
    }
  }
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
	return true;

    if (set_belong_p(def_use_variables, (char *) a_variable))
	/* OK, looks like the variable have been previously initialized: */
	return false;

    return true;
}


/* Warn about uninitialized variables in this statement: */
static bool
flint_check_uninitialized_variables_in_statement(statement s)
{
    list effects_list = load_proper_rw_effects_list(s);

    bool something_said_about_this_statement = false;

    /* It appears that effects are packed by statement number or
       ordering. I assume that to factorize the prettyprint: */
    MAP(EFFECT, an_effect,
	{
	    reference a_reference = effect_any_reference(an_effect);
	    entity a_variable = reference_variable(a_reference);
	    if (action_read_p(effect_action(an_effect))
		&& flint_variable_uninitialize_elsewhere(s, a_variable)) {
		if (flint_no_uninitialized_variable_ouput_yet) {
		    /* Nothing has been output yet: add a banner: */
		    raw_flint_message(false,
				      "\n\nNon-initialized variables:"
				      "\n--------------------------\n");
		    flint_no_uninitialized_variable_ouput_yet = false;
		}
		if (!something_said_about_this_statement) {
		    raw_flint_message(false,
				      "In statement number %d (%d.%d):\n",
				      statement_number(s),
				      ORDERING_NUMBER(statement_ordering(s)),
				      ORDERING_STATEMENT(statement_ordering(s)));
		    something_said_about_this_statement = true;
		}
		raw_flint_message(true,
				  "\t\"%s\" used but not initialized.\n",
				  entity_minimal_name(a_variable));
	    }
	},
	effects_list);

    /* Go on recursion... */
    return true;
}


/* Warn about conservatively uninitialized variables in the module: */
void
flint_uninitialized_variables(graph dependence_graph,
			      statement module_stat)
{
    flint_initialize_statement_def_use_variables(dependence_graph);
    flint_no_uninitialized_variable_ouput_yet = true;
    
    gen_recurse(module_stat,
		statement_domain,
		flint_check_uninitialized_variables_in_statement,
		gen_null);
    
    if (!flint_no_uninitialized_variable_ouput_yet)
	raw_flint_message(false,
			  "\n");
    
    flint_free_statement_def_use_variables();
}
