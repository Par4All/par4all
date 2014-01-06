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

/* 2 phases :
   - recover for loops hidden in while loops:
     see comments before try_to_recover_for_loop_in_a_while()
   - for-loop to do-loop transformation:
     not found (FI)
*/

#include "genC.h"
#include "linear.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "semantics.h"
#include "transformer.h"

#if 0
// Not used yet
#include "dg.h"
/* Instantiation of the dependence graph: */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"
/* Just to be able to use ricedg.h: */
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

static graph dependence_graph;
#endif

/* Find if a variable rv can be considered as an index-like variable for
   the given statement s.

   @return :
   - false if rv is not considered as such;
   - true if yes, change index_relation to the Pvector of the transformer
     that represents the index evolution and initial_rv points to the initial
     variable of rv in the transformer.
*/
static bool
find_simple_for_like_variable(statement s,
			      entity rv,
			      entity * initial_rv,
			      Pvecteur * index_relation) {
  pips_debug(5, "Look for variable %s (%p)\n", entity_global_name(rv), rv);
  transformer t = load_statement_semantic(s);
  ifdebug(5) {
    dump_text(text_for_a_transformer(t, true));
    print_statement(s);
  }
  /* Look for the referenced variable in the transformer argument: */
  FOREACH(ENTITY, e, transformer_arguments(t)) {
     pips_debug(5, "Name of a variable whose change value is in the transformer: %s (%p)\n", entity_global_name(e), e);
    if (rv == e) {
      pips_debug(5, "! Found matching variable in the statement !\n");
      // Find or build a matching initial valued variable (...#init):
      Psysteme ps = predicate_system(transformer_relation(t));
      ifdebug(7) {
	pips_debug(7, "sys %p\n", ps);
	sc_syst_debug(ps);
	text txt = make_text(NIL);
	char crt_line[MAX_LINE_LENGTH];

	system_text_format(crt_line, ">", txt, ps,
			   (char * (*)(Variable)) pips_user_value_name, false);
	fputs(text_to_string(txt), stderr);
      }

      /* Search the equalities: */
      for (Pcontrainte equality = sc_egalites(ps);
	   equality != NULL;
	   equality = equality->succ) {
	/* For the pattern matching: */
	bool found_value_for_current_variable = false;
	bool found_old_value_for_current_variable = false;
	bool found_value_for_another_variable_in_transformer = false;

	pips_debug(5, "Equality : %p\n", equality);
	for (Pvecteur v = contrainte_vecteur(equality);
	     !VECTEUR_NUL_P(v);
	     v = v->succ) {
	  Variable var = var_of(v);
	  Value coeff = val_of(v);
	  pips_debug(5, "Value " VALUE_FMT " ", coeff);
	  if (var==TCST)
	    pips_debug(5, "for vector constant\n");
	  else {
	    pips_debug(5, "for vector variable : %s %s %s\n",
		       pips_user_value_name(var),
		       entity_global_name(var),
		       entity_local_name(var));
	    if (var == e) {
	      pips_debug(5, "We have found a reference to the variable"
			 " in the transformer\n");
	      found_value_for_current_variable = true;
	    }
	    else if (local_old_value_entity_p(var)
		     && value_to_variable(var) == e
		     && value_one_p(value_abs(coeff))) {
	      pips_debug(5, "We have found a reference to the initial value"
			 " of the variable in the transformer"
			 " with a factor +/- 1\n");
	      found_old_value_for_current_variable = true;
	      *initial_rv = var;
	    }
	    else if (gen_in_list_p(var, transformer_arguments(t))) {
	      pips_debug(5, "We have found a reference to another variable"
			 " marked as modified by the transformer\n");
	      /* That means that the variable e evolved according to
		 another variable var, so it is not a simple for
		 loop. Just skipping... */
	      found_value_for_another_variable_in_transformer = true;
	      break;
	    }
	    /* Else it should be a reference to a variable that is loop
	       invariant because not in the variables marked a
	       transformed by the transformer: good news. */
	  }
	}
	if (found_value_for_current_variable
	    && found_old_value_for_current_variable
	    && !found_value_for_another_variable_in_transformer) {
	  pips_debug(4, "We have found a relation of the form:\n"
		     "+/- i + k*i#init + invariant_loop_variables == constant\n"
		     );
	  /* Return the found relation that modelizes the index behaviour: */
	  *index_relation = contrainte_vecteur(equality);
	  return true;
	}
      }
    }
  }
  return false;
}


/* Try to recover "for" loops from "while" loops in a statement.

   The algorithm used is for a statement of this form:

   while(cond)
     body;

   For each variable i used (with read effects) in the expression cond, if
   the loop body has a transformer of the form i == i#init+inc_1+inc_2+...
   and inc_x are loop-invariant (that is that they do not appear in the
   variables changed by the transformet), i is a loop index suitable for a
   simple for-loop.

   Then replace this while-loop with:
   for(i0 = i; cond; i0 += inc_1+inc_2+...) {
     i = i0;
     body;
   }

   We can substitute any reference to i by i0 in cond. If it is not
   possible (cond has some interprocedural side effect on i) the code is
   still correct but less for-loop friendly.

   We use the transformer to deal with some interprocedural loop increment
   (for example if there are some iterator-like constructions).

   Do not deal with loop with only one statement in it right now.

   TODO: could deal with any write effect on i in the loop instead of one
   with a transformer with a form i == i#init+x

   TODO: deal with multiple index

   TODO: verify MAY or MUST?

   TODO: deal with index shifting by cloning the index where it is used,
   such as in:
   i = 0;
   while (i < 100) {
     i++;
     a[i] = i;
   }
   could be transformed to:
   for(i = 0; i < 100; i++) {
     future_i = i + 1;
     a[future_i] = future_i;
   }
   TODO: while(i-->0) {} might not be recognized, as well as
   while(i++<n) {}
 */
static void
try_to_recover_for_loop_in_a_while(whileloop wl) {
  /* Get the englobing statement of the "while" assuming we are called
     from a gen_recurse()-like function: */
  instruction i = (instruction) gen_get_recurse_ancestor(wl);
  statement wls = (statement) gen_get_recurse_ancestor(i);

  pips_debug(9, "While-loop %p, parent (instruction): %p, "
	     "whileloop of the instruction: %p\n", wl, i,
	     instruction_whileloop(i));
  if ((statement_instruction(wls) != i) || (instruction_whileloop(i) != wl))
    pips_internal_error("Cannot get the enclosing statement of the while-loop.");

  /* If it is a "do { } while()" do nothing: */
  if (evaluation_after_p(whileloop_evaluation(wl)))
    return;

  // Use to get the effects of the condition part of the "while":
  list while_proper_effects = load_proper_rw_effects_list(wls);
  // Use to get the effects of the whole loop, with condition and body:
  //list while_cumulated_effects = load_cumulated_rw_effects_list(wls);

  FOREACH(EFFECT, an_effect, while_proper_effects) {
    reference r = effect_any_reference(an_effect);
    ifdebug(5) {
      print_effect(an_effect);
      dump_effect(an_effect);
    }
    pips_debug(5, "%p: %s\n", an_effect, words_to_string(effect_words_reference(r)));
    if (effect_read_p(an_effect)) {
      /* Look for the reference variable in the statement body transformers: */
      entity rv = reference_variable(r);
      entity initial_rv;
      Pvecteur index_evolution;
      pips_debug(5, "Look for variable %s (%p)\n", entity_global_name(rv), rv);
      bool found = find_simple_for_like_variable(whileloop_body(wl), rv,
						 &initial_rv,
						 &index_evolution);
      if (found) {
	ifdebug(3) {
	  pips_debug(3, "Variable %s (%p) is a nice loop-like index"
		     " with relation\n", entity_global_name(rv), rv);
	  vect_dump(index_evolution);
	  vect_print(index_evolution,
		     (get_variable_name_t) pips_user_value_name);
	}
	entity new_index =
	  make_new_scalar_variable_with_prefix(entity_user_name(rv),
					       get_current_module_entity(),
					       // Should use ultimate type?
					       entity_basic(rv));
    AddEntityToCurrentModule(new_index);
	//\domain{Forloop = initialization:expression x condition:expression x increment:expression x body:statement}


	/* Build the initialization part of the for-loop by initializing
	   the new index variable from the old one: */
	expression init = make_assign_expression(entity_to_expression(new_index), entity_to_expression(rv));

	/* Build the conditional part of the for-loop, with the new loop
	   index instead of the old one: */
	expression cond = whileloop_condition(wl);
	cond = substitute_entity_in_expression(rv, new_index, cond);

	/* Build the increment part of the for-loop: */
	expression inc = make_constraint_expression(index_evolution, rv);
	/* Replace old variable#init by the plain one: */
	inc = make_assign_expression(entity_to_expression(new_index), substitute_entity_in_expression(initial_rv, new_index, inc));

	/* Add a statement "old_index = new_index" at the beginning of the
	   loop body to propagate the new index value to the old body: */
	statement copy_new_index_to_old =
	  make_assign_statement(entity_to_expression(rv),
				entity_to_expression(new_index));
	statement for_loop_body = whileloop_body(wl);
	insert_statement(for_loop_body, copy_new_index_to_old, true);

	forloop f = make_forloop(init, cond, inc, for_loop_body);
	/* Modify the enclosing instruction to be a for-loop instead. It
	   works even if we are in a gen_recurse because we are in the
	   bottom-up phase of the recursion. */
	instruction_tag(i) = is_instruction_forloop;
	instruction_forloop(i) = f;
	/* Detach informations of the while-loop before freeing it: */
	whileloop_body(wl) = statement_undefined;
	whileloop_condition(wl) = expression_undefined;
	free_whileloop(wl);
    wl=whileloop_undefined;
    break;
      }
    }
  }
  return;
}


/* Apply recursively for-loop recovering to a given statement. */
static void
recover_for_loop_in_statement(statement s) {
  /* We need to access to the statement containing the current
     while-loops, so ask NewGen gen_recurse to keep this informations for
     us: */
  /* Iterate on all the while-loops: */
  //gen_debug = -1;
  gen_recurse(s,
              /* Since loop statements can be nested, only restructure in
                  a bottom-up way, : */
	      whileloop_domain, gen_true, try_to_recover_for_loop_in_a_while);
  //gen_debug = 0;
}


/* The phase to apply for-loop recovering to a given code module */
bool
recover_for_loop(char * module_name) {
  statement module_statement;

  /* Get the true ressource, not a copy, since we modify it in place. */
  module_statement =
    (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  set_current_module_statement(module_statement);
  entity mod = module_name_to_entity(module_name);
  set_current_module_entity(mod);

  /* Construct the mapping to get the statements associated to the
     dependence graph: */
  set_ordering_to_statement(module_statement);

  /* The proper effect to detect statement memory effects: */
  set_proper_rw_effects((statement_effects)
			db_get_memory_resource(DBR_PROPER_EFFECTS,
					       module_name,
					       true));

  /* To set up the hash table to translate value into value names */
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource(DBR_CUMULATED_EFFECTS,
						  module_name, true));

  /* Get the transformers of the module: */
  set_semantic_map((statement_mapping)
		   db_get_memory_resource(DBR_TRANSFORMERS,
					  module_name,
					  true));

  transformer summary = (transformer)
    db_get_memory_resource(DBR_SUMMARY_TRANSFORMER, module_name, true);

  /* Build all the mapping needed by the transformer usage: */
  module_to_value_mappings(mod);

  /* The summary precondition may be in another module's frame */
  translate_global_values(mod, summary);

  /* Get the data dependence graph: */
  /* The dg is more precise than the chains, so I (RK) guess I should
     remove more code with the dg, specially with array sections and
     so on. */
  /* FI: it's much too expensive; and how can you gain something
   * with scalar variables?
   */
  /*
    dependence_graph =
    (graph) db_get_memory_resource(DBR_DG, module_name, true);
  */

  /* Get the use-def chains */
  /*
    Not used yet
    dependence_graph =
    (graph) db_get_memory_resource(DBR_CHAINS, module_name, true);
  */

  debug_on("RECOVER_FOR_LOOP_DEBUG_LEVEL");

  recover_for_loop_in_statement(module_statement);

  pips_debug(2, "done");

  debug_off();

  /* Reorder the module, because some statements have been deleted.
     Well, the order on the remaining statements should be the same,
     but by reordering the statements, the number are consecutive. Just
     for pretty print... :-) */
  module_reorder(module_statement);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_statement);

  reset_semantic_map();
  reset_cumulated_rw_effects();
  reset_proper_rw_effects();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();
  free_value_mappings();

  /* Should have worked: */
  return true;
}
