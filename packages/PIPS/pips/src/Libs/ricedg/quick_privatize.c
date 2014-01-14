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
/*********************************************************************************/
/* QUICK PRIVATIZATION                                                           */
/*********************************************************************************/

#include "local.h"
#include "transformations.h"

static bool quick_privatize_loop(statement /*stat*/, list /*successors*/);
static bool quick_privatize_statement_pair(statement /*s1*/,
                                           statement /*s2*/,
                                           list /*conflicts*/);

void quick_privatize_graph(graph dep_graph) {
  /* we analyze arcs exiting from loop statements */
  FOREACH (VERTEX, v1, graph_vertices(dep_graph)) {
    statement s1 = vertex_to_statement(v1);
    list successors = vertex_successors(v1);
    if(statement_loop_p(s1)) {
      loop l = statement_loop(s1);
      list locals = loop_locals(l);
      entity ind = loop_index(l);

      if(gen_find_eq(ind, locals) == entity_undefined) {
        if(// entity_privatizable_in_loop_p(ind, l) && /* to be restored when global variables are uniformly treated everywhere. BC.*/
	   quick_privatize_loop(s1, successors)) {
          pips_debug(1, "Index for loop %" PRIdPTR " privatized\n",
              statement_number(s1));
          loop_locals(l) = CONS(ENTITY, ind, locals);
        } else {
          pips_debug(1, "could not privatize loop %" PRIdPTR "\n",
              statement_number(s1));
        }
      }
    }
  }
}

static bool quick_privatize_loop(statement stat, list successors) {
  pips_debug(3, "arcs from %" PRIdPTR "\n", statement_number(stat));
  FOREACH(SUCCESSOR, su, successors) {
    dg_arc_label dal = (dg_arc_label)successor_arc_label(su);
    statement st = vertex_to_statement(successor_vertex(su));

    pips_debug(3, "arcs to %" PRIdPTR "\n", statement_number(st));

    if(!quick_privatize_statement_pair(stat, st, dg_arc_label_conflicts(dal)))
      return (false);
  }

  return (true);
}



static bool quick_privatize_statement_pair(s1, s2, conflicts)
  statement s1, s2;list conflicts; {
  loop l1 = statement_loop(s1);
  entity ind1 = loop_index(l1);
  FOREACH(CONFLICT, c, conflicts) {
    effect f1 = conflict_source(c);
    reference r1 = effect_any_reference(f1);
    entity e1 = reference_variable(r1);

    effect f2 = conflict_sink(c);
    reference r2 = effect_any_reference(f2);
    entity e2 = reference_variable(r2);

    pips_debug(2, "conflict between %s & %s\n",
        entity_name(e1), entity_name(e2));

    /* equivalence or conflict not created by loop index. I give up ! */
    if(e1 != ind1)
      continue;

    if(action_write_p(effect_action(f1)) && action_read_p(effect_action(f2))) {
      /* we must know where this read effect come from. if it
       comes from the loop body, the arc may be ignored. */

      list loops = load_statement_enclosing_loops(s2);

      if(gen_find_eq(s1, loops) == entity_undefined) {
        loop l2;
        entity ind2;
        list range_effects;

        pips_debug(3, "the arc goes outside the loop body.\n");

        if(is_implied_do_index(e1, statement_instruction(s2))) {
          pips_debug(3, "s2 is an implied loop\n");
          return (true);
        }
        if(!statement_loop_p(s2)) {
          pips_debug(3, "s2 not a loop\n");
          return (false);
        }

        /* s2 is a loop. if there are no read effet in the range
         part, ignore this conflict. */
        l2 = statement_loop(s2);
        ind2 = loop_index(l2);
        range_effects = proper_effects_of_range(loop_range(l2));
        FOREACH(EFFECT, e, range_effects) {
          if(reference_variable(effect_any_reference(e)) == ind2
              && action_read_p(effect_action(e))) {

            pips_debug(3, "index read in range expressions\n");

            free_effects(make_effects(range_effects));
            return (false);
          }
        }
        free_effects(make_effects(range_effects));
      }
    }
  }

  return (true);
}

