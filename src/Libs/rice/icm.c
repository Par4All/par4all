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
 * Invariant Code Motion 
 *
 * Olivier Albiez
 */


/*
 * Il faut tester la presence de dependance sur le graphe original et
 * modifier une copie.
 */


#include "local.h"
#include "effects-generic.h"
#include "effects-simple.h"

statement vertex_to_statement(vertex v);


/* Set to 2 if we want to simplify in two passes */
#define NB_SIMPLIFY_PASSES 1


#define FLOW_DEPENDANCE 1
#define ANTI_DEPENDANCE 2
#define OUTPUT_DEPENDANCE 4
#define INPUT_DEPENDANCE 8
#define ALL_DEPENDANCES (FLOW_DEPENDANCE|ANTI_DEPENDANCE|OUTPUT_DEPENDANCE)

/*
 * Definition of local variables
 */
static int reference_level=0;

static bool expression_invariant = false; 
static set /* of entity */ invariant_entities = set_undefined;
//static set /* of statement */ statements_partialy_invariant = set_undefined;


/*********************************************************** PRINT FUNCTIONS */


/*
 * Print a statement_to_effect table.
 */
void 
dump_sef(statement_effects se)
{
    fprintf(stderr, "\n");
    STATEMENT_EFFECTS_MAP(s, e, {
	fprintf(stderr, 
		"%02td (%p) -> (%td) : ", 
		statement_number(s), 
		s,
		gen_length(effects_effects(e)));

	MAP(EFFECT, e, { 
	    print_words(stderr, words_effect(e));
	    fprintf(stderr, "(%p), ", e);
	}, effects_effects(e));
	
	fprintf(stderr, "\n");
    }, se);
}


/*
 * Print a conflict.
 */
static void 
prettyprint_conflict(FILE *fd, conflict c)
{
    fprintf(fd, "\t\tfrom ");
    print_words(fd, words_effect(conflict_source(c)));

    fprintf(fd, " to ");
    print_words(fd, words_effect(conflict_sink(c)));

    fprintf(fd, " at levels ");
    MAP(INT, level, {
	fprintf(fd, " %d", level);
    }, cone_levels(conflict_cone(c)));
    fprintf(fd, "\n");
}


/*
 * Print a successor.
 */
static void 
prettyprint_successor(FILE *fd, successor su)
{
    vertex v = successor_vertex(su);
    fprintf(fd, 
	    "link with %02td :\n", 
	    statement_number(vertex_to_statement(v)));

    MAP(CONFLICT, c, { 
	prettyprint_conflict(fd, c); 
    }, dg_arc_label_conflicts((dg_arc_label) successor_arc_label(su)));
}


/*
 * Print a vertex.
 */
static void 
prettyprint_vertex(FILE *fd, vertex v)
{
    fprintf(fd, 
	    "vertex %02td :\n", 
	    statement_number(vertex_to_statement(v)));
    MAP(SUCCESSOR, s, { 
	prettyprint_successor(fd, s); 
    }, vertex_successors(v));
}


/*
 * Print a entities list.
 */
void
print_list_entities(list /* of entity */ l)
{
    MAP(ENTITY, e, { 
	fprintf(stderr, "%s ", entity_name(e));
    }, l);
}


/*********************************************************** MISC FUNCTIONS */


/*
 * Add statements from the vertex list to the statements set.
 */
static set /* of statement */
vertices_to_statements(list /* of vertex */ vl, 
		       set /* of statement */ss)
{
    MAP(VERTEX, v, {
	ss = set_add_element(ss, ss, (char *) vertex_to_statement(v));
    }, vl);

    return ss;
}


static set /* of entity */
invariant_vertex_to_invariant_entities(vertex v,
    set /* of entity */ rs)
{
  statement st = vertex_to_statement(v);
  list /* of effect */ le = load_proper_rw_effects_list(st);

  MAP(EFFECT, ef, {
      if (effect_write_p(ef)) {
        entity e = reference_variable(effect_any_reference(ef));

        if (!set_belong_p(rs, (char *) v)) {
          rs = set_add_element(rs, rs, (char *) e);
        }
      }
  }, le);

  return rs;
}


/************************************************************ MAPPING TOOLS */


GENERIC_LOCAL_FUNCTION(has_level, persistant_statement_to_int)
GENERIC_LOCAL_MAPPING(has_indices, list, statement)

static list indices = NIL;    
static int depth = 0;


/*
 * Gen_multi_recurse hook.
 * Set the depth and the indices list of the statement.
 */
static bool
statement_mark(statement s)
{
     store_has_level(s, depth);
     store_statement_has_indices(s, gen_nreverse(gen_copy_seq(indices)));
     
     return true;
}


/*
 * Gen_multi_recurse hook.
 * Update the depth and the indices list when enter a loop.
 */
static bool
loop_level_in (loop l) 
{
    entity index = loop_index(l);
    indices = CONS(ENTITY, index, indices);
    ++depth;

    return true;
}


/*
 * Gen_multi_recurse hook.
 * Update the depth and the indices list when exit a loop.
 */
static void
loop_level_out (__attribute__((unused))loop l) 
{
    list new_indices = CDR(indices);
    free(indices);
    indices = new_indices;
    --depth;
}


/***************************************************** DEPENDANCES FUNCTIONS */


/*
 * Test if the conflict correspond to the given dependance type.
 */
static bool
action_dependance_p(conflict c, int dependance_type) 
{
    action s = effect_action(conflict_source(c)) ;
    action k = effect_action(conflict_sink(c)) ;

    return (((dependance_type & FLOW_DEPENDANCE) &&
            ((action_write_p(s) && action_read_p(k)))) ||
            ((dependance_type & ANTI_DEPENDANCE) &&
            ((action_read_p(s) && action_write_p(k)))) ||
            ((dependance_type & OUTPUT_DEPENDANCE) &&
            ((action_write_p(s) && action_write_p(k)))) ||
            ((dependance_type & INPUT_DEPENDANCE) &&
            ((action_read_p(s) && action_read_p(k)))));
}


/* 
 * Test the existence of a given dependence between two vertices v1, v2.
 * 
 * Note: 
 * - dependance_type is a bitfield which contains the kind of dependence,
 * - level is the minimum level wanted.
 */
static bool
dependance_vertices_p(vertex v1, vertex v2, int dependance_type, int level)
{
  ifdebug(9) {
    debug(9, "dependance_vertices_p", "");

    if (dependance_type & FLOW_DEPENDANCE) fprintf(stderr, "F ");
    if (dependance_type & ANTI_DEPENDANCE) fprintf(stderr, "A ");
    if (dependance_type & OUTPUT_DEPENDANCE) fprintf(stderr, "O ");
    if (dependance_type & INPUT_DEPENDANCE) fprintf(stderr, "I ");
    fprintf(stderr, "dep. at level >= %d ", level);
    fprintf(stderr,
        "between %02td and %02td ? ",
        statement_number(vertex_to_statement(v1)),
        statement_number(vertex_to_statement(v2)));
  }

  MAP(SUCCESSOR, su, {
      vertex s = successor_vertex(su);
      if (s == v2) {
        MAP(CONFLICT, c, {
            if (conflict_cone(c) != cone_undefined) {
              MAP(INT, l, {
                  if (l >= level) {
                    if (action_dependance_p(c, dependance_type))
                    {
                      ifdebug(9) { fprintf(stderr, "yes\n"); }
                      return true;
                    }
                  }
              }, cone_levels(conflict_cone(c)));
            }
        }, dg_arc_label_conflicts(successor_arc_label(su)));
      }
  }, vertex_successors(v1));

  ifdebug(9) { fprintf(stderr, "no\n"); }
  return false;
}


/*
 * Test the existance of a given dependance from the vertex v with any other 
 * vertices than v. 
 */
static bool
exist_non_self_dependance_from_vertex_p(vertex v, 
					int dependance_type, 
					int level)
{
    MAP(SUCCESSOR, su, {
	vertex y = successor_vertex(su);
	if (v != y) {
	    if (dependance_vertices_p(v, y, dependance_type, level))
		return true;
	}
    }, vertex_successors(v));

    return false;
}

/*
 * Remove all levels between level_min and level_max from the list of levels.
 */
static list /* of level */
remove_dependance_from_levels(list /* of level */ levels, 
			      int level_min, 
			      int level_max)
{
    list /* of level */ new_levels = NIL;

    MAP(INT, level, {
	if ((level < level_min) || (level > level_max)) {
	    new_levels = gen_nconc(new_levels, CONS(INT, level, NIL));
	}
    }, levels);

    /* memory leak : levels */

    return new_levels;
}


/*
 * Remove all conflicts matching the dependance_type for a level 
 * between level_min and level_max.
 */
static list /* of conflict */
remove_dependance_from_conflicts(list /* of conflict */ conflicts, 
				 int dependance_type, 
				 int level_min, 
				 int level_max)
{
    list /* of conflict */ new_conflicts = NIL;

    ifdebug(7) {
	fprintf(stderr, "Old conflicts :\n");
	MAP(CONFLICT, c, { 
	    prettyprint_conflict(stderr, c); 
	}, conflicts);
    }

    FOREACH(CONFLICT, c, conflicts) {
	if(conflict_cone(c) != cone_undefined) {
	    if (action_dependance_p(c, dependance_type))
	    {
		list /* of level */ levels = cone_levels(conflict_cone(c));

		list /* of level */ new_levels = 
		    remove_dependance_from_levels(levels, 
						  level_min, 
						  level_max);

		if (new_levels != NIL) {
		    cone_levels(conflict_cone(c)) = new_levels;
		    new_conflicts = CONS(CONFLICT, c, new_conflicts);
		}

		/* memory leak : levels */
	    }
	    else {
		new_conflicts = CONS(CONFLICT, c, new_conflicts);
	    }
	}
	else {
	    new_conflicts = CONS(CONFLICT, c, new_conflicts);
	}
    }

    /* memory leak : conflicts */

    ifdebug(7) 
    {
	fprintf(stderr, "New conflicts :\n");
	MAP(CONFLICT, c, { 
	    prettyprint_conflict(stderr, c); 
	}, new_conflicts);
    }

    return new_conflicts;
}


/*
 * Remove all successors matching the given parameters.
 */
static list /* of successor */
remove_dependances_from_successors(list successors, 
				   vertex v2, 
				   int dependance_type, 
				   int level_min, 
				   int level_max)
{
    list /* of successor */ new_successors = NIL;

    ifdebug(7) 
    {
	fprintf(stderr, "Old successors :\n");
	MAP(SUCCESSOR, s, { 
	    prettyprint_successor(stderr, s); 
	}, successors);
    }

    MAP(SUCCESSOR, su, {
	vertex v = successor_vertex(su);
	
	if (v == v2) {
	    list /* of conflict */ lc = 
		dg_arc_label_conflicts(successor_arc_label(su));

	    list /* of conflict */ new_lc = 
		remove_dependance_from_conflicts(lc,
						 dependance_type, 
						 level_min, 
						 level_max);

	    if (new_lc != NIL) {
		dg_arc_label_conflicts(successor_arc_label(su)) = new_lc;
		new_successors = CONS(SUCCESSOR, su, new_successors);
	    }

	    /* memory leak : lc */
	}
	else {
	    new_successors = CONS(SUCCESSOR, su, new_successors);
	}

    }, successors);

    /* memory leak : successors */

    ifdebug(7) 
    {
	fprintf(stderr, "New successors :\n");
	MAP(SUCCESSOR, s, { 
	    prettyprint_successor(stderr, s); 
	}, successors);
    }

    return new_successors;
}


/*
 * Remove all dependances between v1 and v2 matching the given parameters.
 */
static void
remove_dependance(vertex v1, /* Successors of this vertex are updated */
		  vertex v2, 
		  int dependance_type, 
		  int level_min, 
		  int level_max)
{
    list /* of successor */ v1_successors = vertex_successors(v1);

    ifdebug(3) {
	pips_debug(3, "Remove ");
	
	if (dependance_type & FLOW_DEPENDANCE) fprintf(stderr, "F ");
	if (dependance_type & ANTI_DEPENDANCE) fprintf(stderr, "A ");
	if (dependance_type & OUTPUT_DEPENDANCE) fprintf(stderr, "O ");
	if (dependance_type & INPUT_DEPENDANCE) fprintf(stderr, "I ");

	fprintf(stderr, 
		"dep. at level >= %d and level <= %d ", 
		level_min, 
		level_max);

	fprintf(stderr, 
		"between %02td and %02td.\n", 
		statement_number(vertex_to_statement(v1)), 
		statement_number(vertex_to_statement(v2))); 
    } 

    vertex_successors(v1) = 
	remove_dependances_from_successors(v1_successors, 
					   v2, 
					   dependance_type, 
					   level_min, 
					   level_max);

    ifdebug(7) {
	prettyprint_vertex(stderr, v1);
    }
}


/************************************************************ SCCS FUNCTIONS */


/*
 * FindAndTopSortSccs hook.
 * Test if the vertex belong to the region.
 */
static bool 
common_ignore_this_vertex(set /* of statement */ region, vertex v)
{
    return(!set_belong_p(region, (char *) vertex_to_statement(v)));
}


/*
 * FindAndTopSortSccs hook.
 * Test if exist any dependance with a level >= 'level'. 
 */
static bool 
icm_ignore_this_successor(vertex v, 
			  set /* of statement */ region, 
			  successor su, 
			  int level)
{
    if (common_ignore_this_vertex(region, successor_vertex(su)))
	return(true);

    if (!dependance_vertices_p(v, 
			       successor_vertex(su), 
			       ALL_DEPENDANCES, 
			       level))
	return (true);

    return false;
}


/*
 * FindAndTopSortSccs hook.
 * Test if exist a flow dependance with a level >= 'level'. 
 */
static bool 
invariant_ignore_this_successor(vertex v, 
				set /* of statement */ region, 
				successor su, 
				int level)
{
    if (common_ignore_this_vertex(region, successor_vertex(su)))
	return(true);

    if (!dependance_vertices_p(v, 
				successor_vertex(su), 
				FLOW_DEPENDANCE, 
				level))
	return (true);

    return (false);
}


/********************************************************* DG SIMPLIFICATION */


/*
 * Test if a statement depend of any indicies.
 * The 'level' first indicies are not used.
 */
static bool
statement_depend_of_indices_p(statement st, 
    list /* of entity */ indices,
    int level)
{
  list /* of effect */ le = load_proper_rw_effects_list(st);

  ifdebug(6) {
    pips_debug(6, "Statement %02td depend of ", statement_number(st));
    print_list_entities(indices);
  }

  /* Ignore the first indicies... */
  for(; level > reference_level; --level) {
    indices = CDR(indices);
  }

  MAP(ENTITY, index, {
      MAP(EFFECT, ef, {
          entity e = reference_variable(effect_any_reference(ef));
          if (e == index) {

            ifdebug(6) {
              fprintf(stderr,": yes\n");
            }

            return true;
          }
      }, le);
  }, indices);

  ifdebug(6) {
    fprintf(stderr,": no\n");
  }

  return false;
}


static bool
inv_entity_filter(entity e)
{
    expression_invariant &= set_belong_p(invariant_entities, (char *) e);
    if (!expression_invariant) gen_recurse_stop(NULL);
    return true;
}

static bool ref_flt(reference r)
{
  return inv_entity_filter(reference_variable(r));
}

static bool
expressions_invariant_p(list /* of expression */ le)
{
    MAP(EXPRESSION, exp, {
	expression_invariant = true;

	gen_recurse(exp, reference_domain, ref_flt, gen_null);

	if (!expression_invariant) {
	    return false;
	}
    }, le);

    return true;
}


/*
 * Test if the vertex is partially invariant.
 */
static bool
vertex_partially_invariant_p(vertex v, 
    __attribute__((unused)) graph g,
    __attribute__((unused)) int level,
    __attribute__((unused)) set /* of statement */ invariant)
{
  statement st = vertex_to_statement(v);

  MAP(EFFECT, ef, {
      reference ref = effect_any_reference(ef);

      /* Looking for write effects only */
      if (effect_write_p(ef)) {

        /* Which kind of reference we have ? */
        if (reference_indices(ref) != NIL) {
          /* An array access with well known indices */
          if (expressions_invariant_p(reference_indices(ref))) {
            pips_debug(6,
                "statement %02td is partially invariant "
                "(known array access).\n",
                statement_number(st));

            return true;
          }
          else {
            pips_debug(6,
                "statement %02td is not partially invariant "
                "(known array access).\n",
                statement_number(st));

            return false;
          }
        }
        else {
          if(variable_entity_dimension(reference_variable(ref)) != 0) {
            /* An array access with unknow indices */
            pips_debug(6,
                "statement %02td is not partially invariant "
                "(UNKNOWN array access).\n",
                statement_number(st));

            return false;
          }
          else {
            /* A scalar variable */
            pips_debug(6,
                "statement %02td is partially invariant "
                "(scalar access).\n",
                statement_number(st));

            return true;
          }
        }
      }
  }, load_proper_rw_effects_list(st));

  pips_debug(6,
      "statement %02td is not partially invariant.\n",
      statement_number(st));

  return false;
}


/*
 * Test if the vertex is invariant.
 */
static bool
vertex_invariant_p(vertex v, 
    graph g,
    int level,
    set /* of statement */ region,
    set /* of statement */ invariant)
{
  statement st = vertex_to_statement(v);

  /* Test if the statement is dependent of ALL loop indexes >= level */
  if (statement_depend_of_indices_p(st,
      load_statement_has_indices(st),
      level)) {
    pips_debug(6,
        "statement %02td is not invariant (depend of indices).\n",
        statement_number(st));

    return false;
  }

  /* If there is a flow dependence from v to v, then v is not variant */
  if (dependance_vertices_p(v, v, FLOW_DEPENDANCE, level)) {
    pips_debug(6,
        "statement %02td is not invariant (self flow dep).\n",
        statement_number(st));

    return false;
  }

  /* If there is a flow dependence from y to v and if y is not invariant,
   * then v is not invariant */
  MAP(VERTEX, y, {
      if (dependance_vertices_p(y, v, FLOW_DEPENDANCE, level)) {
        statement y_st = vertex_to_statement(y);

        if (!set_belong_p(invariant, (char *) y_st) &&
            set_belong_p(region, (char *) y_st)) {

          pips_debug(6,
              "statement %02td is not invariant "
              "(dep. of %02td).\n",
              statement_number(st),
              statement_number(y_st));

          return false;
        }
      }
  }, graph_vertices(g));

  pips_debug(6,
      "statement %02td is invariant.\n",
      statement_number(st));

  return true;
}


/*
 * Simplify a invariant vertex 'v' using the following rule:
 *
 * if 
 *   exist output(v, v, <= level)
 *   not exist output(v, z, <= level) v != z
 * then 
 *   remove output(v, v, <= level)
 *   foreach y satisfying
 *     exist flow(v -> y, infiny)
 *     exist anti(y -> v, <= level)
 *   do remove anti(y -> v, <= level)
 */
static void
SimplifyInvariantVertex(vertex v, /* Successors of this vertex are updated */
    set /* of statement */ region,
    int level)
{
  set /* of vertex */ matching_vertices = set_make(set_pointer);
  statement st = vertex_to_statement(v);

  if (dependance_vertices_p(v, v, OUTPUT_DEPENDANCE, level) &&
      !exist_non_self_dependance_from_vertex_p(v,
          OUTPUT_DEPENDANCE, level)) {
    FOREACH(SUCCESSOR, su, vertex_successors(v)) {
      vertex y = successor_vertex(su);
      if (!common_ignore_this_vertex(region, y) &&
          dependance_vertices_p(v, y, FLOW_DEPENDANCE, level) &&
          dependance_vertices_p(y, v, ANTI_DEPENDANCE, level)) {

        matching_vertices = set_add_element(matching_vertices,
            matching_vertices,
            (char *) y);
      }
    }

    //remove_dependance(v, v, OUTPUT_DEPENDANCE, 0, load_has_level(st));
    remove_dependance(v, v, OUTPUT_DEPENDANCE, level, load_has_level(st));

    if (!set_empty_p(matching_vertices)) {
      SET_MAP(y, {
          remove_dependance((vertex) y, v, ANTI_DEPENDANCE,
              //0, load_has_level(st));
              level, load_has_level(st));
      }, matching_vertices);
    }
  }

  set_free(matching_vertices);
}


/*
 * Find and simplify invariants statements.
 */
static graph
DoInvariantsStatements(list /* of scc */ lsccs, 
    graph g,
    set /* of statement */ region,
    int level,
    set /* of statement */ partially_invariant)
{
  set /* of statement */ invariant = set_make(set_pointer);

  FOREACH(SCC, s, lsccs) {
    list /* of vertex */ lv = scc_vertices(s);

    if (gen_length(lv) > 1) {
      /* Group of vertices : all are variants */
    }
    else {
      /* One statement... */
      vertex v = VERTEX(CAR(lv));
      statement st = vertex_to_statement(v);

      if (!declaration_statement_p(st)) {
        if (vertex_invariant_p(v, g, level, region, invariant)) {
          /* which is invariant */
          SimplifyInvariantVertex(v, region, level);

          /* Added to the list */
          invariant = set_add_element(invariant,
              invariant,
              (char *) st);

          /* Invariant is partially invariant... */
          partially_invariant = set_add_element(partially_invariant,
              partially_invariant,
              (char *) st);

          invariant_entities =
              invariant_vertex_to_invariant_entities(v,
                  invariant_entities);

        }
        else if (vertex_partially_invariant_p(v, g, level, invariant)) {
          partially_invariant =
              set_add_element(partially_invariant,
                  partially_invariant,
                  (char *) st);
        }
      }
      else {
        //If it's an declaration, so we work on environment domain
        // NL: It's a workaround, and not totally sure that it always work, not enough test case

        /* NL: I can't explain why we have to add the declaration statement
         *     as an invariant statement, but it's seem that it work,
         *     and permit some optimizations...
         */
        /* Added to the list */
        invariant = set_add_element(invariant,
            invariant,
            (char *) st);

        /* Invariant is partially invariant... */
        partially_invariant = set_add_element(partially_invariant,
            partially_invariant,
            (char *) st);

        /* NL: I'm not totally sure of the explanation I propose below
         *     The variable declared inside the loop can be consider as invariant variable
         *     because there values only be important inside the loop and not outside.
         *     so from the outside of the loop these variables doesn't exist and
         *     so can be consider as invariant?
         */
        invariant_entities =
            invariant_vertex_to_invariant_entities(v,
                invariant_entities);
      }
    }
  }

  set_free(invariant);

  return g;
}


/*
 * Test if the vertex is redundant.
 */
static bool
vertex_redundant_p(vertex v, 
    __attribute__((unused)) graph g,
    int level,
    set /* of statement */ region,
    set /* of statement */ partially_invariant,
    set /* of statement */ redundant)
{
  statement st = vertex_to_statement(v);

  /* Test if the statement is depandant of ALL loop indexes >= level */
  /* This condition is not required, but putting a statement depending
       of indicies after the loop is tiedous (setting the value to 
       the bound+1...) */
/*
  if (statement_depend_of_indices_p(st, 
  load_statement_has_indices(st),
  level)) {
  ifdebug(6) { 
  debug(6, "vertex_redundant_p", "");
  fprintf(stderr, 
  "statement %02d is not redundant (depend of indices).\n",
  statement_number(st));
  }

  return false;
  }
*/

  /* Test if we are not always writing at the same adress
       ie. is not partially_invariant. */
  if (!set_belong_p(partially_invariant, (char *) st)) {
    pips_debug(6,
          "statement %02td is not redundant (variable address).\n",
          statement_number(st));

    return false;
  }

  /* If there is a flow dependance from v to y and if y is not redundant,
       then v is not redundant */
  FOREACH(SUCCESSOR, su, vertex_successors(v)) {
    vertex y = successor_vertex(su);
    if (dependance_vertices_p(v, y, FLOW_DEPENDANCE, level)) {
      statement y_st = vertex_to_statement(y);

      if (!set_belong_p(redundant, (char *) y_st) &&
          set_belong_p(region, (char *) y_st)) {

        pips_debug(6,
              "statement %td is not redundant "
              "(dep. of %td).\n",
              statement_number(st),
              statement_number(y_st));

        return false;
      }
    }
  }

  pips_debug(6,
        "statement %td is redundant.\n",
        statement_number(st));
  return true;
}


/*
 * Simplify a redundant vertex 'v' using the following rule:
 */
static void
SimplifyRedundantVertex(vertex v, /* Successors of this vertex are updated */
    set /* of statement */ region,
    int level)
{
  set /* of vertices */ matching_vertices = set_make(set_pointer);
  statement st = vertex_to_statement(v);

  if (dependance_vertices_p(v, v, OUTPUT_DEPENDANCE, level) &&
      !exist_non_self_dependance_from_vertex_p(v,
          OUTPUT_DEPENDANCE, level)) {
    FOREACH(SUCCESSOR, su, vertex_successors(v)) {
      vertex y = successor_vertex(su);
      if (!common_ignore_this_vertex(region, y) &&
          dependance_vertices_p(y, v, FLOW_DEPENDANCE, level) &&
          dependance_vertices_p(v, y, ANTI_DEPENDANCE, level)) {

        matching_vertices = set_add_element(matching_vertices,
            matching_vertices,
            (char *) y);
      }
    }

    remove_dependance(v, v, OUTPUT_DEPENDANCE, 0, load_has_level(st));

    if (!set_empty_p(matching_vertices)) {
      SET_MAP(y, {
          remove_dependance(v, (vertex)y, ANTI_DEPENDANCE,
              0, load_has_level(st));
      }, matching_vertices);
    }
  }

  set_free(matching_vertices);
}


/*
 * Find and simplify redundant statements.
 */
static graph
DoRedundantsStatements(list /* of scc */ lsccs, 
    graph g,
    set /* of statement */ region,
    int level,
    set /* of statement */ partially_invariant)
{
  set /* of statement */ redundant = set_make(set_pointer);

  MAP(SCC, s, {
      list /* of vertex */ lv = scc_vertices(s);

      if (gen_length(lv) > 1) {
        /* Group of vertices : all are no redundant */
      }
      else {
        /* One statement... */
        vertex v = VERTEX(CAR(lv));
        statement st = vertex_to_statement(v);

        if (set_belong_p(partially_invariant, (char *) st)) {
          if (vertex_redundant_p(v, g, level,
              region,
              partially_invariant,
              redundant)) {
            /* which is redundant */
            SimplifyRedundantVertex(v, region, level);

            redundant = set_add_element(redundant,
                redundant,
                (char *) st);
          }
        }
      }
  }, lsccs);

  set_free(redundant);

  return g;
}

static graph SimplifyGraph(graph g, set region, int level, unsigned int count);
static graph SupressDependances(graph g, set region, int level, unsigned int count);

/*
 * Simplify the dependence graph.
 */
static graph
                    (graph g,
    set /* of statement */ region,
    int level,
    unsigned int count)
{
  ifdebug(5) {
    pips_debug(5, "start\n");
    pips_debug(5, "level=%d, count=%d\n", level, count);

    pips_debug(5, "set of statement number studied: ");
    SET_MAP(elt, fprintf(stderr, "%d, ", statement_number((statement) elt)), region);
    fprintf(stderr, "\n");
    ifdebug(8) {
      pips_debug(8, "set of statement studied:\n");
      SET_MAP(elt, print_statement((statement) elt), region);
    }
  }
  list /* of scc */ lsccs;

  /* Find sccs */
  set_sccs_drivers(&common_ignore_this_vertex,
      &icm_ignore_this_successor);
  lsccs = FindAndTopSortSccs(g, region, level);
  reset_sccs_drivers();

  FOREACH(SCC, elmt, lsccs) {
    /* Check if the component is strongly connected */
    if (strongly_connected_p(elmt, level)) {
      set new_region = set_make(set_pointer);
      new_region = vertices_to_statements(scc_vertices(elmt),
          new_region);

      g = SupressDependances(g, new_region, level, count);

      set_free(new_region);
    }
  }

  // No leak
  gen_free_list(lsccs);

  ifdebug(5) {
    pips_debug(5, "end\n");
  }
  return g;
}


/*
 * Supress unneeded dependances.
 */
static graph
SupressDependances(graph g,
    set /* of statement */ region,
    int level,
    unsigned int count)
{
  ifdebug(5) {
    pips_debug(5, "start\n");
    pips_debug(5, "level=%d, count=%d\n", level, count);

    pips_debug(5, "set of statement number studied: ");
    SET_MAP(elt, fprintf(stderr, "%d, ", statement_number((statement) elt)), region);
    fprintf(stderr, "\n");
    ifdebug(8) {
      pips_debug(8, "set of statement studied:\n");
      SET_MAP(elt, print_statement((statement) elt), region);
    }
  }
  list /* of scc */ lsccs;
  set /* of statement */ partially_invariant = set_make(set_pointer);

  /* Find sccs considering only flow dependances */
  set_sccs_drivers(&common_ignore_this_vertex,
      &invariant_ignore_this_successor);
  lsccs = FindAndTopSortSccs(g, region, level);
  reset_sccs_drivers();

  /* Forward simplification */
  g = DoInvariantsStatements(lsccs, g, region, level, partially_invariant);

  /* Backward simplification */
  lsccs = gen_nreverse(lsccs);
  g = DoRedundantsStatements(lsccs, g, region, level, partially_invariant);

  // No leak
  gen_free_list(lsccs);
  set_free(partially_invariant);

  if (count > 1) {
    return SimplifyGraph(g, region, level, count-1);
  }
  else{
    return SimplifyGraph(g, region, level+1, NB_SIMPLIFY_PASSES);
  }
}


/******************************************************** REMOVE DUMMY LOOPS */


DEFINE_LOCAL_STACK(stmt, statement)

static list /* of entity */ depending_indices;
static bool it_depends;


/* 
 * Set whether s depends from enclosing indices
 */
static bool does_it_depend(statement s)
{
    it_depends |= statement_depend_of_indices_p(s, depending_indices, 0);
    if (it_depends) gen_recurse_stop(NULL);
    return true;
}


static bool push_depending_index(loop l)
{
    depending_indices = CONS(ENTITY, loop_index(l), depending_indices);
    return true;
}


static void pop_depending_index(loop l)
{
    list tmp = depending_indices;
    pips_assert("current loop index is poped", 
		loop_index(l)==ENTITY(CAR(depending_indices)));
    depending_indices = CDR(depending_indices);
    CDR(tmp) = NIL;
    free(tmp);
}


static bool drop_it(loop l)
{
  if (execution_parallel_p(loop_execution(l)))
  {
    depending_indices = NIL;
    it_depends = false;
    gen_multi_recurse(l,
        statement_domain, does_it_depend, gen_null,
        loop_domain, push_depending_index, pop_depending_index,
        NULL);
    depending_indices = NIL; /* assert? */
    return !it_depends;
  }

  return false;
}


/*
 * Compute the final expression of the loop index.
 * Follow the ANSI Fortran 77 normalization.

 *  = m1 + MAX(INT((m2 - m1 + m3) / m3), 0) * m3

 */
static expression
compute_final_index_value(expression m1, expression m2, expression m3)
{
    expression result;
    expression E0 = make_op_exp("-", m2, copy_expression(m1));
    expression E1 = make_op_exp("+", E0, copy_expression(m3));
    expression E2 = make_op_exp("/", E1, copy_expression(m3));

    if (expression_constant_p(E2)) {
	int val_E2 = expression_to_int(E2);

	/* max of (val_e2, 0) */
	if (val_E2 > 0) {
	    expression E3 = make_op_exp("*", E2, m3);
	    result = make_op_exp("+", E3, m1);
	}
	else {
	    result = m1;
        free_expression(m3);
	}
    }
    else {
	expression zero = int_to_expression(0);
	expression p_int, E3, E4;

	p_int = MakeUnaryCall(entity_intrinsic(INT_GENERIC_CONVERSION_NAME), 
			      E2);

	E3 = MakeBinaryCall(entity_intrinsic(MAX_OPERATOR_NAME),
			    p_int, 
			    zero);
	E4 = make_op_exp("*", E3, m3);
	result = make_op_exp("+", m1, E4);
    }

    /* memory leak */

    return result;
}


static bool icm_loop_rwt(loop l)
{
  statement head = stmt_head();

  ifdebug(5) {
    fprintf(stderr, "TEST : loop on %s (statement %td):\n",
        entity_name(loop_index(l)),
        statement_number(head));
  }

  if (drop_it(l))
  {
    statement index_statement;
    statement body = loop_body(l);

    expression index, m1, m2, m3;

    m1 = copy_expression(range_lower(loop_range(l)));
    m2 = copy_expression(range_upper(loop_range(l)));
    m3 = copy_expression(range_increment(loop_range(l)));

    /* Assume here that index is a scalar variable... :-) */
    pips_assert("icm_loop_rwt", entity_scalar_p(loop_index(l)));

    index = make_factor_expression(1, loop_index(l));

    index_statement =
        make_assign_statement(index,
            compute_final_index_value(m1, m2, m3));

    statement_instruction(head)=instruction_undefined;
    update_statement_instruction(head,make_instruction_block(make_statement_list(body, index_statement)));

    pips_debug(5, "-> loop on %s removed (statement %td)\n",
        entity_name(loop_index(l)),
        statement_number(head));

    /* memory leak... */
  }
  else {
    pips_debug(5, "-> loop on %s NOT removed (statement %td)\n",
        entity_name(loop_index(l)),
        statement_number(head));
  }

  return true;
}


/*
 * Drop all loops l  matching the pattern:
 *   l is parallel
 *   the body of l doesn't use indicies of the loop l.
 *
 * WARNING : the pattern is correct ????????
 */
static void drop_dummy_loops(statement s)
{
  /* WARNING :
   * We must recompute proper_effects for the program !!!
   * So we directly call the pass !!!!!
   */
  set_methods_for_proper_simple_effects();
  init_proper_rw_effects();
  proper_effects_of_module_statement(s);


  make_stmt_stack();

  gen_multi_recurse(s,
      statement_domain, stmt_filter, stmt_rewrite,
      loop_domain, icm_loop_rwt, gen_null,
      NULL);

  free_stmt_stack();
  reset_proper_rw_effects();
  generic_effects_reset_all_methods();
}


/*********************************************************** REGENERATE CODE */


/*
 * Simplify the dependance graph and regenerate the code.
 *
 * Using the algorithm described in Chapter xxx of Julien Zory's PhD?
 */
static statement icm_codegen(statement stat,
    graph g,
    set /* of statement */ region,
    int level,
    bool task_parallelize_p)
{
  statement result = statement_undefined;
  graph simplified_graph = graph_undefined;

  reference_level = level;

  debug_on("ICM_DEBUG_LEVEL");
  ifdebug(4) {
    pips_debug(9, "ICM_DEBUG_LEVEL start\n");
    pips_debug(4, "on statement:\n");
    print_statement(stat);
  }

  set_proper_rw_effects((statement_effects)
      db_get_memory_resource(DBR_PROPER_EFFECTS,
          get_current_module_name(),
          true));

  /* Compute has_level hash and has_indices tables */

  init_has_level();
  make_has_indices_map();

  indices = NIL;

  gen_multi_recurse(stat,
      loop_domain, loop_level_in, loop_level_out, /* LOOP */
      statement_domain, statement_mark, gen_null, /* STATEMENT */
      NULL);

  gen_free_list(indices);

  /* Simplify the dependance graph */

  simplified_graph = copy_graph(g);

  /* Definir le mapping entre les vertex originaux et les vertex copies */

  ifdebug(4) {
    pips_debug(4, "Original graph:\n");
    prettyprint_dependence_graph(stderr,
        statement_undefined,
        simplified_graph);
  }

  invariant_entities = set_make(set_pointer);

  simplified_graph = SimplifyGraph(simplified_graph,
      region,
      level,
      NB_SIMPLIFY_PASSES);

  set_free(invariant_entities);

  ifdebug(4) {
    pips_debug(4, "Simplified graph:\n");
    prettyprint_dependence_graph(stderr,
        statement_undefined,
        simplified_graph);
  }

  close_has_level();
  free_has_indices_map();
  reset_proper_rw_effects();
  /* CodeGenerate reload the
   * proper_rw_effects table, so we must
   * reset before... */

  pips_debug(9, "ICM_DEBUG_LEVEL stop\n");
  debug_off();

  /* Generate the code (CodeGenerate don't use the first
   * parameter...) */
  result =  CodeGenerate(/* big hack */ statement_undefined,
      simplified_graph,
      region,
      level,
      task_parallelize_p);
  free_graph(simplified_graph);

  ifdebug(4) {
    pips_debug(4, "Intermediate code:\n");
    print_statement(result);
  }

  /* Remove dummy loops. */
  drop_dummy_loops(result);

  ifdebug(4) {
    pips_debug(4, "Final code:\n");
    print_statement(result);
  }

  return result;
}


/*************************************************************** ENTRY POINT */


/* Phase that hoists loop invariant code out of loops.

   @param[in] module_name

   @return true because everything should go fine

   Prepare some stuffs and call icm_codegen...
*/
bool
invariant_code_motion(const char* module_name)
{
  entity module = local_name_to_top_level_entity(module_name);
  statement mod_stat = statement_undefined;
  set_current_module_entity(module);

  set_bool_property( "GENERATE_NESTED_PARALLEL_LOOPS", true );
  set_bool_property( "RICE_DATAFLOW_DEPENDENCE_ONLY", false );

  set_current_module_statement((statement)
      db_get_memory_resource(DBR_CODE,
          module_name,
          true));

  mod_stat = get_current_module_statement();

  set_ordering_to_statement(mod_stat);

  debug_on("ICM_DEBUG_LEVEL");

  ifdebug(7)
  {
    fprintf(stderr,
        "\nTesting NewGen consistency for initial code %s:\n",
        module_name);
    if (statement_consistent_p((statement)mod_stat))
      fprintf(stderr," NewGen consistent statement\n");
  }

  ifdebug(1) {
    pips_debug(1, "original sequential code:\n\n");
    print_statement(mod_stat);
  }

  if (graph_undefined_p(dg)) {
    dg = (graph) db_get_memory_resource(DBR_DG, module_name, true);
  }
  else {
    pips_internal_error("dg should be undefined");
  }

  enclosing = 0;
  rice_statement(mod_stat, 1, &icm_codegen);

  ifdebug(7) {
    fprintf(stderr, "\ntransformed code %s:",module_name);
    if (statement_consistent_p((statement)mod_stat))
      fprintf(stderr," gen consistent ");
  }

  // Uselessly reinitialize ordering_to_statement, even if it not set...
  clean_up_sequences(mod_stat);
  module_reorder(mod_stat);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, mod_stat);

  dg = graph_undefined;
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();

  debug_off();
  return true;
}
