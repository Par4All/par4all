/*

 $Id: loop_fusion.c 15900 2009-12-23 16:00:55Z amini $

 Copyright 1989-2009 MINES ParisTech

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
/* Functions to perform the greedy loop fusion of a loop sequence */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "effects-util.h"
#include "misc.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
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
/* */
#include "ricedg.h"
#include "semantics.h"
#include "transformations.h"
#include "chains.h"
extern bool get_bool_property(string);

/**
 * Fusion configuration
 */
typedef struct fusion_params {
  bool maximize_parallelism; // Prevent sequentializing loop that were parallel
  bool greedy; // Fuse as much a we can, and not only loops that have reuse
} *fusion_params;

/**
 * Structure to hold block used for the fusion selection algorithm.
 * It's used in a sequence of statements to keep track of the precedence
 * constraints between statements while fusing some of them
 */
typedef struct fusion_block {
  int num;
  statement s; // The main statement (header for loops)
  // statements inside the block (in case of loop, header won't belong to)
  set statements;
  set successors; // set of blocks that depend from this one. Precedence constraint
  set rr_successors; // set of blocks that reuse data used in this one, false dep
  set predecessors; // set of blocks this one depends from. Precedence constraint
  set rr_predecessors; // set of blocks that use data reused in this one, false dep
  bool is_a_loop;
}*fusion_block;

/* Newgen list foreach compatibility */
#define fusion_block_TYPE fusion_block
#define fusion_block_CAST(x) ((fusion_block)((x).p))

/**
 * Get node in the DG corresponding to given statement ordering
 */
static hash_table ordering_to_dg_mapping;
static vertex ordering_to_vertex(int ordering) {
  long int lordering = ordering;
  return (vertex)hash_get(ordering_to_dg_mapping, (void *)lordering);
}

/**
 * Just a debug function, might not be here...
 */
static void print_graph(graph dependence_graph) {

  MAP(VERTEX,
      a_vertex,
      {
        statement s1 = vertex_to_statement(a_vertex);

        fprintf( stderr, "Statement %d\n", (int)statement_ordering( s1 ) );
        MAP(SUCCESSOR, a_successor,
            {
              vertex v2 = successor_vertex(a_successor);
              statement s2 = vertex_to_statement(v2);
              dg_arc_label an_arc_label = successor_arc_label(a_successor);
              fprintf(stderr, "\t%d --> %d with conflicts\n", (int)statement_ordering( s1 ), (int)statement_ordering( s2 ) );

              MAP(CONFLICT, a_conflict,
                  {

                    fprintf(stderr, "\t\tfrom ");
                    print_words(stderr, words_effect(conflict_source( a_conflict ) ) );
                    fprintf(stderr, " to ");
                    print_words(stderr, words_effect(conflict_sink( a_conflict ) ) );
                    if( cone_undefined != conflict_cone( a_conflict ) ) {
                      MAP(INT,
                          level,
                          {
                            fprintf(stderr, " cone level %d", level);
                          },
                          cone_levels(conflict_cone(a_conflict)));
                    }
                    fprintf(stderr, "\n");

                  },
                  dg_arc_label_conflicts(an_arc_label));
            },
            vertex_successors(a_vertex));
      },
      graph_vertices(dependence_graph) );

}

/**
 * Debug function that print block informations
 */
static void print_block(fusion_block block) {
  fprintf(stderr, "Block %d , predecessors : ", block->num);
  SET_FOREACH(fusion_block,pred,block->predecessors) {
    fprintf(stderr, "%d, ", pred->num);
  }
  fprintf(stderr, " | successors : ");
  SET_FOREACH(fusion_block,succ,block->successors) {
    fprintf(stderr, "%d, ", succ->num);
  }
  fprintf(stderr, " | rr_predecessors : ");
  SET_FOREACH(fusion_block,rr_pred,block->rr_predecessors) {
    fprintf(stderr, "%d, ", rr_pred->num);
  }
  fprintf(stderr, " | rr_successors : ");
  SET_FOREACH(fusion_block,rr_succ,block->rr_successors) {
    fprintf(stderr, "%d, ", rr_succ->num);
  }
  fprintf(stderr, "\n");
}

/**
 * Debug function that print a list of blocks
 */
static void print_blocks(list /* of fusion_block */ blocks) {
  FOREACH(fusion_block, block, blocks) {
    print_block(block);
  }
}

/**
 * @brief Check that two loop statements have the same bounds
 */
static bool loops_have_same_bounds_p(loop loop1, loop loop2) {
  bool same_p = FALSE;

  range r1 = loop_range(loop1);
  range r2 = loop_range(loop2);

  same_p = range_equal_p(r1, r2);

  return same_p;
}


#if 0
/**
 * @brief Check that two loop have the same header (same index variable and
 * same bounds)
 */
static bool loop_has_same_header_p(loop loop1, loop loop2) {

  entity index1 = loop_index(loop1);
  entity index2 = loop_index(loop2);

  // This assumes no side effects of loop iterations on the bound expressions
  if(loops_have_same_bounds_p(loop1,loop2) && index1 == index2) {
    return true;
  }
  return false;
}
#endif


/**
 * DIRTY HACK
 * Replace entity in effects associated to a statement
 */
struct entity_pair
{
    entity old;
    entity new;
};


void replace_entity_effects_walker(statement s, void *_thecouple ) {
  struct entity_pair *thecouple = _thecouple;
  list effs = load_proper_rw_effects_list( s );
  ifdebug(7) {
    pips_debug(0,"Handling statement :");
    print_statement(s);
    pips_debug(0,"Effects :");
    print_effects(effs);
    fprintf(stderr,"\n");
  }

  FOREACH(effect, eff, effs) {
    replace_entity(eff, thecouple->old, thecouple->new);
  }
  ifdebug(7) {
    pips_debug(0,"Effects after :");
    print_effects(effs);
    fprintf(stderr,"\n");
  }

}



/**
 * @brief Try to fuse the two loop. Dependences are check against the new body
 * but other constraints such as some statement between the two loops are not
 * handled and must be enforced outside.
 *
 * FIXME High leakage
 */
static bool fusion_loops(statement sloop1,
                         set contained_stmts_loop1,
                         statement sloop2,
                         set contained_stmts_loop2,
                         bool maximize_parallelism) {
  pips_assert("Previous is a loop", statement_loop_p( sloop1 ) );
  pips_assert("Current is a loop", statement_loop_p( sloop2 ) );
  bool success = false;



  loop loop1 = statement_loop(sloop1);
  loop loop2 = statement_loop(sloop2);
  statement body_loop1 = loop_body(loop1);
  statement body_loop2 = loop_body(loop2);

  // Check if loops have fusion compatible headers, else abort
  if(!loops_have_same_bounds_p(loop1, loop2)) {
    pips_debug(4,"Fusion aborted because of incompatible loop headers\n");
    return false;
  }

  // If requested, fuse only look of the same kind (parallel/sequential).
  if( maximize_parallelism && ((loop_parallel_p(loop1)
      && !loop_parallel_p(loop2)) || (!loop_parallel_p(loop1)
      && loop_parallel_p(loop2)))) {
    pips_debug(4,"Fusion aborted because of fuse_maximize_parallelism property"
        ", loop_parallel_p(loop1)=>%d | loop_parallel_p(loop2)=>%d\n"
        ,loop_parallel_p(loop1),loop_parallel_p(loop2));

    // Abort to preserve parallelism
    return false;
  }


  entity index1 = loop_index(loop1);
  entity index2 = loop_index(loop2);
  if(index1!=index2) {
    pips_debug(4,"Replace second loop index (%s) with first one (%s)\n",
               entity_name(index2), entity_name(index1));
    // Get all variable referenced in loop2 body
    set ref_entities = get_referenced_entities(loop2);

    // Assert that index1 is not referenced in loop2
    if(set_belong_p(ref_entities,index1)) {
      pips_debug(3,"First loop index (%s) is used in the second loop, we don't"
                 " know how to handle this case !\n",
                 entity_name(index1));
      return false;
    } else {
      // FI: FIXME we should check that index2 is dead on exit of loop2
      replace_entity((void *)body_loop2, index2, index1);

      struct entity_pair thecouple = { index2, index1 };
      gen_context_recurse(body_loop2, &thecouple,
                          statement_domain, gen_true,
                          replace_entity_effects_walker);
    }
    set_free(ref_entities);
  }
  //statement new_body = make_block_with_stmt_if_not_already(body_loop1);
  list seq1;
  list fused;

  if(statement_sequence_p(body_loop1)) {
    seq1 = sequence_statements(statement_sequence(body_loop1));
  } else {
    seq1 = CONS(statement, body_loop1, NIL );
  }

  if(statement_sequence_p(body_loop2)) {
    list seq2 = sequence_statements(statement_sequence(body_loop2));
    fused = gen_concatenate(seq1, seq2);
  } else {
    list seq2 = CONS(statement, body_loop2, NIL );
    fused = gen_concatenate(seq1, seq2);
  }

  // Let's check if the fusion is valid

  // Construct the fused sequence
  statement fused_statement = make_block_statement(fused);
  loop_body( loop1 ) = fused_statement;
  statement_ordering( fused_statement) = 999999999; // FIXME : dirty
  add_ordering_of_the_statement_to_current_mapping(fused_statement);

  // Fix a little bit proper effects so that chains will be happy with it
  store_proper_rw_effects_list(fused_statement, NIL);
  // Stuff for DG
  set_enclosing_loops_map(loops_mapping_of_statement(sloop1));

  // Build chains
  debug_on("CHAINS_DEBUG_LEVEL");
  graph chains = statement_dependence_graph(sloop1);
  debug_off();


  // Build DG
  debug_on("RICEDG_DEBUG_LEVEL");
  graph candidate_dg = compute_dg_on_statement_from_chains(sloop1, chains);
  debug_off();

  ifdebug(5) {
    pips_debug(0, "Candidate CHAINS :\n");
    print_graph(chains);
    pips_debug(0, "Candidate DG :\n");
    print_graph(candidate_dg);
    pips_debug(0, "Candidate fused loop :\n");
    print_statement(sloop1);
  }

  // Cleaning
  reset_enclosing_loops_map();

  // Let's validate the fusion now
  // No write dep between a statement from loop2 to statement from loop1
  success = true;
  FOREACH( vertex, v, graph_vertices(candidate_dg) ) {
    dg_vertex_label dvl = (dg_vertex_label)vertex_vertex_label(v);
    int statement_ordering = dg_vertex_label_statement(dvl);
    statement stmt1 = ordering_to_statement(statement_ordering);
    // Check that the source of the conflict is in the "second" loop body
    if(set_belong_p(contained_stmts_loop2,stmt1)) {
      FOREACH( successor, a_successor, vertex_successors(v) )
      {
        vertex v2 = successor_vertex(a_successor);
        dg_vertex_label dvl2 = (dg_vertex_label)vertex_vertex_label(v2);
        arc_label an_arc_label = successor_arc_label(a_successor);
        int statement_ordering2 = dg_vertex_label_statement(dvl2);
        statement stmt2 = ordering_to_statement(statement_ordering2);

        // Check that the sink of the conflict is in the "first" loop body
        if(set_belong_p(contained_stmts_loop1,stmt2)) {
          FOREACH( conflict, c, dg_arc_label_conflicts(an_arc_label) )
          {
            effect e_sink = conflict_sink(c);
            effect e_source = conflict_source(c);
            ifdebug(6) {
              pips_debug(0,"Considering arc : from statement %d :",statement_ordering);
              print_effect(conflict_source(c));
              pips_debug(0," to statement %d :",statement_ordering2);
              print_effect(conflict_sink(c));
            }
            if(( effect_write_p(e_source) && store_effect_p(e_source))
                || (effect_write_p(e_sink) && store_effect_p(e_sink))) {

              // Inner loop indices conflicts aren't preventing fusion

              if(statement_loop_p(stmt1) && effect_variable(e_source)
                  == loop_index(statement_loop(stmt1))) {
                continue;
              }

              if(statement_loop_p(stmt2) && effect_variable(e_sink)
                  == loop_index(statement_loop(stmt2))) {
                continue;
              }


              ifdebug(6) {
                pips_debug(0,"Arc preventing fusion : from statement %d :",statement_ordering);
                print_effect(conflict_source(c));
                pips_debug(0," to statement %d :",statement_ordering2);
                print_effect(conflict_sink(c));
              }
              success = false;
            }
          }
        } else {
          ifdebug(6) {
            pips_debug(0,"Arc ignored (%d,%d) : from statement %d :",
                       (int)statement_ordering(sloop2),
                       (int)statement_ordering(sloop1),
                       statement_ordering);
            print_statement(ordering_to_statement(statement_ordering));
            pips_debug(0," to statement %d :",statement_ordering2);
            print_statement(ordering_to_statement(statement_ordering2));
          }
        }
      }
    }
  }

  if(success) {
    // Cleaning FIXME
    // Fix real DG
    // Fix statement ordering
    // Fix loop_execution (still parallel ?)
    // If index2 is different from index 1 and if index2 is live on
    // exit, its exit value should be restored by an extra
    // assignment here
     // ...
  } else {
    // FI: this also should be controlled by information about the
    // liveness of both indices; also index1 must not be used in
    // loop2 as a temporary; so the memory effects of loops 2 should
    // be checked before attempting the first subtitution
    if(index1!=index2) {
      replace_entity((void *)body_loop2, index1, index2);
      struct entity_pair thecouple = { index1, index2 };
      gen_context_recurse(body_loop2, &thecouple,
              statement_domain, gen_true, replace_entity_effects_walker);
    }
    loop_body(loop1) = body_loop1;
    // Cleaning FIXME
  }

  ifdebug(3) {
    pips_debug(3, "End of fusion_loops\n\n");
    print_statement(sloop1);
    pips_debug(3, "\n********************\n");
  }
  return success;
}

/**
 * Create an empty block
 */
static fusion_block make_empty_block(int num) {
  fusion_block block = (fusion_block)malloc(sizeof(struct fusion_block));
  block->num = num;
  block->s = NULL;
  block->statements = set_make(set_pointer);
  block->successors = set_make(set_pointer);
  block->rr_successors = set_make(set_pointer);
  block->predecessors = set_make(set_pointer);
  block->rr_predecessors = set_make(set_pointer);
  block->is_a_loop = false;

  return block;
}

/**
 * Add statement 's' to the set 'stmts'. To be called with gen_context_recurse
 * to record all statement in a branch of the IR tree.
 */
static bool record_statements(statement s, set stmts) {
  set_add_element(stmts, stmts, s);
  return true;
}

/**
 * Create a block with statement 's' as a root and given the number 'num'.
 */
static fusion_block make_block_from_statement(statement s, int num) {
  // Create the new block
  fusion_block b = make_empty_block(num);

  // Record the original statement
  b->s = s;

  // Populate the block statements
  gen_context_recurse(s,b->statements,statement_domain,record_statements,gen_true);

  // Mark the block a loop if applicable
  if(statement_loop_p(s)) {
    b->is_a_loop = true;
    // Remove loop header from the list of statements
    set_del_element(b->statements, b->statements, b->s);
  }

  pips_debug(3,"Block created : num %d ; is_a_loop : %d\n",
             b->num,b->is_a_loop);
  return b;
}

/**
 * Find the block owning the statement corresponding to the given ordering
 */
static fusion_block get_block_from_ordering(int ordering, list block_list) {
  statement s = ordering_to_statement(ordering);
  FOREACH(fusion_block, b, block_list) {
    if(set_belong_p(b->statements, s)) {
      return b;
    }
  }
  return NULL;
}

/**
 * Update b by computing the set of successors using the dependence graph and
 * the set of statements inside b
 */
static void compute_successors(fusion_block b, list block_list) {
  pips_assert("Expect successors list to be initially empty",
      set_empty_p(b->successors) && set_empty_p(b->rr_successors));
  // Loop over statements that belong to this block
  SET_FOREACH(statement,s,b->statements)
  {
    int ordering = statement_ordering(s);
    vertex v = ordering_to_vertex(ordering); // Using DG
    pips_debug(5, "  > Statement %d ", ordering);

    if(v != HASH_UNDEFINED_VALUE) {
      // Statement has a node in the graph
      pips_debug(5, " has a vertex in DG !\n");
      // Loop over successors in DG
      FOREACH( SUCCESSOR, a_successor, vertex_successors( v ) )
      {
        // Loop over conflicts between current statement and the successor
        dg_arc_label an_arc_label = successor_arc_label(a_successor);
        FOREACH( CONFLICT, a_conflict,dg_arc_label_conflicts(an_arc_label))
        {
          // We have a dependence
          // ... or not, read after read are not real one when
          // dealing with precedence !
          int sink_ordering = vertex_ordering(successor_vertex(a_successor));
          pips_debug(5, "Considering dependence to statement %d\n",
              sink_ordering);

          // Try to recover the sink block for this dependence.
          // We might not find any, because dependence can be related to
          // a statement outside from current sequence scope or can be related
          // to a loop header, which we ignore.
          fusion_block sink_block = get_block_from_ordering(sink_ordering,
                                                            block_list);
          if(sink_block == NULL) {
            pips_debug(2,"No block found for ordering %d, dependence ignored\n",
                sink_ordering);
          } else {
            // It's a forward pass, we only add precedence on blocks
            // with ordering higher to current one
            if(sink_block->num > b->num) {
              // We have a successor !
              if(action_write_p(effect_action(conflict_sink(a_conflict)))
                  || action_write_p(effect_action(conflict_source(a_conflict)))) {
                // There's a real dependence here
                set_add_element(b->successors,
                                b->successors,
                                (void *)sink_block);
                // Mark current block as a predecessor ;-)
                set_add_element(sink_block->predecessors,
                                sink_block->predecessors,
                                (void *)b);
              } else {
                // Read-read dependence is interesting to fuse, but is not a
                // precedence constraint
                set_add_element(b->rr_successors,
                                b->rr_successors,
                                (void *)sink_block);
                // Mark current block as a rr_predecessor ;-)
                set_add_element(sink_block->rr_predecessors,
                                sink_block->rr_predecessors,
                                (void *)b);
              }

              break; // One conflict with each successor is enough
            }
          }
        }
      }
    }
  }
  // Optimization, do not try two time the same fusion !
  set_difference(b->rr_successors, b->rr_successors, b->successors);
  set_difference(b->rr_predecessors, b->rr_predecessors, b->predecessors);
}

/**
 * Prune the graph so that we have a DAG. There won't be anymore more than
 * one path between two block in the predecessors/successors tree. We keep only
 * longest path, no shortcut :-)
 */
static set prune_successors_tree(fusion_block b) {
  pips_debug(8,"visiting %d\n",b->num);
  set full_succ = set_make(set_pointer);
  SET_FOREACH(fusion_block, succ, b->successors) {
    set full_succ_of_succ = prune_successors_tree(succ);
    full_succ = set_union(full_succ, full_succ, full_succ_of_succ);
    set_free(full_succ_of_succ);
  }
  SET_FOREACH(fusion_block, succ_of_succ, full_succ ) {
    set_del_element(b->successors, b->successors, succ_of_succ);
    set_del_element(b->rr_successors, b->rr_successors, succ_of_succ);
    set_del_element(succ_of_succ->predecessors, succ_of_succ->predecessors, b);
    set_del_element(succ_of_succ->rr_predecessors, succ_of_succ->rr_predecessors, b);
  }

  full_succ = set_union(full_succ, full_succ, b->successors);
  return full_succ;
}

/**
 * Merge two blocks (successors, predecessors, statements).
 */
static void merge_blocks(fusion_block block1, fusion_block block2) {
//FIXME not always the case
//pips_assert("block1->num < block2->num expected",block1->num < block2->num);

  ifdebug(3) {
    pips_debug(3,"Merging blocks :\n");
    print_block(block1);
    print_block(block2);
  }

  // merge predecessors
  set_union(block1->predecessors, block1->predecessors, block2->predecessors);

  // merge rr_predecessors
  set_union(block1->rr_predecessors, block1->rr_predecessors, block2->rr_predecessors);

  // merge successors
  set_union(block1->successors, block1->successors, block2->successors);

  // merge rr_successors
  set_union(block1->rr_successors, block1->rr_successors, block2->rr_successors);

  // merge statement
  set_union(block1->statements, block1->statements, block2->statements);

  // Replace block2 with block1 as a predecessor of his successors
  SET_FOREACH(fusion_block,succ,block2->successors) {
    set_add_element(succ->predecessors, succ->predecessors, block1);
    set_del_element(succ->predecessors, succ->predecessors, block2);
  }
  // Replace block2 with block1 as a predecessor of his rr_successors
  SET_FOREACH(fusion_block,rr_succ,block2->rr_successors) {
    set_add_element(rr_succ->rr_predecessors, rr_succ->rr_predecessors, block1);
    set_del_element(rr_succ->rr_predecessors, rr_succ->rr_predecessors, block2);
  }
  // Replace block2 with block1 as a successor of his predecessors
  SET_FOREACH(fusion_block,pred,block2->predecessors) {
    if(pred != block1) {
      set_add_element(pred->successors, pred->successors, block1);
    }
    set_del_element(pred->successors, pred->successors, block2);
  }
  // Replace block2 with block1 as a successor of his rr_predecessors
  SET_FOREACH(fusion_block,rr_pred,block2->rr_predecessors) {
    if(pred != block1) {
      set_add_element(rr_pred->rr_successors, rr_pred->rr_successors, block1);
    }
    set_del_element(rr_pred->rr_successors, rr_pred->rr_successors, block2);
  }

  // Remove block1 from predecessors and successors of ... block1
  set_del_element(block1->predecessors, block1->predecessors, block1);
  set_del_element(block1->successors, block1->successors, block1);
  // Remove block1 from rr_predecessors and rr_successors of ... block1
  set_del_element(block1->rr_predecessors, block1->rr_predecessors, block1);
  set_del_element(block1->rr_successors, block1->rr_successors, block1);

  block2->num = -1; // Disable block, will be garbage collected

  ifdebug(4) {
    pips_debug(4,"After merge :\n");
    print_block(block1);
    print_block(block2);
  }
}


/**
 * @brief Checks if precedence constraints allow fusing two blocks
 */
static bool fusable_blocks_p( fusion_block b1, fusion_block b2) {
  bool fusable_p = false;
  if(b1!=b2 && b1->num>=0 && b2->num>=0 && b1->is_a_loop && b2->is_a_loop) {
    // Blocks are active and are loops

    if(set_belong_p(b2->successors,b1)) {
      ifdebug(6) {
        pips_debug(6,"b1 is a successor of b2, fusion prevented !\n");
        print_block(b1);
        print_block(b2);
      }
      fusable_p = false;
    } else if(set_belong_p(b1->successors,b2)) {
      // Adjacent blocks are fusable
      ifdebug(6) {
        pips_debug(6,"blocks are fusable because directly connected\n");
        print_block(b1);
        print_block(b2);
      }
      fusable_p = true;
    } else {
      // If there's some constraint, we won't be able to fuse them
      // here is a heavy way to check that, better not to think about
      // algorithm complexity :-(
      pips_debug(6,"Getting full successors for b1 (%d)\n",b1->num);
      set full_succ_b1 = prune_successors_tree(b1);
      if(!set_belong_p(full_succ_b1,b2)) {
        // b2 is not a successors of a successor of a .... of b1
        // look at the opposite !
        pips_debug(6,"Getting full successors for b2 (%d)\n",b2->num);
        set full_succ_b2 = prune_successors_tree(b2);
        if(!set_belong_p(full_succ_b2,b1)) {
          fusable_p = true;
        }
        set_free(full_succ_b2);
      }
      set_free(full_succ_b1);
    }
  }
  return fusable_p;
}


/**
 * Try to fuse two blocks (if they are loop...)
 * @return true if a fusion occured !
 */
static bool fuse_block( fusion_block b1,
                        fusion_block b2,
                        bool maximize_parallelism) {
  bool return_val = FALSE; // Tell is a fusion has occured
  if(!b1->is_a_loop) {
    pips_debug(5,"B1 (%d) is a not a loop, skip !\n",b1->num);
  } else if(!b2->is_a_loop) {
    pips_debug(5,"B2 (%d) is a not a loop, skip !\n",b2->num);
  } else if(b1->num==-1) {
    pips_debug(5,"B2 (%d) is disabled, skip !\n",b1->num);
  } else if(b2->num==-1) {
    pips_debug(5,"B2 (%d) is disabled, skip !\n",b2->num);
  } else {
    // Try to fuse
    pips_debug(4,"Try to fuse %d with %d\n",b1->num, b2->num);
    if(fusion_loops(b1->s,b1->statements, b2->s, b2->statements, maximize_parallelism)) {
      pips_debug(2, "Loop have been fused\n");
      // Now fuse the corresponding blocks
      merge_blocks(b1, b2);
      return_val=TRUE;
    }
  }
  return return_val;
}


/**
 * This function first try to fuse b with its successors (if b is a loop and if
 * there's any loop in the successor list) ; then it recurse on each successor.
 *
 * @param b is the current block
 * @param fuse_count is the number of successful fusion done
 */
static bool try_to_fuse_with_successors(fusion_block b,
                                        int *fuse_count,
                                        bool maximize_parallelism) {
  // First step is to try to fuse with each successor
  if(b->is_a_loop && b->num>=0) {
    pips_debug(5,"Block %d is a loop, try to fuse with successors !\n",b->num);
    SET_FOREACH(fusion_block, succ, b->successors)
    {
      if(fuse_block(b, succ, maximize_parallelism)) {
        /* predecessors and successors set have been modified for the current
         * block... we can no longer continue in this loop, so we stop and let
         * the caller restart the computation
         */
        (*fuse_count)++;
        return true;
      }
    }
  }
  // Second step is recursion on successors (if any)
  SET_FOREACH(fusion_block, succ, b->successors) {
    return try_to_fuse_with_successors(succ, fuse_count, maximize_parallelism);
  }

  return false;
}


/**
 * This function first try to fuse b with its rr_successors (if b is a loop and
 * if there's any loop in the rr_successor list)
 *
 * @param b is the current block
 * @param fuse_count is the number of successful fusion done
 */
static void try_to_fuse_with_rr_successors(fusion_block b,
                                        int *fuse_count,
                                        bool maximize_parallelism) {
  if(b->is_a_loop) {
    pips_debug(5,"Block %d is a loop, try to fuse with rr_successors !\n",b->num);
    SET_FOREACH(fusion_block, succ, b->rr_successors)
    {
      if(fusable_blocks_p(b,succ) &&
          fuse_block(b, succ, maximize_parallelism)) {
        /* predecessors and successors set have been modified for the current
         * block... we can no longer continue in this loop, let's restart
         * current function at the beginning and end this one.
         *
         * FIXME : performance impact may be high ! Should not try to fuse
         * again with the same block
         *
         */
        (*fuse_count)++;
        try_to_fuse_with_rr_successors(b, fuse_count, maximize_parallelism);
        return;
      }
    }
  }

  return;
}


/**
 * This function loop over all possible couple of blocks and then try to fuse
 * all of them that validated precedence constaints
 *
 * @param blocks is the list of available blocks
 * @param fuse_count is the number of successful fusion done
 */
static void fuse_all_possible_blocks(list blocks,
                                     int *fusion_count,
                                     bool maximize_parallelism) {
  FOREACH(fusion_block, b1, blocks) {
    if(b1->is_a_loop) {
      FOREACH(fusion_block, b2, blocks) {
        if(fusable_blocks_p(b1,b2)) {
          if(fuse_block(b1, b2, maximize_parallelism)) {
            (*fusion_count)++;
          }
        }
      }
    }
  }
}


/**
 * Try to fuse every loops in the given sequence
 */
static bool fusion_in_sequence(sequence s, fusion_params params) {

  /*
   * Construct "block" decomposition
   */

  /* Keep track of all blocks created */
  list block_list = NIL;

  // We have to give a number to each block. It'll be used for regenerating
  // the list of statement in the sequence wrt initial order.
  int current_block_number = 1;

  // Keep track of the number of loop founded, to enable or disable next stage
  int number_of_loop = 0;

  // Loop over the list of statements in the sequence and compute blocks
  list stmts = sequence_statements(s);
  FOREACH(statement, st, stmts) {
    fusion_block b = make_block_from_statement(st, current_block_number);
    block_list = gen_cons(b, block_list);
    current_block_number++;
    if(statement_loop_p(st)) {
      number_of_loop++;
    }
  }

  block_list = gen_nreverse(block_list);

  // We only continue now if we have at least 2 loops. What can we fuse else ?
  if(number_of_loop > 1) {
    // Construct now predecessors/successors relationships for all blocks
    FOREACH(fusion_block, block, block_list) {
      compute_successors(block, block_list);
    }
    /*
     *  Prune the graph so that we have a DAG
     */
    FOREACH(fusion_block, block, block_list) {
      if(set_empty_p(block->predecessors)) { // Block has no predecessors
        set full_succ = prune_successors_tree(block);
        set_free(full_succ);
      }
    }

    ifdebug(3) {
      print_blocks(block_list);
    }
    // Loop over blocks and find fusion candidate (loop with compatible header)
    /*
     * Now we call a recursive method on blocks that don't have any
     * predecessor. The others will be visited recursively.
     */
    int fuse_count = 0;
restart_loop: ;
    FOREACH(fusion_block, block, block_list) {
      if(set_empty_p(block->predecessors)) { // Block has no predecessors
        pips_debug(2,
            "Operate on block %d (is_a_loop %d)\n",
            block->num,
            block->is_a_loop);

        if(try_to_fuse_with_successors(block, &fuse_count,params->maximize_parallelism)) {
          // We fused some blocks, let's restart the process !
          // FIXME : we shouldn't have to restart the process, but it'll require
          // hard work in try_to_fuse_with_successors, so we'll do that later...
          goto restart_loop;
        }

      }
    }

    /* Here we try to fuse each block with its rr_successors, this is to benefit
     * from reuse (read-read) !
     */
    FOREACH(fusion_block, block, block_list) {
      if(block->num>=0) { // Block is active
        try_to_fuse_with_rr_successors(block, &fuse_count,params->maximize_parallelism);
      }
    }

    if(params->greedy) {
      /*
       * We allow the user to request a greedy fuse, which mean that we fuse as
       * much as we can, even if there's no reuse !
       */
      ifdebug(3) {
        print_blocks(block_list);
      }
      fuse_all_possible_blocks(block_list, &fuse_count,params->maximize_parallelism);
    }



    if(fuse_count > 0) {
      /*
       * Cleaning : removing old blocks that have been fused
       */
      list block_iter = block_list;
      list prev_block_iter = NIL;
      while(block_iter != NULL) {
        fusion_block block = (fusion_block)REFCAR(block_iter);

        if(block->num < 0) { // to be removed
          if(prev_block_iter==NIL) {
            block_list = CDR(block_iter);
          } else {
            CDR(prev_block_iter) = CDR(block_iter);
          }
          list garbage = block_iter;
          block_iter = CDR(block_iter);
          free(garbage);
        } else {
          prev_block_iter = block_iter;
          block_iter = CDR(block_iter);
        }
      }

      /* Regenerate sequence now
       *  We will process as follow :
       *  - find every eligible block WRT to precedence constraint
       *  - schedule eligible blocks according to their original position
       *  - loop until there's no longer block to handle
       */
      list new_stmts = NIL;

      ifdebug(4) {
        pips_debug(0,"Before regeneration\n");
        print_blocks(block_list);
      }

      // Loop until every block have been regenerated
      int block_count = gen_length(block_list);
      while(block_count > 0) {
restart_generation:
        block_count = 0;
        int active_blocks = 0;
        bool at_least_one_block_scheduled = false;
        // First loop, construct eligible blocks
        FOREACH(fusion_block, block, block_list)
        {
          if(block->num < 0) {
            continue; // block is disabled
          }
          active_blocks++;

          if(set_empty_p(block->predecessors)) { // Block has no predecessors
            // Block is eligible
            ifdebug(3) {
              pips_debug(3,"Eligible : ");
              print_block(block);
            }
            // Schedule block
            new_stmts = CONS(statement,block->s,new_stmts);
            block->num = -1; // Disable block
            at_least_one_block_scheduled = true;

            // Release precedence constraint on successors
            SET_FOREACH(fusion_block,succ,block->successors)
            {
              set_del_element(succ->predecessors, succ->predecessors, block);
            }
            // We have free some constraints, and thus we restart the process
            // to ensure that we generate in an order as close as possible to
            // the original code
            goto restart_generation;
          } else {
            ifdebug(3) {
              pips_debug(3,"Not eligible : ");
              print_block(block);
            }

            // Number of block alive
            block_count++;
          }
        }
        if(!at_least_one_block_scheduled && active_blocks>0) {
          pips_internal_error("No block scheduled, we have interdependence "
              "in the block tree, which means it's not a tree ! Abort...\n");
        }
      }

      // Replace original list with the new one
      sequence_statements( s) = gen_nreverse(new_stmts);
    }
  }
  return true;
}

/**
 * Will try to fuse as many loops as possible in the IR subtree rooted by 's'
 */
static void compute_fusion_on_statement(statement s) {
  // Get user preferences with some properties
  struct fusion_params params;
  params.maximize_parallelism = get_bool_property("LOOP_FUSION_MAXIMIZE_PARALLELISM");
  params.greedy = get_bool_property("LOOP_FUSION_GREEDY");

  // Go on fusion on every sequence of statement founded
  gen_context_recurse( s, &params, sequence_domain, fusion_in_sequence, gen_true);
}

/**
 * PIPSMake entry point for loop fusion
 */
bool loop_fusion(char * module_name) {
  statement module_statement;

  /* Get the true ressource, not a copy. */
  module_statement = (statement)db_get_memory_resource(DBR_CODE,
                                                       module_name,
                                                       TRUE);

  /* Get the data dependence graph (chains) : */
  graph dependence_graph = (graph)db_get_memory_resource(DBR_DG,
                                                         module_name,
                                                         TRUE);

  /* The proper effect to detect the I/O operations: */
  set_proper_rw_effects((statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS,
                                                                  module_name,
                                                                  TRUE));
  set_precondition_map((statement_mapping)db_get_memory_resource(DBR_PRECONDITIONS,
                                                                 module_name,
                                                                 TRUE));
  /* Mandatory for DG construction */
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,
                                                                     module_name,
                                                                     TRUE));

  set_current_module_statement(module_statement);
  set_current_module_entity(module_name_to_entity(module_name));

  set_ordering_to_statement(module_statement);
  ordering_to_dg_mapping = compute_ordering_to_dg_mapping(dependence_graph);

  debug_on("LOOP_FUSION_DEBUG_LEVEL");

  // Here we go ! Let's fuse :-)
  compute_fusion_on_statement(module_statement);

  /* Reorder the module, because some statements have been deleted, and others
   * have been reordered
   */
  module_reorder(module_statement);

  /* Store the new code */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_statement);

  // Free resources
  hash_table_free(ordering_to_dg_mapping);
  ordering_to_dg_mapping = NULL;
  reset_proper_rw_effects();
  reset_cumulated_rw_effects();
  reset_precondition_map();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();

  pips_debug(2, "done for %s\n", module_name);
  debug_off();

  /* Should have worked:
   *
   * Do we want to provide statistics about the number of fused loops?
   *
   * How do we let PyPS know the number of loops fused?
   */
  return TRUE;
}
