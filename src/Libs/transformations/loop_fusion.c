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

// Table that map paths to the list of vertex in the paths
//static hash_table paths_block;
//static int max_path = 1;

typedef struct fusion_block {
  int num;
  statement s; // The main statement (header for loops)
  // statements inside the block (in case of loop, header won't belong to)
  set statements;
  set successors; // set of blocks that depend from this one. Precedence constraint
  set predecessors; // set of blocks this one depends from. Precedence constraint
  bool is_a_loop;
}*fusion_block;

// Newgen foreach on list compatibility
#define fusion_block_TYPE fusion_block
#define fusion_block_CAST(x) ((fusion_block)((x).p))

//static fusion_block current_block = HASH_UNDEFINED_VALUE;
//static bool have_found_a_loop = false;

static list block_list = NULL;

static graph dependence_graph;
static hash_table ordering_to_dg_mapping;

// Forward declaration
static void compute_fusion_on_statement(statement s);

static vertex ordering_to_vertex(int ordering) {
  long int lordering = ordering;
  return (vertex)hash_get(ordering_to_dg_mapping, (void *)lordering);
}

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
 *
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
  fprintf(stderr, "\n");
}

/**
 * @brief Check that two loop have the same header (same index variable and
 * same bounds)
 */
static bool loop_has_same_header_p(statement loop1, statement loop2) {
  pips_assert("Previous is a loop!", statement_loop_p( loop1 ) );
  pips_assert("Current is a loop", statement_loop_p( loop2 ) );

  range r1 = loop_range( statement_loop( loop1 ) );
  range r2 = loop_range( statement_loop( loop2 ) );
  entity index1 = loop_index(statement_loop( loop1 ));
  entity index2 = loop_index(statement_loop( loop2 ));

  // This assumes no side effects of loop iterations on the bound expressions
  if(same_expression_p(range_lower( r1 ), range_lower( r2 ))
      && same_expression_p(range_upper( r1 ), range_upper( r2 ))
      && same_expression_p(range_increment( r1 ), range_increment( r2 ))
     /*&& index1 == index2*/) {
    // Of course, PIPS generates different indices when unrolling
    // loops containing loops...
    if(index1!=index2) {
      // Remap the second loop onto the index of the first loop
      // This is not safe unless index1 does not appear at all in the
      // second loop
      replace_entity((void *)loop2, index2, index1);
    }
    return true;
  }
  return false;
}

// Forward declaration
extern graph statement_dependence_graph(statement s);

/**
 * @brief Try to fuse the two loop. Dependences are check against the new body
 * but other constraints such as some statement between the two loops are not
 * handled and must be enforced outside.
 *
 * FIXME High leakage
 */
static bool fusion_loops(statement loop1, statement loop2) {
  bool success = false;
  if(loop_has_same_header_p(loop1, loop2)) {
    instruction instr_loop1 = statement_instruction( loop1 );
    instruction instr_loop2 = statement_instruction( loop2 );
    statement body_loop1 = loop_body( instruction_loop( instr_loop1 ) );
    statement body_loop2 = loop_body( instruction_loop( instr_loop2 ) );
    statement new_body = make_block_with_stmt_if_not_already(body_loop1);
    instruction instr_body_loop1 = statement_instruction( new_body );
    instruction instr_body_loop2 = statement_instruction( body_loop2 );
    list seq1 = sequence_statements( instruction_sequence ( instr_body_loop1 ) );
    //    list seq2 = sequence_statements( instruction_sequence ( instr_body_loop2 ) );
    list fused;

    //entity index1 = loop_index( instruction_loop( instr_loop1 ) );
    //entity index2 = loop_index( instruction_loop( instr_loop2 ) );

    if(instruction_sequence_p( instr_body_loop2 )) {
      /*
       list
       seq2 =
       sequence_statements( copy_sequence( instruction_sequence ( instr_body_loop2 ) ) );
       replace_entity( seq2, index2, index1 );
       */
      list seq2 =
          sequence_statements( instruction_sequence ( instr_body_loop2 ) );
      fused = gen_concatenate(seq1, seq2);
    } else {
      //      statement body_loop2_with_loop1_index = copy_statement(body_loop2);
      //      replace_entity( body_loop2_with_loop1_index, index2, index1 );
      //      list seq2 = CONS(statement, body_loop2_with_loop1_index, NIL );
      list seq2 = CONS(statement, body_loop2, NIL );
      fused = gen_concatenate(seq1, seq2);
      //      free( seq2 );
    }

    // Let's check if the fusion is valid
    statement fused_statement = make_block_statement(fused);
    loop_body( instruction_loop( instr_loop1 ) ) = fused_statement;
    statement_ordering(fused_statement) = 999999999; // FIXME : dirty
    add_ordering_of_the_statement_to_current_mapping(fused_statement);

    // Fix a little bit proper effects so that chains will be happy with it
    store_proper_rw_effects_list(fused_statement, NIL);
    // Stuff for DG
    set_enclosing_loops_map(loops_mapping_of_statement(loop1));

    // Build chains
    debug_on("CHAINS_DEBUG_LEVEL");
    graph chains = statement_dependence_graph(loop1);
    debug_off();

    // Build DG
    debug_on("RICEDG_DEBUG_LEVEL");
    graph candidate_dg = compute_dg_on_statement_from_chains(loop1, chains);
    debug_off();

    ifdebug(6) {
      pips_debug(6, "DG :\n");
      print_graph(dependence_graph);
      pips_debug(6, "Candidate DG :\n");
      print_graph(candidate_dg);
      pips_debug(6, "Candidate :\n");
      print_statement(loop1);
    }

    // Cleaning
    reset_enclosing_loops_map();

    // Let's validate the fusion now
    // No write dep between a statement from loop2 to statement from loop1
    success = true;
    FOREACH( vertex, v, graph_vertices(candidate_dg) ) {
      dg_vertex_label dvl = (dg_vertex_label)vertex_vertex_label( v );
      int statement_ordering = dg_vertex_label_statement(dvl);
      if(statement_ordering > statement_ordering( loop2 ) && statement_ordering
          != statement_ordering(fused_statement)) {
        FOREACH( successor, a_successor, vertex_successors(v) )
        {
          vertex v2 = successor_vertex( a_successor );
          dg_vertex_label dvl2 = (dg_vertex_label)vertex_vertex_label( v2 );
          arc_label an_arc_label = successor_arc_label(a_successor);
          int statement_ordering2 = dg_vertex_label_statement(dvl2);

          if(statement_ordering2 < statement_ordering( loop2 )
              && statement_ordering2 != statement_ordering( loop1 )) {
            FOREACH( conflict, c, dg_arc_label_conflicts(an_arc_label) )
            {
              if(action_write_p( effect_action( conflict_sink( c ) ) )
                  || action_write_p( effect_action( conflict_source( c ) ) )) {
                success = false;
              }
            }
          }
        }
      }
    }

    if(success) {
      // Cleaning FIXME
      // Fix real DG
      // Fix statement ordering
      // ...
    } else {
      loop_body( instruction_loop( instr_loop1 ) ) = body_loop1;
      // Cleaning FIXME
    }

    ifdebug(3) {
      pips_debug(3, "End of fusion_loops\n\n");
      print_statement(loop1);
      pips_debug(3, "\n********************\n");
    }
  }
  return success;
}

static fusion_block make_empty_block(int num) {
  fusion_block block = (fusion_block)malloc(sizeof(struct fusion_block));
  block->num = num;
  block->s = NULL;
  block->statements = set_make(set_pointer);
  block->successors = set_make(set_pointer);
  block->predecessors = set_make(set_pointer);
  block->is_a_loop = false;

  block_list = gen_cons(block, block_list);

  return block;
}

/*
static void free_block(fusion_block block) {
  set_free(block->statements);
  block->statements = NULL;
  block->successors = NULL;
  block->s = NULL;
  free(block);
}
*/
/*
 static void free_blocks() {
 HASH_MAP(key, b, {
 free_block((fusion_block)b);
 }, block_list);
 set_clear(block_list);
 }*/

/*
 // FIXME Leakage
 static bool clean_statement_to_delete(statement s) {
 SET_FOREACH( statement, to_delete, to_be_deleted ) {
 // Find and delete it
 if(statement_sequence_p(s)) {
 list
 seq =
 sequence_statements( instruction_sequence ( statement_instruction( s ) ) );
 if(gen_in_list_p(to_delete, seq)) {
 pips_debug(5,
 "Has removed statement %d\n",
 (int)statement_ordering( to_delete ));
 gen_remove(&seq, to_delete);
 set_del_element(to_be_deleted, to_be_deleted, to_delete);
 }
 }
 }
 return true;
 }
 */

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

  return b;
}

/**
 * Try to recover the block corresponding to the given statement ordering
 */

/*
 static fusion_block get_block_from_ordering(int ordering) {
 fusion_block block = NULL;

 statement s = ordering_to_statement(ordering);

 // Add order dependences constraints
 HASH_MAP ( block_number, hblock,
 {
 fusion_block candidateBlock = (fusion_block) hblock;
 if(set_belong_p(candidateBlock,s)) {
 block=candidateBlock;
 break;
 }
 }
 , blocks );
 pips_assert("Block not found from ordering", block!=NULL );
 return block;
 }*/

/*
static int min(int a, int b) {
  return (a > b) ? b : a;
}
static int max(int a, int b) {
  return (a < b) ? b : a;
}
*/

/**
 * Find the block owning the statement corresponding to the given ordering
 */
static fusion_block get_block_from_ordering(int ordering) {
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
static void compute_successors(fusion_block b) {
  pips_assert("Expect successors list to be initially empty",
      set_empty_p(b->successors));
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
        dg_arc_label an_arc_label = successor_arc_label( a_successor );
        FOREACH( CONFLICT, a_conflict,dg_arc_label_conflicts(an_arc_label))
        {
          // We have a dependence
          // ... or not, read after read are not real one when
          // dealing with precedence
          if(store_effect_p(conflict_sink( a_conflict ))
              && store_effect_p(conflict_source( a_conflict ))
              && (action_write_p( effect_action( conflict_sink( a_conflict ) ) )
                  || action_write_p( effect_action( conflict_source( a_conflict ) ) ))) {

            int sink_ordering =
                vertex_to_ordering(successor_vertex( a_successor ));
            pips_debug(5, "Considering dependence to statement %d\n",
                sink_ordering);

            // Try to recover the sink block for this dependence.
            // We might not find any, because dependence can be related to
            // a statement outside from current sequence scope or can be related
            // to a loop header, which we ignore.
            fusion_block sink_block = get_block_from_ordering(sink_ordering);
            if(sink_block == NULL) {
              pips_debug(2,"No block found for ordering %d, dependence ignored",
                  sink_ordering);
            } else {
              // It's a forward pass, we only add precedence on blocks
              // with ordering higher to current one
              if(sink_block->num > b->num) {
                // We have a successor !
                set_add_element(b->successors,
                                b->successors,
                                (void *)sink_block);
                // Mark current block as a predecessor ;-)
                set_add_element(sink_block->predecessors,
                                sink_block->predecessors,
                                (void *)b);

                break; // One conflict with each successor is enough
              }
            }
          }
        }
      }
    }
  }
}

/**
 * Prune the graph so that we have a real tree. There won't be anymore more than
 * one path between two block in the predecessors/successors tree. We keep only
 * longest path, no shortcut :-)
 */
static set prune_successors_tree( fusion_block b ) {
  set full_succ = set_make(set_pointer);
  SET_FOREACH(fusion_block, succ, b->successors)
  {
    set full_succ_of_succ = prune_successors_tree(succ);
    full_succ = set_union(full_succ,full_succ,full_succ_of_succ);
  }
  SET_FOREACH(fusion_block, succ_of_succ, full_succ )
  {
    set_del_element(b->successors,b->successors,succ_of_succ);
  }

  full_succ = set_union(full_succ,full_succ,b->successors);
  return full_succ;
}



/**
 * Merge two blocks (successors, predecessors, statements).
 */
static void merge_blocks(fusion_block block1, fusion_block block2) {
  pips_assert("block1->num < block2->num expected",block1->num < block2->num);

  ifdebug(3) {
    pips_debug(3,"Merging blocks :\n");
    print_block(block1);
    print_block(block2);
  }

  // merge predecessors
  set_union(block1->predecessors, block1->predecessors, block2->predecessors);

  // merge successors
  set_union(block1->successors, block1->successors, block2->successors);

  // merge statement
  set_union(block1->statements, block1->statements, block2->statements);
  // Replace block2 with block1 as a predecessor of his successors
  SET_FOREACH(fusion_block,succ,block2->successors) {
    set_add_element(succ->predecessors, succ->predecessors, block1);
    set_del_element(succ->predecessors, succ->predecessors, block2);
  }
  // Replace block2 with block1 as a successor of his predecessors
  SET_FOREACH(fusion_block,pred,block2->predecessors) {
    if(pred != block1) {
      set_add_element(pred->successors, pred->successors, block1);
    }
    set_del_element(pred->successors, pred->successors, block2);
  }

  // Remove block1 from predecessors of ... block1
  set_del_element(block1->predecessors, block1->predecessors, block1);

  block2->num = -1; // Disable block, will be garbage collected

  ifdebug(4) {
    pips_debug(4,"After merge :\n");
    print_block(block1);
    print_block(block2);
  }

}

/**
 * This function first try to fuse b with its successors (if b is a loop and if
 * there's any loop in the successor list) ; then it recurse on each successor.
 *
 * @param b is the current block
 * @param fuse_count is the number of successful fusion done
 */
static void try_to_fuse_with_successors(fusion_block b, int *fuse_count) {
  // First step is to try to fuse with each successor
  if(b->is_a_loop) {
    SET_FOREACH(fusion_block, succ, b->successors)
    {
      if(succ->is_a_loop) {
        // Try to fuse
        if(fusion_loops(b->s, succ->s)) {
          pips_debug(2, "Loop have been fused\n");
          // Now fuse the corresponding blocks
          merge_blocks(b, succ);
          (*fuse_count)++;

          /* predecessors and successors set have been modified for the current
           * block... we can no longer continue in this loop, let's restart
           * current function at the beginning and end this one.
           *
           * FIXME : performance impact may be high ! Should not try to fuse
           * again with the same block
           *
           */
          try_to_fuse_with_successors(b, fuse_count);
          return;
        }
      }
    }
  }
  // Second step is recursion on successors (if any)
  SET_FOREACH(fusion_block, succ, b->successors) {
    try_to_fuse_with_successors(succ, fuse_count);
  }

  return;
}

/*
static bool order_blocks(fusion_block b1, fusion_block b2) {
  if(b1->num < b2->num)
    return -1;
  else
    return 1;
}
*/

/**
 * Try to fuse every loop in the given sequence
 */
static bool fusion_in_sequence(sequence s) {

  /*
   * Construct "block" decomposition
   */

  // List of blocks
  block_list = NIL;

  // We have to give a number to each block. It'll be used for regenerating
  // the list of statement in the sequence wrt initial order.
  int current_block_number = 1;

  // Keep track of the number of loop founded, to enable or disable next stage
  int number_of_loop = 0;

  // Loop over the list of statements in the sequence and compute blocks
  list stmts = sequence_statements(s);
  FOREACH(statement, st, stmts) {
    make_block_from_statement(st, current_block_number);
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
      compute_successors(block);
    }

    /*
     *  Prune the graph so that we have a real tree
     */
    FOREACH(fusion_block, block, block_list) {
      if(set_empty_p(block->predecessors)) { // Block has no predecessors
        prune_successors_tree(block);
      }
    }


    ifdebug(3) {
      FOREACH(fusion_block, block, block_list)
      {
        pips_debug(3,"Block %d, predecessors : ",block->num);
        SET_FOREACH(fusion_block,pred,block->predecessors) {
          fprintf(stderr, "%d, ", pred->num);
        }
        fprintf(stderr, "\n");
      }
    }
    // Loop over blocks and find fusion candidate (loop with compatible header)
    /*
     * Now we call a recursive method on blocks that don't have any
     * predecessor. The others will be visited recursively.
     */
    int fuse_count = 0;
    FOREACH(fusion_block, block, block_list) {
      if(set_empty_p(block->predecessors)) { // Block has no predecessors
        pips_debug(2,
            "Operate on block %d (is_a_loop %d)\n",
            block->num,
            block->is_a_loop);

        try_to_fuse_with_successors(block, &fuse_count);

      }
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
          CDR(prev_block_iter) = CDR(block_iter); // prev_block_iter cannot be null
          free(block_iter);
          block_iter = CDR(prev_block_iter);
          continue;
        }
        prev_block_iter = block_iter;
        block_iter = CDR(block_iter);
      }

      /* Regenerate sequence now
       *  We will process as follow :
       *  - find every eligible block WRT to precedence constraint
       *  - schedule eligible blocks according to their original position
       *  - loop until there's no longer block to handle
       */
      list new_stmts = NIL;

      // Loop until every block have been regenerated
      int block_count = gen_length(block_list);
      while(block_count > 0) {
        block_count = 0;
	//int eligible_block_idx = 0;
        // First loop, construct eligible blocks
        FOREACH(fusion_block, block, block_list)
        {
          if(block->num < 0) {
            continue; // block is disabled
          }

          if(set_empty_p(block->predecessors)) { // Block has no predecessors
            // Block is eligible
            ifdebug(3) {
              pips_debug(3,"Eligible : ");
              print_block(block);
            }
            // Schedule block
            new_stmts = CONS(statement,block->s,new_stmts);
            block->num = -1; // Disable block

            // Release precedence constraint on successors
            SET_FOREACH(fusion_block,succ,block->successors)
            {
              set_del_element(succ->predecessors, succ->predecessors, block);
            }
          } else {
            ifdebug(3) {
              pips_debug(3,"Not eligible : ");
              print_block(block);
            }

            // Number of block alive
            block_count++;
          }
        }
      }

      // Replace original list with the new one
      sequence_statements(s) = gen_nreverse(new_stmts);
    }
  }
  return true;
}

/**
 * Will try to fuse as many loops as possible in the IR subtree rooted by 's'
 */
static void compute_fusion_on_statement(statement s) {
  // Go on fusion on every sequence of statement founded
  gen_recurse( s, sequence_domain, fusion_in_sequence, gen_true);
}

bool loop_fusion(char * module_name) {
  statement module_statement;

  /* Get the true ressource, not a copy. */
  module_statement = (statement)db_get_memory_resource(DBR_CODE,
                                                       module_name,
                                                       TRUE);

  /* Get the data dependence graph (chains) : */
  dependence_graph = (graph)db_get_memory_resource(DBR_DG, module_name, TRUE);

  /* The proper effect to detect the I/O operations: */
  set_proper_rw_effects((statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS,
                                                                  module_name,
                                                                  TRUE));
  set_precondition_map((statement_mapping)db_get_memory_resource(DBR_PRECONDITIONS,
                                                                 module_name,
                                                                 TRUE));
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,
                                                                     module_name,
                                                                     TRUE));

  set_current_module_statement(module_statement);
  set_current_module_entity(module_name_to_entity(module_name));

  set_ordering_to_statement(module_statement);
  ordering_to_dg_mapping = compute_ordering_to_dg_mapping(dependence_graph);

  debug_on("LOOP_FUSION_DEBUG_LEVEL");

  compute_fusion_on_statement(module_statement);

  hash_table_free(ordering_to_dg_mapping);
  ordering_to_dg_mapping = NULL;

  //  print_graph( dependence_graph );

  /* Reorder the module, because some statements have been deleted.
   Well, the order on the remaining statements should be the same,
   but by reordering the statements, the number are consecutive. Just
   for pretty print... :-) */
  module_reorder(module_statement);

  debug(2, "loop_fusion", "done for %s\n", module_name);

  debug_off();

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_statement);

  reset_proper_rw_effects();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();

  /* Should have worked: */
  return TRUE;
}
