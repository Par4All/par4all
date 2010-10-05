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
static hash_table paths_block;
static int max_path = 1;

typedef struct fusion_block {
  int ordering;
  int end; // last statement ordering
  set paths; // last statement ordering
  bool is_a_loop;
}*fusion_block;

static int current_block_number = 1;
static fusion_block current_block = HASH_UNDEFINED_VALUE;
static hash_table blocks;
static bool have_found_a_loop = false;

static graph dependence_graph;
static hash_table ordering_to_dg_mapping;

static set to_be_deleted;

// Forward declaration
static void compute_fusion_on_statement(statement s);

static vertex ordering_to_vertex(int ordering) {
  return (vertex)hash_get(ordering_to_dg_mapping, (void *)ordering);
}

void prettyprint_paths(FILE * fd) {

  ifdebug(8) {
    /* There is no guarantee that the ordering_to_statement() hash table is the proper one */
    print_ordering_to_statement();
  }

  fprintf(fd, "digraph {\n");

  HASH_MAP ( block_number1, hblock1,
      {
        fusion_block block1 = (fusion_block)hblock1;
        fprintf( fd, "(%d) %d -> %d : ", block_number1, block1->ordering, block1->end );
        SET_FOREACH( int, path, block1->paths ) {
          fprintf( fd, " %d ", path );
        }
        fprintf( fd, "\n" );
      }, blocks );
  fprintf(fd, "\n}\n");
}

void prettyprint_dot_paths_graph(FILE * fd) {

  ifdebug(8) {
    /* There is no guarantee that the ordering_to_statement() hash table is the proper one */
    print_ordering_to_statement();
  }

  fprintf(fd, "digraph {\n");

  for (int block_number1 = 1; block_number1 <= hash_table_entry_count(blocks); block_number1++) {
    fusion_block block1 = hash_get(blocks, (void *)block_number1);
    if(block1 != HASH_UNDEFINED_VALUE) {
      set seen_paths = set_make(set_pointer);
      if(set_empty_p(block1->paths)) {
        fprintf(fd,
                " %d -> %d [label=\"empty\"]",
                block1->ordering,
                block1->ordering);
      } else {
        for (int block_number2 = 1; block_number2 <= hash_table_entry_count(blocks); block_number2++) {
          fusion_block block2 = hash_get(blocks, (void *)block_number2);
          if(block2 != HASH_UNDEFINED_VALUE) {
            if(block1->ordering < block2->ordering) {
              if(!set_empty_p(block2->paths)) {
                set intersect_path = set_make(set_pointer);
                set_intersection(intersect_path, block1->paths, block2->paths);

                set difference_path = set_make(set_pointer);

                set_difference(difference_path, intersect_path, seen_paths);
                if(!set_empty_p(difference_path)) {
                  fprintf(fd,
                          "%d->%d [label=\"",
                          block1->ordering,
                          block2->ordering);
                  SET_FOREACH( int, path, difference_path ) {
                    fprintf(fd, "%d,", path);
                  }
                  fprintf(fd, "\"]");
                }
                set_union(seen_paths, difference_path, seen_paths);
                set_free(intersect_path);
                intersect_path = NULL;
                set_free(difference_path);
                difference_path = NULL;
              }
              fprintf(fd, ";\n");
            }
          }
        }
      }
      set_free(seen_paths);
      seen_paths = NULL;
    }
  }
  fprintf(fd, "\n}\n");
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

static bool loop_has_same_header_p(statement loop1, statement loop2) {
  pips_assert("Previous is not a loop !!", statement_loop_p( loop1 ) );
  pips_assert("Current is not a loop !!", statement_loop_p( loop2 ) );

  range r1 = loop_range( statement_loop( loop1 ) );
  range r2 = loop_range( statement_loop( loop2 ) );
  entity index1 = loop_index(statement_loop( loop1 ));
  entity index2 = loop_index(statement_loop( loop2 ));

  if(same_expression_p(range_lower( r1 ), range_lower( r2 ))
      && same_expression_p(range_upper( r1 ), range_upper( r2 ))
      && same_expression_p(range_increment( r1 ), range_increment( r2 ))
      && index1 == index2) {
    return true;
  }
  return false;
}

extern graph statement_dependence_graph(statement s);

// FIXME High leakage
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

    entity index1 = loop_index( instruction_loop( instr_loop1 ) );
    entity index2 = loop_index( instruction_loop( instr_loop2 ) );

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

    // Let's check if fusion is valid
    statement fused_statement = make_block_statement(fused);
    loop_body( instruction_loop( instr_loop1 ) ) = fused_statement;
    statement_ordering(fused_statement) = 999999999; // FIXME : dirty
    add_ordering_of_the_statement_to_current_mapping(fused_statement);

    // Fix a little bit proper effects so that chains will be happy with it
    store_proper_rw_effects_list(fused_statement, NIL);
    // Stuff for DG
    set_enclosing_loops_map(loops_mapping_of_statement(loop1));

    // Build chains
    graph chains = statement_dependence_graph(loop1);
    // Build DG
    graph candidate_dg = compute_dg_on_statement_from_chains(loop1, chains);
    fprintf(stderr, "DG :\n");
    print_graph(dependence_graph);
    fprintf(stderr, "Candidate DG :\n");
    print_graph(candidate_dg);
    fprintf(stderr, "Candidate :\n");
    print_statement(loop1);

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

    fprintf(stderr, "End of fusion_loops\n\n");
    print_statement(loop1);
    fprintf(stderr, "\n********************\n");
  }
  return success;
}

static bool block_empty_p(int block_number) {
  return hash_get(blocks, (char *)block_number) == HASH_UNDEFINED_VALUE;
}

static fusion_block make_empty_block(int ordering) {
  fusion_block block = (fusion_block)malloc(sizeof(struct fusion_block));
  block->ordering = ordering;
  block->end = -1;
  block->paths = set_make(set_int);
  block->is_a_loop = false;
  return block;
}

static void free_block(fusion_block block) {
  set_free(block->paths);
  block->paths = NULL;
  free(block);
}

static void free_blocks() {
  HASH_MAP(key, b, {
        free_block((fusion_block *)b);
      }, blocks);
  hash_table_clear(blocks);
}

static void new_block(int ordering) {
  if(current_block != HASH_UNDEFINED_VALUE) {
    current_block->end = ordering;
  }
  current_block = make_empty_block(ordering);
  ifdebug(2) {
    fprintf(stderr,
            "Statement %d produce block %d\n",
            ordering,
            current_block_number);
  }
  hash_put(blocks, (char *)current_block_number, (char *)current_block);
  current_block_number++;
}

// Leakage
static bool clean_statement_to_delete(statement s) {
  SET_FOREACH( statement, to_delete, to_be_deleted ) {
    // Find and delete it
    if(statement_sequence_p(s)) {
      list
          seq =
              sequence_statements( instruction_sequence ( statement_instruction( s ) ) );
      if(gen_in_list_p(to_delete, seq)) {
        fprintf(stderr,
                "Has removed statement %d\n",
                statement_ordering( to_delete ));
        gen_remove(&seq, to_delete);
        set_del_element(to_be_deleted, to_be_deleted, to_delete);
      }
    }
  }
  return true;
}

static int last_statement = 1;
static bool make_blocks_from_statement(statement s) {
  bool is_not_a_loop = true;

  if(statement_loop_p(s)) {
    is_not_a_loop = false;
    have_found_a_loop = true;
    new_block(statement_ordering( s ));
    current_block->is_a_loop = true;
  } else {
    if(current_block == HASH_UNDEFINED_VALUE || current_block->is_a_loop) {
      new_block(statement_ordering( s ));
    }
  }
  return is_not_a_loop;
}

static bool find_last_statement_ordering(statement s) {
  last_statement = statement_ordering( s );
  return true;
}


// Get the list of block that belong to this path
static set get_blocks_for_path(int path) {
  pips_assert("Path must be positive", path > 0 );
  set blocks_in_this_path = (set)hash_get(paths_block, (void *)path);
  if(blocks_in_this_path == HASH_UNDEFINED_VALUE) {
    blocks_in_this_path = set_make(set_pointer);
    hash_put(paths_block, (void *)path, (void *)blocks_in_this_path);
  }
  return blocks_in_this_path;
}

static void add_a_path_to_predecessors(int ordering, set predecessors, int path) {
  ifdebug(2) {
    fprintf(stderr, "Retro propagate from %d\nPaths : ", ordering);
    SET_FOREACH( fusion_block, predecessor, predecessors ) {
      fprintf(stderr, "%d ", predecessor->ordering);
    }
    fprintf(stderr, "\n");
  }

  set blocks = get_blocks_for_path(path);
  SET_FOREACH( fusion_block, block, predecessors )
  {
    if(block->ordering <= ordering) {

      // Add the paths to the list of paths this block belongs to
      pips_assert("Path must be positive", path > 0 );
      set_add_element(block->paths, block->paths, (void *)path);

      // Add current block to this path
      set_add_element(blocks, blocks, (void *)block);
    }
  }
}

static fusion_block get_block_from_ordering(int ordering) {
  fusion_block block = NULL;

  // Add order dependences constraints
  HASH_MAP ( block_number, hblock,
      {
        block = (fusion_block) hblock;
        if( ordering >= block->ordering && ordering < block->end ) {
          break;
        }
      }
      , blocks );
  pips_assert("Block not found from ordering", block!=NULL );
  return block;
}

static void add_a_block_to_the_path(int path,
                                    fusion_block block,
                                    set predecessors) {
  // Check if this block already belongs to the path
  if(!set_belong_p(block->paths, (void *)path)) {

    // Add the paths to the list of paths this block belongs to
    ifdebug(2) {
      fprintf(stderr, "Adding path %d to block %d\n", path, block->ordering);
    }
    set_add_element(block->paths, block->paths, (void *)path);

    // Get the list of block that belong to this path
    set blocks_in_this_path = get_blocks_for_path(path);

    // Add current block to this path
    ifdebug(2) {
      fprintf(stderr, "Adding block %d to path %d\n", block->ordering, path);
    }
    set_add_element(blocks_in_this_path, blocks_in_this_path, (void *)block);

    // If we have more than one successors, we have to branch current path
    // so we create a new path for each successor and retro-propagate it to
    // predecessors

    // Keep the block already added from here
    set blocks_seen = set_make(set_pointer);

    // Loop over statements in current block
    for (int s = block->ordering; s < block->end; s++) {
      vertex v = ordering_to_vertex(s);
      // Statement has a node in the graph
      ifdebug(9) {
        fprintf(stderr, "  > Statement %d ", block->ordering);
      }
      if(v != HASH_UNDEFINED_VALUE) {
        ifdebug(9) {
          fprintf(stderr, " has a vertex in DG !\n");
        }
        // Loop over successors
        FOREACH( SUCCESSOR, a_successor, vertex_successors( v ) )
        {
          // Loop over conflicts between current statement and the successor
          dg_arc_label an_arc_label = successor_arc_label( a_successor );
          FOREACH( CONFLICT, a_conflict,dg_arc_label_conflicts(an_arc_label))
          {
            // We have a dependence
            // ... or not, write after write are not real one when
            // dealing with precedence
            if(action_write_p( effect_action( conflict_sink( a_conflict ) ) )) {
              fusion_block
                  sink_block =
                      get_block_from_ordering(vertex_to_ordering(successor_vertex( a_successor )));

              // It's a forward pass
              if(sink_block->ordering > block->ordering
                  && !set_belong_p(blocks_seen, sink_block)) {

                ifdebug(2) {
                  fprintf(stderr,
                          "From %d, Recurse on %d with path %d\n",
                          block->ordering,
                          sink_block->ordering,
                          path);
                }
                // Add to blocks seen
                set_add_element(blocks_seen, blocks_seen, (void *)sink_block);

                // Recurse on this successor
                set_add_element(predecessors, predecessors, (void *)block);
                add_a_block_to_the_path(path, sink_block, predecessors);

                // Retro-propagate to predecessors
                add_a_path_to_predecessors(block->ordering, predecessors, path);

                // New path for next successor
                max_path++;
                path = max_path;

                // Get the list of block that belong to this path
                blocks_in_this_path = get_blocks_for_path(path);

                break; // One conflict with each successor is enough
              }
            }
          }
        }
      }
    }
    set_free(blocks_seen);
    blocks_seen = NULL;
  }
}

static int min(int a, int b) {
  return (a > b) ? b : a;
}
static int max(int a, int b) {
  return (a < b) ? b : a;
}

static void merge_blocks(fusion_block block1, fusion_block block2) {
  block1->ordering = min(block1->ordering, block2->ordering);
  block1->end = max(block1->end, block2->end);

  // Add path
  set diff = set_make(set_pointer);
  diff = set_difference(diff, block2->paths, block1->paths);
  SET_FOREACH( int, new_path, diff ) {
    //Add the block to the path
    set blocks_in_new_path = get_blocks_for_path(new_path);
    set_add_element(blocks_in_new_path, blocks_in_new_path, (void *)block1);

    // Add new path to the block
    set_add_element(block1->paths, block1->paths, (void *)new_path);
  }
  SET_FOREACH( int, old_path, block2->paths )
  {
    set blocks_in_path = get_blocks_for_path(old_path);
    set_del_element(blocks_in_path, blocks_in_path, block2);
  }
}

static bool fusion_in_sequence(statement s) {
  if(statement_sequence_p(s)) {


    // Construct block decomposition for current statement
    have_found_a_loop = false;
    current_block_number = 1;
    current_block = HASH_UNDEFINED_VALUE;
    blocks = hash_table_make(hash_int, 0);
//    gen_recurse( s , statement_domain, make_blocks_from_statement, gen_true);
    list stmts = sequence_statements(statement_sequence(s));
    FOREACH(statement, st, stmts) {
      make_blocks_from_statement(st);
    }
    // Find last statement to close block
    gen_recurse( s , statement_domain, find_last_statement_ordering, gen_true);
    current_block->end = last_statement;


    if(have_found_a_loop) {
      // Record junk loops header that has been fused
      to_be_deleted = set_make(set_pointer);

      // Graph path
      paths_block = hash_table_make(hash_int, 0);

      // Construct the graph

      // Add order dependences constraints
      for (int block_number = 1; block_number <= hash_table_entry_count(blocks); block_number++) {
        fusion_block block = hash_get(blocks, (void *)block_number);
        if(!(block == HASH_UNDEFINED_VALUE)) {
          if(set_empty_p(block->paths)) {
            ifdebug(2) {
              fprintf(stderr,
                      "Block %d is now head of path %d\n",
                      block_number,
                      max_path);
            }

            set predecessors = set_make(set_pointer);
            add_a_block_to_the_path(max_path, block, predecessors);
            set_free(predecessors);
            predecessors = NULL;

            max_path++;
          }
        } else {
          pips_debug(1, "No block for number %d\n", block_number);
        }
      }

      prettyprint_dot_paths_graph(stderr);
      prettyprint_paths(stderr);

      // Loop over blocks and find fusion candidate (loop with compatible header)

      // Loop over precedence paths to fuse adjacent loops first

      fusion_block previous_block = HASH_UNDEFINED_VALUE;
      for (int block_number = 1; block_number <= hash_table_entry_count(blocks); block_number++) {
        fusion_block block = hash_get(blocks, (void *)block_number);
        if(block != HASH_UNDEFINED_VALUE) {
          fprintf(stderr,
                  "We have a block, ordering %d, is_a_loop %d\n",
                  block->ordering,
                  block->is_a_loop);
          if(block->is_a_loop) {

            if(previous_block != HASH_UNDEFINED_VALUE) {
              statement loop1 = ordering_to_statement(previous_block->ordering);
              statement loop2 = ordering_to_statement(block->ordering);
              if(fusion_loops(loop1, loop2)) {
                fprintf(stderr, "Loop have been fused\n");
                // loops have been fused, now fuse the corresponding blocks
                merge_blocks(previous_block, block);

                // Add second old loop to be deleted
                set_add_element(to_be_deleted, to_be_deleted, loop2);

              } else {
                previous_block = block;
              }
            } else {
              previous_block = block;
            }
          }
        }
      }
      // Fusion if possible

      // Keep track of including statement numbering

      // Free blocks
      free_blocks();

      // Free graph
      HASH_MAP ( __unused, blocks,
          {
            set_free((set)blocks);
          }
          , paths_block);
      hash_table_free(paths_block);
      paths_block = NULL;


      // Do some cleaning :)
      gen_recurse( s, statement_domain, gen_true, clean_statement_to_delete );
      pips_assert( "Cleanings list must be empty !", set_empty_p( to_be_deleted ) );
      set_free(to_be_deleted);
      to_be_deleted = NULL;


      // Recursion stage
      // if has_fused_at_least_once, self gen_recurse
      // Return false



    }
  }
  return true;
}

static void compute_fusion_on_statement(statement s) {
  //    current_loop_level = 1;
  current_block_number = 1;
  //    enclosing_loop_blocking = hash_table_make(hash_pointer, 0);
  //    set_current_block();


  // Go on fusion for current block decomposition
  gen_recurse( s, statement_domain, fusion_in_sequence, gen_true);
  ;

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
