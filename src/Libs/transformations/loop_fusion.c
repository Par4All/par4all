/*

 $Id$

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
#include "effects-convex.h"
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
#include "transformer.h"
#include "chains.h"
extern bool get_bool_property(string);
extern int get_int_property(string);

/**
 * Fusion configuration
 */
typedef struct fusion_params {
  bool maximize_parallelism; // Prevent sequentializing loop that were parallel
  bool greedy; // Fuse as much a we can, and not only loops that have reuse
  bool coarse; // Use a coarse grain algorithm
  unsigned int max_fused_per_loop; // Threshold to limit the number of fusion per loop
} *fusion_params;
struct fusion_block;
typedef struct fusion_block **fbset;


// Forward declaration
static bool fusion_loops(statement sloop1,
                         set contained_stmts_loop1,
                         statement sloop2,
                         set contained_stmts_loop2,
                         bool maximize_parallelism,
                         bool coarse_grain);





/**
 * Structure to hold block used for the fusion selection algorithm.
 * It's used in a sequence of statements to keep track of the precedence
 * constraints between statements while fusing some of them
 */
typedef struct fusion_block {
  int num;
  int id;    // real original num
  statement s; // The main statement (header for loops)
  // statements inside the block (in case of loop, header won't belong to)
  set statements;
  fbset successors; // set of blocks that depend from this one. Precedence constraint
  fbset rr_successors; // set of blocks that reuse data used in this one, false dep
  fbset predecessors; // set of blocks this one depends from. Precedence constraint
  fbset rr_predecessors; // set of blocks that use data reused in this one, false dep
  bool is_a_loop;
  int count_fusion; // Count the number of fusion that have occur
  bool processed; // Flag that indicates if the block have already been processed
  // (multiple paths can lead to multiple costly tried of the same fusion)
}*fusion_block;

/* Newgen list foreach compatibility */
#define fusion_block_TYPE fusion_block
#define fusion_block_CAST(x) ((fusion_block)((x).p))

static int max_num;
#ifdef __SSE2__
#include <xmmintrin.h>
static inline void fbset_clear(fbset self) {
    __m128i z = _mm_setzero_si128();
    for(fbset iter = self, end=self+max_num;iter!=end;iter+=sizeof(__m128i)/sizeof(fusion_block))
        _mm_store_si128((__m128i*)iter,z);
}
static inline fbset fbset_make() {
    fbset self =_mm_malloc(max_num*sizeof(fusion_block),32);
    fbset_clear(self);
    return self;
}
static inline void fbset_free(fbset fb) {
    _mm_free(fb);
}
static void fbset_union(fbset self, fbset other) {
    for(size_t i=0;i<max_num;i+=sizeof(__m128i)/sizeof(fusion_block)){
        __m128i s = _mm_load_si128((__m128i*)&self[i]),
                o = _mm_load_si128((__m128i*)&other[i]);
        s=_mm_or_si128(s,o);
        _mm_store_si128((__m128i*)&self[i],s);
    }
}
#else
static inline void fbset_clear(fbset self) {
    memset(self,0,sizeof(*self)*max_num);
}
static inline fbset fbset_make() {
    return calloc(max_num,sizeof(fusion_block));
}
static inline void fbset_free(fbset fb) {
    free(fb);
}
static void fbset_union(fbset self, fbset other) {
    for(size_t i=0;i<max_num;i++)
        if(other[i])
            self[i]=other[i];
}
#endif

static void fbset_difference(fbset self, fbset other) {
    for(size_t i=0;i<max_num;i++)
        if(other[i])
            self[i]=NULL;
}
static inline void fbset_del_element(fbset self, fusion_block e) {
    assert(e->id>=0);
    self[e->id]=NULL;
}
static inline void fbset_add_element(fbset self, fusion_block e) {
    assert(e->id>=0);
    self[e->id]=e;
}
static bool fbset_belong_p(fbset self, fusion_block e) {
    assert(e->id>=0);
    return self[e->id]!=NULL;
}
static bool fbset_empty_p(fbset self) {
    for(size_t i=0;i<max_num;i++)
        if(self[i])
            return false;
    return true;
}

#define FBSET_FOREACH(e,s) \
    fusion_block e;\
    for(size_t __i=0;__i<max_num;__i++)\
        if((e=s[__i]))

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
  fprintf(stderr, "Block %d (fused %d times), predecessors : ",
          block->num, block->count_fusion);
  FBSET_FOREACH(pred,block->predecessors) {
    fprintf(stderr, "%d, ", pred->num);
  }
  fprintf(stderr, " | successors : ");
  FBSET_FOREACH(succ,block->successors) {
    fprintf(stderr, "%d, ", succ->num);
  }
  fprintf(stderr, " | rr_predecessors : ");
  FBSET_FOREACH(rr_pred,block->rr_predecessors) {
    fprintf(stderr, "%d, ", rr_pred->num);
  }
  fprintf(stderr, " | rr_successors : ");
  FBSET_FOREACH(rr_succ,block->rr_successors) {
    fprintf(stderr, "%d, ", rr_succ->num);
  }
  fprintf(stderr, "\n");
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
  bool same_p = false;

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
    pips_debug(7,"Handling statement :");
    print_statement(s);
    pips_debug(7,"Effects :");
    print_effects(effs);
    fprintf(stderr,"\n");
  }

  FOREACH(effect, eff, effs) {
    replace_entity(eff, thecouple->old, thecouple->new);
  }
  ifdebug(7) {
    pips_debug(7,"Effects after :");
    print_effects(effs);
    fprintf(stderr,"\n");
  }

}

/* temporary block statement for candidate fused body */
static statement fused_statement = statement_undefined;

/* current ordering for generated statement */
static int next_ordering = 999999;


/*
 * Allocate a temporary block statement for sequence.
 */
static statement make_temporary_fused_statement(list sequence) {
  if(statement_undefined_p(fused_statement)) {
    // Construct the fused sequence
    fused_statement = make_block_statement(sequence);
    statement_ordering( fused_statement) = next_ordering++; // FIXME : dirty
    pips_assert("ordering defined", ordering_to_statement_initialized_p());
    overwrite_ordering_of_the_statement_to_current_mapping(fused_statement);

    // Fix a little bit proper effects so that chains will be happy with it
    store_proper_rw_effects_list(fused_statement, NIL);
  } else {
    sequence_statements(instruction_sequence(statement_instruction(fused_statement))) = sequence;
  }
  return fused_statement;
}

/*
 *
 */
static void free_temporary_fused_statement() {
  if(!statement_undefined_p(fused_statement)) {

    //sefault ! Don't know why...
    //delete_rw_effects(fused_statement);
    //
    sequence_statements(instruction_sequence(statement_instruction(fused_statement))) = NIL;
    free_statement(fused_statement);
    fused_statement = statement_undefined;
  }

}



static bool coarse_fusable_loops_p(statement sloop1,
                         statement sloop2,
                         bool maximize_parallelism) {
  loop loop1 = statement_loop(sloop1);
  statement inner_stat1 = loop_body(loop1);
  loop loop2 = statement_loop(sloop2);
  statement inner_stat2 = loop_body(loop2);

  if (statement_may_contain_exiting_intrinsic_call_p(inner_stat1)
      || statement_may_contain_exiting_intrinsic_call_p(inner_stat2))
    return false;

  /* get the loop body preconditions */
  transformer body_prec1 = load_statement_precondition(inner_stat1);
  transformer body_prec2 = load_statement_precondition(inner_stat2);

  /* do not declare as parallel a loop which is never executed */
  if (transformer_empty_p(body_prec1) || transformer_empty_p(body_prec2)) {
      pips_debug(1, "non feasible inner statement (empty precondition), abort fusion\n");
      return false;
  }

  /* ...needed by TestCoupleOfReferences(): */
  list l_enclosing_loops1 = CONS(STATEMENT, sloop1, NIL);
  //list l_enclosing_loops2 = CONS(STATEMENT, sloop2, NIL);

  /* Get the loop invariant regions for the loop body: */
  list l_reg1 = load_invariant_rw_effects_list(inner_stat1);
  list l_reg2 = load_invariant_rw_effects_list(inner_stat2);

  /* To store potential conflicts to study: */
  list l_conflicts = NIL;

  /* To keep track of a current conflict disabling the parallelization: */
  bool may_conflicts_p = false;

  pips_debug(1,"begin\n");

  pips_debug(1,"building conflicts\n");
  ifdebug(2) {
    fprintf(stderr, "original invariant regions:\n");
    print_regions(l_reg1);
    print_regions(l_reg2);
  }

  /* First, builds list of conflicts: */
  FOREACH(EFFECT, reg1, l_reg1) {
    entity e1 = effect_entity(reg1);
    if (e1!=loop_index(loop1)
        && gen_chunk_undefined_p(gen_find_eq(e1,loop_locals(loop1))) // Ignore private variable
        && store_effect_p(reg1) // Ignore non memory effect
        ) {
      reference r = effect_any_reference(reg1);
      int d1 = gen_length(reference_indices(r));
      conflict conf = conflict_undefined;

      /* Search for a write-read/read-write conflict */
      FOREACH(EFFECT, reg2, l_reg2) {
        reference r2 = effect_any_reference(reg2);
        int d2 = gen_length(reference_indices(r2));
        entity e1 = region_entity(reg1);
        entity e2 = region_entity(reg2);

        if (store_effect_p(reg2)
            && ( (d1<=d2 && region_write_p(reg1) && region_read_p(reg2))
                || (d1>=d2 && region_read_p(reg1) && region_write_p(reg2))
                || (region_write_p(reg1) && region_write_p(reg2))
              )
            && same_entity_p(e1,e2) // String manipulation at the end
            ) {
          /* Add a write-read conflict */
          conf = make_conflict(reg1, reg2, cone_undefined);
          l_conflicts = gen_nconc(l_conflicts, CONS(CONFLICT, conf, NIL));
        }
      }
    }
  }

  /* THEN, TESTS CONFLICTS */
  pips_debug(1,"testing conflicts\n");
  /* We want to test for write/read and read/write dependences at the same
   * time. */
  Finds2s1 = true;
  FOREACH(CONFLICT, conf, l_conflicts) {
    effect reg1 = conflict_source(conf);
    effect reg2 = conflict_sink(conf);
    list levels = NIL;
    list levelsop = NIL;
    Ptsg gs = SG_UNDEFINED;
    Ptsg gsop = SG_UNDEFINED;

    ifdebug(2) {
      fprintf(stderr, "testing conflict from:\n");
      print_region(reg1);
      fprintf(stderr, "\tto:\n");
      print_region(reg2);
    }

    // Patch the region to mimic the same loop index
    entity i1 = loop_index(loop1);
    entity i2 = loop_index(loop2);
    if(i1!=i2) {
      reg2=copy_effect(reg2);
      replace_entity(reg2, loop_index(loop2), loop_index(loop1));
      list l_reg2 = CONS(REGION, reg2, NIL);

      // First project to eliminate the index that can have been added because
      // of the preconditions, this should be safe because it is not used in the
      // body
      list tmp_i = CONS(entity,i1, NIL );
      project_regions_along_variables(l_reg2, tmp_i);
      free(tmp_i);

      // Substitue i2 with i1 in the regions associated to loop2's body
      all_regions_variable_rename(l_reg2,i2,i1);
    }


    /** CHEAT on the ordering !
     *  We make that in order that the dependence test believe that statements
     *  from the first loop comes before statements from the second loop.
     *  It may not be the case after a loop of reordering due to previous fusion
     */
    intptr_t ordering1 = statement_ordering(inner_stat1);
    statement_ordering(inner_stat1) = 1;
    intptr_t ordering2 = statement_ordering(inner_stat2);
    statement_ordering(inner_stat2) = 2;

    /* Use the function TestCoupleOfReferences from ricedg. */
    /* We only consider one loop at a time, disconnected from
     * the other enclosing and inner loops. Thus l_enclosing_loops
     * only contains the current loop statement.
     * The list of loop variants is empty, because we use loop invariant
     * regions (they have been composed by the loop transformer).
     */
    levels = TestCoupleOfReferences(l_enclosing_loops1, region_system(reg1),
                                    inner_stat1, reg1, effect_any_reference(reg1),
                                    l_enclosing_loops1, region_system(reg2),
                                    inner_stat2, reg2, effect_any_reference(reg2),
                                    NIL, &gs, &levelsop, &gsop);

    // Restore the ordering
    statement_ordering(inner_stat1) = ordering1;
    statement_ordering(inner_stat2) = ordering2;


    ifdebug(2) {
      fprintf(stderr, "result:\n");
      if (ENDP(levels) && ENDP(levelsop))
        fprintf(stderr, "\tno dependence\n");

      if (!ENDP(levels)) {
        fprintf(stderr, "\tdependence at levels: ");
        FOREACH(INT, l, levels) {
          fprintf(stderr, " %d", l);
        }
        fprintf(stderr, "\n");

        if (!SG_UNDEFINED_P(gs)) {
          Psysteme sc = SC_UNDEFINED;
          fprintf(stderr, "\tdependence cone:\n");
          sg_fprint_as_dense(stderr, gs, gs->base);
          sc = sg_to_sc_chernikova(gs);
          fprintf(stderr,"\tcorresponding linear system:\n");
          sc_fprint(stderr,sc,(get_variable_name_t)entity_local_name);
          sc_rm(sc);
        }
      }
      if (!ENDP(levelsop)) {
        fprintf(stderr, "\topposite dependence at levels: ");
        FOREACH(INT, l, levelsop)
        fprintf(stderr, " %d", l);
        fprintf(stderr, "\n");

        if (!SG_UNDEFINED_P(gsop)) {
          Psysteme sc = SC_UNDEFINED;
          fprintf(stderr, "\tdependence cone:\n");
          sg_fprint_as_dense(stderr, gsop, gsop->base);
          sc = sg_to_sc_chernikova(gsop);
          fprintf(stderr,"\tcorresponding linear system:\n");
          sc_fprint(stderr,sc,(get_variable_name_t)entity_local_name);
          sc_rm(sc);
        }
      }
    }

    /* If we have a level we may be into trouble */
    if (!ENDP(levels)) {
      // Here are the forward dependences, carried or not...
      FOREACH(INT, l, levels) {
        if(l==1) {
          // This is a loop carried dependence, break the parallelism but safe !
          pips_debug(1,"Loop carried forward dependence, break parallelism ");
          if(loop_parallel_p(loop1) || loop_parallel_p(loop2)) {
            if(maximize_parallelism) {
              ifdebug(1) {
                fprintf(stderr,"then it is fusion preventing!\n");
              }
              may_conflicts_p = true;
            } else {
              ifdebug(1) {
                fprintf(stderr," but both loops are sequential, then fuse!\n");
              }
            }
          } else ifdebug(1) {
            fprintf(stderr," but fuse anyway!\n");
          }
        } else if(l==2) {
          // This is a  loop independent dependence, seems safe to me !
          pips_debug(1,"Loop independent forward dependence, safe...\n");
          may_conflicts_p = false;
        } else {
          pips_user_error("I don't know what to do with a dependence level of "
              "%d here!\n",l);
        }
      }
    }
    if (!ENDP(levelsop)) {
      // Here are the backward dependences, carried or not...
      FOREACH(INT, l, levelsop) {
        if(l==1) {
          // This is a loop carried dependence, not preventing fusion but
          // breaking the parallelism
          pips_debug(1,"Loop carried backward dependence, fusion preventing !\n");
          may_conflicts_p = true;
        } else if(l==2) {
          // This is a  non-carried dependence backware dependence
          // Hey wait a minute, how is it possible ???
          pips_user_error("Loop independent backward dependence... weird !\n");
        } else {
          pips_user_error("I don't know what to do with a dependence level of "
              "%d here !\n",l);
        }
      }
    }

    gen_free_list(levels);
    gen_free_list(levelsop);
    if (!SG_UNDEFINED_P(gs))
      sg_rm(gs);
    if (!SG_UNDEFINED_P(gsop))
      sg_rm(gsop);


    if(may_conflicts_p)
      break;
  }

  /* Finally, free conflicts */
  pips_debug(1,"freeing conflicts\n");
  FOREACH(CONFLICT, c, l_conflicts) {
    conflict_source(c) = effect_undefined;
    conflict_sink(c) = effect_undefined;
    free_conflict(c);
  }
  gen_free_list(l_conflicts);

  gen_free_list(l_enclosing_loops1);

  pips_debug(1,"end\n");

  return !may_conflicts_p;

}


/**
 *
 */
static bool coarse_fusion_loops(statement sloop1,
                         statement sloop2,
                         bool maximize_parallelism) {
  loop loop1 = statement_loop(sloop1);
  loop loop2 = statement_loop(sloop2);
  statement body_loop1 = loop_body(loop1);
  statement body_loop2 = loop_body(loop2);

  // Check if loops have fusion compatible headers, else abort
  if(!loops_have_same_bounds_p(loop1, loop2)) {
    pips_debug(4,"Fusion aborted because of incompatible loop headers\n");
    return false;
  }

  bool coarse_fusable_p = coarse_fusable_loops_p(sloop1,sloop2,maximize_parallelism);

  if(coarse_fusable_p) {
    pips_debug(2,"Fuse the loops now\n");
    entity index1 = loop_index(loop1);
    entity index2 = loop_index(loop2);
    if(index1!=index2) {
      set ref_entities = get_referenced_entities(loop2);
      // Assert that index1 is not referenced in loop2
      if(set_belong_p(ref_entities,index1)) {
        pips_debug(3,"First loop index (%s) is used in the second loop, we don't"
                   " know how to handle this case !\n",
                   entity_name(index1));
        return false;
      }
    }

    // Merge loop locals
    FOREACH(ENTITY,e,loop_locals(loop2)) {

      if(e != loop_index(loop2) && !gen_in_list_p(e,loop_locals(loop1))) {
        loop_locals(loop1) = CONS(ENTITY,e,loop_locals(loop1));
      }
    }

    ifdebug(3) {
      pips_debug(3,"Before fusion : ");
      print_statement(sloop1);
      print_statement(sloop2);
    }

    /* Here a lot of things are broken :
     *  - The regions associated to both body must be merged
     *  - The ordering is broken if a new sequence is created :-(
     *  - ?
     */
    intptr_t ordering =statement_ordering(body_loop1); // Save ordering
    // Append body 2 to body 1
    insert_statement(body_loop1,body_loop2,false);
    statement_ordering(body_loop1) = ordering;

    // FIXME insert_statement() does a bad job here :-(
    // I should fix it but I'm lazy now so use the bazooka:
    clean_up_sequences(body_loop1);

    // Merge regions list for body
    list l_reg1 = load_invariant_rw_effects_list(body_loop1);
    list l_reg2 = load_invariant_rw_effects_list(body_loop2);
    if(loop_index(loop1)!=loop_index(loop2)) {
      // Patch the regions to be expressed with the correct loop index
      all_regions_variable_rename(l_reg2,index2,index1);
      // FI: FIXME we should check that index2 is dead on exit of loop2
      replace_entity((void *)body_loop1, index2, index1);
    }
    gen_nconc(l_reg1,l_reg2);
    store_invariant_rw_effects_list(body_loop1,l_reg1);
    store_invariant_rw_effects_list(body_loop2,NIL); // No sharing

    // Free the body_loop2
    // Hum, only the sequence, not the inner statements
    // Keep the leak for now... FIXME


    ifdebug(3) {
      pips_debug(3,"After fusion : ");
      print_statement(sloop1);
    }
  }

  return coarse_fusable_p;
}


/**
 * @brief Try to fuse the two liront naoop recomputing a DG !
 * Dependences are check against the new body
 * but other constraints such as some statement between the two loops are not
 * handled and must be enforced outside.
 *
 * FIXME High leakage
 */
static bool fine_fusion_loops(statement sloop1,
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



  entity index1 = loop_index(loop1);
  entity index2 = loop_index(loop2);
  if(index1!=index2) {
    pips_debug(4,"Replace second loop index (%s) with first one (%s)\n",
               entity_name(index2), entity_name(index1));
    // Get all variable referenced in loop2 body to find if index1 is referenced
    // This could be optimized by a search for index1 and abort if found.
    // this would avoid building a set
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

      // Replace entities in effects
      struct entity_pair thecouple = { index2, index1 };
      gen_context_recurse(body_loop2, &thecouple,
                          statement_domain, gen_true,
                          replace_entity_effects_walker);
    }
    set_free(ref_entities);
  }


  // Be sure that loop bodies are encapsulated in sequence
  if(!statement_sequence_p(body_loop1)) {
    loop_body(loop1) = make_block_statement(CONS(statement, body_loop1, NIL ));
    body_loop1 = loop_body(loop1);
    statement_ordering( body_loop1 ) = next_ordering++; // FIXME : dirty
    overwrite_ordering_of_the_statement_to_current_mapping( body_loop1 );
    // Fix a little bit proper effects so that chains will be happy with it
    store_proper_rw_effects_list(body_loop1, NIL); // FIXME should lead to a call to delete_rw_effects();

  }

  if(!statement_sequence_p(body_loop2)) {
    loop_body(loop2) = make_block_statement(CONS(statement, body_loop2, NIL ));
    body_loop2 = loop_body(loop2);
    statement_ordering( body_loop2 ) = next_ordering++; // FIXME : dirty
    overwrite_ordering_of_the_statement_to_current_mapping( body_loop2 );
    // Fix a little bit proper effects so that chains will be happy with it
    store_proper_rw_effects_list(body_loop2, NIL); // FIXME should lead to a call to delete_rw_effects();
  }

  // Build a list with the statements from loop 1 followed by stmts for loop 2
  list seq1 = gen_copy_seq(sequence_statements(statement_sequence(body_loop1)));
  list seq2 = gen_copy_seq(sequence_statements(statement_sequence(body_loop2)));
  list fused = gen_nconc(seq1, seq2);


  // Let's check if the fusion is valid

  // Construct the fused sequence
  loop_body( loop1 ) = make_temporary_fused_statement(fused);

  // Stuff for Chains and DG
  set_enclosing_loops_map(loops_mapping_of_statement(sloop1));


  // Build chains
  // do not debug on/off all the time : costly (read environment + atoi )?
  //debug_on("CHAINS_DEBUG_LEVEL");
  graph candidate_dg = statement_dependence_graph(sloop1);
  //debug_off();

  ifdebug(5) {
    pips_debug(5, "Candidate CHAINS :\n");
    print_graph(candidate_dg);
  }

  // Build DG
  // do not debug on/off all the time : costly (read environment + atoi )?
  //debug_on("RICEDG_DEBUG_LEVEL");
  candidate_dg = compute_dg_on_statement_from_chains_in_place(sloop1, candidate_dg);
  //debug_off();

  // Cleaning
  clean_enclosing_loops();
  reset_enclosing_loops_map();

  ifdebug(5) {
    pips_debug(5, "Candidate DG :\n");
    print_graph(candidate_dg);
    pips_debug(5, "Candidate fused loop :\n");
    print_statement(sloop1);
  }


  // Let's validate the fusion now
  // No write dep between a statement from loop2 to statement from loop1
  success = true;
  FOREACH( vertex, v, graph_vertices(candidate_dg) ) {
    dg_vertex_label dvl = (dg_vertex_label)vertex_vertex_label(v);
    int statement_ordering = dg_vertex_label_statement(dvl);
    statement stmt1 = ordering_to_statement(statement_ordering);

    // Look if there's a loop carried dependence, that would be bad, but only
    // for parallel loop nest !
    if(maximize_parallelism && loop_parallel_p(loop1)) {
      FOREACH( successor, a_successor, vertex_successors(v))
      {
        vertex v2 = successor_vertex(a_successor);
        dg_vertex_label dvl2 = (dg_vertex_label) vertex_vertex_label(v2);
        arc_label dal = successor_arc_label(a_successor);
        int statement_ordering2 = dg_vertex_label_statement(dvl2);
        statement stmt2 = ordering_to_statement(statement_ordering2);
        FOREACH( conflict, c, dg_arc_label_conflicts(dal)) {
          effect e_sink = conflict_sink(c);
          effect e_source = conflict_source(c);
          if((effect_write_p(e_source) && store_effect_p(e_source))
              || (effect_write_p(e_sink) && store_effect_p(e_sink))) {

            // Inner loop indices conflicts aren't preventing fusion
            if(statement_loop_p(stmt1) &&
                effect_variable(e_source) == loop_index(statement_loop(stmt1))) {
              continue;
            }
            if(statement_loop_p(stmt2) &&
                effect_variable(e_sink) == loop_index(statement_loop(stmt2))) {
              continue;
            }

            // Scalar can make any conflict ! The loop are parallel thus it has
            // to be private :-)
            if(loop_parallel_p(loop2) &&
                (effect_scalar_p(e_sink) || effect_scalar_p(e_source))) {
              continue;
            }

            // Get the levels and try to find out if the fused loop carries the
            // conflict
            if(cone_undefined != conflict_cone(c)) {
              list levels = cone_levels(conflict_cone(c));
              FOREACH(INT, l, levels) {
                if(l == 1) {
                  // Hum seems bad... This a loop carried dependence !
                  success = false;
                  ifdebug(2) {
                    pips_debug(2,"This loop carried dependence is breaking parallism !\n");
                    fprintf(stderr, "From : ");
                    print_effect(e_source);
                    fprintf(stderr, "to : ");
                    print_effect(e_sink);
                  }
                  break;
                }
              }
            }
          }
          if(!success) break;
        }
      }
    }
    if(!success) {
      break;
    }
    // Check that the source of the conflict is in the "second" loop body
    if(set_belong_p(contained_stmts_loop2,stmt1)) {
      FOREACH( successor, a_successor, vertex_successors(v))
      {
        vertex v2 = successor_vertex(a_successor);
        dg_vertex_label dvl2 = (dg_vertex_label) vertex_vertex_label(v2);
        arc_label an_arc_label = successor_arc_label(a_successor);
        int statement_ordering2 = dg_vertex_label_statement(dvl2);
        statement stmt2 = ordering_to_statement(statement_ordering2);

        // Check that the sink of the conflict is in the "first" loop body
        if(set_belong_p(contained_stmts_loop1, stmt2)) {
          FOREACH( conflict, c, dg_arc_label_conflicts(an_arc_label))
          {
            effect e_sink = conflict_sink(c);
            effect e_source = conflict_source(c);
            ifdebug(6) {
              pips_debug(6,
                         "Considering arc : from statement %d :",
                         statement_ordering);
              print_effect(conflict_source(c));
              pips_debug(6, " to statement %d :", statement_ordering2);
              print_effect(conflict_sink(c));
            }
            if((effect_write_p(e_source) && store_effect_p(e_source))
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
                pips_debug(6,
                           "Arc preventing fusion : from statement %d :",
                           statement_ordering);
                print_effect(conflict_source(c));
                pips_debug(6, " to statement %d :", statement_ordering2);
                print_effect(conflict_sink(c));
              }
              success = false;
            }
          }
        } else {
          ifdebug(6) {
            pips_debug(6,
                       "Arc ignored (%d,%d) : from statement %d :",
                       (int)statement_ordering(sloop2), (int)statement_ordering(sloop1), statement_ordering);
            print_statement(ordering_to_statement(statement_ordering));
            pips_debug(6, " to statement %d :", statement_ordering2);
            print_statement(ordering_to_statement(statement_ordering2));
          }
        }
      }
    }
  }

      // No longer need the DG
  free_graph(candidate_dg);


  bool inner_success = false;
  if(success && get_bool_property("LOOP_FUSION_KEEP_PERFECT_PARALLEL_LOOP_NESTS")
      && loop_parallel_p(loop1) && loop_parallel_p(loop2)
      ) {
    // Check if we have perfect loop nests, and thus prevents losing parallelism
    statement inner_loop1 = get_first_inner_perfectly_nested_loop(body_loop1);
    statement inner_loop2 = get_first_inner_perfectly_nested_loop(body_loop2);
    if(!statement_undefined_p(inner_loop1) && !statement_undefined_p(inner_loop2)) {
      pips_debug(4,"Ensure that we don't break parallel loop nests !\n");
      if(loop_parallel_p(statement_loop(inner_loop1)) &&
          loop_parallel_p(statement_loop(inner_loop2))) {
        // Record statements
        set stmts1 = set_make(set_pointer);
        gen_context_recurse(inner_loop1,stmts1,statement_domain,record_statements,gen_true);

        set stmts2 = set_make(set_pointer);
        gen_context_recurse(inner_loop2,stmts2,statement_domain,record_statements,gen_true);

        pips_debug(4,"Try to fuse inner loops !\n");
        success = fusion_loops(inner_loop1,stmts1,inner_loop2,stmts2,maximize_parallelism,false);
        if(success) {
          inner_success = true;
          pips_debug(4,"Inner loops fused :-)\n");
        } else {
          pips_debug(4,"Inner loops not fusable :-(\n");
        }

        set_free(stmts1);
        set_free(stmts2);
      } else if(loop_parallel_p(statement_loop(inner_loop1)) ||
                loop_parallel_p(statement_loop(inner_loop2))) {
        success = false;
      }

    } else if((!statement_undefined_p(inner_loop1) && loop_parallel_p(statement_loop(inner_loop1))) ||
        (!statement_undefined_p(inner_loop2) && loop_parallel_p(statement_loop(inner_loop2)))) {
      // We have one perfect loop nest deeper than the other, prevent fusion !
      success = false;
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
    loop_body(loop1) = body_loop1;

    // Merge loop locals
    FOREACH(ENTITY,e,loop_locals(loop2)) {

      if(e != loop_index(loop2) && !gen_in_list_p(e,loop_locals(loop1))) {
        loop_locals(loop1) = CONS(ENTITY,e,loop_locals(loop1));
      }
    }

    if(!inner_success) { // Usual case
      gen_free_list(sequence_statements(statement_sequence(body_loop1)));
      sequence_statements(statement_sequence(body_loop1)) = fused;
      gen_free_list(sequence_statements(statement_sequence(body_loop2)));
      sequence_statements(statement_sequence(body_loop2)) = NIL;
      //free_statement(sloop2); SG causes lost comments and lost extensions, MA should check this
    } else {
      // Inner loops have been fused
      gen_free_list(fused);
    }
  } else {
    // FI: this also should be controlled by information about the
    // liveness of both indices; also index1 must not be used in
    // loop2 as a temporary; so the memory effects of loops 2 should
    // be checked before attempting the first substitution
    if(index1!=index2) {
      replace_entity((void *)body_loop2, index1, index2);
      struct entity_pair thecouple = { index1, index2 };
      gen_context_recurse(body_loop2, &thecouple,
              statement_domain, gen_true, replace_entity_effects_walker);
    }
    // Cleaning FIXME
    loop_body(loop1) = body_loop1;
    gen_free_list(fused);
  }

  ifdebug(3) {
    pips_debug(3, "End of fusion_loops\n\n");
    print_statement(sloop1);
    pips_debug(3, "\n********************\n");
  }
  return success;
}



/**
 * @brief Try to fuse the two loop. Dependences are check against the new body
 * but other constraints such as some statement between the two loops are not
 * handled and must be enforced outside.
 *
 */
static bool fusion_loops(statement sloop1,
                         set contained_stmts_loop1,
                         statement sloop2,
                         set contained_stmts_loop2,
                         bool maximize_parallelism,
                         bool coarse_grain) {
  pips_assert("Previous is a loop", statement_loop_p( sloop1 ) );
  pips_assert("Current is a loop", statement_loop_p( sloop2 ) );
  loop loop1 = statement_loop(sloop1);
  loop loop2 = statement_loop(sloop2);

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

  if(coarse_grain) {
    return coarse_fusion_loops(sloop1,sloop2,maximize_parallelism);
  } else {
    return fine_fusion_loops(sloop1,contained_stmts_loop1,sloop2,contained_stmts_loop2,
                    maximize_parallelism);
  }
}


/**
 * Create an empty block
 */
static fusion_block make_empty_block(int num) {
  fusion_block block = (fusion_block)malloc(sizeof(struct fusion_block));
  block->num = num;
  block->id = num;
  block->s = NULL;
  block->statements = set_make(set_pointer);
  block->successors = fbset_make();
  block->rr_successors = fbset_make();
  block->predecessors = fbset_make();
  block->rr_predecessors = fbset_make();
  block->is_a_loop = false;
  block->count_fusion = 0;
  block->processed = false;

  return block;
}


static void free_block_list(list blocks) {
  FOREACH(fusion_block,b,blocks) {
    set_free(b->statements);
    fbset_free(b->successors);
    fbset_free(b->rr_successors);
    fbset_free(b->predecessors);
    fbset_free(b->rr_predecessors);
    free(b);
  }
  gen_free_list(blocks);
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
      fbset_empty_p(b->successors) && fbset_empty_p(b->rr_successors));
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
                fbset_add_element(b->successors,
                                (void *)sink_block);
                // Mark current block as a predecessor ;-)
                fbset_add_element(sink_block->predecessors,
                                (void *)b);
              } else {
                // Read-read dependence is interesting to fuse, but is not a
                // precedence constraint
                fbset_add_element(b->rr_successors,
                                (void *)sink_block);
                // Mark current block as a rr_predecessor ;-)
                fbset_add_element(sink_block->rr_predecessors,
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
  fbset_difference(b->rr_successors, b->successors);
  fbset_difference(b->rr_predecessors, b->predecessors);
}


/**
 * Prune the graph so that we have a DAG. There won't be anymore more than
 * one path between two block in the predecessors/successors tree. We keep only
 * longest path, no shortcut :-)
 */
static void prune_successors_tree_aux(fusion_block b, fbset full_succ) {
    pips_debug(8,"visiting %d\n",b->num);

    fbset full_succ_of_succ = fbset_make();
    FBSET_FOREACH(succ, b->successors) {
        prune_successors_tree_aux(succ, full_succ_of_succ);
        fbset_union(full_succ,full_succ_of_succ);
        fbset_clear(full_succ_of_succ);
    }
    fbset_free(full_succ_of_succ);

    FBSET_FOREACH(succ_of_succ,full_succ){
        fbset_del_element(succ_of_succ->predecessors, b);
        fbset_del_element(succ_of_succ->rr_predecessors, b);
        fbset_del_element(b->successors,succ_of_succ);
        fbset_del_element(b->rr_successors,succ_of_succ);
    }
    fbset_union(full_succ,b->successors);
}

static fbset prune_successors_tree(fusion_block b) {
    pips_debug(8,"visiting %d\n",b->num);
    fbset full_succ = fbset_make();
    prune_successors_tree_aux(b, full_succ);
    return full_succ;
}


static void get_all_path_heads(fusion_block b, set heads) {
  if(fbset_empty_p(b->predecessors)) {
    set_add_element(heads,heads,b);
  } else {
    FBSET_FOREACH(pred,b->predecessors) {
      get_all_path_heads(pred,heads);
    }
  }
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
  fbset_union(block1->predecessors, block2->predecessors);
  // merge rr_predecessors
  fbset_union(block1->rr_predecessors, block2->rr_predecessors);
  // merge successors
  fbset_union( block1->successors, block2->successors);
  // merge rr_successors
  fbset_union(block1->rr_successors, block2->rr_successors);
  // merge statement
  set_union(block1->statements, block1->statements, block2->statements);

  // Replace block2 with block1 as a predecessor of his successors
  FBSET_FOREACH(succ,block2->successors) {
      fbset_add_element(succ->predecessors, block1);
      fbset_del_element(succ->predecessors, block2);
  }

  // Replace block2 with block1 as a predecessor of his rr_successors
  FBSET_FOREACH(rr_succ,block2->rr_successors) {
      fbset_add_element(rr_succ->rr_predecessors, block1);
      fbset_del_element(rr_succ->rr_predecessors, block2);
  }

  // Replace block2 with block1 as a successor of his predecessors
  FBSET_FOREACH(pred,block2->predecessors) {
      if(pred != block1) {
          fbset_add_element(pred->successors, block1);
      }
      fbset_del_element(pred->successors, block2);
  }

  // Replace block2 with block1 as a successor of his rr_predecessors
  FBSET_FOREACH(rr_pred,block2->rr_predecessors) {
      if(rr_pred != block1) {
          fbset_add_element(rr_pred->rr_successors, block1);
      }
      fbset_del_element(rr_pred->rr_successors, block2);
  }

  // Remove block1 from predecessors and successors of ... block1
  fbset_del_element(block1->predecessors, block1);
  fbset_del_element(block1->successors, block1);
  // Remove block1 from rr_predecessors and rr_successors of ... block1
  fbset_del_element(block1->rr_predecessors, block1);
  fbset_del_element(block1->rr_successors, block1);


  block2->num = -1; // Disable block, will be garbage collected

  // Fix the graph to be a tree
  // Fixme : Heavy :-(
  set heads = set_make(set_pointer);
  get_all_path_heads(block1,heads);
  SET_FOREACH(fusion_block,b,heads) {
    fbset_free(prune_successors_tree(b));
  }
  set_free(heads);

  // Do not loose comments and extensions
  if(!empty_comments_p(statement_comments(block2->s)) &&
          !blank_string_p(statement_comments(block2->s)))
      append_comments_to_statement(block1->s, statement_comments(block2->s));
  extensions_extension(statement_extensions(block1->s))=
      gen_nconc(extensions_extension(statement_extensions(block1->s)), gen_full_copy_list(extensions_extension(statement_extensions(block2->s))));

  block1->count_fusion++;

  ifdebug(4) {
    pips_debug(4,"After merge :\n");
    print_block(block1);
    print_block(block2);
  }
}


/**
 * @brief Checks if precedence constraints allow fusing two blocks
 */
static bool fusable_blocks_p( fusion_block b1, fusion_block b2, unsigned int fuse_limit) {
  bool fusable_p = false;
  if(b1!=b2 && b1->num>=0 && b2->num>=0 && b1->is_a_loop && b2->is_a_loop
      && b1->count_fusion<fuse_limit && b2->count_fusion<fuse_limit) {
    // Blocks are active and are loops

    if(fbset_belong_p(b2->successors,b1)) {
      ifdebug(6) {
        pips_debug(6,"b1 is a successor of b2, fusion prevented !\n");
        print_block(b1);
        print_block(b2);
      }
      fusable_p = false;
    } else if(fbset_belong_p(b1->successors,b2)) {
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
      fusion_block* full_succ_b1 = prune_successors_tree(b1);
      if(!fbset_belong_p(full_succ_b1,b2)) {
        // b2 is not a successors of a successor of a .... of b1
        // look at the opposite !
        pips_debug(6,"Getting full successors for b2 (%d)\n",b2->num);
        fusion_block* full_succ_b2 = prune_successors_tree(b2);
        if(!fbset_belong_p(full_succ_b2,b1)) {
          fusable_p = true;
        }
        fbset_free(full_succ_b2);
      }
      fbset_free(full_succ_b1);
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
                        bool maximize_parallelism,
                        bool coarse) {
  bool return_val = false; // Tell is a fusion has occured
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
    if(fusion_loops(b1->s,b1->statements, b2->s, b2->statements, maximize_parallelism, coarse)) {
      pips_debug(2, "Loop have been fused\n");
      // Now fuse the corresponding blocks
      merge_blocks(b1, b2);
      return_val=true;
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
                                        bool maximize_parallelism,
                                        unsigned int fuse_limit,
                                        bool coarse) {
  // First step is to try to fuse with each successor
  if(!b->processed && b->is_a_loop && b->num>=0 && b->count_fusion < fuse_limit) {
    pips_debug(5,"Block %d is a loop, try to fuse with successors !\n",b->num);
    FBSET_FOREACH( succ, b->successors)
    {
      pips_debug(6,"Handling successor : %d\n",succ->num);
      if(fuse_block(b, succ, maximize_parallelism,coarse)) {
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
  FBSET_FOREACH(succ, b->successors) {
    if(try_to_fuse_with_successors(succ,
                                   fuse_count,
                                   maximize_parallelism,
                                   fuse_limit,
                                   coarse))
      return true;
  }

  b->processed = false;
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
                                        bool maximize_parallelism,
                                        unsigned int fuse_limit,
                                        bool coarse) {
  if(b->is_a_loop && b->count_fusion < fuse_limit) {
    pips_debug(5,"Block %d is a loop, try to fuse with rr_successors !\n",b->num);
    FBSET_FOREACH(succ, b->rr_successors)
    {
      if(fusable_blocks_p(b,succ,fuse_limit) &&
          fuse_block(b, succ, maximize_parallelism,coarse)) {
        /* predecessors and successors set have been modified for the current
         * block... we can no longer continue in this loop, let's restart
         * current function at the beginning and end this one.
         *
         * FIXME : performance impact may be high ! Should not try to fuse
         * again with the same block
         *
         */
        (*fuse_count)++;
        try_to_fuse_with_rr_successors(b,
                                       fuse_count,
                                       maximize_parallelism,
                                       fuse_limit,
                                       coarse);
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
                                     bool maximize_parallelism,
                                     unsigned int fuse_limit,
                                     bool coarse) {
  FOREACH(fusion_block, b1, blocks) {
    if(b1->is_a_loop && b1->count_fusion<fuse_limit) {
      FOREACH(fusion_block, b2, blocks) {
        if(fusable_blocks_p(b1,b2,fuse_limit)) {
          if(fuse_block(b1, b2, maximize_parallelism,coarse)) {
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


  // Keep track of the number of loop founded, to enable or disable next stage
  int number_of_loop = 0, i=0;

  // Loop over the list of statements in the sequence and compute blocks
  list stmts = sequence_statements(s);
  // We have to give a number to each block. It'll be used for regenerating
  // the list of statement in the sequence wrt initial order.
  max_num = gen_length(stmts);
#ifdef __SSE2__
  max_num=4*((max_num+3)/4);
#endif
  FOREACH(statement, st, stmts) {
    fusion_block b = make_block_from_statement(st, i);
    block_list = gen_cons(b, block_list);
    i++;
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
      if(fbset_empty_p(block->predecessors)) { // Block has no predecessors
        fbset_free(prune_successors_tree(block));
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
      if(fbset_empty_p(block->predecessors)) { // Block has no predecessors
        pips_debug(2,
            "Operate on block %d (is_a_loop %d)\n",
            block->num,
            block->is_a_loop);

        if(try_to_fuse_with_successors(block,
                                       &fuse_count,
                                       params->maximize_parallelism,
                                       params->max_fused_per_loop,
                                       params->coarse)) {
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
        try_to_fuse_with_rr_successors(block,
                                       &fuse_count,
                                       params->maximize_parallelism,
                                       params->max_fused_per_loop,
                                       params->coarse);
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
      fuse_all_possible_blocks(block_list,
                               &fuse_count,
                               params->maximize_parallelism,
                               params->max_fused_per_loop,
                               params->coarse);
    }



    if(fuse_count > 0) {
      /*
       * Cleaning : removing old blocks that have been fused
       */
      list block_iter = block_list;
      list prev_block_iter = NIL;
      while(block_iter != NULL) {
        fusion_block block = (fusion_block)CAR(block_iter).p;

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

      ifdebug(3) {
        pips_debug(3,"Before regeneration\n");
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

          if(fbset_empty_p(block->predecessors)) { // Block has no predecessors
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
            FBSET_FOREACH(succ,block->successors)
            {
              fbset_del_element(succ->predecessors, block);
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

  // No leak
  free_block_list(block_list);

  return true;
}

/**
 * Will try to fuse as many loops as possible in the IR subtree rooted by 's'
 */
static void compute_fusion_on_statement(statement s, bool coarse) {
  // Get user preferences with some properties
  struct fusion_params params;
  params.maximize_parallelism = get_bool_property("LOOP_FUSION_MAXIMIZE_PARALLELISM");
  params.coarse = coarse;
  params.greedy = get_bool_property("LOOP_FUSION_GREEDY");
  params.max_fused_per_loop = get_int_property("LOOP_FUSION_MAX_FUSED_PER_LOOP");

  // Go on fusion on every sequence of statement founded
  gen_context_recurse( s, &params, sequence_domain, fusion_in_sequence, gen_true);
}

/**
 * Loop fusion main entry point
 */
bool module_loop_fusion(char * module_name, bool region_based) {
  statement module_statement;
  graph dependence_graph;

  /* Get the true ressource, not a copy. */
  module_statement = (statement)db_get_memory_resource(DBR_CODE,
                                                       module_name,
                                                       true);
  set_ordering_to_statement(module_statement);

  set_current_module_statement(module_statement);
  set_current_module_entity(module_name_to_entity(module_name));

  /* The proper effect to detect the I/O operations: */
  set_proper_rw_effects((statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS,
                                                                  module_name,
                                                                  true));

  /* Mandatory for DG construction and module_to_value_mapping() */
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,
                                                                     module_name,
                                                                     true));

  /* Get the data dependence graph */
  dependence_graph = (graph)db_get_memory_resource(DBR_DG,
                                                         module_name,
                                                         true);

  ordering_to_dg_mapping = compute_ordering_to_dg_mapping(dependence_graph);


  if(region_based) {
    /* use preconditions to check that loop bodies are not dead code
     */
    set_precondition_map((statement_mapping)db_get_memory_resource(DBR_PRECONDITIONS,
                                                                   module_name,
                                                                   true));

    /* Build mapping between variables and semantics informations: */
    module_to_value_mappings(module_name_to_entity(module_name));

    /* Get and use invariant read/write regions */
    set_invariant_rw_effects((statement_effects) db_get_memory_resource(DBR_INV_REGIONS,
                                                                        module_name,
                                                                        true));
  }




  debug_on("LOOP_FUSION_DEBUG_LEVEL");

  // Here we go ! Let's fuse :-)
  compute_fusion_on_statement(module_statement,region_based);

  /* Reorder the module, because some statements have been deleted, and others
   * have been reordered
   */
  module_reorder(module_statement);

  /* Store the new code */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_statement);

  // Free resources
  free_temporary_fused_statement(); // Free...
  hash_table_free(ordering_to_dg_mapping);
  ordering_to_dg_mapping = NULL;
  reset_proper_rw_effects();
  reset_cumulated_rw_effects();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();

  if(region_based) {
    reset_precondition_map();
    reset_invariant_rw_effects();
    free_value_mappings();
  }

  pips_debug(2, "done for %s\n", module_name);
  debug_off();

  /* Should have worked:
   *
   * Do we want to provide statistics about the number of fused loops?
   *
   * How do we let PyPS know the number of loops fused?
   */
  return true;
}


/**
 * Loop fusion with DG ; PIPSMake entry point
 */
bool loop_fusion(char * module_name) {
  return module_loop_fusion(module_name, false);
}

/**
 * Loop fusion with Regions ; PIPSMake entry point
 */
bool loop_fusion_with_regions(char * module_name) {
  return module_loop_fusion(module_name, true);
}

