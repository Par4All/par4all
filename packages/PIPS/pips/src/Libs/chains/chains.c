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
/* -- usedef.c

 Computes (approximate) use/def chains of a structured control graph.

 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"

typedef void * arc_label;
typedef void * vertex_label;

#include "graph.h"
#include "dg.h"

#include "misc.h"
#include "properties.h"

#include "ri-util.h"
#include "effects-util.h"

#include "ricedg.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "chains.h"
#include "pipsdbm.h"

#include "resources.h"


/* Some forward declarations */
static void reset_effects();
static list load_statement_effects( statement s );
static void inout_control();
static void inout_statement();
//static void usedef_control(); declared but never defined.
static void genref_statement();
static bool dd_du( effect fin, effect fout );
static bool ud( effect fin, effect fout );
static void add_conflicts(effect fin,
                          statement stout,
                          vertex vout,
                          cons *effect_outs,
                          bool(*which)());

/* Macro to create set */
#define MAKE_STATEMENT_SET() (set_make( set_pointer ))

/* Default sizes for corresponding sets. This is automatically adjusted
 according to needs by the hash table package. */
#define INIT_STATEMENT_SIZE 20
#define INIT_ENTITY_SIZE 10

/* Mapping from effects to the associated statement */
static hash_table effects2statement;

/* Gen maps each statement to the effects it generates. */
static hash_table Gen;
#define GEN(st) ((set)hash_get( Gen, (char *)st ))

/* Refs maps each statement to the effects it references. */
static hash_table Ref;
#define REF(st) ((set)hash_get( Ref, (char *)st ))

/* current_defs is the set of DEF at the current point of the computation */
static set current_defs;
static set current_refs;

/* Def_in maps each statement to the statements that are in-coming the statement
 * It's only used for unstructured, current_defs is used for structured parts */
static hash_table Def_in;
#define DEF_IN(st) ((set)hash_get(Def_in, (char *)st))

/* Def_out maps each statement to the statements that are out-coming the statement
 * It's only used for unstructured, current_defs is used for structured parts */
static hash_table Def_out;
#define DEF_OUT(st) ((set)hash_get(Def_out, (char *)st))

/* Ref_in maps each statement to the effects that are in-coming the statement
 * It's only used for unstructured, current_refs is used for structured parts */
static hash_table Ref_in;
#define REF_IN(st) ((set)hash_get(Ref_in, (char *)st))

/* Ref_out maps each statement to the effects that are out-coming the statement
 * It's only used for unstructured, current_refs is used for structured parts */
static hash_table Ref_out;
#define REF_OUT(st) ((set)hash_get(Ref_out, (char *)st))

/* dg is the dependency graph ; FIXME : should not be static global ? */
static graph dg;

/* Vertex_statement maps each statement to its vertex in the dependency graph. */
static hash_table Vertex_statement;

/* Some properties */
static bool one_trip_do_p;
static bool keep_read_read_dependences_p;
static bool mask_effects_p;
static bool dataflow_dependence_only_p;

/* Access functions for debug only */

/**
 * @brief displays on stderr, the MSG followed by the set of statement numbers
 * associated to effects present in the set S.
 *
 * @param msg a string to be display at the beginning
 * @param s the set of effects to display
 */
static void local_print_statement_set( string msg, set s ) {
  fprintf( stderr, "\t%s ", msg );
  SET_FOREACH( effect, eff, s) {
    fprintf( stderr,
             ",%p (%td) ",
             eff,
             statement_number( (statement)hash_get(effects2statement,eff) ) );
    print_effect( (effect) eff );
  }
  fprintf( stderr, "\n" );
}

/**
 * @return the vertex associated to the statement ST in the dependence graph DG.
 */
static vertex vertex_statement( statement st ) {
  vertex v = (vertex) hash_get( Vertex_statement, (char *) st );
  pips_assert( "vertex_statement", v != (vertex)HASH_UNDEFINED_VALUE );
  return ( v );
}

/**
 * @brief Initializes the global data structures needed for usedef computation
 * of one statement.
 *
 * @param st the statement to initialize
 * @return always true (gen_recurse continuation)
 */
static bool init_one_statement( statement st ) {

  if ( GEN( st ) == (set) HASH_UNDEFINED_VALUE ) {
    /* First initialization (normal use) */

    ifdebug(2) {
      fprintf( stderr,
               "Init statement %td with effects %p, ordering %tx\n",
               statement_number( st ),
               load_statement_effects( st ),
               statement_ordering(st) );
      print_effects( load_statement_effects( st ) );
    }

    /* Create a new vertex in the DG for the statement: */
    dg_vertex_label l;
    vertex v;
    l = make_dg_vertex_label( statement_ordering(st), sccflags_undefined );
    v = make_vertex( l, NIL );
    hash_put( Vertex_statement, (char *) st, (char *) v );
    graph_vertices( dg ) = CONS( VERTEX, v, graph_vertices( dg ));

    /* If the statement has never been seen, allocate new sets:
     * This could be optimized : {DEF,REF}_{IN,OUT} are
     * needed only for unstructured */
    hash_put( Gen, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Ref, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Def_in, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Def_out, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Ref_in, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Ref_out, (char *) st, (char *) MAKE_STATEMENT_SET() );

    /* FI: regions are not really proper effects...
     I should use proper effects for non-call instruction
     I need a global variable
     let's kludge this for the time being, 5 August 1992
     */
    if ( !instruction_block_p(statement_instruction(st)) ) {
      FOREACH(EFFECT, e, load_statement_effects(st) )
      {
        hash_put( effects2statement, (char *) e, (char *) st );
      }
    }
  } else {
    /* If the statement has already been seen, reset the sets associated
     to it: */
    set_clear( GEN( st ) );
    set_clear( REF( st ) );
    set_clear( DEF_OUT( st ) );
    set_clear( DEF_IN( st ) );
    set_clear( REF_OUT( st ) );
    set_clear( REF_IN( st ) );
  }
  /* Go on recursing down: */
  return true;
}

/* The genref_xxx functions implement the computation of GEN and REF
 sets from Aho, Sethi and Ullman "Compilers" (p. 612). This is
 slightly more complex since we use a structured control graph, thus
 fixed point computations can be recursively required (the correctness
 of this is probable, although not proven, as far as I can tell). */

/* KILL_STATEMENT updates the KILL set of statement ST on entity LHS. Only
 effects that modify one reference (i.e., assignments) are killed (see
 above). Write effects on arrays (see IDXS) don't kill the definitions of
 the array. Equivalence with non-array entities is managed properly. */
/**
 * @brief Kill the GEN set with a set of KILL effects
 * @param gen the set to filter, modified by side effects
 * @param killers the "killer" effects
 */
static void kill_effects(set gen, set killers) {
    set killed = MAKE_STATEMENT_SET();
    SET_FOREACH(effect,killer,killers) {
        /* A killer effect is exact and is a write (environment effects kills !!) */
        if ( action_write_p(effect_action(killer))
	     // these tests are optimizations to avoid redundant tests inside the
	     // SET_FOREACH loop;
	     // they allow to call first_exact_scalar_effect_certainly_includes_second_effect_p
	     // instead of first_effect_certainly_includes_second_effect_p
	     // if the latter is enhanced, these tests should be updated accordingly.
	     && effect_exact_p(killer) && effect_scalar_p(killer) ) {
            SET_FOREACH(effect,e,gen) {
                /* We only kill store effect */
                if(store_effect_p(e)) {
                    /* We avoid a self killing */
                    if( e != killer
			&& first_exact_scalar_effect_certainly_includes_second_effect_p(killer, e) ) {
                        set_add_element( killed, killed, e );
                    }
                }
            }
            set_difference(gen,gen,killed);
            set_clear(killed);
        }
    }
    set_free(killed);
}

/**
 * @brief Compute Gen and Ref set for a single statement
 * @param st the statement to compute
 */
static void genref_one_statement( statement st ) {
  set gen = GEN( st );
  set ref = REF( st );
  set_clear( gen );
  set_clear( ref );
  /* Loop over effects to find which one will generates
   * or references some variables
   */
  FOREACH( effect, e, load_statement_effects(st) )
  {
    action a = effect_action( e );

    if ( action_write_p( a ) ) {
      pips_assert("Effect isn't map to this statement !",
          st == hash_get(effects2statement, e));

      /* A write effect will always generate a definition */
      set_add_element( gen, gen, (char *) e );
    } else if ( action_read_p( a ) ) {
      /* A read effect will always generate a reference */
      set_add_element( ref, ref, (char *) e );
    } else {
      /* Secure programming */
      pips_internal_error("Unknow action for effect : "
          "neither a read nor a write !");
    }
  }
}

/**
 * @brief Compute Gen and Ref set for a test
 * @param t the test
 * @param s the statement to compute
 */
static void genref_test( test t, statement s ) {
  statement st = test_true(t);
  statement sf = test_false(t);
  set ref = REF( s );

  /* compute true path */
  genref_statement( st );
  /* compute false path */
  genref_statement( sf );

  /* Combine the two path to summarize the test */
  set_union( GEN( s ), GEN( st ), GEN( sf ) );
  set_union( ref, ref, REF( sf ) );
  set_union( ref, ref, REF( st ) );
}

/**
 * @brief MASK_EFFECTS masks the effects in S according to the locals L.
 * @param s the set of effects
 * @param l the locals to mask
 */
static void mask_effects( set s, list l ) {
  cons *to_mask = NIL;
  /*
   * Loop over effect and check if they affect a local variable
   * We mask only read effects (FIXME : why ?)
   */
  SET_FOREACH(effect, f, s) {
    action a = effect_action( f );

    if ( action_read_p( a ) ) {
      FOREACH( entity, e, l )
      {
        if ( effect_may_read_or_write_memory_paths_from_entity_p( f, e ) ) {
          /* Register which one we have to mask */
          to_mask = CONS( effect, f, to_mask );
        }
      }
    }
  }
  /* Do the masking */
  FOREACH( effect, f, to_mask ) {
    set_del_element( s, s, (char *) f );
  }

  /* We free because we are good programmers and we don't leak ;-) */
  gen_free_list( to_mask );
}

/**
 * @brief Compute Gen and Ref set for any loop (do, for, while,...)
 * @description It has to deal specially with the loop variable which is not
 * managed in the Dragon book. Effect masking is performed
 * on locals (i.e., gen&ref sets are pruned from local definitions).
 *
 * For DO loop, if loops are at least one trip (set by property "ONE_TRIP_DO"),
 * then statements are always killed by execution of loop body.
 *
 * @param l the loop to compute
 * @param st the statement that hold the loop
 *
 */
static void genref_any_loop( statement body,
                              statement st,
                              list locals,
                              bool one_trip_do_p ) {
  set gen = GEN( st );
  set ref = REF( st );

  /* Compute genref on the loop body */
  genref_statement( body );

  /* Summarize the body to the statement that hold the loop */
  set_union( gen, gen, GEN( body ) );
  set_union( ref, ref, REF( body ) );

  /* Filter effects on local variables */
  if ( mask_effects_p ) {
    mask_effects( gen, locals );
    mask_effects( ref, locals );
  }

}

/**
 * @brief Compute Gen and Ref set for a "do" loop
 * @description see genref_any_loop()
 *
 * @param l the loop to compute
 * @param st the statement that hold the loop
 */
static void genref_loop( loop l, statement st ) {
  statement body = loop_body( l );

  /* Building locals list */
  list llocals = loop_locals( l );
  list slocals = statement_declarations(st);
  list locals = gen_nconc( gen_copy_seq( llocals ), gen_copy_seq( slocals ) );

  /* Call the generic function handling all kind of loop */
  genref_any_loop( body, st, locals, one_trip_do_p );

  /* We free because we are good programmers and we don't leak ;-) */
  gen_free_list( locals );
}

/**
 * @brief Compute Gen and Ref set for a "for" loop
 * @description see genref_any_loop()
 *
 * @param l the loop to compute
 * @param st the statement that hold the loop
 *
 */
static void genref_forloop( forloop l, statement st ) {
  statement body = forloop_body( l );
  list locals = statement_declarations(st);

  /* Call the generic function handling all kind of loop */
  genref_any_loop( body, st, locals, one_trip_do_p );

}

/**
 * @brief Compute Gen and Ref set for a "while" loop
 * @description see genref_any_loop()
 *
 * @param l the loop to compute
 * @param st the statement that hold the loop
 */
static void genref_whileloop( whileloop l, statement st ) {
  statement body = whileloop_body( l );
  list locals = statement_declarations(st);

  /* Call the generic function handling all kind of loop */
  genref_any_loop( body, st, locals, one_trip_do_p );
}

/**
 * @brief Compute Gen and Ref set for a block
 * @description The Dragon book only deals with a sequence of two statements.
 * here we generalize to lists, via recursion. Statement are processed in
 * reversed order (i.e. on the descending phase of recursion)
 *
 * @param sts the list of statements inside the block
 * @param st the statement that hold the block
 *
 */
static void genref_block( cons *sts, statement st ) {
  /* Summarize for the block */
  set gen_st = GEN( st );
  set ref_st = REF( st );

  /* loop over statements inside the block */
  FOREACH( statement, one, sts ) {

    genref_statement( one );

    // We no longer use a "kill set" FIXME : one_trip_do, test, ....
    kill_effects(gen_st,GEN(one));
    set_union(gen_st,GEN(one),gen_st);

    set_union( ref_st, REF( one ), ref_st );

  }
  // Gen for the block doesn't include locally declared variable
  if ( mask_effects_p) {
    mask_effects( gen_st, statement_declarations(st) );
    mask_effects( ref_st, statement_declarations(st) );
  }
}

/**
 * @brief Compute Gen and Ref set for an unstructured
 * @description computes the gens, refs, and kills set of the unstructured by
 * recursing on INOUT_CONTROL. The gens&refs can then be inferred.
 * The situation for the kills is more fishy; for the moment, we just keep
 * the kills that are common to all statements of the control graph.
 *
 * @param u is the unstructured
 * @param st is the statement that hold the unstructured
 */
static void genref_unstructured( unstructured u, statement st ) {
  control c = unstructured_control( u );
  statement exit = control_statement( unstructured_exit( u ));
  cons *blocs = NIL;

  set_clear( DEF_IN( st ) );
  /* recursing */
  CONTROL_MAP( c, {statement st = control_statement( c );
        genref_statement( st );
        },
      c, blocs );

  if ( set_undefined_p( DEF_OUT( exit )) ) {
    set ref = MAKE_STATEMENT_SET();
    set empty = MAKE_STATEMENT_SET();
    CONTROL_MAP( cc, {
          set_union( ref, ref, REF( control_statement( cc )));
        }, c, blocs );
    set_assign( REF( st ), ref );
    set_free( ref );
    set_assign( GEN( st ), empty );
    set_free( empty );
  } else {
    set_assign( GEN( st ), DEF_OUT( exit ) );
    set_difference( REF( st ), REF_OUT( exit ), REF_IN( st ) );
  }
  gen_free_list( blocs );
}

/**
 * @brief Does the dispatch and recursion loop
 * @description
 *
 * @param i is the instruction
 * @param st is the statement that hold the instruction
 */
static void genref_instruction( instruction i, statement st ) {
  cons *pc;
  test t;
  loop l;

  switch ( instruction_tag(i) ) {
    case is_instruction_block:
      genref_block( pc = instruction_block(i), st );
      break;
    case is_instruction_test:
      t = instruction_test(i);
      genref_test( t, st );
      break;
    case is_instruction_loop:
      l = instruction_loop(i);
      genref_loop( l, st );
      break;
    case is_instruction_whileloop: {
      whileloop l = instruction_whileloop(i);
      genref_whileloop( l, st );
      break;
    }
    case is_instruction_forloop: {
      forloop l = instruction_forloop(i);
      genref_forloop( l, st );
      break;
    }
    case is_instruction_unstructured:
      genref_unstructured( instruction_unstructured( i ), st );
      break;
    case is_instruction_call:
    case is_instruction_expression:
    case is_instruction_goto:
      /* no recursion for these kind of instruction */
      break;
    default:
      pips_internal_error("unexpected tag %d", instruction_tag(i));
  }
}

/**
 * @brief Compute recursively Gen and Ref set for a statement.
 *  This is the main entry to this task.
 * @description computes the gens and refs set of the statement by
 * recursing
 *
 * @param s is the statement to compute
 */
static void genref_statement( statement s ) {

  /* Compute genref using effects associated to the statement itself */
  genref_one_statement( s );

  ifdebug(2) {
    debug( 2,
           "genref_statement",
           "Result genref_one_statement for Statement %p [%s]:\n",
           s,
           statement_identification( s ) );
    local_print_statement_set( "GEN", GEN( s ) );
    local_print_statement_set( "REF", REF( s ) );
  }

  /* Recurse inside the statement (relevant for blocks, loop, ...) */
  genref_instruction( statement_instruction(s), s );

  ifdebug(2) {
    debug( 2,
           "genref_statement",
           "Result for Statement %p [%s]:\n",
           s,
           statement_identification( s ) );
    ;
    local_print_statement_set( "GEN", GEN( s ) );
    local_print_statement_set( "REF", REF( s ) );
  }
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the block
 * @param sts is the list of statement inside the block
 */
static void inout_block( statement st, cons *sts ) {

  /* loop over statements inside the block */
  FOREACH( statement, one, sts ) {
    /* Compute the outs from the ins for this statement */
    inout_statement( one );
  }
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the test
 * @param t is the test
 */
static void inout_test( statement s, test t ) {
  statement st = test_true( t );
  statement sf = test_false( t );

  // Save DEF and REF
  set def_in = set_dup(current_defs);
  set ref_in = set_dup(current_refs);

  /*
   * Compute true path
   */
  inout_statement( st );

  // Save DEF and REF
  set def_out = current_defs;
  set ref_out = current_refs;

  // Restore DEF and REF
  current_defs = def_in;
  current_refs = ref_in;

  /*
   * Compute false path
   */
  inout_statement( sf );

  /* Compute the out for the test */
  set_union( current_defs, current_defs, def_out );
  set_union( current_refs, current_refs, ref_out );

  // Free temporary set
  set_free(def_out);
  set_free(ref_out);

}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the loop
 * @param lo is the loop
 * @param one_trip_do_p tell if there is always at least one trip (do loop only)
 */
static void inout_any_loop( statement st, statement body, bool one_trip_do_p ) {
  set def_in;
  set ref_in;

  if ( ! one_trip_do_p ) {
    // Save DEF and REF
    def_in = set_dup(current_defs);
    ref_in = set_dup(current_refs);
  }

  /* Compute "in" sets for the loop body */
  set_union( current_defs, current_defs, GEN( st ) );
  set_union( current_refs, current_refs, REF( st ) );

  /* Compute loop body */
  inout_statement( body );

  /* Compute "out" sets for the loop */
  if ( ! one_trip_do_p ) {
    /* Body is not always done, approximation by union */
    set_union( current_defs, current_defs, def_in );
    set_union( current_refs, current_refs, ref_in );
    // Free temporary set
    set_free(def_in);
    set_free(ref_in);
  }
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the loop
 * @param lo is the loop
 */
static void inout_loop( statement st, loop lo ) {
  // Try to detect loop executed at least once trivial case
  bool executed_once_p = one_trip_do_p;
  if(!executed_once_p) executed_once_p = loop_executed_at_least_once_p(lo);

  inout_any_loop( st, loop_body( lo ), executed_once_p );
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the loop
 * @param t is the loop
 */
static void inout_whileloop( statement st, whileloop wl ) {
  statement body = whileloop_body( wl );
  inout_any_loop( st, body, false );
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the loop
 * @param t is the loop
 */
static void inout_forloop( statement st, forloop fl ) {
  statement body = forloop_body( fl );
  inout_any_loop( st, body, one_trip_do_p );
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the call
 * @param c is the call (unused)
 */
static void inout_call( statement st, call __attribute__ ((unused)) c ) {
  /* Compute "out" sets  */

  kill_effects(current_defs,GEN(st));
  set_union( current_defs, GEN( st ), current_defs);

  kill_effects(current_refs,GEN(st));
  set_union( current_refs, REF( st ), current_refs);
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the unstructured
 * @param u is the unstructured
 */
static void inout_unstructured( statement st, unstructured u ) {
  control c = unstructured_control( u );
  control exit = unstructured_exit( u );
  statement s_exit = control_statement( exit );

  /* Compute "in" sets */
  set_assign( DEF_IN( control_statement( c )), current_defs );
  set_assign( REF_IN( control_statement( c )), current_refs );

  /* Compute the unstructured */
  inout_control( c );

  /* Compute "out" sets */
  if ( set_undefined_p( DEF_OUT( s_exit )) ) {
    list blocs = NIL;
    set ref = MAKE_STATEMENT_SET();
    set empty = MAKE_STATEMENT_SET();

    CONTROL_MAP( cc, {
          set_union( ref, ref, REF( control_statement( cc )));
        }, c, blocs );
    set_assign( REF_OUT( st ), ref );
    set_assign( DEF_OUT( st ), empty );

    set_free( ref );
    set_free( empty );
    gen_free_list( blocs );
  } else {
    set_assign( DEF_OUT( st ), DEF_OUT( s_exit ) );
    set_assign( REF_OUT( st ), REF_OUT( s_exit ) );
  }

  // Exit from unstructured, restore global DEF
  set_assign( current_defs, DEF_OUT( st ) );
  set_assign( current_refs, REF_OUT( st ) );

}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement to compute
 */
static void inout_statement( statement st ) {
  instruction i;
  static int indent = 0;

  ifdebug(2) {
    pips_debug( 2,
             "%*s> Computing DEF and REF for statement %p (%td %td):\n"
             "current_defs %p, current_refs %p\n",
             indent++,
             "",
             st,
             statement_number( st ),
             statement_ordering( st ),
             current_defs,
             current_refs );
    local_print_statement_set( "DEF_IN", current_defs );
    local_print_statement_set( "REF_IN", current_refs );
  }
  /* Compute Def-Def and Def-Use conflicts from Def_in set */
  vertex statement_vertex = vertex_statement( st );
  cons *effects = load_statement_effects( st );
  SET_FOREACH( effect, defIn, current_defs) {
    add_conflicts( defIn, st, statement_vertex, effects, dd_du );
  }
  /* Compute Use-Def conflicts from Ref_in set */
  SET_FOREACH( effect, refIn, current_refs) {
    add_conflicts( refIn, st, statement_vertex, effects, ud );
  }

  ifdebug(3) {
    pips_debug( 3,
             "%*s> After add conflicts for statement %p (%td %td):\n"
             "current_defs %p, current_refs %p\n",
             indent,
             "",
             st,
             statement_number( st ),
             statement_ordering( st ),
             current_defs,
             current_refs );
    local_print_statement_set( "DEF_IN", current_defs );
    local_print_statement_set( "REF_IN", current_refs );
  }

  /* Compute "out" sets for the instruction : recursion */
  switch ( instruction_tag( i = statement_instruction( st )) ) {
    case is_instruction_block:
      inout_block( st, instruction_block( i ) );
      break;
    case is_instruction_test:
      inout_test( st, instruction_test( i ) );
      break;
    case is_instruction_loop:
      inout_loop( st, instruction_loop( i ) );
      break;
    case is_instruction_whileloop:
      inout_whileloop( st, instruction_whileloop( i ) );
      break;
    case is_instruction_forloop:
      inout_forloop( st, instruction_forloop( i ) );
      break;
    case is_instruction_call:
      inout_call( st, instruction_call( i ) );
      break;
    case is_instruction_expression:
      /* The second argument is not used */
      inout_call( st, (call) instruction_expression( i ) );
      break;
    case is_instruction_goto:
      pips_internal_error("Unexpected tag %d", i );
      break;
    case is_instruction_unstructured:
      inout_unstructured( st, instruction_unstructured( i ) );
      break;
    default:
      pips_internal_error("Unknown tag %d", instruction_tag(i) );
  }
  ifdebug(2) {
    pips_debug( 2,
             "%*s> Statement %p (%td %td):\n"
             "current_defs %p, current_refs %p\n",
             indent--,
             "",
             st,
             statement_number( st ),
             statement_ordering( st ),
             current_defs,
             current_refs );
    local_print_statement_set( "DEF_OUT", DEF_OUT( st ) );
    local_print_statement_set( "REF_OUT", REF_OUT( st ) );
  }

}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 * @description It computes the in and out sets of the structured control
 * graph. This is done by fixed point iteration (see Dragon book, p. 625),
 * except that the in set of CT is not empty (this explains the set_union
 * instead of set_assign in the fixed point computation loop on IN( st )).
 * Once again, the correctness of this modification is not proven.
 *
 * @param ct is the control to compute
 */
static void inout_control( control ct ) {
  bool change;
  set d_oldout = MAKE_STATEMENT_SET();
  set d_out = MAKE_STATEMENT_SET();
  set r_oldout = MAKE_STATEMENT_SET();
  set r_out = MAKE_STATEMENT_SET();
  cons *blocs = NIL;


  ifdebug(2) {
    fprintf( stderr, "Computing DEF_IN and OUT of control %p entering", ct );
    local_print_statement_set( "", DEF_IN( control_statement( ct )) );
  }

  CONTROL_MAP( c, {statement st = control_statement( c );
        set_assign( DEF_OUT( st ), GEN( st ));
        set_assign( REF_OUT( st ), REF( st ));

        if( c != ct ) {
          set_clear( DEF_IN( st ));
          set_clear( REF_OUT( st ));
        }},
      ct, blocs );

  for ( change = true; change; ) {
    ifdebug(3) {
      fprintf( stderr, "Iterating on %p ...\n", ct );
    }
    change = false;

    CONTROL_MAP( b,
        { statement st = control_statement( b );

          set_clear( d_out );
          set_clear( r_out );
          MAPL( preds, {control pred = CONTROL( CAR( preds ));
                statement pst = control_statement( pred );

                set_union( d_out, d_out, DEF_OUT( pst ));
                set_union( r_out, r_out, REF_OUT( pst ));},
              control_predecessors( b ));
          set_union( DEF_IN( st ), DEF_IN( st ), d_out );
          set_union( REF_IN( st ), REF_IN( st ), r_out );
          set_assign( d_oldout, DEF_OUT( st ));
          set_union( DEF_OUT( st ), GEN( st ),DEF_IN( st ));
                     // FIXME set_difference( diff, DEF_IN( st ), KILL( st )));
          set_assign( r_oldout, REF_OUT( st ));
          set_union( REF_OUT( st ), REF( st ), REF_IN( st ));
          change |= (!set_equal_p( d_oldout, DEF_OUT( st )) ||
              !set_equal_p( r_oldout, REF_OUT( st )));
        }, ct, blocs );
  }

  CONTROL_MAP( c, {
                   statement st = control_statement( c );
                   /* Prepare "in" sets */
                   set_assign( current_defs, DEF_IN( st ) );
                   set_assign( current_refs, REF_IN( st ) );

                   inout_statement( st);

                   /* Restore out sets */
                   set_assign( DEF_OUT( st ), current_defs );
                   set_assign( REF_OUT( st ), current_refs );
                  },
      ct, blocs );
  set_free( d_oldout );
  set_free( d_out );
  set_free( r_oldout );
  set_free( r_out );
  gen_free_list( blocs );

}

/**
 * @brief adds the conflict FIN->FOUT in the list CFS
 * @description adds the conflict FIN->FOUT in the list CFS (if it's
 * not already there, and apparently even if it's already there for
 * performance reason...).
 *
 * @param fin is the incoming effect
 * @param fout is the sink effect
 * @param cfs is the list of conflict to update
 */
static cons *pushnew_conflict( effect fin, effect fout, cons *cfs ) {
  conflict c;

  /* FI: This is a bottleneck for large modules with lots of callees and
   long effect lists. Let's assume that the pairs (fin, fout) are
   always unique because the effects are sets. */

  /*
   MAPL( cs, {
   conflict c = CONFLICT( CAR( cs )) ;

   if( conflict_source( c )==fin && conflict_sink( c )==fout ) {
   return( cfs ) ;
   }
   }, cfs ) ;
   */

  /* create the conflict */
  c = make_conflict( fin, fout, cone_undefined );
  ifdebug(2) {
    fprintf( stderr,
             "Adding %s->%s\n",
             entity_name( effect_entity( fin )),
             entity_name( effect_entity( fout )) );
  }
  /* Add the conflict to the list */
  return ( CONS( CONFLICT, c, cfs ) );
}

/**
 * @brief DD_DU detects Def/Def, Def/Use conflicts between effects FIN and FOUT.
 */
static bool dd_du( effect fin, effect fout ) {
  bool conflict_p = action_write_p( effect_action( fin ));

  if ( dataflow_dependence_only_p ) {
    conflict_p = conflict_p && action_read_p( effect_action( fout ) );
  }

  return conflict_p;
}

/**
 * @brief UD detects Use/Def conflicts between effects FIN and FOUT.
 */
static bool ud( effect fin, effect fout ) {
  return ( action_read_p( effect_action( fin ))
      && ( action_write_p( effect_action( fout )) || keep_read_read_dependences_p ) );
}

/**
 * @brief adds conflict arcs to the dependence graph dg between the in-coming
 * statement STIN and the out-going STOUT.
 * @description Note that output dependencies are not minimal
 * (e.g., i = s ; s = ... ; s = ...) creates an oo-dep between the i assignment
 * and the last s assignment.
 */
static void add_conflicts(effect fin,
                          statement stout,
                          vertex vout,
                          cons *effect_outs,
                          bool(*which)()) {
  vertex vin;
  cons *cs = NIL;

  ifdebug(2) {
    statement stin = hash_get( effects2statement, fin );
    _int stin_o = statement_ordering(stin);
    _int stout_o = statement_ordering(stout);
    fprintf( stderr,
	     "Conflicts %td (%td,%td) (%p) -> %td (%td,%td) (%p) %s"
	     " for \"%s\"\n",
	     statement_number(stin),
	     ORDERING_NUMBER(stin_o),
	     ORDERING_STATEMENT(stin_o),
	     stin,
	     statement_number(stout),
	     ORDERING_NUMBER(stout_o),
	     ORDERING_STATEMENT(stout_o),
	     stout,
	     ( which == ud ) ? "ud" : "dd_du",
	     entity_local_name(effect_to_entity(fin)));
  }

  /* To speed up pushnew_conflict() without damaging the integrity of
   the use-def chains */
  ifdebug(1) {
    statement stin = hash_get( effects2statement, fin );
    cons *effect_ins = load_statement_effects( stin );
    if ( !gen_once_p( effect_ins ) ) {
      pips_internal_error("effect_ins are redundant");
    }
    if ( !gen_once_p( effect_outs ) ) {
      pips_internal_error("effect_outs are redundant");
    }
  }
  FOREACH(EFFECT, fout, effect_outs) {
    ifdebug(2) {
      print_effect(fin);
      fprintf(stderr," -> ");
      print_effect(fout);
      fprintf(stderr,"\n");
    }

    // We want to check the conflict even with read/read, because we already
    // asserted what we want before (ud/du/dd)
    if((*which)(fin, fout) && effects_might_conflict_even_read_only_p(fin, fout)) {
      entity ein = effect_entity(fin);
      bool ein_abstract_location_p = entity_abstract_location_p(ein);
      entity eout = effect_entity(fout);
      bool add_conflict_p = true;

      /* I think that most of this is hazardous when mixes of
         pointers, arrays and structs are involved. However, there
         should be no pointer anymore with CONSTANT_PATH_EFFECTS set
         to TRUE!  We need to think more about all of this. BC.
      */
      if(ein_abstract_location_p) {
#if 0 //SG: as is, this code is equivalent to doing nothing
	    pips_debug(2, "abstract location case \n");
        entity alout = variable_to_abstract_location(eout);
	/* this test is not correct, rout should be converted to an abstract location, not eout. BC. */
        if(abstract_locations_may_conflict_p(ein, alout))
          add_conflict_p = true;
#endif
      } else {
        reference rin = effect_any_reference(fin);
        int din = gen_length(reference_indices(rin));
        reference rout = effect_any_reference(fout);
        int dout = gen_length(reference_indices(rout));
        type tin = ultimate_type(entity_type(ein));
        type tout = ultimate_type(entity_type(eout));
        if(pointer_type_p(tin) && pointer_type_p(tout)) {
	  pips_debug(2, "pointer type case \n");
          /* Second version due to accuracy improvements in effect
         computation */
          if(din == dout) {
            /* This is the standard case */
            add_conflict_p = true;
          } else if(din < dout) {
            /* I'm not sure this can be the case because of
             * effects_might_conflict_even_read_only_p(fin, fout) */
            /* a write on the shorter memory access path conflicts
             * with the longer one. If a[i] is written, then a[i][j]
             * depends on it. If a[i] is read, no conflict */
            add_conflict_p = action_write_p(effect_action(fin));
          } else /* dout > din */{
            /* same explanation as above */
            add_conflict_p = action_write_p(effect_action(fout));
          }
        } else {
          /* Why should we limit this test to pointers? Structures,
           * structures of arrays and arrays of structures with
           * pointers embedded somewhere must behave in the very same
           * way. Why not unify the two cases? Because we have not
           * spent enough time thinking about memory access paths. */
          if(din < dout) {
            /* a write on the shorter memory access path conflicts
             * with the longer one. If a[i] is written, then a[i][j]
             * depends on it. If a[i] is read, no conflict */
            add_conflict_p = action_write_p(effect_action(fin));
          } else if(dout < din) {
            /* same explanation as above */
            add_conflict_p = action_write_p(effect_action(fout));
          }
        }
      }

      if(add_conflict_p) {
	pips_debug(2, "add_conflicts_p is true; checking conflicts with loop indices\n");
        bool remove_this_conflict_p = false;
        if(!ein_abstract_location_p && store_effect_p(fin)) {
          /* Here we filter effect on loop indices except for abstract
           locations */
          list loops = load_statement_enclosing_loops(stout);
          FOREACH( statement, el, loops ) {
            entity il = loop_index(statement_loop(el));
	    pips_debug(2, "checking conflict with %s\n", entity_name(il));
            remove_this_conflict_p |= entities_may_conflict_p(ein, il);
	    pips_debug(2, "remove_this_conflict_ p = %s\n", bool_to_string(remove_this_conflict_p));
          }
        }
        if(!remove_this_conflict_p) {
          cs = pushnew_conflict(fin, fout, cs);
        }
      }
    }
  }

  /* Add conflicts */
  if(!ENDP( cs )) {
    statement stin = hash_get( effects2statement, fin );
    vin = vertex_statement( stin );

    /* The sink vertex in the graph */
    successor sout = successor_undefined;
    /* Try first to find an existing vertex for this statement */
    FOREACH( successor, s, vertex_successors( vin ) ) {
      if(successor_vertex(s) == vout) {
        sout = s;
        break;
      }
    }
    if(successor_undefined_p(sout)) {
      /* There is no sink vertex for this statement, create one */
      sout = make_successor(make_dg_arc_label(cs), vout);
      vertex_successors( vin )
          = CONS( SUCCESSOR, sout, vertex_successors( vin ));
    } else {
      /* Use existing vertex for this statement */
      gen_nconc(successor_arc_label(sout), cs);

    }
  }

}

/**
 * @brief Compute from a given statement, the dependency graph.
 * @description Statement s is assumed "controlized", i.e. GOTO have been
 * replaced by unstructured.
 *
 * FIXME FI: this function is bugged. As Pierre said, you have to start with
 * an unstructured for the use-def chain computation to be correct.
 */
graph statement_dependence_graph( statement s ) {
  /* Initialize some properties */
  one_trip_do_p = get_bool_property( "ONE_TRIP_DO" );
  keep_read_read_dependences_p = get_bool_property( "KEEP_READ_READ_DEPENDENCE" );
  mask_effects_p = get_bool_property("CHAINS_MASK_EFFECTS");
  dataflow_dependence_only_p = get_bool_property("CHAINS_DATAFLOW_DEPENDENCE_ONLY");

  /* Initialize global hashtables */
  effects2statement = hash_table_make( hash_pointer, 0 );
  Gen = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Ref = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Def_in = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Def_out = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Ref_in = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Ref_out = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Vertex_statement = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  current_defs = set_make(set_pointer);
  current_refs = set_make(set_pointer);

  /* Initialize the dg */
  dg = make_graph( NIL );

  /* Initialize data structures for all the statements

   It recursively initializes the sets of gens, ins and outs for
   the statements that appear in st. Note that not only call statements are
   there, but also enclosing statements (e.g, blocks and loops). */
  gen_recurse(s, statement_domain,
      init_one_statement,
      gen_identity);

  /* Compute genref phase */
  genref_statement( s );

  /* Compute inout phase and create conflicts*/
  inout_statement( s );

#define TABLE_FREE(t)							\
  {HASH_MAP( k, v, {set_free( (set)v ) ;}, t ) ; hash_table_free(t);}

  TABLE_FREE(Gen);
  TABLE_FREE(Ref);
  TABLE_FREE(Def_in);
  TABLE_FREE(Def_out);
  TABLE_FREE(Ref_in);
  TABLE_FREE(Ref_out);

  hash_table_free( Vertex_statement );
  hash_table_free( effects2statement );

  set_free(current_defs);
  set_free(current_refs);

  return ( dg );
}

/* functions for effects maps */
static bool rgch = false;
static bool iorgch = false;

static list load_statement_effects( statement s ) {
  instruction inst = statement_instruction( s );
  tag t = instruction_tag( inst );
  tag call_t;
  list le = NIL;

  switch ( t ) {
    case is_instruction_call:
      call_t
          = value_tag(entity_initial(call_function( instruction_call( inst ))));
      if ( iorgch && ( call_t == is_value_code ) ) {
        list l_in = load_statement_in_regions( s );
        list l_out = load_statement_out_regions( s );
        le = gen_append( l_in, l_out );
        break;
      }
      /* else, flow thru! */
    case is_instruction_expression:
      /* FI: I wonder about iorgch; I would expect the same kind of
       stuff for the case expression, which may also hide a call to
       a user defined function, for instance via a for loop
       construct. */
    case is_instruction_block:
    case is_instruction_test:
    case is_instruction_loop:
    case is_instruction_whileloop:
    case is_instruction_forloop:
    case is_instruction_unstructured:
      le = load_proper_rw_effects_list( s );
      break;
    case is_instruction_goto:
      pips_internal_error("Go to statement in CODE data structure %d", t );
      break;
    default:
      pips_internal_error("Unknown tag %d", t );
  }

  return le;
}

/* Select the type of effects used to compute dependence chains */
static void set_effects( char *module_name, enum chain_type use ) {
  switch ( use ) {

    case USE_PROPER_EFFECTS:
      rgch = false;
      iorgch = false;
      string pe =
          db_get_memory_resource( DBR_PROPER_EFFECTS, module_name, true );
      set_proper_rw_effects( (statement_effects) pe );
      break;

      /* In fact, we use proper regions only; proper regions of
       * calls are similar to regions, except for expressions given
       * as arguments, whose regions are simply appended to the list
       * (non convex hull).  For simple statements (assignments),
       * proper regions contain the list elementary regions (there
       * is no summarization, i.e no convex hull).  For loops and
       * tests, proper regions contain the elements accessed in the
       * tests and loop range. BC.
       */
    case USE_REGIONS:
      rgch = true;
      iorgch = false;
      string pr =
          db_get_memory_resource( DBR_PROPER_REGIONS, module_name, true );
      set_proper_rw_effects( (statement_effects) pr );
      break;

      /* For experimental purpose only */
    case USE_IN_OUT_REGIONS:
      rgch = false;
      iorgch = true;
      /* Proper regions */
      string iopr = db_get_memory_resource( DBR_PROPER_REGIONS,
                                            module_name,
                                            true );
      set_proper_rw_effects( (statement_effects) iopr );

      /* in regions */
      string ir = db_get_memory_resource( DBR_IN_REGIONS, module_name, true );
      set_in_effects( (statement_effects) ir );

      /* out regions */
      string or = db_get_memory_resource( DBR_OUT_REGIONS, module_name, true );
      set_out_effects( (statement_effects) or );
      break;

    default:
      pips_internal_error("ill. parameter use = %d", use );
  }

}

static void reset_effects() {
  reset_proper_rw_effects( );
  if ( iorgch ) {
    reset_in_effects( );
    reset_out_effects( );
  }
}

/**
 * @brief Compute chain dependence for a module according different kinds of
 * store-like effects.
 *
 * @param module_name is the name of the module we want to compute the chains
 * @param use the type of effects we want to use to compute the dependence
 * chains
 *
 * @return true because we are very confident it works :-)
 */
bool chains( char * module_name, enum chain_type use ) {
  statement module_stat;
  graph module_graph;
  void print_graph();

  set_current_module_statement( (statement) db_get_memory_resource( DBR_CODE,
                                                                    module_name,
                                                                    true ) );
  module_stat = get_current_module_statement( );
  set_current_module_entity( local_name_to_top_level_entity( module_name ) );
  /* set_entity_to_size(); should be performed at the workspace level */

  debug_on("CHAINS_DEBUG_LEVEL");

  pips_debug(1, "finding enclosing loops ...\n");
  set_enclosing_loops_map( loops_mapping_of_statement( module_stat ) );


  set_effects( module_name, use );
  set_conflict_testing_properties();

  module_graph = statement_dependence_graph( module_stat );

  ifdebug(2) {
    set_ordering_to_statement( module_stat );
    prettyprint_dependence_graph( stderr, module_stat, module_graph );
    reset_ordering_to_statement( );
  }

  debug_off();

  DB_PUT_MEMORY_RESOURCE(DBR_CHAINS, module_name, (char*) module_graph);

  reset_effects( );
  clean_enclosing_loops( );
  reset_current_module_statement( );
  reset_current_module_entity( );

  return true;
}

/**
 * @brief Phase to compute atomic chains based on proper effects (simple memory
 accesses)
 */
bool atomic_chains( char * module_name ) {
  return chains( module_name, USE_PROPER_EFFECTS );
}

/**
 * @brief Phase to compute atomic chains based on array regions
 */
bool region_chains( char * module_name ) {
  return chains( module_name, USE_REGIONS );
}

/**
 * @brief Phase to compute atomic chains based on in-out array regions
 */
bool in_out_regions_chains( char * module_name ) {
  return chains( module_name, USE_IN_OUT_REGIONS );
}
