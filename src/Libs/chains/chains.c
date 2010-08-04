/*

 $Id$

 Copyright 1989-2010 MINES ParisTech

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
static void usedef_control();
static void genkill_statement();

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

/* Kill maps each statement to the effects it kills. */
static hash_table Kill;
#define KILL(st) ((set)hash_get( Kill, (char *)st ))

/* Def_in maps each statement to the statements that are in-coming the statement */
static hash_table Def_in;
#define DEF_IN(st) ((set)hash_get(Def_in, (char *)st))

/* Def_out maps each statement to the statements that are out-coming the statement */
static hash_table Def_out;
#define DEF_OUT(st) ((set)hash_get(Def_out, (char *)st))

/* Ref_in maps each statement to the effects that are in-coming the
 statement */
static hash_table Ref_in;
#define REF_IN(st) ((set)hash_get(Ref_in, (char *)st))

/* Ref_out maps each statement to the effects that are out-coming the statement */
static hash_table Ref_out;
#define REF_OUT(st) ((set)hash_get(Ref_out, (char *)st))

/* dg is the dependency graph ; FIXME : should not be static global ? */
static graph dg;

/* Vertex_statement maps each statement to its vertex in the dependency graph. */
static hash_table Vertex_statement;

static bool one_trip_do;
static bool keep_read_read_dependences;

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
 * @return always TRUE (gen_recurse continuation)
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

    /* If the statement has never been seen, allocate new sets: */
    hash_put( Gen, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Ref, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Def_in, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Def_out, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Ref_in, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Ref_out, (char *) st, (char *) MAKE_STATEMENT_SET() );
    hash_put( Kill, (char *) st, (char *) MAKE_STATEMENT_SET() );

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
    set_clear( KILL( st ) );
    set_clear( DEF_OUT( st ) );
    set_clear( DEF_IN( st ) );
    set_clear( REF_OUT( st ) );
    set_clear( REF_IN( st ) );
  }
  /* Go on recursing down: */
  return TRUE;
}

/* The GENKILL_xxx functions implement the computation of GEN, REF and 
 KILL sets
 from Aho, Sethi and Ullman "Compilers" (p. 612). This is slightly
 more complex since we use a structured control graph, thus fixed
 point computations can be recursively required (the correctness of
 this is probable, although not proven, as far as I can tell). */

/* KILL_STATEMENT updates the KILL set of statement ST on entity LHS. Only
 effects that modify one reference (i.e., assignments) are killed (see
 above). Write effects on arrays (see IDXS) don't kill the definitions of
 the array. Equivalence with non-array entities is managed properly. */

/**
 * @brief Update the set of effects that an effect might kill
 * @param kill the set to initialize
 * @param e the "killer" effect
 */
static void kill_effect( set kill, effect e ) {
  if ( action_write_p(effect_action(e))
      && approximation_must_p(effect_approximation(e)) ) {
    HASH_MAP(theEffect,theStatement, {
          /* We avoid a self killing */
          if( e != theEffect ) {
            /* Check if there is a must conflict
             * Avoid using effects_must_conflict_p() because we want to be
             * able to kill "may" effects.
             */
            cell cell1 = effect_cell(e);
            cell cell2 = effect_cell((effect)theEffect);
            /* Check that the cells conflicts */
            if ( cells_must_conflict_p( cell1, cell2 ) ) {
              set_add_element( kill, kill, theEffect );
            }
          }
        },effects2statement)
  }
}

/**
 * @brief Compute Gen, Ref, and Kill set for a single statement
 * @param st the statement to compute
 */
static void genkill_one_statement( statement st ) {
  set gen = GEN( st );
  set ref = REF( st );
  set kill = KILL( st );
  set_clear( gen );
  set_clear( ref );
  set_clear( kill );
  /* Loop over effects to find which one will kill some others, generate
   * or reference some variables
   */
  FOREACH( effect, e, load_statement_effects(st) )
  {
    action a = effect_action( e );

    if ( action_write_p( a ) ) {
      /* This effects is a write, it may kill some others effects ! */
      pips_assert("Effect isn't map to this statement !",
          st == hash_get(effects2statement, e));
      kill_effect( kill, e );

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
 * @brief Compute Gen, Ref, and Kill set for a test
 * @param t the test
 * @param s the statement to compute
 */
static void genkill_test( test t, statement s ) {
  statement st = test_true(t);
  statement sf = test_false(t);
  set ref = REF( s );

  /* compute true path */
  genkill_statement( st );
  /* compute false path */
  genkill_statement( sf );

  /* Combine the two path to summarize the test */
  set_union( GEN( s ), GEN( st ), GEN( sf ) );
  set_union( ref, ref, REF( sf ) );
  set_union( ref, ref, REF( st ) );
  set_intersection( KILL( s ), KILL( st ), KILL( sf ) );
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
        if ( effect_may_conflict_with_entity_p( f, e ) ) {
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
 * @brief Compute Gen, Ref, and Kill set for any loop (do, for, while,...)
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
static void genkill_any_loop( statement body,
                              statement st,
                              list locals,
                              bool one_trip_do ) {
  set gen = GEN( st );
  set ref = REF( st );

  /* Compute genkill on the loop body */
  genkill_statement( body );

  /* Summarize the body to the statement that hold the loop */
  set_union( gen, gen, GEN( body ) );
  set_union( ref, ref, REF( body ) );

  /* This is used only for do loop */
  if ( one_trip_do ) {
    /* If we assume the loop is always done at least one time, we can use
     * kill information from loop body.
     */
    set_union( KILL( st ), KILL( st ), KILL( body ) );
  }

  /* Filter effects on local variables */
  if ( get_bool_property( "CHAINS_MASK_EFFECTS" ) ) {
    mask_effects( gen, locals );
    mask_effects( ref, locals );
  }

}

/**
 * @brief Compute Gen, Ref, and Kill set for a "do" loop
 * @description see genkill_any_loop()
 *
 * @param l the loop to compute
 * @param st the statement that hold the loop
 */
static void genkill_loop( loop l, statement st ) {
  statement body = loop_body( l );

  /* Building locals list */
  list llocals = loop_locals( l );
  list slocals = statement_declarations(st);
  list locals = gen_nconc( gen_copy_seq( llocals ), gen_copy_seq( slocals ) );

  /* Call the generic function handling all kind of loop */
  genkill_any_loop( body, st, locals, one_trip_do );

  /* We free because we are good programmers and we don't leak ;-) */
  gen_free_list( locals );
}

/**
 * @brief Compute Gen, Ref, and Kill set for a "for" loop
 * @description see genkill_any_loop()
 *
 * @param l the loop to compute
 * @param st the statement that hold the loop
 *
 */
static void genkill_forloop( forloop l, statement st ) {
  statement body = forloop_body( l );
  list locals = statement_declarations(st);

  /* Call the generic function handling all kind of loop */
  genkill_any_loop( body, st, locals, one_trip_do );

}

/**
 * @brief Compute Gen, Ref, and Kill set for a "while" loop
 * @description see genkill_any_loop()
 *
 * @param l the loop to compute
 * @param st the statement that hold the loop
 */
static void genkill_whileloop( whileloop l, statement st ) {
  statement body = whileloop_body( l );
  list locals = statement_declarations(st);

  /* Call the generic function handling all kind of loop */
  genkill_any_loop( body, st, locals, one_trip_do );
}

/**
 * @brief Compute Gen, Ref, and Kill set for a block
 * @description The Dragon book only deals with a sequence of two statements.
 * here we generalize to lists, via recursion. Statement are processed in
 * reversed order (i.e. on the descending phase of recursion)
 *
 * @param sts the list of statements inside the block
 * @param st the statement that hold the block
 *
 */
static void genkill_block( cons *sts, statement st ) {
  statement one;

  /* If we are not at the end of the block */
  if ( !ENDP( sts ) ) {
    /* Recurse till the end of the block */
    genkill_block( CDR( sts ), st );

    /* End of recursion, process current statement */
    genkill_statement( one = STATEMENT( CAR( sts )) );

    /* Summarize for the block */
    set diff = MAKE_STATEMENT_SET();
    set gen = MAKE_STATEMENT_SET();
    set ref = MAKE_STATEMENT_SET();
    set kill_st = KILL( st );
    set gen_st = GEN( st );
    set ref_st = REF( st );

    set_difference( diff, GEN( one ), kill_st );
    set_union( gen, gen_st, diff );
    set_difference( diff, KILL( one ), gen_st );
    set_union( kill_st, kill_st, diff );
    set_assign( gen_st, gen );
    set_union( ref_st, REF( one ), ref_st );

    /* Memory freeing */
    set_free( diff );
    set_free( gen );
    set_free( ref );

    /* FIXME : This should be done after all recursion for performance... */
    if ( get_bool_property( "CHAINS_MASK_EFFECTS" ) ) {
      mask_effects( gen_st, statement_declarations(st) );
      mask_effects( ref_st, statement_declarations(st) );
    }

  }
}

/**
 * @brief Compute Gen, Ref, and Kill set for an unstructured
 * @description computes the gens, refs, and kills set of the unstructured by
 * recursing on INOUT_CONTROL. The gens&refs can then be inferred.
 * The situation for the kills is more fishy; for the moment, we just keep
 * the kills that are common to all statements of the control graph.
 *
 * @param u is the unstructured
 * @param st is the statement that hold the unstructured
 */
static void genkill_unstructured( unstructured u, statement st ) {
  control c = unstructured_control( u );
  statement exit = control_statement( unstructured_exit( u ));
  set kill = MAKE_STATEMENT_SET();
  cons *blocs = NIL;

  set_clear( DEF_IN( st ) );
  /* recursing
   * FIXME : Mehdi : it seems to me not to be the right way to call an
   * inout function here. Is it a hack ?
   * We should do a more straight recursion. Something like :
   CONTROL_MAP( cc, {statement st = control_statement( cc );
   genkill_statement( st );
   }, c, blocs );
   */
  inout_control( c );

  /* Kill set will be the intersection of all kill inside the unstructured
   * FIXME : kill is initially empty, so the intersection will ALWAYS be empty !
   */
  CONTROL_MAP( cc, {
        set_intersection( kill, kill, KILL( control_statement( cc )));
      }, c, blocs );

  /* FIXME : why not use directly KILL( st ) instead of using a temporary set ? */
  set_assign( KILL( st ), kill );
  set_free( kill );

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
static void genkill_instruction( instruction i, statement st ) {
  cons *pc;
  test t;
  loop l;

  switch ( instruction_tag(i) ) {
    case is_instruction_block:
      genkill_block( pc = instruction_block(i), st );
      break;
    case is_instruction_test:
      t = instruction_test(i);
      genkill_test( t, st );
      break;
    case is_instruction_loop:
      l = instruction_loop(i);
      genkill_loop( l, st );
      break;
    case is_instruction_whileloop: {
      whileloop l = instruction_whileloop(i);
      genkill_whileloop( l, st );
      break;
    }
    case is_instruction_forloop: {
      forloop l = instruction_forloop(i);
      genkill_forloop( l, st );
      break;
    }
    case is_instruction_unstructured:
      genkill_unstructured( instruction_unstructured( i ), st );
      break;
    case is_instruction_call:
    case is_instruction_expression:
    case is_instruction_goto:
      /* no recursion for these kind of instruction */
      break;
    default:
      pips_internal_error("unexpected tag %d\n", instruction_tag(i));
  }
}

/**
 * @brief Compute recursively Gen, Ref, and Kill set for a statement.
 *  This is the main entry to this task.
 * @description computes the gens, refs, and kills set of the statement by
 * recursing
 *
 * @param s is the statement to compute
 */
static void genkill_statement( statement s ) {

  /* Compute genkill using effects associated to the statement itself */
  genkill_one_statement( s );

  ifdebug(2) {
    debug( 2,
           "genkill_statement",
           "Result genkill_one_statement for Statement %p [%s]:\n",
           s,
           statement_identification( s ) );
    local_print_statement_set( "GEN", GEN( s ) );
    local_print_statement_set( "REF", REF( s ) );
    local_print_statement_set( "KILL", KILL( s ) );
  }

  /* Recurse inside the statement (relevant for blocks, loop, ...) */
  genkill_instruction( statement_instruction(s), s );

  ifdebug(2) {
    debug( 2,
           "genkill_statement",
           "Result for Statement %p [%s]:\n",
           s,
           statement_identification( s ) );
    ;
    local_print_statement_set( "GEN", GEN( s ) );
    local_print_statement_set( "REF", REF( s ) );
    local_print_statement_set( "KILL", KILL( s ) );
  }
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the block
 * @param sts is the list of statement inside the block
 */
static void inout_block( statement st, cons *sts ) {
  /* The initial in for the block */
  set def_in = DEF_IN( st );
  set ref_in = REF_IN( st );

  ifdebug(1) {
    mem_spy_begin( );
  }
  /* loop over statements inside the block */
  FOREACH( statement, one, sts ) {
    /* propagate "in" sets */
    set_assign( DEF_IN( one ), def_in );
    set_assign( REF_IN( one ), ref_in );

    /* Compute the outs from the ins for this statement */
    inout_statement( one );

    /* Get the ins for next statement */
    def_in = DEF_OUT( one );
    set_union( ref_in, ref_in, REF_OUT( one ) );
  }

  /* The outs for the whole block */
  set_assign( DEF_OUT( st ), def_in );
  set_assign( REF_OUT( st ), ref_in );

  ifdebug(1) {
    mem_spy_end( "inout_block" );
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

  ifdebug(1) {
    mem_spy_begin( );
  }

  /*
   * Compute true path
   */
  /* init "in" sets for this path */
  set_assign( DEF_IN( st ), DEF_IN( s ) );
  set_assign( REF_IN( st ), REF_IN( s ) );
  /* Compute the path */
  inout_statement( st );

  /*
   * Compute false path
   */
  /* init "in" sets for this path */
  set_assign( DEF_IN( sf ), DEF_IN( s ) );
  set_assign( REF_IN( sf ), REF_IN( s ) );
  /* Compute the path */
  inout_statement( sf );

  /* Compute the out for the test */
  set_union( DEF_OUT( s ), DEF_OUT( st ), DEF_OUT( sf ) );
  set_union( REF_OUT( s ), REF_OUT( st ), REF_OUT( sf ) );

  ifdebug(1) {
    mem_spy_end( "inout_test" );
  }
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the loop
 * @param lo is the loop
 * @param one_trip_do tell if there is always at least one trip (do loop only)
 */
static void inout_any_loop( statement st, statement body, bool one_trip_do ) {
  set diff = MAKE_STATEMENT_SET();

  ifdebug(1) {
    mem_spy_begin( );
  }

  /* Compute "in" sets for the loop body
   * FIXME : diff temporary can be avoided ! */
  set_union( DEF_IN( body ), GEN( st ), set_difference( diff,
                                                        DEF_IN( st ),
                                                        KILL( st ) ) );
  set_free( diff );
  set_union( REF_IN( body ), REF_IN( st ), REF( st ) );

  /* Compute loop body */
  inout_statement( body );

  /* Compute "out" sets for the loop */
  if ( one_trip_do ) {
    /* Body is always done at least one time */
    set_assign( DEF_OUT( st ), DEF_OUT( body ) );
    set_assign( REF_OUT( st ), REF_OUT( body ) );
  } else {
    /* Body is not always done */
    set_union( DEF_OUT( st ), DEF_OUT( body ), DEF_IN( st ) );
    set_union( REF_OUT( st ), REF_OUT( body ), REF_IN( st ) );
  }

  ifdebug(1) {
    mem_spy_end( "inout_loop" );
  }
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the loop
 * @param lo is the loop
 */
static void inout_loop( statement st, loop lo ) {
  statement body = loop_body( lo );
  inout_any_loop( st, body, one_trip_do );
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the loop
 * @param t is the loop
 */
static void inout_whileloop( statement st, whileloop wl ) {
  statement body = whileloop_body( wl );
  inout_any_loop( st, body, FALSE );
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the loop
 * @param t is the loop
 */
static void inout_forloop( statement st, forloop fl ) {
  statement body = forloop_body( fl );
  inout_any_loop( st, body, one_trip_do );
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement that hold the call
 * @param c is the call (unused)
 */
static void inout_call( statement st, call __attribute__ ((unused)) c ) {
  set diff = MAKE_STATEMENT_SET();

  ifdebug(1) {
    mem_spy_begin( );
  }

  /* Compute "out" sets
   * FIXME diff temporary set can be avoided */
  set_union( DEF_OUT( st ), GEN( st ), set_difference( diff,
                                                       DEF_IN( st ),
                                                       KILL( st ) ) );
  set_union( REF_OUT( st ), REF_IN( st ), REF( st ) );
  set_free( diff );

  ifdebug(1) {
    mem_spy_end( "inout_call" );
  }
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

  ifdebug(1) {
    mem_spy_begin( );
  }

  /* Compute "in" sets */
  set_assign( DEF_IN( control_statement( c )), DEF_IN( st ) );
  set_assign( REF_IN( control_statement( c )), REF_IN( st ) );

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

  ifdebug(1) {
    mem_spy_end( "inout_unstructured" );
  }
}

/**
 * @brief Propagates in sets of ST (which is inherited) to compute the out sets.
 *
 * @param st is the statement to compute
 */
static void inout_statement( statement st ) {
  instruction i;
  static int indent = 0;

  /*
   ifdebug(1) {
   mem_spy_begin();
   }
   */

  ifdebug(2) {
    fprintf( stderr,
             "%*s> Computing DEF_IN and OUT of statement %p (%td %td):\n",
             indent++,
             "",
             st,
             statement_number( st ),
             statement_ordering( st ) );
    local_print_statement_set( "DEF_IN", DEF_IN( st ) );
    local_print_statement_set( "DEF_OUT", DEF_OUT( st ) );
    local_print_statement_set( "REF_IN", REF_IN( st ) );
    local_print_statement_set( "REF_OUT", REF_OUT( st ) );
  }

  /* Compute defaults "out" sets ; will be overwrite most of the time
   * FIXME : is it really useful ?? */
  set_assign( DEF_OUT( st ), GEN( st ) );
  set_assign( REF_OUT( st ), REF( st ) );

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
      pips_error( "inout_statement", "Unexpected tag %d\n", i );
      break;
    case is_instruction_unstructured:
      inout_unstructured( st, instruction_unstructured( i ) );
      break;
    default:
      pips_internal_error("Unknown tag %d\n", instruction_tag(i) );
  }
  ifdebug(2) {
    fprintf( stderr,
             "%*s> Statement %p (%td %td):\n",
             indent--,
             "",
             st,
             statement_number( st ),
             statement_ordering( st ) );
    local_print_statement_set( "DEF_IN", DEF_IN( st ) );
    local_print_statement_set( "DEF_OUT", DEF_OUT( st ) );
    local_print_statement_set( "REF_IN", REF_IN( st ) );
    local_print_statement_set( "REF_OUT", REF_OUT( st ) );
  }

  /*
   ifdebug(1) {
   mem_spy_end("inout_statement");
   }
   */
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
  set diff = MAKE_STATEMENT_SET();
  cons *blocs = NIL;

  ifdebug(1) {
    mem_spy_begin( );
    mem_spy_begin( );
  }

  ifdebug(2) {
    fprintf( stderr, "Computing DEF_IN and OUT of control %p entering", ct );
    local_print_statement_set( "", DEF_IN( control_statement( ct )) );
  }

  CONTROL_MAP( c, {statement st = control_statement( c );

        /* FIXME : I don't think genkill should be called inside inout phase ! */
        genkill_statement( st );
        set_assign( DEF_OUT( st ), GEN( st ));
        set_assign( REF_OUT( st ), REF( st ));

        if( c != ct ) {
          set_clear( DEF_IN( st ));
          set_clear( REF_OUT( st ));
        }},
      ct, blocs );

  ifdebug(1) {
    mem_spy_end( "inout_control: phase 1" );
    mem_spy_begin( );
  }

  for ( change = TRUE; change; ) {
    ifdebug(3) {
      fprintf( stderr, "Iterating on %p ...\n", ct );
    }
    change = FALSE;

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
          set_union( DEF_OUT( st ), GEN( st ),
              set_difference( diff, DEF_IN( st ), KILL( st )));
          set_assign( r_oldout, REF_OUT( st ));
          set_union( REF_OUT( st ), REF( st ), REF_IN( st ));
          change |= (!set_equal_p( d_oldout, DEF_OUT( st )) ||
              !set_equal_p( r_oldout, REF_OUT( st )));
        }, ct, blocs );
  }

  ifdebug(1) {
    mem_spy_end( "inout_control: phase 2 (fix-point)" );
    mem_spy_begin( );
  }
  CONTROL_MAP( c, {inout_statement( control_statement( c ));},
      ct, blocs );
  set_free( d_oldout );
  set_free( d_out );
  set_free( r_oldout );
  set_free( r_out );
  set_free( diff );
  gen_free_list( blocs );

  ifdebug(1) {
    mem_spy_end( "inout_control: phase 3" );
    mem_spy_end( "inout_control" );
  }
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

  if ( get_bool_property( "CHAINS_DATAFLOW_DEPENDENCE_ONLY" ) ) {
    conflict_p = conflict_p && action_read_p( effect_action( fout ) );
  }

  return conflict_p;
}

/**
 * @brief UD detects Use/Def conflicts between effects FIN and FOUT.
 */
static bool ud( effect fin, effect fout ) {
  return ( action_read_p( effect_action( fin ))
      && ( action_write_p( effect_action( fout )) || keep_read_read_dependences ) );
}

/**
 * @brief adds conflict arcs to the dependence graph dg between the in-coming
 * statement STIN and the out-going STOUT.
 * @description Note that output dependencies are not minimal
 * (e.g., i = s ; s = ... ; s = ...) creates an oo-dep between the i assignment
 * and the last s assignment.
 */
static void add_conflicts( effect fin, statement stout, bool(*which)() ) {
  vertex vin;
  vertex vout = vertex_statement( stout );
  statement stin = hash_get( effects2statement, fin );
  cons *effect_outs = load_statement_effects( stout );
  cons *cs = NIL;

  ifdebug(1) {
    mem_spy_begin( );
  }

  ifdebug(2) {
    _int stin_o = statement_ordering(stin);
    _int stout_o = statement_ordering(stout);
    fprintf( stderr,
	     "Conflicts %td (%td,%td) (%p) -> %td (%td,%td) (%p) %s\n",
	     statement_number(stin),
	     ORDERING_NUMBER(stin_o),
	     ORDERING_STATEMENT(stin_o),
	     stin,
	     statement_number(stout),
	     ORDERING_NUMBER(stout_o),
	     ORDERING_STATEMENT(stout_o),
	     stout,
	     ( which == ud ) ? "ud" : "dd_du" );
  }
  vin = vertex_statement( stin );

  /* To speed up pushnew_conflict() without damaging the integrity of
   the use-def chains */
  ifdebug(1) {
    cons *effect_ins = load_statement_effects( stin );
    if ( !gen_once_p( effect_ins ) ) {
      pips_internal_error("effect_ins are redundant\n");
    }
    if ( !gen_once_p( effect_outs ) ) {
      pips_internal_error("effect_outs are redundant\n");
    }
  }
  FOREACH(EFFECT, fout, effect_outs) {
    ifdebug(2) {
      print_effect(fin);
      fprintf(stderr," -> ");
      print_effect(fout);
      fprintf(stderr,"\n");
    }

    if ( effects_may_conflict_p( fin, fout ) && ( *which )( fin, fout ) ) {
      entity ein = effect_entity( fin );
      entity eout = effect_entity( fout );
      type tin = ultimate_type( entity_type(ein) );
      type tout = ultimate_type( entity_type(eout) );
      reference rin = effect_any_reference(fin);
      int din = gen_length( reference_indices(rin) );
      reference rout = effect_any_reference(fout);
      int dout = gen_length( reference_indices(rout) );
      bool add_conflict_p = TRUE;

      if ( pointer_type_p( tin ) && pointer_type_p( tout ) ) {

	/* Second version due to accuracy improvements in effect
	   computation */
	if ( din == dout ) {
	  /* This is the standard case */
	  add_conflict_p = TRUE;
	} else if ( din < dout ) {
	  /* a write on the shorter memory access path conflicts
           with the longer one. If a[i] is written, then a[i][j]
           depends on it. If a[i] is read, no conflict */
	  add_conflict_p = action_write_p(effect_action(fin));
	} else /* dout < din */{
	  /* same explanation as above */
	  add_conflict_p = action_write_p(effect_action(fout));
	}
      } else {
	/* Why should we limit this test to pointers? Structures,
	   structures of arrays and arrays of structures with
	   pointers embedded somewhere must behave in the very same
	   way. Why not unify the two cases? Because we have not
	   spent enough time thinking about memory access paths. */
	if ( din < dout ) {
	  /* a write on the shorter memory access path conflicts
	     with the longer one. If a[i] is written, then a[i][j]
	     depends on it. If a[i] is read, no conflict */
	  add_conflict_p = action_write_p(effect_action(fin));
	} else if ( dout < din ) {
	  /* same explanation as above */
	  add_conflict_p = action_write_p(effect_action(fout));
	}
      }

      if ( add_conflict_p ) {
	bool remove_this_conflict_p = FALSE;

	/* Here we filter effect on loop indices */
	list loops = load_statement_enclosing_loops( stout );
	FOREACH( statement, el, loops ) {
	  entity il = loop_index(statement_loop(el));
	  remove_this_conflict_p |= entities_may_conflict_p( ein, il );
	}

	if ( !remove_this_conflict_p )
	  cs = pushnew_conflict( fin, fout, cs );
      }
    }
  }

  /* Add conflicts */
  if ( !ENDP( cs ) ) {

    /* The sink vertex in the graph */
    successor sout = successor_undefined;
    /* Try first to find an existing vertex for this statement */
    FOREACH( successor, s, vertex_successors( vin ) ) {
      if ( successor_vertex(s) == vout ) {
	sout = s;
	break;
      }
    }
    if ( successor_undefined_p(sout) ) {
      /* There is no sink vertex for this statement, create one */
      sout = make_successor( make_dg_arc_label( cs ), vout );
      vertex_successors( vin )
	= CONS( SUCCESSOR, sout, vertex_successors( vin ));
    } else {
      /* Use existing vertex for this statement */
      gen_nconc( successor_arc_label(sout), cs );

    }
  }

  ifdebug(1) {
    mem_spy_end( "add_conflicts" );
  }
}

/**
 * @brief Updates the dependence graph between the statement ST and
 *  its "in" sets. Only calls & expression are taken into account.
 */
static void usedef_statement( st )
  statement st; {

  instruction i = statement_instruction( st );
  tag t;
  /* Compute Def-Def and Def-Use conflicts from Def_in set */
  SET_FOREACH( effect, defIn, DEF_IN( st )) {
    add_conflicts( defIn, st, dd_du );
  }
  /* Compute Use-Def conflicts from Ref_in set */
  SET_FOREACH( effect, refIn, REF_IN( st )) {
    add_conflicts( refIn, st, ud );
  }

  /* Recurse on instruction inside statement */
  switch ( t = instruction_tag( i ) ) {
    case is_instruction_block:
      MAPL( sts, {usedef_statement( STATEMENT( CAR( sts )));},
          instruction_block( i ))
      ;
      break;
    case is_instruction_test:
      usedef_statement( test_true( instruction_test( i )) );
      usedef_statement( test_false( instruction_test( i )) );
      break;
    case is_instruction_loop:
      usedef_statement( loop_body( instruction_loop( i )) );
      break;
    case is_instruction_whileloop:
      usedef_statement( whileloop_body( instruction_whileloop( i )) );
      break;
    case is_instruction_forloop:
      usedef_statement( forloop_body( instruction_forloop( i )) );
      break;
    case is_instruction_call:
    case is_instruction_expression:
    case is_instruction_goto: // should lead to a core dump
      break;
    case is_instruction_unstructured:
      usedef_control( unstructured_control( instruction_unstructured( i )) );
      break;
    default:
      pips_error( "usedef_statement", "Unknown tag %d\n", t );
  }
}

/**
 *  @brief USEDEF_CONTROL updates the dependence graph between each of the
 *  nodes of C and their "in" sets.
 *  Only calls & expression are taken into account.
 */
static void usedef_control( c )
  control c; {
  cons *blocs = NIL;

  ifdebug(1) {
    mem_spy_begin( );
  }

  CONTROL_MAP( n, {usedef_statement( control_statement( n ));},
      c, blocs );
  gen_free_list( blocs );

  ifdebug(1) {
    mem_spy_end( "usedef_control" );
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
  one_trip_do = get_bool_property( "ONE_TRIP_DO" );
  keep_read_read_dependences = get_bool_property( "KEEP_READ_READ_DEPENDENCE" );

  /* FIXME
   * OBSOLETE
   * disambiguate_constant_subscripts
   *   = get_bool_property( "CHAINS_DISAMBIGUATE_CONSTANT_SUBSCRIPTS" );
   */

  ifdebug(1) {
    mem_spy_begin( );
    mem_spy_begin( );
  }

  /* Initialize global hashtables */
  effects2statement = hash_table_make( hash_pointer, 0 );
  Gen = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Ref = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Kill = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Def_in = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Def_out = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Ref_in = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Ref_out = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );
  Vertex_statement = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE );

  /* Initialize the dg */
  dg = make_graph( NIL );

  /* Initialize data structures for all the statements

   It recursively initializes the sets of gens, kills, ins and outs for
   the statements that appear in st. Note that not only call statements are
   there, but also enclosing statements (e.g, blocks and loops). */
  gen_recurse(s, statement_domain,
      init_one_statement,
      gen_identity);

  /* Compute genkill phase */
  genkill_statement( s );

  /* Compute inout phase */
  inout_statement( s );

  /* Compute usedef phase */
  usedef_statement( s );

#define TABLE_FREE(t)							\
  {HASH_MAP( k, v, {set_free( (set)v ) ;}, t ) ; hash_table_free(t);}

  ifdebug(1) {
    mem_spy_end( "dependence_graph: after computation" );
    mem_spy_begin( );
  }

  TABLE_FREE(Gen);
  TABLE_FREE(Ref);
  TABLE_FREE(Kill);
  TABLE_FREE(Def_in);
  TABLE_FREE(Def_out);
  TABLE_FREE(Ref_in);
  TABLE_FREE(Ref_out);

  hash_table_free( Vertex_statement );
  hash_table_free( effects2statement );

  ifdebug(1) {
    mem_spy_end( "dependence graph: after freeing the hash tables" );
    mem_spy_end( "dependence_graph" );
  }

  return ( dg );
}

/* functions for effects maps */
static bool rgch = FALSE;
static bool iorgch = FALSE;

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
      pips_internal_error( "Go to statement in CODE data structure %d\n", t );
      break;
    default:
      pips_internal_error( "Unknown tag %d\n", t );
  }

  return le;
}

/* Select the type of effects used to compute dependence chains */
static void set_effects( char *module_name, enum chain_type use ) {
  switch ( use ) {

    case USE_PROPER_EFFECTS:
      rgch = FALSE;
      iorgch = FALSE;
      string pe =
          db_get_memory_resource( DBR_PROPER_EFFECTS, module_name, TRUE );
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
      rgch = TRUE;
      iorgch = FALSE;
      string pr =
          db_get_memory_resource( DBR_PROPER_REGIONS, module_name, TRUE );
      set_proper_rw_effects( (statement_effects) pr );
      break;

      /* For experimental purpose only */
    case USE_IN_OUT_REGIONS:
      rgch = FALSE;
      iorgch = TRUE;
      /* Proper regions */
      string iopr = db_get_memory_resource( DBR_PROPER_REGIONS,
                                            module_name,
                                            TRUE );
      set_proper_rw_effects( (statement_effects) iopr );

      /* in regions */
      string ir = db_get_memory_resource( DBR_IN_REGIONS, module_name, TRUE );
      set_in_effects( (statement_effects) ir );

      /* out regions */
      string or = db_get_memory_resource( DBR_OUT_REGIONS, module_name, TRUE );
      set_out_effects( (statement_effects) or );
      break;

    default:
      pips_error( "set_effects", "ill. parameter use = %d\n", use );
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
 * @return TRUE because we are very confident it works :-)
 */
bool chains( char * module_name, enum chain_type use ) {
  statement module_stat;
  instruction module_inst;
  graph module_graph;
  void print_graph();

  debug_on("CHAINS_DEBUG_LEVEL");

  ifdebug(1) {
    mem_spy_init( 0, 100000., NET_MEASURE, 0 );
    mem_spy_begin( );
    mem_spy_begin( );
  }

  debug_off();

  set_current_module_statement( (statement) db_get_memory_resource( DBR_CODE,
                                                                    module_name,
                                                                    TRUE ) );
  module_stat = get_current_module_statement( );
  set_current_module_entity( local_name_to_top_level_entity( module_name ) );
  /* set_entity_to_size(); should be performed at the workspace level */

  debug_on("CHAINS_DEBUG_LEVEL");

  ifdebug(1) {
    mem_spy_end( "Chains: resources loaded" );
    mem_spy_begin( );
  }

  pips_debug(1, "finding enclosing loops ...\n");
  set_enclosing_loops_map( loops_mapping_of_statement( module_stat ) );

  module_inst = statement_instruction(module_stat);

  set_effects( module_name, use );

  ifdebug(1) {
    mem_spy_end( "Chains: Preamble" );
    mem_spy_begin( );
  }

  module_graph = statement_dependence_graph( module_stat );

  ifdebug(1) {
    mem_spy_end( "Chains: Computation" );
    mem_spy_begin( );
  }

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

  debug_on("CHAINS_DEBUG_LEVEL");
  ifdebug(1) {
    mem_spy_end( "Chains: Deallocation" );
    mem_spy_end( "Chains" );
    mem_spy_reset( );
  }
  debug_off();

  return TRUE;
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
