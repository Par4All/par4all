/* -- usedef.c

   Computes (approximate) use/def chains of a structured control graph.

   - If a statement has more than one def (e.g., call statements), then
     it is never killed. This is a conservative assumption, but doesn't
     catch killings like:
     
     i,j = 3 ; if foo then i=1 ; j=2 else i=3 ; j=4 ; endif.

     where after the test, the (i,j) assignment is still alive. 

   - Usedef chains are only built on assignments (would be straightforward
     to adapt). */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "text.h"
#include "database.h"

typedef void * arc_label;
typedef void * vertex_label;

#include "graph.h"
#include "dg.h"

#include "misc.h"
#include "properties.h"

#include "ri-util.h"
#include "control.h"
#include "prettyprint.h"

/* #include "ricedg.h" */
extern void prettyprint_dependence_graph(FILE*, statement, graph);

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "chains.h"
#include "pipsdbm.h"

#include "constants.h"
#include "resources.h"

#define MAKE_STATEMENT_SET() (set_make( set_pointer ))

/* Default sizes for corresponding sets. This is automatically adjusted
   according to needs by the hash table package. */

#define INIT_STATEMENT_SIZE 20
#define INIT_ENTITY_SIZE 10


static hash_table Gen ;			/* Gen maps each statement to the 
					   statements it generates. */

#define GEN(st) ((set)hash_get( Gen, (char *)st ))

static hash_table Ref ;			/* Refs maps each statement to the 
					   statements it references. */

#define REF(st) ((set)hash_get( Ref, (char *)st ))

static hash_table Kill ;		/* Kill maps each statement to the 
					   statements it kills. */

#define KILL(st) ((set)hash_get( Kill, (char *)st ))

static hash_table Defs ;		/* Defs maps each entity to the set of 
					   statements it is defined by. */

#define DEFS(e) ((set)hash_get( Defs, (char *)e ))

static hash_table Def_in ;		/* Def_in maps each statement to the 
					   statements that are in-coming the 
					   statement */

#define DEF_IN(st) ((set)hash_get(Def_in, (char *)st))

static hash_table Def_out ;		/* Def_out maps each statement to the 
					   statements that are out-coming 
					   the statement */ 

#define DEF_OUT(st) ((set)hash_get(Def_out, (char *)st))

static hash_table Ref_in ;		/* Ref_in maps each statement to the 
					   statements that are in-coming the 
					   statement */

#define REF_IN(st) ((set)hash_get(Ref_in, (char *)st))

static hash_table Ref_out ;		/* Ref_out maps each statement to the 
					   statements that are out-coming 
					   the statement */ 

#define REF_OUT(st) ((set)hash_get(Ref_out, (char *)st))

static graph dg ;			/* dg is the dependency graph */

static hash_table Vertex_statement ;	/* Vertex_statement maps each 
					   statemen to its vertex in the 
					   dependency graph. */

static bool one_trip_do ;
static bool keep_read_read_dependences ;
static bool disambiguate_constant_subscripts ;

static void set_effects(char *module_name, int use);
static void reset_effects();
static list load_statement_effects(statement st);

static void local_print_statement_set(char *msg, set s);

/* VERTEX_STATEMENT returns the vertex associated to the statement ST in the
   dependence graph DG. */

static vertex vertex_statement( st ) 
statement st ;
{
    vertex v = (vertex)hash_get(Vertex_statement,(char *)st) ;

    pips_assert( "vertex_statement", v != (vertex)HASH_UNDEFINED_VALUE ) ;
    return( v ) ;
}

static void add_entity_to_defs( e, st )
entity e ;
statement st ;
{
    storage s = entity_storage( e ) ;
    set d = DEFS( e ) ;

    if( d == (set) HASH_UNDEFINED_VALUE ) {
	hash_put( Defs, (char *)e, (char *)(d = MAKE_STATEMENT_SET()));
    }
    if( set_belong_p( d, (char *)st )) {
	return ;
    }
    set_add_element( d, d, (char *)st ) ;

    if( storage_ram_p( s )) {
	MAPL( es, {add_entity_to_defs( ENTITY( CAR( es )), st );},
	      ram_shared( storage_ram( s ))) ;
    }
}

/* INIT_ONE_STATEMENT initializes the global data structures needed
   for usedef computation of one statement ST. */

static void init_one_statement( st )
statement st ;
{
    if( GEN( st ) == (set) HASH_UNDEFINED_VALUE ) {
	dg_vertex_label l ;
	vertex v ;

	ifdebug(2) {
	    fprintf(stderr, "Init statement %d with effects %p\n", 
		    statement_number( st ), load_statement_effects(st) );
	    print_effects( load_statement_effects(st) ) ;
	}
	hash_put( Gen, (char *)st, (char *)MAKE_STATEMENT_SET()) ;
	hash_put( Ref, (char *)st, (char *)MAKE_STATEMENT_SET()) ;
	hash_put( Kill, (char *)st, (char *)MAKE_STATEMENT_SET()) ;
	hash_put( Def_in, (char *)st, (char *)MAKE_STATEMENT_SET()) ;
	hash_put( Def_out, (char *)st, (char *)MAKE_STATEMENT_SET()) ;
	hash_put( Ref_in, (char *)st, (char *)MAKE_STATEMENT_SET()) ;
	hash_put( Ref_out, (char *)st, (char *)MAKE_STATEMENT_SET()) ;

	l = make_dg_vertex_label(statement_ordering(st), 
				 sccflags_undefined ) ;
	v = make_vertex( l, NIL ) ;
	hash_put( Vertex_statement, (char *)st, (char *)v ) ;
	graph_vertices( dg ) = CONS( VERTEX, v, graph_vertices( dg )) ;

	/* FI: regions are not really proper effects... 
	   I should use proper effects for non-call instruction
	   I need a global variable
	   let's kludge this for the time being, 5 August 1992
	   */
	if(!instruction_block_p(statement_instruction(st))) {
	    MAPL( peffects, {
		effect e = EFFECT( CAR( peffects )) ;
	      
		if( action_write_p( effect_action( e )) &&
		   approximation_must_p( effect_approximation( e ))) {
		    add_entity_to_defs( effect_entity( e ), st );
		}
	    }, load_statement_effects(st) ) ;
	}
    }
    else {
	set_clear( GEN( st )) ;
	set_clear( REF( st )) ;
	set_clear( KILL( st )) ;
	set_clear( DEF_OUT( st )) ;
	set_clear( DEF_IN( st )) ;
	set_clear( REF_OUT( st )) ;
	set_clear( REF_IN( st )) ;
    }
}

/* INIT_STATEMENT recursively initializes the sets of gens,
   kills, ins and outs for the statements that appear in ST. Moreover, the
   set of defs are computed; note that not only call statements are there,
   but also enclosing statements (e.g, blocks and loops). */

static void init_control() ;

static void init_statement( st )
statement st ;
{
    instruction i ;

    init_one_statement( st ) ;

    switch( instruction_tag( i = statement_instruction( st ))) {
    case is_instruction_block:
	MAPL( sts, {init_statement( STATEMENT( CAR( sts )));},
	      instruction_block( i )) ;
	break ;
    case is_instruction_test:
	init_statement( test_true( instruction_test( i )));
	init_statement( test_false( instruction_test( i )));
	break ;
    case is_instruction_loop:
	init_statement( loop_body( instruction_loop( i ))) ;
	break ;
    case is_instruction_whileloop:
	init_statement(whileloop_body( instruction_whileloop( i ))) ;
	break ;
    case is_instruction_goto: 
    case is_instruction_call:
	break ;
    case is_instruction_unstructured:
	init_control( unstructured_control( instruction_unstructured( i ))) ;
	break ;
    default:
	pips_error( "init_statement", "Unknown tag %d\n", 
		    instruction_tag( i )) ;
    }
}

static void init_control( ct )
control ct ;
{
    cons *blocs = NIL ;

    ifdebug(1) {
	mem_spy_begin();
    }

    CONTROL_MAP( c, {init_statement( control_statement( c ));},
		 ct, blocs ) ;
    gen_free_list( blocs ) ;

    ifdebug(1) {
	mem_spy_end("init_control");
    }
}

/* The GENKILL_xxx functions implement the computation of GEN, REF and 
   KILL sets
   from Aho, Sethi and Ullman "Compilers" (p. 612). This is slightly
   more complex since we use a structured control graph, thus fixed
   point computations can be recursively required (the correctness of
   this is probable, although not proven, as far as I can tell). */

static void genkill_statement() ;

/* KILL_STATEMENT updates the KILL set of statement ST on entity LHS. Only
   effects that modify one reference (i.e., assignments) are killed (see
   above). Write effects on arrays (see IDXS) don't kill the definitions of 
   the array. Equivalence with non-array entities is managed properly. */

static void kill_statement( kill, st, lhs, idxs )
set kill ;
statement st ;
entity lhs ;
cons *idxs ;
{
    cons *shared ;

    if( !ENDP( idxs )) {
	return ;
    }
    shared = (storage_ram_p( entity_storage( lhs )) ?
	      gen_copy_seq( ram_shared( storage_ram( entity_storage( lhs )))) :
	      NIL) ;
    shared = CONS( ENTITY, lhs, shared) ;
    MAPL( mods, {entity mod = ENTITY( CAR( mods )) ;
		 type t = entity_type( mod ) ;

		 if( !type_variable_p( t ) ||
		     !ENDP( variable_dimensions( type_variable( t )))) {
 		     continue ;
		 }
		 SET_MAP( v, {
		     statement def = (statement)v ;
		     int nb_writes = 0 ;
		     approximation the_write = approximation_undefined;

		     MAPL( es, {effect e = EFFECT( CAR( es )) ;

				if( action_write_p( effect_action( e ))) {
				    the_write = effect_approximation( e ) ;
				    nb_writes++ ;
				}},
			  load_statement_effects(def) ) ;
		     pips_assert( "kill_call", nb_writes > 0 ) ;

		     if(nb_writes == 1 && 
			approximation_must_p( the_write ) &&
			def != st ) {
			 set_add_element( kill, kill, v ) ;
		     }}, DEFS( mod )) ;},
	 shared ) ;
    gen_free_list( shared ) ;
}

/* GENKILL_ONE_STATEMENT loops over the entities written in the 
   statement ST. */

static void genkill_one_statement( st )
statement st ;
{
    bool does_write = FALSE ;
    bool does_read = FALSE ;
    set gen = GEN( st ) ;
    set ref = REF( st ) ;
    set kill = KILL( st ) ;

    set_clear( kill ) ;
    set_clear( gen ) ;
    set_clear( ref ) ;

    MAPL( peffects, {
	effect e = EFFECT( CAR( peffects )) ;
	action a = effect_action( e ) ;
		    
	if(action_write_p( a ) &&
	   approximation_must_p( effect_approximation( e ))) {
	    reference r = effect_reference( e ) ;

	    kill_statement(kill, 
			   st, 
			   reference_variable( r ),
			   reference_indices( r )) ;
	}
	does_write |= action_write_p( a ) ;
	does_read |= action_read_p( a ) ;
    }, load_statement_effects(st) ) ;

    if( does_write ) {
	set_add_element( gen, gen, (char *)st ) ;
    }
    if( does_read ) {
	set_add_element( ref, ref, (char *)st ) ;
    }
}

static void genkill_test( t, s )
test t;
statement s ;
{
    statement st = test_true(t);
    statement sf = test_false(t);
    set ref = REF( s ) ;

    genkill_statement( st );
    genkill_statement( sf );
    set_union( GEN( s ), GEN( st ), GEN( sf )) ;
    set_union( ref, ref, REF( sf )) ;
    set_union( ref, ref, REF( st )) ;
    set_intersection( KILL( s ), KILL( st ), KILL( sf )) ;
}

					/*  */
/* MASK_EFFECTS masks the effects in S according to the locals L. */

static void mask_effects( s, l )
set s ;
cons *l ;
{
    cons *to_mask = NIL ;

    SET_MAP( v, {
	statement s = (statement)v ;
	bool mask = TRUE ;

	MAPL( fs, {
	    effect f = EFFECT( CAR( fs )) ;
	    action a = effect_action( f ) ;
			    
	    if(action_write_p( a ) &&
	       gen_find_eq( effect_entity( f ), l ) == entity_undefined) {
		mask = FALSE ;
		break ;
	    }}, load_statement_effects(s) ) ;
	if( mask ) {
	    to_mask = CONS( STATEMENT, s, to_mask ) ;
	}}, s ) ;
    MAPL( ss, {set_del_element( s, s, (char *)STATEMENT( CAR( ss )));}, 
	  to_mask ) ;
    gen_free_list( to_mask ) ;
}

/* GENKILL_LOOP has to deal specially with the loop variable which is not
   managed in the Dragon book. If loops are at least one trip, then statements
   are always killed by execution of loop body. Effect masking is performed
   on locals (i.e., gen set is pruned from local definitions). */

static void genkill_loop( l, st )
loop l ;
statement st ;
{
    statement b = loop_body( l ) ;
    set gen = GEN( st ) ;
    set ref = REF( st ) ;
    cons *locals = loop_locals( l ) ;

    genkill_statement( b ) ;

    set_union( gen, gen,  GEN( b )) ;
    set_union( ref, ref, REF( b )) ;

    if( one_trip_do ) {
	set_union( KILL( st ), KILL( st ), KILL( b )) ;
    }
    if( get_bool_property( "CHAINS_MASK_EFFECTS" )) {
	mask_effects( gen, locals ) ;
	mask_effects( ref, locals ) ;
    }
}

static void genkill_whileloop( l, st )
whileloop l ;
statement st ;
{
    statement b = whileloop_body( l ) ;
    set gen = GEN( st ) ;
    set ref = REF( st ) ;

    genkill_statement( b ) ;
    
    set_union( gen, gen,  GEN( b )) ;
    set_union( ref, ref, REF( b )) ;
   
}


/* The Dragon book only deals with a sequence of two statements. 
   GENKILL_BLOCK generalizes to lists, via recursion. ST holds the being
   computed sets of gens, refs and kills. */

static void genkill_block( sts, st )
cons *sts ;
statement st ;
{
    statement one ;
    
    if( !ENDP( sts )) {
	set diff = MAKE_STATEMENT_SET() ;
	set gen = MAKE_STATEMENT_SET() ;
	set ref = MAKE_STATEMENT_SET() ;
	set kill_st = KILL( st ) ;
	set gen_st = GEN( st ) ;
	set ref_st = REF( st ) ;
	
	genkill_block( CDR( sts ), st ) ;
	genkill_statement( one = STATEMENT( CAR( sts ))) ;
	set_difference( diff, GEN( one ), kill_st) ;
	set_union( gen, gen_st, diff ) ;
	set_difference( diff, KILL( one ), gen_st) ;
	set_union( kill_st, kill_st, diff ) ;
	set_assign( gen_st, gen ) ;
	set_union( ref_st, REF( one ), ref_st ) ;
	set_free( diff ) ;
	set_free( gen ) ;
	set_free( ref ) ;
    }
}

/* GENKILL_UNSTRUCTURED computes the gens and kills of the unstructured U by
   recursing on INOUT_CONTROL. The gens can then be inferred. The situation
   for the kills is more fishy; for the moment, we just keep the kills
   that are common to all statements of the control graph. */

static void inout_control() ;

static void genkill_unstructured( u, st )
unstructured u ;
statement st ;
{
    control c = unstructured_control( u ) ;
    statement exit = control_statement( unstructured_exit( u )) ;
    set kill = MAKE_STATEMENT_SET() ;
    cons *blocs = NIL ;

    set_clear( DEF_IN( st )) ;
    inout_control( c ) ;

    CONTROL_MAP( cc, {
	set_intersection( kill, kill, KILL( control_statement( cc ))) ;
    }, c, blocs ) ;
    set_assign( KILL( st ), kill ) ;
    set_free( kill ) ;

    if( set_undefined_p( DEF_OUT( exit ))) {
	set ref = MAKE_STATEMENT_SET() ;
	set empty = MAKE_STATEMENT_SET() ;
	CONTROL_MAP( cc, {
	    set_union( ref, ref, REF( control_statement( cc ))) ;
	}, c, blocs ) ;
	set_assign( REF( st ), ref ) ;
	set_free( ref ) ;
	set_assign( GEN( st ), empty) ;
	set_free( empty ) ;
    }
    else {
	set_assign( GEN( st ), DEF_OUT( exit )) ;
	set_difference( REF( st ), REF_OUT( exit ), REF_IN( st )) ;
    }
    gen_free_list( blocs ) ;
}

/* GENKILL_INSTRUCTION does the dispatch and recursion loop. */

static void genkill_instruction( i, st )
instruction i ;
statement st ;
{
    cons *pc;
    test t;
    loop l;

    switch(instruction_tag(i)) {
    case is_instruction_block:
	genkill_block( pc = instruction_block(i), st ) ;
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
    case is_instruction_unstructured:
	genkill_unstructured( instruction_unstructured( i ), st ) ;
	break ;
    case is_instruction_call:
    case is_instruction_goto:
	break ;
    default:
	pips_error( "genkill_instruction",
		    "unexpected tag %d\n", instruction_tag(i));
    }
}

/* GENKILL_STATEMENT computes the sets of gens and kills of the statement
   S. This is the main entry to this task. */

static void genkill_statement(s)
statement s;
{
    genkill_one_statement( s ) ;
    genkill_instruction( statement_instruction(s), s ) ;
    ifdebug(2) {
	debug(2, "genkill_statement", "Result for Statement %p [%s]:\n", 
		s, statement_identification(s));;
	local_print_statement_set( "GEN", GEN( s )) ;
	local_print_statement_set( "REF", REF( s )) ;
	local_print_statement_set( "KILL", KILL( s )) ;
    }
}

/* INOUT_STATEMENT propagates the in set of ST (which is inherited) to
   compute the out set. This works in tandem with INOUT_CONTROL. */

static void inout_statement() ;

static void inout_block( st, l )
statement st ;
cons *l ;
{
    set def_in = DEF_IN( st ) ;
    set ref_in = REF_IN( st ) ;
    statement one ;

    ifdebug(1) {
	mem_spy_begin();
    }

    MAPL( sts, {one = STATEMENT( CAR( sts )) ;

		set_assign( DEF_IN( one ), def_in ) ;
		set_assign( REF_IN( one ), ref_in ) ;
		inout_statement( one ) ;
		def_in = DEF_OUT( one );
		set_union( ref_in, ref_in, REF_OUT( one )) ;},
	 l ) ;
    set_assign( DEF_OUT( st ), def_in ) ;
    set_assign( REF_OUT( st ), ref_in ) ;

    ifdebug(1) {
	mem_spy_end("inout_block");
    }
}


static void inout_test( st, t )
statement st ;
test t ;
{
    statement sv = test_true( t ) ;
    statement sf = test_false( t ) ;

    ifdebug(1) {
	mem_spy_begin();
    }

    set_assign( DEF_IN( sv ), DEF_IN( st )) ;
    set_assign( REF_IN( sv ), REF_IN( st )) ;
    inout_statement( sv );
    set_assign( DEF_IN( sf ), DEF_IN( st )) ;
    set_assign( REF_IN( sf ), REF_IN( st )) ;
    inout_statement( sf );
    set_union( DEF_OUT( st ), DEF_OUT( sv ), DEF_OUT( sf )) ;
    set_union( REF_OUT( st ), REF_OUT( sv ), REF_OUT( sf )) ;

    ifdebug(1) {
	mem_spy_end("inout_test");
    }
}

static void inout_loop( st, lo )
statement st ;
loop lo ;
{
    statement l = loop_body( lo ) ;
    set diff = MAKE_STATEMENT_SET() ;

    ifdebug(1) {
	mem_spy_begin();
    }

    set_union( DEF_IN( l ), GEN( st ),
	       set_difference( diff, DEF_IN( st ), KILL( st )));
    set_free( diff ) ;
    set_union( REF_IN( l ), REF_IN( st ), REF( st )) ;
    inout_statement( l ) ;

    if( one_trip_do ) {
	set_assign( DEF_OUT( st ), DEF_OUT( l )) ;
	set_assign( REF_OUT( st ), REF_OUT( l )) ;
    }
    else {
	set_union( DEF_OUT( st ), DEF_OUT( l ), DEF_IN( st )) ;
	set_union( REF_OUT( st ), REF_OUT( l ), REF_IN( st )) ;
    }

    ifdebug(1) {
	mem_spy_end("inout_loop");
    }
}

static void inout_whileloop( st, lo )
statement st ;
whileloop lo ;
{
    statement l = whileloop_body( lo ) ;
    set diff = MAKE_STATEMENT_SET() ;

    set_union( DEF_IN( l ), GEN( st ),
	       set_difference( diff, DEF_IN( st ), KILL( st )));
    set_free( diff ) ;
    set_union( REF_IN( l ), REF_IN( st ), REF( st )) ;
    inout_statement( l ) ;

    set_union( DEF_OUT( st ), DEF_OUT( l ), DEF_IN( st )) ;
    set_union( REF_OUT( st ), REF_OUT( l ), REF_IN( st )) ;
  
  
}
static void inout_call(statement st, call c)
{
    set diff = MAKE_STATEMENT_SET() ;

    ifdebug(1) {
	mem_spy_begin();
    }

    set_union( DEF_OUT( st ), GEN( st ),
	       set_difference( diff, DEF_IN( st ), KILL( st )));
    set_union( REF_OUT( st ), REF_IN( st ), REF( st )) ;
    set_free( diff ) ;

    ifdebug(1) {
	mem_spy_end("inout_call");
    }
}

static void inout_unstructured( st, u )
statement st ;
unstructured u ;
{
    control c = unstructured_control( u ) ;
    control exit = unstructured_exit( u ) ;
    statement s_exit =  control_statement( exit ) ;

    ifdebug(1) {
	mem_spy_begin();
    }

    set_assign( DEF_IN( control_statement( c )), DEF_IN( st )) ;
    set_assign( REF_IN( control_statement( c )), REF_IN( st )) ;
    inout_control( c ) ;

    if( set_undefined_p( DEF_OUT( s_exit ))) {
	list blocs = NIL ;
	set ref = MAKE_STATEMENT_SET() ;
	set empty = MAKE_STATEMENT_SET() ;

	CONTROL_MAP( cc, {
	    set_union( ref, ref, REF( control_statement( cc ))) ;
	}, c, blocs ) ;
	set_assign( REF_OUT( st ), ref ) ;
	set_assign( DEF_OUT( st ), empty) ;

	set_free( ref ) ;
	set_free( empty ) ;
	gen_free_list( blocs ) ;
    }
    else {
	set_assign( DEF_OUT( st ), DEF_OUT( s_exit )) ;
	set_assign( REF_OUT( st ), REF_OUT( s_exit )) ;
    }

    ifdebug(1) {
	mem_spy_end("inout_unstructured");
    }
}					

static void inout_statement( st )
statement st ;
{
    instruction i ;
    static int indent = 0 ;

    /*
    ifdebug(1) {
	mem_spy_begin();
    }
    */
    
    ifdebug(2) {
	fprintf( stderr, "%*s> Statement %p (%d):\n", 
		 indent++, "", st, statement_number( st )) ;
	local_print_statement_set( "DEF_IN", DEF_IN( st )) ;
	local_print_statement_set( "DEF_OUT", DEF_OUT( st )) ;
	local_print_statement_set( "REF_IN", REF_IN( st )) ;
	local_print_statement_set( "REF_OUT", REF_OUT( st )) ;
    }
    switch( instruction_tag( i = statement_instruction( st ))) {
    case is_instruction_block: 
	inout_block( st, instruction_block( i )) ;
	break ;
    case is_instruction_test:
	inout_test( st, instruction_test( i )) ;
	break ;
    case is_instruction_loop:
	inout_loop( st, instruction_loop( i )) ;
	break ;
    case is_instruction_whileloop:
	inout_whileloop( st, instruction_whileloop( i )) ;
	break ;
    case is_instruction_call:
	inout_call( st, instruction_call( i )) ;
	break ;
    case is_instruction_goto: 
	pips_error( "inout_statement", "Unexpected tag %d\n", i ) ;
	break ;
    case is_instruction_unstructured:
	inout_unstructured( st, instruction_unstructured( i )) ;
	break ;
    default:
	pips_error( "inout_statement", "Unknown tag %d\n", i ) ;
    }
    ifdebug(2) {
	fprintf( stderr, "%*s> Statement %p (%d):\n", 
		 indent--, "", st, statement_number( st )) ;
	local_print_statement_set( "DEF_IN", DEF_IN( st )) ;
	local_print_statement_set( "DEF_OUT", DEF_OUT( st )) ;
	local_print_statement_set( "REF_IN", REF_IN( st )) ;
	local_print_statement_set( "REF_OUT", REF_OUT( st )) ;
    }

    /*
    ifdebug(1) {
	mem_spy_end("inout_statement");
    }
    */
}

/* INOUT_CONTROL computes the in and out sets of the structured control
   graph. This is done by fixed point iteration (see Dragon book, p. 625),
   except that the in set of CT is not empty (this explains the set_union
   instead of set_assign in the fixed point computation loop on IN( st )).
   Once again, the correctness of this modification is not proven. */

static void inout_control( ct )
control ct ;
{
    bool change ;
    set d_oldout = MAKE_STATEMENT_SET() ;
    set d_out = MAKE_STATEMENT_SET() ;
    set r_oldout = MAKE_STATEMENT_SET() ;
    set r_out = MAKE_STATEMENT_SET() ;
    set diff = MAKE_STATEMENT_SET() ;
    cons *blocs = NIL ;

    ifdebug(1) {
	mem_spy_begin();
	mem_spy_begin();
    }

    ifdebug(2) {
	fprintf(stderr, "Computing DEF_IN and OUT of control %p entering", 
		ct ) ;
	local_print_statement_set( "", DEF_IN( control_statement( ct ))) ;
    }
    CONTROL_MAP( c, {statement st = control_statement( c ) ;

		     genkill_statement( st ) ;
		     set_assign( DEF_OUT( st ), GEN( st )) ;
		     set_assign( REF_OUT( st ), REF( st )) ;

		     if( c != ct ) {
			 set_clear( DEF_IN( st )) ;
			 set_clear( REF_OUT( st )) ;
		     }},
		 ct, blocs ) ;

    ifdebug(1) {
	mem_spy_end("inout_control: phase 1");
	mem_spy_begin();
    }

    for( change = TRUE ; change ; ) {
	ifdebug(3) {
 	    fprintf( stderr, "Iterating on %p ...\n", ct ) ;
	}
	change = FALSE ;

	CONTROL_MAP( b, 
	    {statement st = control_statement( b ) ;

	     set_clear( d_out ) ;
	     set_clear( r_out ) ;
	     MAPL( preds, {control pred = CONTROL( CAR( preds )) ;
			   statement pst = control_statement( pred ) ;

			   set_union( d_out, d_out, DEF_OUT( pst )) ;
			   set_union( r_out, r_out, REF_OUT( pst )) ;},
		  control_predecessors( b )) ;
	     set_union( DEF_IN( st ), DEF_IN( st ), d_out ) ;
	     set_union( REF_IN( st ), REF_IN( st ), r_out ) ;
	     set_assign( d_oldout, DEF_OUT( st )) ;
	     set_union( DEF_OUT( st ), GEN( st ), 
		       set_difference( diff, DEF_IN( st ), KILL( st ))) ;
	     set_assign( r_oldout, REF_OUT( st )) ;
	     set_union( REF_OUT( st ), REF( st ), REF_IN( st )) ;
	     change |= (!set_equal( d_oldout, DEF_OUT( st )) ||
			!set_equal( r_oldout, REF_OUT( st )));
	 }, ct, blocs ) ;
    }

    ifdebug(1) {
	mem_spy_end("inout_control: phase 2 (fix-point)");
	mem_spy_begin();
    }
    CONTROL_MAP( c, {inout_statement( control_statement( c ));},
		 ct, blocs ) ;
    set_free( d_oldout ) ;
    set_free( d_out ) ;
    set_free( r_oldout ) ;
    set_free( r_out ) ;
    set_free( diff ) ;
    gen_free_list( blocs ) ;

    ifdebug(1) {
	mem_spy_end("inout_control: phase 3");
	mem_spy_end("inout_control");
    }
}

/* PUSHNEW_CONFLICTS adds the conflict FIN->FOUT in the list CS (if it's
   not already there. */

static cons *pushnew_conflict( fin, fout, cfs )
effect fin, fout ;
cons *cfs ;
{
    conflict c ;

    MAPL( cs, {
	conflict c = CONFLICT( CAR( cs )) ;

	if( conflict_source( c )==fin && conflict_sink( c )==fout ) {
	    return( cfs ) ;
	}
    }, cfs ) ;
    c = make_conflict( fin, fout, cone_undefined) ;
    ifdebug(2) {
	fprintf( stderr, "Adding %s->%s\n", 
		entity_name( effect_entity( fin )),
		entity_name( effect_entity( fout ))) ;
    }
    return( CONS( CONFLICT, c, cfs )) ;
}

/* DD_DU detects Def/Def, Def/Use conflicts between effects FIN and FOUT. */

static bool dd_du( fin, fout )
effect fin, fout ;
{
    return(action_write_p( effect_action( fin )) &&
	   ((get_bool_property( "CHAINS_DATAFLOW_DEPENDENCE_ONLY" )) ?
	    action_read_p( effect_action( fout )) :
	    TRUE )) ;
}

/* UD detects Use/Def conflicts between effects FIN and FOUT. */

static bool ud( fin, fout )
effect fin, fout ;
{
    return( action_read_p( effect_action( fin )) &&
	    (action_write_p( effect_action( fout )) ||
	     keep_read_read_dependences )) ;
}

/* ADD_CONFLICTS adds conflitc arcs to the dependence graph dg between the
   in-coming statement STIN and the out-going STOUT. 
   Note that output dependencies
   are not minimal (e.g., i = s ; s = ... ; s = ...) creates an oo-dep between
   the i assingment and the last s assignment. */

static void add_conflicts( stin, stout, which )
statement stin, stout ;
bool (*which)() ;
{
    vertex vin ;
    vertex vout = vertex_statement( stout ) ;
    cons *effect_ins = load_statement_effects( stin ) ;
    cons *effect_outs = load_statement_effects( stout ) ;
    cons *cs = NIL ;

    ifdebug(1) {
	mem_spy_begin();
    }

    ifdebug(2) {
	fprintf( stderr, "Conflicts %d(%p) -> %d(%p) %s\n",
		statement_number( stin ), stin,
		statement_number( stout ), stout,
		(which == ud) ? "ud" : "dd_du" ) ;
    }
    vin = vertex_statement( stin ) ;

    MAPL( fins, {
	effect fin = EFFECT( CAR( fins )) ;
	entity ein = effect_entity( fin ) ;

	MAPL(fouts, {
	    effect fout = EFFECT( CAR( fouts )) ;
	    entity eout = effect_entity( fout ) ;

	    if( entity_conflict_p( ein, eout ) && (*which)( fin, fout )) {
		list loops = load_statement_enclosing_loops(stout);
		reference rin = effect_reference(fin);
		reference rout = effect_reference(fout);
		bool remove_this_conflict_p = disambiguate_constant_subscripts?
		    references_do_not_conflict_p(rin, rout): FALSE;

		MAPL(pl, {
		    statement el = STATEMENT(CAR(pl));
		    entity il = loop_index(statement_loop(el));
		    remove_this_conflict_p |= entity_conflict_p(ein, il);
		}, loops);

		if (!remove_this_conflict_p)
		    cs = pushnew_conflict( fin, fout, cs ) ;
	    }
	}, effect_outs );
    }, effect_ins ) ;

    /* il faut verifier que l'arc vin/vout n'existe pas deja !! */
    if( !ENDP( cs )) {
	successor s = make_successor( make_dg_arc_label( cs ), vout ) ;
	    
	vertex_successors( vin ) = 
		CONS( SUCCESSOR, s, vertex_successors( vin )) ;
    }

    ifdebug(1) {
	mem_spy_end("add_conflicts");
    }
}

/* USEDEF_STATEMENT updates the dependence graph between the statement ST and
   its in set. Only calls are taken into account. */

static void usedef_control() ;

static void usedef_statement( st )
statement st ;
{

    instruction i = statement_instruction( st ) ;
    tag t ;

    SET_MAP( in, {add_conflicts( (statement)in, st, dd_du ) ;}, DEF_IN( st )) ;
    SET_MAP( in, {add_conflicts( (statement)in, st, ud ) ;}, REF_IN( st )) ;

    switch( t = instruction_tag( i )) {
    case is_instruction_block: 
	MAPL( sts, {usedef_statement( STATEMENT( CAR( sts ))) ;},
	      instruction_block( i )) ;
	break ;
    case is_instruction_test: 
	usedef_statement( test_true( instruction_test( i )));
	usedef_statement( test_false( instruction_test( i )) );
	break ;
    case is_instruction_loop: 
	usedef_statement( loop_body( instruction_loop( i ))) ;
	break ;
    case is_instruction_whileloop: 
	usedef_statement( whileloop_body( instruction_whileloop( i ))) ;
	break ;
    case is_instruction_call: 
    case is_instruction_goto: 
	break ;
    case is_instruction_unstructured:
	usedef_control( unstructured_control( instruction_unstructured( i ))) ;
	break ;
    default:
	pips_error( "usedef_statement", "Unknown tag %d\n", t ) ;
    }
}

/* USEDEF_CONTROL does the same thing on each of the nodes of C. */

static void usedef_control( c )
control c ;
{
    cons *blocs = NIL ;

    ifdebug(1) {
	mem_spy_begin();
    }

    CONTROL_MAP( n, {usedef_statement( control_statement( n ));},
		 c, blocs ) ;
    gen_free_list( blocs ) ;

    ifdebug(1) {
	mem_spy_end("usedef_control");
    }
}


#define USE_PROPER_EFFECTS 1
#define USE_REGIONS 2
#define USE_IN_OUT_REGIONS 3

bool atomic_chains(module_name)
char *module_name;
{
    return chains(module_name, USE_PROPER_EFFECTS);
}

bool region_chains(module_name)
char *module_name;
{
    return chains(module_name, USE_REGIONS);
}

bool in_out_regions_chains(module_name)
char *module_name;
{
    return chains(module_name, USE_IN_OUT_REGIONS);
}


bool 
chains(char *module_name, int use)
{
    statement module_stat;
    instruction module_inst;
    control module_cont = control_undefined ;
    graph module_graph;
    void print_graph() ;

    debug_on("CHAINS_DEBUG_LEVEL");

    ifdebug(1) {
	mem_spy_init(0, 100000., NET_MEASURE, 0);
	mem_spy_begin();
	mem_spy_begin();
    }

    debug_off();

    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, TRUE) );
    module_stat = get_current_module_statement();
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    /* set_entity_to_size(); should be performed at the workspace level */

    debug_on("CHAINS_DEBUG_LEVEL");

    ifdebug(1) {
	mem_spy_end("Chains: resources loaded");
	mem_spy_begin();
    }
   
    pips_debug(1, "finding enclosing loops ...\n");
    set_enclosing_loops_map( loops_mapping_of_statement(module_stat) );

    module_inst = statement_instruction(module_stat);

    if (! instruction_unstructured_p(module_inst)) {
	module_cont = make_control(module_stat, NIL, NIL) ;

	module_inst = make_instruction(is_instruction_unstructured,
				       make_unstructured(module_cont,
							 module_cont)) ;
    }
    set_effects(module_name,use);

    ifdebug(1) {
	mem_spy_end("Chains: Preamble");
	mem_spy_begin();
    }

    module_graph = 
	dependence_graph(instruction_unstructured(module_inst));

    ifdebug(1) {
	mem_spy_end("Chains: Computation");
	mem_spy_begin();
    }

    ifdebug(2)
	prettyprint_dependence_graph(stderr, module_stat, module_graph);

    debug_off();

    if( !control_undefined_p( module_cont )) {
	control_statement (module_cont) = statement_undefined;
	free_instruction( module_inst );
    }

    DB_PUT_MEMORY_RESOURCE(DBR_CHAINS, module_name, (char*) module_graph);

    reset_effects();
    clean_enclosing_loops();
    reset_current_module_statement();
    reset_current_module_entity();
    /* reset_entity_to_size(); */

    debug_on("CHAINS_DEBUG_LEVEL");
    ifdebug(1) {
	mem_spy_end("Chains: Deallocation");
	mem_spy_end("Chains");
	mem_spy_reset();
    }
    debug_off();

    return TRUE;
}



/* DEPENDENCE_GRAPH computes, from the control graph U, the dependency 
   graph. */

graph dependence_graph( u )
unstructured u ;
{
    control c = unstructured_control( u ) ;

    one_trip_do = get_bool_property( "ONE_TRIP_DO" ) ;
    keep_read_read_dependences = 
	    get_bool_property( "KEEP_READ_READ_DEPENDENCE" ) ;
    disambiguate_constant_subscripts = 
	    get_bool_property( "CHAINS_DISAMBIGUATE_CONSTANT_SUBSCRIPTS" ) ;

    ifdebug(1) {
	mem_spy_begin();
	mem_spy_begin();
    }

    Gen = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Ref = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Kill = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Def_in = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Def_out = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Ref_in = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Ref_out = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Defs = hash_table_make( hash_pointer, INIT_ENTITY_SIZE ) ;
    Vertex_statement = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    dg = make_graph( NIL ) ;
    init_control( c ) ;
    inout_control( c ) ;
    usedef_control( c ) ;

#define TABLE_FREE(t) \
    {HASH_MAP( k, v, {set_free( (set)v ) ;}, t ) ; hash_table_free(t);}

    ifdebug(1) {
	mem_spy_end("dependence_graph: after computation");
	mem_spy_begin();
    }

    TABLE_FREE( Gen ) ;
    TABLE_FREE( Ref ) ;
    TABLE_FREE( Kill ) ;
    TABLE_FREE( Def_in ) ;
    TABLE_FREE( Def_out ) ;
    TABLE_FREE( Ref_in ) ;
    TABLE_FREE( Ref_out ) ;
    TABLE_FREE( Defs ) ;

    hash_table_free( Vertex_statement ) ;


    ifdebug(1) {
	mem_spy_end("dependence graph: after freeing the hash tables");
	mem_spy_end("dependence_graph");
    }

    return( dg ) ;
}



/* STATEMENT_DEPENDENCE_GRAPH computes, from the statement s, the dependency 
 * graph. Statement s is assumed "controlized", i.e. GOTO have been replaced
 * by unstructured.
 *
 * FI: this function is bugged. As Pierre said, you have to start with
 * an unstructured for the use-def chain computation to be correct.
 */



graph statement_dependence_graph(statement s)
{
    one_trip_do = get_bool_property( "ONE_TRIP_DO" ) ;
    keep_read_read_dependences = 
	    get_bool_property( "KEEP_READ_READ_DEPENDENCE" ) ;

    Gen = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Ref = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Kill = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Def_in = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Def_out = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Ref_in = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Ref_out = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;
    Defs = hash_table_make( hash_pointer, INIT_ENTITY_SIZE ) ;
    Vertex_statement = hash_table_make( hash_pointer, INIT_STATEMENT_SIZE ) ;

    dg = make_graph( NIL ) ;
    init_statement( s ) ;
    inout_statement( s ) ;
    usedef_statement( s ) ;
    return( dg ) ;
}


/* functions for effects maps */

static bool rgch = FALSE;
static bool iorgch = FALSE;

static list load_statement_effects(st)
statement st;
{
    instruction inst = statement_instruction( st );
    tag t = instruction_tag( inst );
    tag call_t;
    list le = NIL;
    
    switch( t ) {
    case is_instruction_call: 
	call_t =
	    value_tag(entity_initial(call_function( instruction_call( inst ))));
	if ( iorgch && (call_t == is_value_code)) 
	{
	    list l_in = load_statement_in_regions(st); 
	    list l_out = load_statement_out_regions(st);
	    le = gen_append(l_in,l_out);
	    break;
	}
    case is_instruction_block: 
    case is_instruction_test: 
    case is_instruction_loop: 
    case is_instruction_whileloop: 
    case is_instruction_goto: 
    case is_instruction_unstructured:
	le = load_proper_rw_effects_list(st);
	break ;
    default:
	pips_error( "load_statement_effects", "Unknown tag %d\n", t ) ;
    }
    
    return le;
}


static void set_effects(module_name, use)
char *module_name;
int use;
{
    switch(use) {

    case USE_PROPER_EFFECTS: 
	rgch = FALSE;
	iorgch = FALSE;
	set_proper_rw_effects((statement_effects)
               db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE));
	break;

	/* In fact, we use proper regions only; proper regions of calls are 
         * similar to regions, except for expressions given as arguments,
	 * whose regions are simply appended to the list (non convex hull).
	 * For simple statements (assignments), proper regions contain the list 
	 * elementary regions (there is no summarization, i.e no convex hull).
	 * For loops and tests, proper regions contain the elements accessed in
	 * the tests and loop range. BC. 
	 */
    case USE_REGIONS: 
	rgch = TRUE;
	iorgch = FALSE;
	set_proper_rw_effects((statement_effects) 
	       db_get_memory_resource(DBR_PROPER_REGIONS, module_name, TRUE));
	break;

	/* For experimental purpose only */
    case USE_IN_OUT_REGIONS: 
	rgch = FALSE;
	iorgch = TRUE;
	set_proper_rw_effects((statement_effects) 
	       db_get_memory_resource(DBR_PROPER_REGIONS, module_name, TRUE));
	set_in_effects((statement_effects) 
	       db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE));
	set_out_effects((statement_effects) 
	       db_get_memory_resource(DBR_OUT_REGIONS, module_name, TRUE));
	break;    

    default: pips_error("set_effects", "ill. parameter use = %d\n", use);
    }

}


static void reset_effects()
{
    reset_proper_rw_effects();
    if (iorgch) {
	reset_in_effects();
	reset_out_effects();
    }
}


/* Acces functions for debug only */

/* PRINT_STATEMENT_SET displays on stderr, the MSG followed by the set
   of statement numbers present in the set S. */

static void local_print_statement_set( msg, s )
char *msg ;
set s ;
{
    fprintf( stderr, "\t%s ", msg ) ;
    SET_MAP( st, {
	fprintf(stderr, ",%p (%d) ", 
		st, statement_number( (statement)st ));
    }, s ) ;
    fprintf( stderr, "\n" );
}

hash_table kill_table()
{
    return( Kill ) ;
}

hash_table gen_table()
{
    return( Gen ) ;
}

hash_table def_in_table()
{
    return( Def_in ) ;
}

hash_table def_out_table()
{
    return( Def_out ) ;
}

hash_table ref_in_table()
{
    return( Def_in ) ;
}

hash_table ref_out_table()
{
    return( Def_out ) ;
}

hash_table defs_table()
{
    return( Defs ) ;
}





