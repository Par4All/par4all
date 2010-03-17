(*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@isatis.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*)

open Array

datatype chunk 
	= undefined
	| unit 
	| bool of bool
	| char of string
	| int of int 
	| real of real 
	| string of string
	| list of chunk list
(*		| set of chunk set 		*)
	| vector of chunk array	

fun 	chunk_undefined_p undefined = true 
|	chunk_undefined_p _ = false

exception Newgen

(*

extern chunk *Gen_tabulated_[] ;
extern struct binding Domains[], *Tabulated_bp ;


#define TABULATED_MAP(_x,_code,_dom) \
	{int _tabulated_map_i=0 ; \
	 chunk *_tabulated_map_t = Gen_tabulated_[Domains[_dom].index] ; \
         chunk *_x ; \
	 for(;_tabulated_map_i<6007;_tabulated_map_i++) {\
		if( (_x=(_tabulated_map_t+_tabulated_map_i)->p) != \
		     chunk_undefined ) \
			_code ;}}

/* The root of the chunk read with READ_CHUNK. */

extern chunk *Read_chunk ;

/* Function interface for user applications. */

extern int gen_debug ;

#define GEN_DBG_TRAV_LEAF 1
#define GEN_DBG_TRAV_SIMPLE 2
#define GEN_DBG_TRAV_OBJECT 4
#define GEN_DBG_CHECK 8

#define GEN_DBG_TRAV \
     (GEN_DBG_TRAV_LEAF|GEN_DBG_TRAV_SIMPLE|GEN_DBG_TRAV_OBJECT)

extern void gen_free () ;
extern int gen_free_tabulated() ;
extern void gen_write() ;
extern int gen_write_tabulated() ;
extern void gen_read_spec() ;
extern chunk *gen_read() ;
extern int gen_read_tabulated() ;
extern int gen_read_and_check_tabulated() ;
extern chunk *gen_make_array() ;
extern chunk *gen_alloc() ;
extern void gen_init_external() ;
extern chunk *gen_check() ;
extern void gen_clear_tabulated_element() ;
extern chunk *gen_copy_tree() ;
extern int gen_consistent_p() ;

/* Since C is not-orthogonal (chunk1 == chunk2 is prohibited), this one is
   needed. */

#ifndef MEMORY_INCLUDED
#include <memory.h>
#define MEMORY_INCLUDED
#endif

#define gen_equal(lhs,rhs) (memcmp((lhs),(rhs))==0)

/* GEN_CHECK can be used to test run-time coherence of Newgen values. */

#ifdef GEN_CHECK
#undef GEN_CHECK
#define GEN_CHECK(e,t) gen_check((e),(t)),e
#else
#define GEN_CHECK(e,t) (e)
#endif

#include "mapping.h"

#endif
*)
