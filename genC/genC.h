/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@isatis.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*/


#ifndef GENC_INCLUDED
#define GENC_INCLUDED

/* genC.h */

#define NEWGEN

/*
 * This is the include file to be used for the generation of C code.
 */
#include <sys/stdtypes.h>
#include <stdio.h>
#include "malloc.h"                    /* for debug with malloclib */
#include "newgen_assert.h"

#include "newgen_types.h"
#include "newgen_set.h"

/*
 * The size of the management information inside each Newgen object (in chunks)
 */

#define HEADER 1
#define HEADER_SIZE (sizeof( chunk )*HEADER)

/*
 * A chunk is used to store every object. It has to be able to store, at least,
 * a (CHUNK *) and every inlinable value. To use a union is a trick to enable
 * the assignment opereator and the ability of passing and returning them as
 * values (for function): this requires a sufficiently clever compiler !

 * Note that the field name of inlinable types have to begin with the same
 * letter as the type itself (this can be fixed if necessary but why bother).
 * This characteristic is used by the Newgen code generator.
 */

typedef union chunk {
	unit u ;
	bool b ;
	char c ;
	int i ;
	float f ;
	string s ;
	struct cons *l ;
	set t ;
	hash_table h ;
	union chunk *p ;
} chunk ;

#define chunk_undefined ((chunk *)(-16))
#define chunk_undefined_p(c) ((c)==chunk_undefined)

#define UNIT(x) "You don't want to take the value of a unit type, do you !"
#define BOOL(x) ((x).b)
#define CHAR(x) ((x).c)
#define INT(x) ((x).i)
#define FLOAT(x) ((x).f)
#define STRING(x) ((x).s)
#define CONSP(x) ((x).l)
#define SETP(x) ((x).t)
#define CHUNK(x) ((x).p)

#include "newgen_list.h"
#include "newgen_stack.h"

/* The implementation of tabulated domains */

extern chunk *Gen_tabulated_[] ;
extern struct binding Domains[], *Tabulated_bp ;


#define TABULATED_MAP(_x,_code,_dom) \
	{int _tabulated_map_i=0 ; \
	 chunk *_tabulated_map_t = Gen_tabulated_[Domains[_dom].index] ; \
         chunk *_x ; \
	 for(;_tabulated_map_i<max_tabulated_elements();_tabulated_map_i++) {\
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
#define GEN_DBG_QUICK_RECURSE 16

#define GEN_DBG_TRAV \
     (GEN_DBG_TRAV_LEAF|GEN_DBG_TRAV_SIMPLE|GEN_DBG_TRAV_OBJECT)

extern void gen_free GEN_PROTO(( chunk * )) ;
extern int gen_free_tabulated GEN_PROTO(( int )) ;
extern void gen_write GEN_PROTO(( FILE *, chunk * )) ;
extern int gen_write_tabulated GEN_PROTO(( FILE *, int )) ;
extern void gen_read_spec GEN_PROTO(()) ; /* instead of ... */
extern chunk *gen_read GEN_PROTO(( FILE * )) ;
extern int gen_read_tabulated GEN_PROTO(( FILE *, int )) ;
extern int gen_read_and_check_tabulated GEN_PROTO(( FILE *, int )) ;
extern chunk *gen_make_array GEN_PROTO(( int )) ;
extern chunk *gen_alloc GEN_PROTO(()) ; /* was ... */
extern void gen_init_external GEN_PROTO((int, 
					 char *(*)(), void (*)(), 
					 void (*)(), char *(*)() )) ;
extern chunk *gen_check GEN_PROTO(( chunk *, int )) ;
extern void gen_clear_tabulated_element GEN_PROTO(( chunk * )) ;
extern chunk *gen_copy_tree GEN_PROTO(( chunk * )) ;
extern int gen_consistent_p GEN_PROTO(( chunk * )) ;
extern char *alloc GEN_PROTO ((int )) ;
extern bool gen_true GEN_PROTO((chunk *)) ; /* instead of previous ... */
extern void gen_null GEN_PROTO((chunk *)) ; /* was ... */

extern void gen_recurse_stop GEN_PROTO((chunk *));
extern void gen_multi_recurse GEN_PROTO(()); /* was ... */
extern void gen_recurse GEN_PROTO((chunk *,
				   int, 
				   bool (*)( chunk * ), 
				   void (*)( chunk * ))) ;

/* Since C is not-orthogonal (chunk1 == chunk2 is prohibited),
 * this one is needed.
 */

#ifndef MEMORY_INCLUDED
#include <memory.h>
#define MEMORY_INCLUDED
#endif

#define gen_equal(lhs,rhs) (memcmp((lhs),(rhs))==0)

/* GEN_CHECK can be used to test run-time coherence of Newgen values.
 */

#ifdef GEN_CHECK
#undef GEN_CHECK
#define GEN_CHECK(e,t) (gen_check((e),(t)),e)
#define GEN_CHECK_ALLOC 1
#else
#define GEN_CHECK(e,t) (e)
#define GEN_CHECK_ALLOC 0
#endif

#include "newgen_generic-mapping.h"
#include "newgen_generic_stack.h"
#include "newgen_map.h"

#endif
