/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

  This file is part of NewGen.

  NewGen is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or any later version.

  NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU General Public License along with
  NewGen.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef GENC_INCLUDED
#define GENC_INCLUDED
#define NEWGEN

/*
 * This is the include file to be used for the generation of C code.
 */
/* #include <sys/stdtypes.h> */
#include <stdio.h>
#include <stdint.h>
#include <strings.h>
/* extern char * strdup(const char *);*/

#include "newgen_assert.h"

#include "newgen_types.h"
#include "newgen_set.h"

/* The size of the management information inside each Newgen object
 * (in gen_chunks)
 */

#define GEN_HEADER (1)
#define GEN_HEADER_SIZE (sizeof(gen_chunk)*GEN_HEADER)

/* A gen_chunk is used to store every object. It has to be able to store,
 * at least, a (CHUNK *) and every inlinable value. To use a union is a
 * trick to enable the assignment opereator and the ability of passing and
 * returning them as values (for function): this requires a sufficiently
 * clever compiler !  * Note that the field name of inlinable types have
 * to begin with the same letter as the type itself (this can be fixed if
 * necessary but why bother).  This characteristic is used by the Newgen
 * code generator.
 */

typedef union gen_chunk {
  unit u;
  bool b;
  char c;
  _int i;
  float f;
  string s;
  void * e; /* for externals */
  struct cons * l;
  set t;
  hash_table h;
  union gen_chunk * p;
} gen_chunk, *gen_chunkp;

#define gen_chunk_undefined ((gen_chunk *)(-16))
#define gen_chunk_undefined_p(c) ((c)==gen_chunk_undefined)

/* obsolete
 */
#define chunk_undefined gen_chunk_undefined
#define chunk_undefined_p(c) gen_chunk_undefined_p(c)

#define UNIT(x) "You don't want to take the value of a unit type, do you?"
#define BOOL(x) ((x).b)
#define CHAR(x) ((x).c)
#define INT(x) ((x).i)
#define FLOAT(x) ((x).f)
#define STRING(x) ((x).s)
#define CONSP(x) ((x).l)
#define SETP(x) ((x).t)
#define CHUNK(x) ((x).p)
#define HASH(x) ((x).h)
#define CHUNKP(x) ((x).p)
#define LIST(x) ((x).l)
#define SET(c) ((x).t)

/* for the MAP macro to handle simple types correctly. FC.
 */
#define UNIT_TYPE "You don't want a unit type, do you?"
#define BOOL_TYPE bool
#define CHAR_TYPE char
#define INT_TYPE int
#define FLOAT_TYPE float
#define STRING_TYPE string

#define CONSP_TYPE list
#define LIST_TYPE list
#define SETP_TYPE set
#define SET_TYPE set
#define CHUNK_TYPE gen_chunkp
#define CHUNKP_TYPE gen_chunkp
#define HASH_TYPE hash_table

/* some other macros need the domain number to keep track of the type.
 * they are provided here for the internal types.
 */
enum internal_type {
  unit_domain = 1, /**< Start at 1 to be able to iterate on them with
		      gen_recurse() functions */
  bool_domain,
  char_domain,
  int_domain,
  intptr_t_domain = int_domain,
  _int_domain = int_domain,
  float_domain,
  string_domain
};

/* utils for typed cons */
#define BOOL_NEWGEN_DOMAIN (bool_domain)
#define CHAR_NEWGEN_DOMAIN (char_domain)
#define INT_NEWGEN_DOMAIN (int_domain)
#define FLOAT_NEWGEN_DOMAIN (float_domain)
#define STRING_NEWGEN_DOMAIN (string_domain)

#define bool_NEWGEN_DOMAIN (bool_domain)
#define char_NEWGEN_DOMAIN (char_domain)
#define int_NEWGEN_DOMAIN (int_domain)
#define float_NEWGEN_DOMAIN (float_domain)
#define string_NEWGEN_DOMAIN (string_domain)

#define LIST_NEWGEN_DOMAIN (-1) /* means unknown type... */

#include "newgen_list.h"
#include "newgen_stack.h"
#include "newgen_string_buffer.h"
#include "newgen_auto_string.h"

/* Function interface for user applications. */

extern int gen_debug ;

#define GEN_DBG_TRAV_LEAF 1
#define GEN_DBG_TRAV_SIMPLE 2
#define GEN_DBG_TRAV_OBJECT 4
#define GEN_DBG_CHECK 8
#define GEN_DBG_RECURSE 16

#define GEN_DBG_TRAV \
       (GEN_DBG_TRAV_LEAF|GEN_DBG_TRAV_SIMPLE|GEN_DBG_TRAV_OBJECT)

extern void gen_free GEN_PROTO(( gen_chunk * )) ;
extern int gen_free_tabulated GEN_PROTO(( int )) ;
extern void gen_write GEN_PROTO(( FILE *, gen_chunk * )) ;
extern int gen_write_tabulated GEN_PROTO(( FILE *, int )) ;
extern void gen_read_spec GEN_PROTO((char *, ...)) ;
extern gen_chunk *gen_read GEN_PROTO(( FILE * )) ;
extern int gen_read_tabulated GEN_PROTO(( FILE *, int )) ;
extern int gen_read_and_check_tabulated GEN_PROTO(( FILE *, int )) ;
extern gen_chunk *gen_make_array GEN_PROTO(( int )) ;
extern gen_chunk *gen_alloc GEN_PROTO((int, int, int, ...)) ; 
extern char * alloc GEN_PROTO((int));

/* exported type translation functions. */
extern int gen_type_translation_old_to_actual GEN_PROTO((int));
extern int gen_type_translation_actual_to_old GEN_PROTO((int));
extern void gen_type_translation_reset GEN_PROTO((void));
extern void gen_type_translation_default GEN_PROTO((void));
extern void gen_type_translation_read GEN_PROTO((string));
extern void gen_type_translation_write GEN_PROTO((string));

extern void gen_init_external GEN_PROTO((int, 
					 void*(*)(FILE*, int(*)(void)), 
					 void (*)(FILE*, void*), 
					 void (*)(void*), 
					 void* (*)(void*), 
					 int (*)(void*))) ;
extern gen_chunk *gen_check GEN_PROTO(( gen_chunk *, int )) ;
extern bool gen_sharing_p GEN_PROTO((gen_chunk *, gen_chunk *));
extern int gen_type GEN_PROTO((gen_chunk *)) ;
extern string gen_domain_name GEN_PROTO((int)) ;
extern void gen_clear_tabulated_element GEN_PROTO((gen_chunk *)) ;
extern gen_chunk *gen_copy_tree GEN_PROTO((gen_chunk *)) ;
extern gen_chunk *gen_copy_tree_with_sharing GEN_PROTO((gen_chunk *)) ;
extern int gen_consistent_p GEN_PROTO(( gen_chunk * )) ;
extern int gen_tabulated_consistent_p GEN_PROTO((int));
extern int gen_allocated_memory GEN_PROTO((gen_chunk*));
extern int gen_defined_p GEN_PROTO((gen_chunk *));

/*  recursion and utilities
 */
extern bool gen_true GEN_PROTO((gen_chunk *)) ;
extern bool gen_false GEN_PROTO((gen_chunk *)) ;
extern void gen_null GEN_PROTO((void *)) ;
extern void gen_core GEN_PROTO((void *)) ;

extern void gen_recurse_stop GEN_PROTO((void *));
extern void gen_multi_recurse GEN_PROTO((void *, ...));
extern void gen_context_multi_recurse GEN_PROTO((void *, void *,...));

#define gen_recurse(s,d,f,r) \
        gen_multi_recurse(s,d,f,r,NULL)

#define gen_context_recurse(s,c,d,f,r) \
        gen_context_multi_recurse(s,c,d,f,r,NULL)

extern gen_chunk gen_get_recurse_previous_visited_object();
extern gen_chunk gen_get_recurse_ancestor(void * object);
extern void gen_start_recurse_ancestor_tracking();
extern void gen_stop_recurse_ancestor_tracking();


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

/* this macro does about the same as gen_check, but inlined and safer.
 * the item *must* be some newgen allocated structure.
 */
#define NEWGEN_CHECK_TYPE(dom, item)					\
  {									\
    _int __type = dom, __itype;						\
    void * __item = item;						\
    message_assert("valid required domaine number",			\
		   __type>0 && __type<MAX_DOMAIN);			\
    if (Domains[__type].domain &&					\
	Domains[__type].domain->co.type==CONSTRUCTED_DT) {		\
      message_assert("some item", __item!=NULL);			\
      message_assert("item is defined", __item!=gen_chunk_undefined);	\
      __itype = ((gen_chunk*) __item)->i;				\
      if (__itype!=__type) {						\
	message_assert("valid item domain number",			\
		       __itype>0 && __itype<MAX_DOMAIN);		\
	fprintf(stderr, "type error: expecting %s, got %s\n",		\
		Domains[__type].name, Domains[__itype].name);		\
	message_assert("check type", __itype==__type);			\
      }									\
    }									\
    /* else should I say something? Hmmm... */				\
  }

#include "newgen_map.h"
#include "newgen_array.h"

#include "newgen_generic_mapping.h"
#include "newgen_generic_stack.h"
#include "newgen_generic_function.h"

#endif
