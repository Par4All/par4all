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

/* $RCSfile: newgen_list.h,v $ ($Date: 1995/07/20 17:05:43 $, )
 * version $Revision$
 * got on %D%, %T%
 *
 *  These are the functions defined in the Newgen list library. 
 */

#ifndef LIST_INCLUDED
#define LIST_INCLUDED

#include <sys/stdtypes.h>   /* for size_t, used in malloc.h from malloclib */
#include "malloc.h"

typedef struct cons { 
  gen_chunk car; 
  struct cons *cdr ;
} cons ;
typedef cons *list ;

#define NIL ((cons *)NULL)
#define POP(l) ((l)=(l)->cdr)
#define ENDP(l) ((l)==NIL)

#define list_undefined ((cons *)-3)
#define list_undefined_p(c) ((c)==list_undefined)

#define MAX_NESTED_CONS 10

extern cons *Gen_cp_[] ;
extern cons **gen_cp_ ;

#define CAR(pcons) ((pcons)->car)
#define CDR(pcons) ((pcons)->cdr)

#define IN_STACK(i,s,e,res) \
(((i++)>=(s))? \
 (fprintf(stderr,"Newgen: Too deeply nested IN_STACK\n"),exit(-1),(res)): \
 (e))

#define IN_HEAP(i,s,e,res) ((i)=(s *)alloc( sizeof(s)), (e))
 	    
#define CONS(type,x,l) \
IN_STACK(gen_cp_, &Gen_cp_[MAX_NESTED_CONS], \
	 IN_HEAP(*(gen_cp_-1), struct cons, \
		 (type((*(gen_cp_-1))->car)=(x),\
		  (*(gen_cp_-1))->cdr=(l),\
		  (*--gen_cp_)), \
		 *gen_cp_), \
	 *gen_cp_)

#define MAPL(_map_list_cp,_code,_l) \
	{cons* _map_list_cp = (_l) ; \
	for(;!ENDP(_map_list_cp);POP(_map_list_cp)) _code;}

#define MAP(CASTER, _map_chunkp, _map_code, _map_list) \
{ list _map_chunkp##_consp = _map_list; gen_chunk * _map_chunkp;\
  for(; !ENDP(_map_chunkp##_consp); POP(_map_chunkp##_consp))\
  { _map_chunckp = CASTER(CAR(_map_chunkp##_consp)); _code; }}

/* Fonctions de list.c 
 */
extern cons *gen_append GEN_PROTO(( cons *, cons *)) ;
extern cons *gen_concatenate GEN_PROTO(( cons *, cons * )) ;
extern void gen_copy GEN_PROTO(( gen_chunk *, gen_chunk *)) ;
extern cons *gen_copy_seq GEN_PROTO(( cons * )) ;
extern int gen_eq GEN_PROTO(( gen_chunk *, gen_chunk * )) ;
extern gen_chunk *gen_car GEN_PROTO((list));
extern gen_chunk *gen_identity GEN_PROTO((gen_chunk*));
extern gen_chunk *gen_find GEN_PROTO((gen_chunk *, cons *, 
				  bool (*)(), gen_chunk *(*)() )) ;
extern gen_chunk *gen_find_from_end GEN_PROTO((gen_chunk *, cons *, 
				  bool (*)(), gen_chunk *(*)() )) ;
extern gen_chunk *gen_find_eq GEN_PROTO(( gen_chunk *, cons * )) ;
extern gen_chunk *gen_find_if GEN_PROTO(( bool (*)(), cons *,
					 gen_chunk *(*)())) ;
extern gen_chunk *gen_find_if_from_end GEN_PROTO((bool (*)(), cons *, 
					      gen_chunk *(*)())) ;
extern gen_chunk *gen_find_tabulated GEN_PROTO(( char *, int )) ;
extern cons *gen_filter_tabulated GEN_PROTO(( int (*)(), int )) ;
extern void gen_free_list GEN_PROTO(( cons *)) ;
extern cons *gen_last GEN_PROTO(( cons * )) ;
extern int gen_length GEN_PROTO(( cons * )) ;
extern void gen_map GEN_PROTO(( void (*)(), list )) ;
extern void gen_mapl GEN_PROTO(( void (*)(), cons * )) ;
extern void gen_mapc_tabulated GEN_PROTO(( void (*)(), int )) ;
extern cons *gen_nconc GEN_PROTO(( cons *, cons * )) ;
extern cons *gen_nreverse GEN_PROTO(( cons * )) ;
extern gen_chunk gen_nth GEN_PROTO(( int, list )) ;
extern cons *gen_nthcdr GEN_PROTO(( int, list )) ;
extern char *gen_reduce GEN_PROTO(( char *, char *(*)(), cons * )) ;
extern void gen_remove GEN_PROTO(( cons **, gen_chunk * )) ;
extern cons *gen_some  GEN_PROTO(( bool(*)(), cons * )) ;
extern void gen_insert_after GEN_PROTO((gen_chunk *, gen_chunk *, cons *)) ;
extern list gen_once GEN_PROTO((gen_chunk *, list));
extern bool gen_in_list_p GEN_PROTO((gen_chunk *, list));
extern void gen_sort_list GEN_PROTO((list, int (*)())) ;
extern void gen_closure GEN_PROTO((list (*)(), list));
extern list gen_make_list GEN_PROTO(());

#endif
