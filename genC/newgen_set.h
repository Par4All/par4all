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


#ifndef SET_INCLUDED
#define SET_INCLUDED
/* - set.h
   
   Pierre Jouvelot (3 Avril 1989)

   Set (of CHAR *) package interface. 

   WARNING: You aren't allowed to use = or == between sets. Always use 
   SET_ASSIGN and SET_EQUAL. */

#include "newgen_types.h"
#include "newgen_hash.h"

typedef enum {
    set_string = hash_string ,
    set_int = hash_int,
    set_pointer = hash_pointer
} set_type ;

typedef struct {
    hash_table table ;
    set_type type ;
} set_chunk, *set ;

#define set_undefined ((set)(-16))
#define set_undefined_p(s) ((s)==set_undefined)

#define SET_MAP(element,code,set) \
    { HASH_MAP(_set_map_key, element, code, (set)->table); }

/* functions declared in set.c */

extern set set_add_element GEN_PROTO (( set, set, char *)) ;
extern set set_del_element GEN_PROTO(( set, set, char *)) ;
extern set set_assign GEN_PROTO(( set, set )) ;
extern bool set_belong_p GEN_PROTO(( set, char *)) ;
extern void set_clear GEN_PROTO(( set )) ;
extern set set_difference GEN_PROTO(( set, set, set )) ;
extern bool set_equal GEN_PROTO(( set, set )) ;
extern void set_free GEN_PROTO(( set )) ;
extern set set_intersection GEN_PROTO(( set, set, set )) ;
extern set set_make GEN_PROTO(( set_type )) ;
extern set set_singleton GEN_PROTO(( set_type, char * )) ;
extern set set_union GEN_PROTO(( set, set, set )) ;
extern bool set_empty_p GEN_PROTO(( set )) ;
extern void gen_set_closure GEN_PROTO(( void(*)(char*,set), set )) ;
extern void gen_set_closure_iterate GEN_PROTO((void(*)(char*,set), set, bool));
extern int set_own_allocated_memory GEN_PROTO((set));

#endif
