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


#ifndef MAP_INCLUDED
#define MAP_INCLUDED

/* $RCSfile: newgen_map.h,v $ ($Date: 1995/04/06 17:12:56 $, ) 
 * version $Revision$
 * got on %D%, %T%
 *
 * These are the functions defined in the Newgen map library. 
 */

#define MAX_NESTED_HASH 10

extern gen_chunk Gen_hash_[] ;
gen_chunk *gen_hash_ ;

#define HASH_GET(start,image,h,k) \
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 ((gen_hash_-1)->start=(k), \
	  (((gen_chunk *)hash_get((h),(char *)(gen_hash_-- -1)))->image)), \
	 gen_hash_->image)

#define HASH_BOUND_P(start, image, h, k)\
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 ((gen_hash_-1)->start=(k), \
	  (bool) (hash_defined_p((h), (char *)(gen_hash_-- -1)))), \
	 (bool) gen_hash_)

#define HASH_UPDATE(start,image,h,k,v) \
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 ((gen_hash_-1)->start=(k), \
	  IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
		   ((gen_hash_-1)->image=(v), \
		    *((gen_chunk *)hash_get((h),(char *)(gen_hash_-2)))= \
		    *(gen_hash_-1), \
		    gen_hash_ -=2, \
		    (h)), \
		   (h))), \
	 (h))

#define HASH_EXTEND(start,image,h,k,v) \
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 IN_HEAP((gen_hash_-1)->p, gen_chunk, \
		 ((gen_hash_-1)->p->start=(k), \
		  IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
			   IN_HEAP((gen_hash_-1)->p, gen_chunk, \
				   ((gen_hash_-1)->p->image=(v), \
				    hash_put((h), \
					     (char *)(gen_hash_-2)->p,\
					     (char *)(gen_hash_-1)->p), \
				    gen_hash_-=2, \
				    (h)), \
				   (h)), \
			   (h)), \
		  (h)), \
		 (h)), \
	 (h))

#define FUNCTION_MAP(typename, start, image, k, v, code, fun) \
    {\
    hash_table _map_hash_h = (fun+1)->h ;\
    register hash_entry_pointer _map_hash_p = hash_table_array(_map_hash_h) ;\
    hash_entry_pointer _map_hash_end = \
	    hash_table_array(_map_hash_h)+hash_table_size(_map_hash_h) ;\
    for( ; _map_hash_p<_map_hash_end ; _map_hash_p++ ) { \
	if( hash_entry_key(_map_hash_p) !=HASH_ENTRY_FREE && \
            hash_entry_key(_map_hash_p) !=HASH_ENTRY_FREE_FOR_PUT) { \
	    typename##_key_type k = ((gen_chunk*)hash_entry_key(_map_hash_p))->start ; \
	    typename##_value_type v = ((gen_chunk*)hash_entry_val(_map_hash_p))->image;\
            code ; }}}

#endif
