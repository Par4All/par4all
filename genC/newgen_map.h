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

/* $RCSfile: newgen_map.h,v $ ($Date: 1998/04/11 11:25:09 $, ) 
 * version $Revision$
 * got on %D%, %T%
 *
 * These are the functions defined in the Newgen map library. 
 */

#define MAX_NESTED_HASH 10

extern gen_chunk Gen_hash_[] ;
extern gen_chunk *gen_hash_ ;

#define HASH_GET(start,image,h,k) \
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 ((gen_hash_-1)->s=(char*)(k), \
	  (((gen_chunk *)hash_get((h),(char *)(gen_hash_-- -1)))->image)), \
	 gen_hash_->image)

#define HASH_BOUND_P(start, image, h, k)\
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 ((gen_hash_-1)->s=(char*)(k), \
	  (bool) (hash_defined_p((h), (char *)(gen_hash_-- -1)))), \
	 (bool) gen_hash_)

#define HASH_UPDATE(start,image,h,k,v) \
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 ((gen_hash_-1)->s=(char*)(k), \
	  IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
		   ((gen_hash_-1)->s=(char*)(v), \
		    *((gen_chunk *)hash_get((h),(char*)(gen_hash_-2)))= \
		    *(gen_hash_-1), \
		    gen_hash_ -=2, \
		    (h)), \
		   (h))), \
	 (h))

#define HASH_EXTEND(start,image,h,k,v) \
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 IN_HEAP((gen_hash_-1)->p, gen_chunk, \
		 ((gen_hash_-1)->p->s=(char*)(k), \
		  IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
			   IN_HEAP((gen_hash_-1)->p, gen_chunk, \
				   ((gen_hash_-1)->p->s=(char*)(v), \
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

#define HASH_DELETE(start,image,h,k) \
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 ((gen_hash_-1)->s=(char*)(k), \
	  (((gen_chunk *)hash_del((h),(char *)(gen_hash_-- -1)))->image)), \
	 gen_hash_->image)

#define FUNCTION_MAP(typename, start, image, k, v, code, fun) \
    { hash_table _map_hash_h = (fun+1)->h ;\
      hash_entry_pointer _map_hash_p = NULL; \
      char *_map_k; char *_map_v; \
      while ((_map_hash_p = \
	   hash_table_scan(_map_hash_h,_map_hash_p,&_map_k,&_map_v))) { \
        typename##_key_type k = (typename##_key_type)((gen_chunk*)_map_k)->start ; \
        typename##_value_type v = (typename##_value_type)((gen_chunk*)_map_v)->image;\
        code ; }}

#endif
