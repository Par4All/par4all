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

/* -- map.h

   These are the functions defined in the Newgen map library. 

*/

#define MAX_NESTED_HASH 10

extern gen_chunk Gen_hash_[] ;
gen_chunk *gen_hash_ ;

#define HASH_GET(start,image,h,k) \
IN_STACK(gen_hash_, &Gen_hash_[MAX_NESTED_HASH], \
	 ((gen_hash_-1)->start=(k), \
	  (((gen_chunk *)hash_get((h),(char *)(gen_hash_-- -1)))->image)), \
	 gen_hash_->image)

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

#endif
