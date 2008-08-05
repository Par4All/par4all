/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@cri.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*/


#ifndef newgen_map_included
#define newgen_map_included

/* $Id$
 *
 * These are the functions defined in the Newgen map library. 
 */

#define HASH_GET(start,image,h,k)		\
  hash_map_get((h), (void*)(k))
#define HASH_BOUND_P(start, image, h, k)	\
  hash_map_defined_p((h), (void*)(k))
#define HASH_UPDATE(start,image,h,k,v)		\
  hash_map_update((h), (void*)(k), (void*)(v))
#define HASH_EXTEND(start,image,h,k,v)		\
  hash_map_put((h), (void*)(k), (void*)(v))
#define HASH_DELETE(start,image,h,k)		\
  hash_map_del((h), (void*)(k))

#define FUNCTION_MAP(typename, start, image, k, v, code, fun)		\
  {									\
    hash_table _map_hash_h = ((gen_chunk*)fun+1)->h ;			\
    void * _map_hash_p = NULL;						\
    void * _map_k; void * _map_v;					\
    while ((_map_hash_p =						\
	    hash_table_scan(_map_hash_h,_map_hash_p,&_map_k,&_map_v)))	\
      {									\
	typename##_key_type k =						\
	  (typename##_key_type)((gen_chunk*)_map_k)->start ;		\
	typename##_value_type v =					\
	  (typename##_value_type)((gen_chunk*)_map_v)->image;		\
	code ;								\
      }									\
  }

#endif /* newgen_map_included */
