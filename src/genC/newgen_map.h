/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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

#ifndef newgen_map_included
#define newgen_map_included

/*
 * These are the functions defined in the Newgen map library. 
 */

#define HASH_GET(start,image,h,k)				\
  hash_map_get((const hash_table)(h), (const void *)(k))
#define HASH_BOUND_P(start, image, h, k)			\
  hash_map_defined_p((const hash_table)(h), (const void *)(k))
#define HASH_UPDATE(start,image,h,k,v)				\
  hash_map_update((h), (const void *)(k), (const void *)(v))
#define HASH_EXTEND(start,image,h,k,v)				\
  hash_map_put((h), (const void *)(k), (const void *)(v))
#define HASH_DELETE(start,image,h,k)		\
  hash_map_del((h), (const void *)(k))

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
