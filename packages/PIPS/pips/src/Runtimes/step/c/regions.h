#ifndef __REGIONS_H__
#define __REGIONS_H__

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#ifndef INLINE
#define INLINE extern inline
#endif
#include "array.h"

#ifndef INDEX_TYPE
#define INDEX_TYPE intmax_t
#endif

#ifndef MAX
#define MAX(X,Y) ((X)>(Y)) ? X:Y
#endif
#ifndef MIN
#define MIN(X,Y) ((X)<(Y)) ? X:Y
#endif


#define BOUNDS(d) (2*(d))
#define LOW(i) (2*(i))
#define UP(i) (2*(i)+1)

#define BOUNDS_2_START(dim, bounds_array, bounds_sub_array, start_size_type, start_sub_array) \
  {uint32_t d_; for (d_=0; d_<dim; d_++){					\
      (start_sub_array)[d_] = (start_size_type)((bounds_sub_array)[LOW(d_)] - (bounds_array)[LOW(d_)]);}}

#define BOUNDS_2_END(dim, bounds_array, bounds_sub_array, start_size_type, end_sub_array) \
  {uint32_t d_; for (d_=0; d_<dim; d_++){					\
      (end_sub_array)[d_] = (start_size_type)((bounds_sub_array)[UP(d_)] - (bounds_array)[LOW(d_)]);}}

#define BOUNDS_2_SIZES(nbdims, bounds, bounds_sub_array, start_size_type, sizes_sub_array) \
  {uint32_t d_; for (d_=0; d_<nbdims; d_++){					\
      (sizes_sub_array)[d_] = (start_size_type)(1 + (bounds_sub_array)[UP(d_)] - (bounds_sub_array)[LOW(d_)]);}}

/*
  A region R of dimension d with indexes of type INDEX_TYPE is a array :
  INDEX_TYPE R[BOUNDS(d)]
  with :
  R[LOW(i)]and R[UP(i)] the bounds index for ith dimension


  The type composedRegions defines a set of simple regions of a same dimension :
  - uint32_t get_userArrayDims(composedRegion *r) is the number of dimensions of one simple region

  - size_t get_nb_simpleRegions(composedRegion *r) is the number of simple regions in the set
  - get_simpleRegion(composedRegion *r, i) returns the ith region of r
*/

typedef struct
{
  Array simpleRegionArray; /* bounds */
  uint32_t userArrayDims; /* dim */
} composedRegion;


INLINE bool rg_simpleRegion_equal_p(uint32_t nbdims, INDEX_TYPE *bounds_r1, INDEX_TYPE *bounds_r2)
{
  uint32_t d; bool equal = true;
  for (d=0; equal && (d<nbdims); d++)
    equal=((bounds_r1[LOW(d)] == bounds_r2[LOW(d)]) && (bounds_r1[UP(d)] == bounds_r2[UP(d)]));
  return equal;
}

#define rg_get_simpleRegion(compReg, i) ((INDEX_TYPE*)(&(array_get_data_from_index(&((compReg)->simpleRegionArray), INDEX_TYPE, i))))
INLINE size_t rg_get_nb_simpleRegions(composedRegion *compReg){return compReg->simpleRegionArray.len;}
INLINE uint32_t rg_get_userArrayDims(composedRegion *compReg){return compReg->userArrayDims;}


INLINE void rg_composedRegion_set(composedRegion *r, const uint32_t nbdims)
{r->userArrayDims = nbdims;array_sized_set(&(r->simpleRegionArray), sizeof(INDEX_TYPE)*BOUNDS(nbdims));}
INLINE void rg_composedRegion_reset(composedRegion *compReg, const INDEX_TYPE *bounds, const uint32_t nb_regions)
{array_reset(&(compReg->simpleRegionArray), bounds, nb_regions);}
INLINE void rg_composedRegion_unset(composedRegion *compReg)
{array_unset(&(compReg->simpleRegionArray));}
INLINE bool rg_composedRegion_empty_p(composedRegion *compReg)
{return rg_get_nb_simpleRegions(compReg)==0;}

void rg_composedRegion_print(composedRegion *compReg);
composedRegion * rg_composedRegion_union(composedRegion *compReg1, composedRegion *compReg2);        // compReg1 = compReg1 union compReg2
composedRegion * rg_composedRegion_intersection(composedRegion *compReg1, composedRegion *compReg2); // compReg1 = compReg1 inter compReg2
composedRegion * rg_composedRegion_difference(composedRegion *compReg1, composedRegion *compReg2);   // compReg1 = compReg1 minus compReg2
composedRegion * rg_composedRegion_simplify(composedRegion *compReg, composedRegion *compReg_box);   // transforme compReg as a set of disjoined regions

void rg_simpleRegion_print(uint32_t nbdims, INDEX_TYPE *myregion);
#endif //__REGIONS_H__
