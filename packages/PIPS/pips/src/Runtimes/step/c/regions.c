#define INLINE static inline
#include "regions.h"

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "trace.h"

/*#####################################################################
  REGION 
#####################################################################*/
void rg_simpleRegion_print(uint32_t nbdims, INDEX_TYPE *myregion)
{
  uint32_t d;

  IN_TRACE("nbdims =%d, myregion = %p", nbdims, myregion);

  for(d=0; d<nbdims; d++)
    printf("dim %d: [%lld; %lld] ", d, (long long int)myregion[LOW(d)], (long long int)myregion[UP(d)] );

  printf("\n");

  OUT_TRACE("end");
}

static void rg_simpleRegion_copy(uint32_t nbdims, INDEX_TYPE *bounds_from, INDEX_TYPE *bounds_to)
{
  memcpy(bounds_to, bounds_from, sizeof(INDEX_TYPE)*BOUNDS(nbdims));
}

static void rg_simpleRegion_intersection(uint32_t nbdims, INDEX_TYPE *bounds_r1, INDEX_TYPE *bounds_r2, composedRegion *reg_inter)
{
  uint32_t d;
  INDEX_TYPE bounds_inter[BOUNDS(nbdims)];
  bool empty = false;

  assert(nbdims == rg_get_userArrayDims(reg_inter));

  for (d=0; !empty && d<nbdims; d++)
    {
      bounds_inter[LOW(d)] = MAX(bounds_r1[LOW(d)], bounds_r2[LOW(d)]);
      bounds_inter[UP(d)] = MIN(bounds_r1[UP(d)], bounds_r2[UP(d)]);
      empty |= bounds_inter[LOW(d)] > bounds_inter[UP(d)];
    }

  if (!empty)
    {
      composedRegion inter;
      rg_composedRegion_set(&inter, nbdims);
      rg_composedRegion_reset(&inter, bounds_inter, 1);
      rg_composedRegion_union(reg_inter, &inter);
      rg_composedRegion_unset(&inter);
    }
}

static void rg_simpleRegion_difference(uint32_t nbdims, INDEX_TYPE *bounds_r1, INDEX_TYPE *bounds_r2, composedRegion *reg_diff)
{
  uint32_t d, nb_chunk;
  composedRegion reg_inter;
  composedRegion chunks;
  INDEX_TYPE new_chunks[1<<nbdims][BOUNDS(nbdims)];

  assert(nbdims == rg_get_userArrayDims(reg_diff));

  rg_composedRegion_set(&chunks, nbdims);
  rg_composedRegion_set(&reg_inter, nbdims);
  rg_simpleRegion_intersection(nbdims, bounds_r1, bounds_r2, &reg_inter);

  if(rg_composedRegion_empty_p(&reg_inter))
    rg_composedRegion_reset(&chunks, bounds_r1, 1);
  else
    {
      assert(rg_get_nb_simpleRegions(&reg_inter) == 1);
      INDEX_TYPE bounds_remaining[BOUNDS(nbdims)];
      INDEX_TYPE *bounds_inter = rg_get_simpleRegion(&reg_inter,0);

      nb_chunk = 0;

      if (!rg_simpleRegion_equal_p(nbdims, bounds_r1, bounds_inter))
      {
	rg_simpleRegion_copy(nbdims, bounds_r1, bounds_remaining);
	
	for (d=0; d<nbdims; d++)
	  {
	    if (bounds_inter[UP(d)] < bounds_remaining[UP(d)])
	      {
		rg_simpleRegion_copy(nbdims, bounds_remaining, new_chunks[nb_chunk]);
		new_chunks[nb_chunk][LOW(d)] = bounds_inter[UP(d)] +1;
		bounds_remaining[UP(d)] = bounds_inter[UP(d)];
		nb_chunk++;
	      }
	    if (bounds_inter[LOW(d)] > bounds_remaining[LOW(d)])
	      {
		rg_simpleRegion_copy(nbdims, bounds_remaining, new_chunks[nb_chunk]);
		new_chunks[nb_chunk][UP(d)] = bounds_inter[LOW(d)] -1;
		bounds_remaining[LOW(d)] = bounds_inter[LOW(d)];
		nb_chunk++;
	      }
	  }
      }
      rg_composedRegion_reset(&chunks, *new_chunks, nb_chunk);
    }

  rg_composedRegion_union(reg_diff, &chunks);
  rg_composedRegion_unset(&chunks);
  rg_composedRegion_unset(&reg_inter);
}

/*#####################################################################
  REGIONS 
#####################################################################*/

void rg_composedRegion_print(composedRegion *myregions)
{
  
  uint32_t nbdims;
  size_t i, nbregions;

  IN_TRACE("myregions = %p", myregions);
  
  assert(myregions);
  nbdims = (uint32_t)rg_get_userArrayDims(myregions);
  nbregions = rg_get_nb_simpleRegions(myregions);
  printf("nbdims = %u, nbregions = %llu\n", nbdims, (long long unsigned)nbregions);
  
  if(rg_composedRegion_empty_p(myregions))
    printf("\t[composedRegion Empty]\n");
  else
    for(i = 0; i < nbregions; i++)
      {
	printf("\t region#%d ", (int)i);
	rg_simpleRegion_print(nbdims, rg_get_simpleRegion(myregions, i));
      }

  OUT_TRACE("end");
}

/*
  r1 = r1 union r2
*/
composedRegion * rg_composedRegion_union(composedRegion *r1, composedRegion *r2)
{
  uint32_t d1;
  uint32_t d2;

  IN_TRACE("r1 = %p, r2 = %p", r1, r2);

  assert(r1 && r2);

  d1 = rg_get_userArrayDims(r1);
  d2 = rg_get_userArrayDims(r2);

  TRACE_P("d1 = %d, d2 = %d", d1, d2);
  assert( d1 == d2 );

  if(r1 != r2)
    array_append_vals(&(r1->simpleRegionArray), r2->simpleRegionArray.data, rg_get_nb_simpleRegions(r2));

  OUT_TRACE("r1= %p", r1);
  return r1;
}

/*
  r1 = r1 inter r2
*/
composedRegion * rg_composedRegion_intersection(composedRegion *r1, composedRegion *r2)
{
  uint32_t d1, d2;
  size_t id_r1, id_r2;
  uint32_t nbdims;
  composedRegion reg_inter, tmp_inter;
  IN_TRACE("r1 = %p, r2 = %p", r1, r2);

  assert(r1 && r2);

  d1 = rg_get_userArrayDims(r1);
  d2 = rg_get_userArrayDims(r2);

  TRACE_P("d1 = %d, d2 = %d", d1, d2);
  assert( d1 == d2 );

  nbdims = rg_get_userArrayDims(r1);
  
  /*
    reg_inter = empty
    foreach R1 element of r1
       foreach R2 element of r2
          tmp_inter = R1 inter R2
          reg_inter = reg_inter union tmp_inter
  */

  rg_composedRegion_set(&reg_inter, nbdims);
  rg_composedRegion_set(&tmp_inter, nbdims);

  for (id_r1=0; id_r1<rg_get_nb_simpleRegions(r1); id_r1++)
    for (id_r2=0; id_r2<rg_get_nb_simpleRegions(r2); id_r2++)
      {
	INDEX_TYPE *bounds_R1;
	INDEX_TYPE *bounds_R2;

	bounds_R1 = rg_get_simpleRegion(r1, id_r1);
	bounds_R2 = rg_get_simpleRegion(r2, id_r2);

	rg_composedRegion_reset(&tmp_inter, NULL, 0);
	rg_simpleRegion_intersection(nbdims, bounds_R1, bounds_R2, &tmp_inter);

	if (!rg_composedRegion_empty_p(&tmp_inter))
	  rg_composedRegion_union(&reg_inter, &tmp_inter);
      }

  // r1=reg_inter
  rg_composedRegion_reset(r1, NULL, 0);
  rg_composedRegion_union(r1, &reg_inter);

  rg_composedRegion_unset(&tmp_inter);
  rg_composedRegion_unset(&reg_inter);

  OUT_TRACE("r1= %p", r1);
  return r1;
}

/*
  r1 = r1 minus r2
*/
composedRegion * rg_composedRegion_difference(composedRegion *r1, composedRegion *r2)
{
  uint32_t d1, d2;
  size_t id_r, id_r2;
  uint32_t nbdims;
  composedRegion reg_diff, tmp_diff;
  IN_TRACE("r1 = %p, r2 = %p", r1, r2);

  assert(r1 && r2);
  
  d1 = rg_get_userArrayDims(r1);
  d2 = rg_get_userArrayDims(r2);

  TRACE_P("d1 = %d, d2 = %d", d1, d2);
  assert( d1 == d2 );

  nbdims = rg_get_userArrayDims(r1);

  rg_composedRegion_set(&tmp_diff, nbdims);
  rg_composedRegion_set(&reg_diff, nbdims);
  
  if(r1 != r2)
    {
      /*
	reg_diff = r1
	foreach R2 element of r2 to remove from reg_diff
	   tmp_diff = empty
	   foreach R element of reg_diff
	      tmp_diff = tmp_diff union (R \ R2)
	   reg_diff=tmp_diff
      */
      rg_composedRegion_union(&reg_diff, r1); 

      for(id_r2=0; id_r2< rg_get_nb_simpleRegions(r2); id_r2++)
	{
	  INDEX_TYPE *bounds_r2 = rg_get_simpleRegion(r2, id_r2);

	  rg_composedRegion_reset(&tmp_diff, NULL, 0);

	  for (id_r=0; id_r<rg_get_nb_simpleRegions(&reg_diff); id_r++)
	    rg_simpleRegion_difference(nbdims, rg_get_simpleRegion(&reg_diff, id_r), bounds_r2, &tmp_diff);
	  
	  rg_composedRegion_reset(&reg_diff, tmp_diff.simpleRegionArray.data, tmp_diff.simpleRegionArray.len);
	}
    }

  // r1=reg_diff
  rg_composedRegion_reset(r1, NULL, 0);
  rg_composedRegion_union(r1, &reg_diff);

  rg_composedRegion_unset(&tmp_diff);
  rg_composedRegion_unset(&reg_diff);

  OUT_TRACE("r1= %p", r1);
  return r1;
}


/*
  Transforme r comme un ensemble de regions disjointes

  A ameliorer : fusionner les regions disjointes et diminuer le nombre d'elements de l'ensemble
*/
composedRegion * rg_composedRegion_simplify(composedRegion *r, composedRegion *r_box)
{
  uint32_t nbdims;
  composedRegion complementary;
  
  IN_TRACE("r = %p, r_box = %p", r, r_box);

  assert(r);
  nbdims = rg_get_userArrayDims(r);
  INDEX_TYPE bounds_box[BOUNDS(nbdims)];

  if (r_box)
    {
      assert(rg_get_nb_simpleRegions(r_box) == 1 && rg_get_userArrayDims(r_box) == nbdims);
      rg_simpleRegion_copy(nbdims, rg_get_simpleRegion(r_box,0), bounds_box);
    }
  else
    {
      size_t id_region;

      rg_simpleRegion_copy(nbdims, rg_get_simpleRegion(r,0), bounds_box);

      for (id_region = 1; id_region < rg_get_nb_simpleRegions(r); id_region++)
	{
	  uint32_t d;
	  INDEX_TYPE *bounds = rg_get_simpleRegion(r, id_region);

	  for (d = 0; d < nbdims; d++)
	    {
	      bounds_box[LOW(d)] = MIN(bounds_box[LOW(d)], bounds[LOW(d)]);
	      bounds_box[UP(d)] = MAX(bounds_box[UP(d)], bounds[UP(d)]);
	    }
	}
    }

  rg_composedRegion_set(&complementary, nbdims);

  rg_composedRegion_reset(&complementary, bounds_box, 1);
  rg_composedRegion_difference(&complementary, r);
  
  rg_composedRegion_reset(r, bounds_box, 1);
  rg_composedRegion_difference(r, &complementary);

  rg_composedRegion_unset(&complementary);

  OUT_TRACE("r = %p", r);
  return r;
}

#ifdef TEST_REGIONS
int main()
{
  int i, r;
  int nbdims = 2;
  int nb_regions = 6;
  INDEX_TYPE bounds_array[nb_regions*BOUNDS(nbdims)];
  composedRegion myregions;
  composedRegion regA, regB;

  /* set traces with commsize = 0 and rank = 0*/
  SET_TRACES("traces", NULL, 1, 0);

  printf("To create 6 regions\n");
  printf("PART I) Creating 6 arrays containing region bounds\n");
  r=0;
  bounds_array[r*BOUNDS(nbdims)+LOW(0)] = 0;
  bounds_array[r*BOUNDS(nbdims)+UP(0)] = 10;
  bounds_array[r*BOUNDS(nbdims)+LOW(1)] = 10;
  bounds_array[r*BOUNDS(nbdims)+UP(1)] = 100;
  r=1;
  bounds_array[r*BOUNDS(nbdims)+LOW(0)] = 2;
  bounds_array[r*BOUNDS(nbdims)+UP(0)] = 20;
  bounds_array[r*BOUNDS(nbdims)+LOW(1)] = 20;
  bounds_array[r*BOUNDS(nbdims)+UP(1)] = 200;
  r=2;
  bounds_array[r*BOUNDS(nbdims)+LOW(0)] = 2;
  bounds_array[r*BOUNDS(nbdims)+UP(0)] = 5;
  bounds_array[r*BOUNDS(nbdims)+LOW(1)] = 20;
  bounds_array[r*BOUNDS(nbdims)+UP(1)] = 50;
  r=3;
  bounds_array[r*BOUNDS(nbdims)+LOW(0)] = 2;
  bounds_array[r*BOUNDS(nbdims)+UP(0)] = 10;
  bounds_array[r*BOUNDS(nbdims)+LOW(1)] = 10;
  bounds_array[r*BOUNDS(nbdims)+UP(1)] = 19;
  r=4;
  bounds_array[r*BOUNDS(nbdims)+LOW(0)] = 0;
  bounds_array[r*BOUNDS(nbdims)+UP(0)] = 2;
  bounds_array[r*BOUNDS(nbdims)+LOW(1)] = 10;
  bounds_array[r*BOUNDS(nbdims)+UP(1)] = 20;
  r=5;
  bounds_array[r*BOUNDS(nbdims)+LOW(0)] = 0;
  bounds_array[r*BOUNDS(nbdims)+UP(0)] = 1;
  bounds_array[r*BOUNDS(nbdims)+LOW(1)] = 20;
  bounds_array[r*BOUNDS(nbdims)+UP(1)] = 100;
  
  for (i=0; i<nb_regions; i++)
    {
      printf("Printing bounds of region %d: ", i);
      rg_simpleRegion_print(nbdims, &(bounds_array[i*BOUNDS(nbdims)]));
    }

  printf("PART II) Creating regions data structure\n");  
  printf("\n Creating a region structure: myregions\n");
  rg_composedRegion_set(&myregions, nbdims);
  
  printf("Test is myregions empty ? %d\n", rg_composedRegion_empty_p(&myregions));  
  printf("Printing myregions...\n");  
  rg_composedRegion_print(&myregions);
  
  
  printf("\nAdd two regions to myregions (first two regions described in bounds_array)...\n");
  /* FSC inutile ? deja fait plus haut
     rg_composedRegion_set(&myregions, nbdims); */

  /* Pourquoi dois-je faire d'abord un set puis un reset???? Dans ce cas, il me semble que reset est mal choisi. */

  rg_composedRegion_reset(&myregions, bounds_array, 2);
  printf("is empty ? %d\n", rg_composedRegion_empty_p(&myregions));
  rg_composedRegion_print(&myregions);
      
  printf("\n Creating 2 region structures: regA and regB\n");
  rg_composedRegion_set(&regA, nbdims);
  rg_composedRegion_set(&regB, nbdims);
  
  printf("\nAdd 1 region in regA\n");
  rg_composedRegion_reset(&regA, bounds_array, 1);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("\nAdd 1 region in regB\n");
  rg_composedRegion_reset(&regB, &(bounds_array[BOUNDS(nbdims)]), 1);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("\n");

  printf("Union of regA and regB in regA\n");
  rg_composedRegion_union(&regA, &regB);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("\n");

  printf("regA=union(regA,myregions)\n");
  rg_composedRegion_union(&regA, &myregions);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("\n");

  printf("regA=inter(regA,regB)\n");
  rg_composedRegion_intersection(&regA, &regB);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("\n");
  
  printf("\nInitialize regA with 1 region\n");
  rg_composedRegion_reset(&regA, bounds_array, 1);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("\n");

  printf("\nInitialize regB with 1 region\n");
  rg_composedRegion_reset(&regB, &(bounds_array[BOUNDS(nbdims)]), 1);
  printf("Union of regB and regB in regB\n");
  rg_composedRegion_union(&regB, &regB);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("\n");

  printf("regA=diff(regA,regB)\n");
  rg_composedRegion_difference(&regA, &regB);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("\n");

  printf("\nInitialize regB with 3 regions\n");
  rg_composedRegion_reset(&regB, &(bounds_array[3*BOUNDS(nbdims)]), 3);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("regB=simplify(regB)\n");
  rg_composedRegion_simplify(&regB, NULL);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("\n");

  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("regB=inter(regB,regA)\n");
  rg_composedRegion_intersection(&regB, &regA);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("\n");
  
  printf("\nreset regB\n");
  rg_composedRegion_reset(&regB, &(bounds_array[BOUNDS(nbdims)]), 1);
  printf("Union of regB and regB in regB\n");
  rg_composedRegion_union(&regB, &regB);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("regB=diff(regB,regB)\n");
  rg_composedRegion_difference(&regB, &regB);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("\n");

  printf("\nInitialize regA with 1 region\n");
  rg_composedRegion_reset(&regA, bounds_array, 1);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("reset regB\n");
  rg_composedRegion_reset(&regB, &(bounds_array[2*BOUNDS(nbdims)]), 1);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("regA=diff(regA,regB)\n");
  rg_composedRegion_difference(&regA, &regB);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("\n");

  printf("\nInitialize regA with 1 region\n");
  rg_composedRegion_reset(&regA, bounds_array, 1);
  printf("Union of regA and myregions in regA\n");
  rg_composedRegion_union(&regA, &myregions);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  printf("\nInitialize regB with 1 region\n");
  rg_composedRegion_reset(&regB, &(bounds_array[2*BOUNDS(nbdims)]), 1);
  printf("Union of regB and myregions in regB\n");
  rg_composedRegion_union(&regB, &myregions);
  printf("Printing regB...\n");  
  rg_composedRegion_print(&regB);
  printf("regA=diff(regA,regB)\n");
  rg_composedRegion_difference(&regA, &regB);
  printf("Printing regA...\n");  
  rg_composedRegion_print(&regA);
  
  return 0;
}

#endif
