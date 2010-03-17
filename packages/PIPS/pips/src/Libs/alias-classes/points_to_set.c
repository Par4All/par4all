/* private implementation of points_to set.

   points_to_equal_p to determine if two points_to relations are equal (same
   source, same sink, same relation)

   points_to_rank   how to compute rank for a points_to element

*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "alias-classes.h"

#define INITIAL_SET_SIZE 10

int compare_entities_without_scope(const entity *pe1, const entity *pe2)
{
  int
    null_1 = (*pe1==(entity)NULL),
    null_2 = (*pe2==(entity)NULL);

  if (null_1 || null_2)
    return(null_2-null_1);
  else {
    /* FI: Which sorting do you want? */

    string s1 = entity_name(*pe1);
    string s2 = entity_name(*pe2);

    return strcmp(strdup(s1),strdup(s2));
  }
}

entity location_entity(cell c)
{
  reference r = cell_to_reference(c);
  entity e = reference_variable(r);

  return e;
}

/* Returns a copy of the reference in cell */
reference location_reference(cell c)
{
  return copy_reference(cell_to_reference(c));
}


/*return true if two acces_path are equals*/
bool locations_equal_p(cell acc1, cell acc2)
{
  reference r1 = cell_to_reference(acc1);
  reference r2 = cell_to_reference(acc2);

  return reference_equal_p(r1,r2);
}

/* returns true if two relations are equals. (maybe added to ri-util later)*/
int points_to_equal_p( const void * vpt1, const void*  vpt2)
{
  points_to pt1 = (points_to) vpt1;
  points_to pt2 = (points_to) vpt2;
  cell c1 = points_to_source(pt1);
  cell c2 = points_to_source(pt2);
  cell c3 = points_to_sink(pt1);
  cell c4 = points_to_sink(pt2);
  bool cmp1 = true, cmp2 = true, cmp3 = false;
  int rlt=0;
  // if (compare_entities_without_scope(&e1_source, &e2_source)== 0)
  
	cmp1 = locations_equal_p(c1,c2);
	cmp2 = locations_equal_p(c3,c4);
	cmp3 = (approximation_exact_p(points_to_approximation(pt1)) && approximation_exact_p(points_to_approximation(pt2))) ||
		(approximation_may_p(points_to_approximation(pt1))&& approximation_may_p(points_to_approximation(pt2)));
    
	bool cmp =cmp1 && cmp2 && cmp3;

	return cmp;
}


/* create a key which is a concatenation of the source's name, the
  sink's name and the approximation of their relation(may or exact)*/
_uint points_to_rank( const void *  vpt, size_t size)
{
  points_to pt= (points_to)vpt;
  cell source = points_to_source(pt);
  cell sink = points_to_sink(pt);
  approximation rel = points_to_approximation(pt);
  tag rel_tag = approximation_tag(rel);
  string s = strdup(i2a(rel_tag));
  string key = strdup(concatenate(entity_name(location_entity(source)),
				  " ",
				  entity_name(location_entity(sink)),
				  s,
				  NULL));

  return hash_string_rank(key,size);
}

/* FI->AM:nowuseless
points_to_path access_points_to_path(access a )
{
  points_to_path p = points_to_path_undefined;
  if(access_referencing_p(a))
    p = access_referencing(a);
  if(access_dereferencing_p(a))
    p = access_dereferencing(a);
  if(access_addressing_p(a))
    p = access_addressing(a);
  return copy_points_to_path(p);
}
*/

/*print a points-to for debug*/
void print_points_to(FILE * fd, const void* vpt)
{
  points_to pt = (points_to)vpt;
  cell source = points_to_source(pt);
  cell sink = points_to_sink(pt);
  approximation app = points_to_approximation(pt);

  // fetch the access type of the source
  // entity e1 = access_entity(source);
  // fetch the access type of the sink
  // entity e2 = access_entity(sink);
  reference r1 = cell_to_reference(source);
  reference r2 = cell_to_reference(sink);

  print_reference(r1);
  fprintf(fd,"->");
  print_reference(r2);
  fprintf(fd," (%d)\n", approximation_tag(app));
}

/*print a points-to set for debug*/
void print_points_to_set(FILE *fd, string what,  set s)
{
  fprintf(fd,"points-to set %s:\n", what);
  SET_MAP(elt, print_points_to(fd, (points_to) elt),s);
  fprintf(fd, "\n");
}


