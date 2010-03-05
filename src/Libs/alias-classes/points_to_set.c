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
//#include "alias-classes.h"

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

entity access_entity(access acc)
{
  points_to_path p = points_to_path_undefined;
  reference r = reference_undefined;
  entity e = entity_undefined;
 
  if(access_referencing_p(acc)){
    p = access_referencing(acc);
    r = points_to_path_reference(p);
    e = reference_variable(copy_reference(r));
  }
  if(access_dereferencing_p(acc)){
    p = access_dereferencing(acc);
    r = points_to_path_reference(p);
    e = reference_variable(copy_reference(r));
  }
  if(access_addressing_p(acc)){
    p = access_addressing(acc);
    r = points_to_path_reference(p);
    e = reference_variable(r);
  }

  return e;

}

reference access_reference(access acc)
{
  points_to_path p = points_to_path_undefined;
  reference r = reference_undefined;
  if(access_referencing_p(acc)){
    p = access_referencing(acc);
    r = points_to_path_reference(p);
  }
  if(access_dereferencing_p(acc)){
    p = access_dereferencing(acc);
    r = points_to_path_reference(p);
  }
  if(access_addressing_p(acc)){
    p = access_addressing(acc);
    r = points_to_path_reference(p);
  }
  return copy_reference(r);
}


/*return true if two acces_path are equals*/
bool access_equal_p(access acc1, access acc2)
{
  reference r1 = access_reference(acc1);
  reference r2 = access_reference(acc2);
 
  if (reference_equal_p(r1,r2)== true)
    return true;
     else
       return false;

}

/* returns true if two relations are equals. (maybe added to ri-util later)*/
int points_to_equal_p( const void * vpt1, const void*  vpt2)
{
  points_to pt1 = (points_to)vpt1;
  points_to pt2 = (points_to)vpt2;
  points_to_consistent_p(pt1);
  points_to_consistent_p(pt2);
  access a1 = points_to_source(pt1);
  access a2 = points_to_source(pt2);
  reference r1_source = access_reference(a1);
  reference r2_source = access_reference(a2);
  access a3 = points_to_sink(pt1);
  access a4 = points_to_sink(pt2);
  reference r1_sink = access_reference(a3);
  reference r2_sink = access_reference(a4);
  bool cmp1 = true, cmp2 = true, cmp3 = false;
  int rlt=0;
  // if (compare_entities_without_scope(&e1_source, &e2_source)== 0)
  if (reference_equal_p(r1_source,r2_source)== true)
    cmp1 = true;
     else
       cmp1 = false;
	if (reference_equal_p(r1_sink,r2_sink)== true)
//  if (compare_entities_without_scope(reference_variable(r1_sink),reference_variable(r2_sink))== 0)
    cmp2= true;
  else
    cmp2 = false;

  if ((approximation_exact_p(points_to_relation(pt1))) &&
      (approximation_exact_p( points_to_relation(pt2))))
    cmp3 = true;
  if ((approximation_may_p(points_to_relation(pt1))) &&
      (approximation_may_p( points_to_relation(pt2))) )
    cmp3 = true;

  bool cmp =cmp1 && cmp2 && cmp3;

  if(cmp == true)
    rlt = TRUE;
  else
    rlt = FALSE;
  return rlt;
}


/* create a key which is a conctanetion of the source's name, the
  sink's name and the approximation of their relation(may or exact)*/
_uint points_to_rank( const void *  vpt, size_t size)
{
  points_to pt= (points_to)vpt;
  access source = points_to_source(pt);
  access sink = points_to_sink(pt);
  approximation rel = points_to_relation(pt);
  tag rel_tag = approximation_tag(rel);
  string s = strdup(i2a(rel_tag));
	string key = strdup(concatenate(entity_name(access_entity(source)),
																	" ",
																	entity_name(access_entity(sink)),
																	s,
																	NULL));

	
  return hash_string_rank(key,size);
}

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

/*print a points-to for debug*/
void print_points_to(FILE * fd, const void* vpt)
{
  points_to pt = (points_to)vpt;
  access source = points_to_source(pt);
  access sink = points_to_sink(pt);
  approximation app = points_to_relation(pt);

  // fetch the access type of the source
  // entity e1 = access_entity(source);
  points_to_path p1 = access_referencing(source);
  // fetch the access type of the sink
  // entity e2 = access_entity(sink);
  points_to_path p2 = access_referencing(sink);
  reference r1 =points_to_path_reference(p1);
  reference r2 =points_to_path_reference(p2);
 
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


