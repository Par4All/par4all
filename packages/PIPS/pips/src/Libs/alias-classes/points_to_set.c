/* private implementation of points_to set.

   points_to_equal_p to determine if two points_to relations are equal (same
   source, same sink, same relation)

   points_to_rank   how to compute rank for a points_to element

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "syntax.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "pipsmake.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "transformations.h"
#include "preprocessor.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"
#include "genC.h"

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

    return strcmp(s1,s2);
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
  return cell_equal_p(acc1, acc2);
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
  //int rlt=0;
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
  reference sro = cell_to_reference(source);
  reference sri = cell_to_reference(sink);
  string s1 = strdup(words_to_string(words_reference(sro, NIL)));
  string s2 = strdup(words_to_string(words_reference(sri, NIL)));
 /*  string key = strdup(concatenate(entity_name(location_entity(source)), */
/* 				  " ", */
/* 				  entity_name(location_entity(sink)), */
/* 				  s, */
/* 				  NULL)); */
 
  string key = strdup(concatenate(s1,
				  " ",
				  s2,
				  s,
				  NULL));
  return hash_string_rank(key,size);
}

set points_to_projection(set pts, list  l)
{
  FOREACH(entity, e, l){
    reference r = make_reference(e, NIL);
    cell c = make_cell_reference(r);
    SET_FOREACH(points_to, pt, pts){
      if(points_to_compare_cell(points_to_source(pt), c) && !variable_static_p(e))
	set_del_element(pts, pts, (void*)pt);
      if(points_to_compare_cell(points_to_sink(pt), c) && !variable_static_p(e) ){
	reference r = cell_to_reference(points_to_source(pt));
	entity e = reference_variable(r);
	pips_user_warning("Dangling pointer %s \n", entity_user_name(e));
	list lhs = CONS(CELL, points_to_source(pt), NIL);
	pts = points_to_nowhere_typed(lhs, pts);
      }
    }
  }
  return pts;
}


/*print a points-to for debug*/
void print_points_to(const points_to pt)
{
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
  fprintf(stderr,"->");
  print_reference(r2);
  fprintf(stderr," (%d)\n", approximation_tag(app));
}

/*print a points-to set for debug*/
void print_points_to_set(string what,  set s)
{
  fprintf(stderr,"points-to set %s:\n", what);
  SET_MAP(elt, print_points_to((points_to) elt), s);
  fprintf(stderr, "\n");
}


