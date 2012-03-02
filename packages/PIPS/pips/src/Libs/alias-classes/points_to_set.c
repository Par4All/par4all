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
  bool cmp1 = locations_equal_p(c1,c2);
  bool cmp2 = locations_equal_p(c3,c4);
  bool cmp3 = ( approximation_exact_p( points_to_approximation(pt1) )
	   &&
	   approximation_exact_p( points_to_approximation(pt2) )
	   ) || ( approximation_may_p( points_to_approximation(pt1) )
		  &&
		  approximation_may_p( points_to_approximation(pt2) )
		  );

  return cmp1 && cmp2 && cmp3;
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
  string key = strdup(concatenate(s1,
				  " ",
				  s2,
				  s,
				  NULL));
  return hash_string_rank(key,size);
}

/* Used to remove local variables and keep static one from blocks */
set points_to_block_projection(set pts, list  l)
{
  set res = set_generic_make(set_private, points_to_equal_p,
				      points_to_rank);
  set_assign(res, pts);
  FOREACH(ENTITY, e, l) {
    SET_FOREACH(points_to, pt, pts){
      cell source = points_to_source(pt);
      cell sink = points_to_sink(pt);
      entity e_sr = reference_variable(cell_to_reference(source));
      entity e_sk = reference_variable(cell_to_reference(sink));
      if(e == e_sr && (!(variable_static_p(e_sr) || top_level_entity_p(e_sr) || heap_cell_p(source))))
	set_del_element(res, res, (void*)pt);
      
      else if(e == e_sk && (!(variable_static_p(e_sk) || top_level_entity_p(e_sk) || heap_cell_p(sink)))){
	pips_user_warning("Dangling pointer %s \n", entity_user_name(e_sr));
	list lhs = CONS(CELL, source, NIL);
	res = points_to_nowhere_typed(lhs, res);
      }
    }
  }
  return res;
}


set points_to_function_projection(set pts)
{
  set res = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  set_assign(res, pts);

  SET_FOREACH(points_to, pt, pts){
    if(cell_out_of_scope_p(points_to_source(pt)))
      set_del_element(res, res, (void*)pt);
  }
  return res;
}

/* Return true if a cell is out of scope */
bool cell_out_of_scope_p(cell c)
{
  reference r = cell_to_reference(c);
  entity e = reference_variable(r);
  return !(variable_static_p(e) ||  entity_stub_sink_p(e) || top_level_entity_p(e) || entity_heap_location_p(e));
}

/*print a points-to for debug*/
void print_points_to(const points_to pt)
{
  cell source = points_to_source(pt);
  cell sink = points_to_sink(pt);
  approximation app = points_to_approximation(pt);
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


/* test if a cell appear as a source in a set of points-to */
bool source_in_set_p(cell source, set s)
{
  bool in_p = false;
  SET_FOREACH ( points_to, pt, s ) {
    /* if( opkill_may_vreference(source, points_to_source(pt) )) */
    if(cell_equal_p(source, points_to_source(pt)))
      return true;
  }
  return in_p;
}

/* test if a cell appear as a sink in a set of points-to */
bool sink_in_set_p(cell sink, set s)
{
  bool in_p = false;
  SET_FOREACH ( points_to, pt, s ) {
    if( cell_equal_p(points_to_sink(pt),sink) )
      in_p = true;
  }
  return in_p;
}


/* merge two points-to sets; required to compute
   the points-to set of the if control statements. */
set merge_points_to_set(set s1, set s2) {
  set Definite_set = set_generic_make(set_private, points_to_equal_p,
				      points_to_rank);
  set Possible_set = set_generic_make(set_private, points_to_equal_p,
				      points_to_rank);
  set Intersection_set = set_generic_make(set_private, points_to_equal_p,
					  points_to_rank);
  set Union_set = set_generic_make(set_private, points_to_equal_p,
				   points_to_rank);
  set Merge_set = set_generic_make(set_private, points_to_equal_p,
				   points_to_rank);

  Intersection_set = set_intersection(Intersection_set, s1, s2);
  Union_set = set_union(Union_set, s1, s2);

  SET_FOREACH ( points_to, i, Intersection_set ) {
    if ( approximation_tag(points_to_approximation(i)) == 2 )
      Definite_set = set_add_element(Definite_set,Definite_set,
				     (void*) i );
  }

  SET_FOREACH ( points_to, j, Union_set ) {
    if ( ! set_belong_p(Definite_set, (void*)j) ) {
      points_to pt = make_points_to(points_to_source(j), points_to_sink(j),
				    make_approximation_may(),
				    make_descriptor_none());
      Possible_set = set_add_element(Possible_set, Possible_set,(void*) pt);
    }
  }

  Merge_set = set_clear(Merge_set);
  Merge_set = set_union(Merge_set, Possible_set, Definite_set);

  set_clear(Intersection_set);
  set_clear(Union_set);
  set_clear(Possible_set);
  set_clear(Definite_set);
  set_free(Definite_set);
  set_free(Possible_set);
  set_free(Intersection_set);
  set_free(Union_set);
  return Merge_set;
}

/* Change the all the exact points-to relations to may relations */
set exact_to_may_points_to_set(set s)
{
  SET_FOREACH ( points_to, pt, s ) {
    if(approximation_exact_p(points_to_approximation(pt)))
      points_to_approximation(pt) = make_approximation_may();
  }
  return s;
}


bool cell_in_list_p(cell c, const list lx)
{
  list l = (list) lx;
  for (; !ENDP(l); POP(l))
    if (points_to_compare_cell(CELL(CAR(l)), c)) return true; /* found! */

  return false; /* else no found */
}

bool points_to_compare_cell(cell c1, cell c2){

  int i = 0;
  reference r1 = cell_to_reference(c1);
  reference r2 = cell_to_reference(c2);
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  list sl1 = NIL, sl2 = NIL;
  extern const char* entity_minimal_user_name(entity);

  i = strcmp(entity_minimal_user_name(v1), entity_minimal_user_name(v2));
  if(i==0) {
    sl1 = reference_indices(r1);
    sl2 = reference_indices(r2);
    int i1 = gen_length(sl1);
    int i2 = gen_length(sl2);

    i = i2>i1? 1 : (i2<i1? -1 : 0);

    for(;i==0 && !ENDP(sl1); POP(sl1), POP(sl2)){
      expression se1 = EXPRESSION(CAR(sl1));
      expression se2 = EXPRESSION(CAR(sl2));
      if(expression_constant_p(se1) && expression_constant_p(se2)){
	int i1 = expression_to_int(se1);
	int i2 = expression_to_int(se2);
	i = i2>i1? 1 : (i2<i1? -1 : 0);
      }else{
	string s1 = words_to_string(words_expression(se1, NIL));
	string s2 = words_to_string(words_expression(se2, NIL));
	i = strcmp(s1, s2);
      }
    }
  }

  return (i== 0 ? true: false) ;
}



/* Order the two points-to relations according to the alphabetical
   order of the underlying variables. Return -1, 0, or 1. */
int points_to_compare_location(void * vpt1, void * vpt2) {
  int i = 0;
  points_to pt1 = *((points_to *) vpt1);
  points_to pt2 = *((points_to *) vpt2);

  cell c1so = points_to_source(pt1);
  cell c2so = points_to_source(pt2);
  cell c1si = points_to_sink(pt1);
  cell c2si = points_to_sink(pt2);

  // FI: bypass of GAP case
  reference r1so = cell_to_reference(c1so);
  reference r2so = cell_to_reference(c2so);
  reference r1si = cell_to_reference(c1si);
  reference r2si = cell_to_reference(c2si);

  entity v1so = reference_variable(r1so);
  entity v2so = reference_variable(r2so);
  entity v1si = reference_variable(r1si);
  entity v2si = reference_variable(r2si);
  list sl1 = NIL, sl2 = NIL;
  // FI: memory leak? generation of a new string?
  extern const char* entity_minimal_user_name(entity);

  i = strcmp(entity_minimal_user_name(v1so), entity_minimal_user_name(v2so));
  if(i==0) {
    i = strcmp(entity_minimal_user_name(v1si), entity_minimal_user_name(v2si));
    if(i==0) {
      sl1 = reference_indices(r1so);
      sl2 = reference_indices(r2so);
      int i1 = gen_length(sl1);
      int i2 = gen_length(sl2);

      i = i2>i1? 1 : (i2<i1? -1 : 0);

      if(i==0) {
	list sli1 = reference_indices(r1si);
	list sli2 = reference_indices(r2si);
	int i1 = gen_length(sli1);
	int i2 = gen_length(sli2);

	i = i2>i1? 1 : (i2<i1? -1 : 0);
	if(i==0) {
	  for(;i==0 && !ENDP(sl1); POP(sl1), POP(sl2)){
	    expression se1 = EXPRESSION(CAR(sl1));
	    expression se2 = EXPRESSION(CAR(sl2));
	    if(expression_constant_p(se1) && expression_constant_p(se2)){
	      int i1 = expression_to_int(se1);
	      int i2 = expression_to_int(se2);
	      i = i2>i1? 1 : (i2<i1? -1 : 0);
	      if(i==0){
		for(;i==0 && !ENDP(sli1); POP(sli1), POP(sli2)){
		  expression sei1 = EXPRESSION(CAR(sli1));
		  expression sei2 = EXPRESSION(CAR(sli2));
		  if(expression_constant_p(sei1) && expression_constant_p(sei2)){
		    int i1 = expression_to_int(sei1);
		    int i2 = expression_to_int(sei2);
		    i = i2>i1? 1 : (i2<i1? -1 : 0);
		  }else{
		    string s1 = words_to_string(words_expression(se1, NIL));
		    string s2 = words_to_string(words_expression(se2, NIL));
		    i = strcmp(s1, s2);
		  }
		}
	      }
	    }else{
	      string s1 = words_to_string(words_expression(se1, NIL));
	      string s2 = words_to_string(words_expression(se2, NIL));
	      i = strcmp(s1, s2);
	    }
	  }
	}
      }
    }
  }
  return i;
}
