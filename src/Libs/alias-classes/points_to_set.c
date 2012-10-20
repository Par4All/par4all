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

/* returns true if two points-to arcs "vpt1" and "vpt2" are equal.
 *
 * Used to build sets of points-to using the set library of Newgen
 */
int points_to_equal_p( const void * vpt1, const void*  vpt2)
{
  points_to pt1 = (points_to) vpt1;
  points_to pt2 = (points_to) vpt2;

 //same source
  cell c1 = points_to_source(pt1);
  cell c2 = points_to_source(pt2);
  bool cmp1 = locations_equal_p(c1,c2);

 // same sink
  cell c3 = points_to_sink(pt1);
  cell c4 = points_to_sink(pt2);
  bool cmp2 = locations_equal_p(c3,c4);

 // same approximation
  approximation a1 = points_to_approximation(pt1);
  approximation a2 = points_to_approximation(pt2);
  // FI: must is forgotten...
  //bool cmp3 = (approximation_exact_p(a1) && approximation_exact_p(a2))
  //  || ( approximation_may_p(a1) && approximation_may_p(a2));
  // FI: should we identify "exact" and "must"?
  bool cmp3 = (approximation_tag(a1)==approximation_tag(a2));
  bool cmp = cmp1 && cmp2 && cmp3;

  ifdebug(8) {
    printf("%s for pt1=%p and pt2=%p\n", bool_to_string(cmp), pt1, pt2);
    print_points_to(pt1);
    print_points_to(pt2);
  }

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
  string key = strdup(concatenate(s1,
				  " ",
				  s2,
				  s,
				  NULL));
  _uint rank = hash_string_rank(key,size); 
  free(key);
  return rank;
}

/* create a string which is a concatenation of the source's name, the
 * sink's name and the approximation of their relation(may or exact).
 *
 * The same string is used by the function points_to_rank()
 */
string points_to_name(const points_to pt)
{
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
  return key;
}

/* Create a string which is the cell reference in C syntax.
 *
 * A new string is allocated.
 */
string points_to_cell_name(cell source)
{
  reference sro = cell_to_reference(source);
  string key = strdup(words_to_string(words_reference(sro, NIL)));

  return key;
}

/* Remove from "pts" arcs based on at least one local entity in list
 * "l" and preserve those based on static and global entities. This
 * function is called when exiting a statement block.
 *
 * Detection of dangling pointers.
 *
 * Detection of memory leaks. Could be skipped when dealing with the
 * "main" module, but the information would have to be passed down
 * thru an extra parameter.
 *
 * Side-effects on argument "pts".
 */
set points_to_set_block_projection(set pts, list  l, bool main_p)
{
  list pls = NIL; // Possibly lost sinks
  list new_l = NIL; // new arcs to add
  list old_l = NIL; // old arcs to remove
  FOREACH(ENTITY, e, l) {
    type uet = entity_basic_concrete_type(e);
    // The use of "sink_p" is only an optimization
    bool sink_p = pointer_type_p(uet)
      || array_of_pointers_type_p(uet)
      || struct_type_p(uet)
      || array_of_struct_type_p(uet);

    SET_FOREACH(points_to, pt, pts){
      cell source = points_to_source(pt);
      cell sink = points_to_sink(pt);
      entity e_sr = reference_variable(cell_to_reference(source));
      entity e_sk = reference_variable(cell_to_reference(sink));

      if(sink_p && e == e_sr && (!(variable_static_p(e_sr) || top_level_entity_p(e_sr) || heap_cell_p(source)))) {
	set_del_element(pts, pts, (void*)pt);
	if(heap_cell_p(sink)) {
	  /* Check for memory leaks */
	  pls = CONS(CELL, sink, pls);
	}
	// FI: memory leak? sink should be copied and pt be freed...
      }
      else if(e == e_sk
	      && (!(variable_static_p(e_sk)
		    || top_level_entity_p(e_sk)
		    || heap_cell_p(sink)))) {
	if(gen_in_list_p(e_sr, l)) {
	  /* Both the sink and the source disappear: the arc is removed */
	  set_del_element(pts, pts, (void*)pt);
	}
	else {
	  pips_user_warning("Dangling pointer \"%s\" towards \"%s\".\n",
			    entity_user_name(e_sr),
			    entity_user_name(e_sk));
	  // list lhs = CONS(CELL, source, NIL);
	  // pts = points_to_nowhere_typed(lhs, pts);
	  bool to_be_freed;
	  type sink_t = points_to_cell_to_type(sink, &to_be_freed);
	  type n_sink_t = copy_type(sink_t);
	  if(to_be_freed) free_type(sink_t);
	  approximation a = points_to_approximation(pt);
	  points_to npt = make_points_to(copy_cell(source),
					 make_typed_nowhere_cell(n_sink_t),
					 copy_approximation(a),
					 make_descriptor_none());
	  // Since we are looping on pts... no side-effects on pts
	  new_l = CONS(POINTS_TO, npt, new_l);
	  old_l = CONS(POINTS_TO, pt, old_l);
	}
      }
    }
  }

  FOREACH(POINTS_TO, npt, new_l)
    add_arc_to_simple_pt_map(npt, pts);
  gen_free_list(new_l);
  FOREACH(POINTS_TO, pt, old_l)
    remove_arc_from_simple_pt_map(pt, pts);
  gen_free_list(old_l);

  if(!main_p) {
    /* Any memory leak? */
    FOREACH(CELL, c, pls) {
      if(!sink_in_set_p(c, pts)) {
	reference r = cell_any_reference(c);
	entity b = reference_variable(r);
	pips_user_warning("Memory leak for bucket \"%s\".\n",
			  entity_name(b));
      }
    }
  }
  return pts;
}

/* Remove all arcs starting from e because e has been assigned a new value */
set points_to_source_projection(set pts, entity e)
{
  list pls = NIL; // Possibly lost sinks

  SET_FOREACH(points_to, pt, pts) {
    cell source = points_to_source(pt);
    cell sink = points_to_sink(pt);
    entity e_sr = reference_variable(cell_to_reference(source));
    //entity e_sk = reference_variable(cell_to_reference(sink));

    if(e == e_sr) {
      set_del_element(pts, pts, (void*)pt);
      if(heap_cell_p(sink)) {
	/* Check for memory leaks */
	pls = CONS(CELL, sink, pls);
      }
    }
  }

  /* Any memory leak? */
  FOREACH(CELL, c, pls) {
    if(!sink_in_set_p(c, pts)) {
      reference r = cell_any_reference(c);
      entity b = reference_variable(r);
      pips_user_warning("Memory leak for bucket \"%s\".\n",
			entity_name(b));
    }
  }
  return pts;
}

/* Remove all arcs in "ptg" starting from "c" */
points_to_graph points_to_cell_source_projection(points_to_graph ptg, cell c)
{
  set pts = points_to_graph_set(ptg);

  SET_FOREACH(points_to, pt, pts) {
    cell source = points_to_source(pt);

    if(cell_equal_p(source, c)) {
      set_del_element(pts, pts, (void*)pt);
    }
  }

  return ptg;
}

/* FI: side-effects to be used explictly in this function */
set points_to_function_projection(set pts)
{
  set res = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  set_assign(res, pts);
  /* Do we have a useful return value? */
  type ft = ultimate_type(entity_type(get_current_module_entity()));
  type rt = ultimate_type(functional_result(type_functional(ft)));
  entity rv = entity_undefined;
  if(pointer_type_p(rt))
    rv = function_to_return_value(get_current_module_entity());

  SET_FOREACH(points_to, pt, pts) {
    if(cell_out_of_scope_p(points_to_source(pt))) {
      /* Preserve the return value */
      reference r = cell_any_reference(points_to_source(pt));
      entity v = reference_variable(r);
      if(rv!=v) {
	/* Also preserve nowhere generated by free although the
	   pointer itself is not written. */
	cell sink = points_to_sink(pt);
	if(!nowhere_cell_p(sink))
	  set_del_element(res, res, (void*)pt);
      }
    }
  }
  return res;
}

/* Return true if a cell is out of scope
 *
 * FI: I add formal parameters as in scope variables...
 *
 * FI: I remove formal parameters because this function is used to
 * compute the OUT set. The modified or not values of formal parameter
 * are not relevant. If they have not been modified, the useful
 * information is already available in the IN set (oops, see next
 * comment below). If they have been modified, they are no longer
 * reachable and must be projected.
 *
 * FI: Unfortunately, some information gathered about the input
 * parametrs during the function analysis is lost. For instance, a
 * pointer must be different from NULL (e.g. see argv03.c). But if you
 * do not know if the pointer has been written or not, you do not know
 * if the information is usable or not. This is also an issue for
 * interprocedural analysis: can the result always be trusted for any
 * actual input context?
 */
bool cell_out_of_scope_p(cell c)
{
  reference r = cell_to_reference(c);
  entity e = reference_variable(r);
  return !(variable_static_p(e) ||  entity_stub_sink_p(e) || top_level_entity_p(e) || entity_heap_location_p(e) /*|| formal_parameter_p(e)*/ );
}

/* print a points-to arc for debug */
void print_or_dump_points_to(const points_to pt, bool print_p)
{
  if(points_to_undefined_p(pt))
    (void) fprintf(stderr,"POINTS_TO UNDEFINED\n");
  // For debugging with gdb, dynamic type checking
  else if(points_to_domain_number(pt)!=points_to_domain)
    (void) fprintf(stderr,"Arg. \"pt\"is not a points_to.\n");
  else {
    cell source = points_to_source(pt);
    cell sink = points_to_sink(pt);
    approximation app = points_to_approximation(pt);
    reference r1 = cell_to_reference(source);
    reference r2 = cell_to_reference(sink);

    fprintf(stderr,"%p ", pt);
    print_reference(r1);
    fprintf(stderr,"->");
    print_reference(r2);
    fprintf(stderr," (%s)", approximation_to_string(app));
    if(!print_p) {
      fprintf(stderr," [%p, %p]", points_to_source(pt), points_to_sink(pt));
    }
    fprintf(stderr,"\n");
  }
}

void print_points_to(const points_to pt)
{
  print_or_dump_points_to(pt, true);
}

void dump_points_to(const points_to pt)
{
  print_or_dump_points_to(pt, false);
}

/* Print a set of points-to for debug */
void print_or_dump_points_to_set(string what,  set s, bool print_p)
{
  fprintf(stderr,"points-to set %s:\n", what);
  if(set_undefined_p(s))
    fprintf(stderr, "undefined set\n");
  else if(s==NULL)
    fprintf(stderr, "uninitialized set\n");
  else if(set_size(s)==0)
    fprintf(stderr, "empty set\n");
  else {
    if(print_p) {
      SET_MAP(elt, print_points_to((points_to) elt), s);
    }
    else {
      SET_MAP(elt, dump_points_to((points_to) elt), s);
    }
  }
  fprintf(stderr, "\n");
}

void print_points_to_set(string what,  set s)
{
  print_or_dump_points_to_set(what, s, true);
}

void dump_points_to_set(string what,  set s)
{
  print_or_dump_points_to_set(what, s, false);
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
/* test if a cell "source" appears as a source in a set of points-to */
bool source_subset_in_set_p(cell source, set s)
{
  bool in_p = false;
  SET_FOREACH ( points_to, pt, s ) {
    /* if( opkill_may_vreference(source, points_to_source(pt) )) */
    if(cell_equal_p(source, points_to_source(pt))) {
      in_p = true;
      break;
    }
    else if(cell_included_p(points_to_source(pt), source)) {
      in_p = true;
      break;
    }
  }
  return in_p;
}

bool source_in_graph_p(cell source, points_to_graph s)
{
  return source_in_set_p(source, points_to_graph_set(s));
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

/* The approximation is not taken into account
 *
 * It might be faster to look up the different points-to arcs that can
 * be made with source, sink and any approximation.
 */
points_to find_arc_in_points_to_set(cell source, cell sink, pt_map ptm)
{
  // FI: no longer compatible with definition of pt_map as set
  set s = points_to_graph_set(ptm);
  points_to fpt = points_to_undefined;
  SET_FOREACH(points_to, pt, s) {
    if(cell_equal_p(points_to_source(pt), source)
       && cell_equal_p(points_to_sink(pt), sink) ) {
      fpt = pt;
      break;
    }
  }
  return fpt;
}

/* source is assumed to be either nowhere/undefined or anywhere, it
 * may be typed or not.
 *
 * Shouldn't we create NULL pointers if the corresponding property is set?
 * Does it matter for anywhere/nowhere?
 *
 * pts must be updated with the new arc(s).
 */
list anywhere_source_to_sinks(cell source, pt_map pts)
{
    /* FI: we should return an anywhere cell with the proper type */
    /* FI: should we add the corresponding arcs in pts? */
    /* FI: should we take care of typed anywhere as well? */

  list sinks = NIL;
  reference r = cell_any_reference(source);
  entity v = reference_variable(r);
  // FI: it would be better to print the reference... words_reference()...
  // FI: sinks==NIL would be interpreted as a bug...
  // If we dereference a nowhere/undefined, we should end up
  // anywhere to please gcc and Fabien Coelho
  type vt = ultimate_type(entity_type(v));
  if(type_variable_p(vt)) {
    if(pointer_type_p(vt)) {
      // FI: should nt be freed?
      type nt = type_to_pointed_type(vt);
      cell c = make_anywhere_points_to_cell(nt);
      sinks = CONS(CELL, c, NIL);
      points_to pt = make_points_to(copy_cell(source), c,
				    make_approximation_may(),
				    make_descriptor_none());
      pts = add_arc_to_pt_map(pt, pts);
    }
    else if(struct_type_p(vt)) {
      variable vrt = type_variable(vt);
      basic b = variable_basic(vrt);
      entity se = basic_derived(b);
      type st = entity_type(se);
      pips_assert("se is an internal struct", type_struct_p(st));
      list fl = type_struct(st);
      FOREACH(ENTITY, f, fl) {
	type ft = entity_type(f);
	type uft = compute_basic_concrete_type(ft); // FI: to be freed...
	if(pointer_type_p(uft)) {
	  cell nsource = copy_cell(source); // FI: memory leak?
	  nsource = points_to_cell_add_field_dimension(nsource, f);
	  type nt = type_to_pointed_type(uft);
	  cell nsink = make_anywhere_points_to_cell(nt);
	  points_to pt = make_points_to(nsource, nsink, 
					make_approximation_may(),
					make_descriptor_none());
	  pts = add_arc_to_pt_map(pt, pts);
	  //sinks = source_to_sinks(source, pts, false);
	  sinks = CONS(CELL, nsink, NIL);
	}
	else if(struct_type_p(uft)) {
	  cell nsource = copy_cell(source); // FI: memory leak?
	  nsource = points_to_cell_add_field_dimension(nsource, f);
	  sinks = anywhere_source_to_sinks(nsource, pts);
	  //pips_internal_error("Not implemented yet.\n");
	}
	else if(array_type_p(uft)) {
	  variable uftv = type_variable(uft);
	  basic uftb = variable_basic(uftv);
	  if(basic_pointer_p(uftb)) {
	    cell nsource = copy_cell(source); // FI: memory leak?
	    reference r = cell_any_reference(nsource);
	    reference_add_zero_subscripts(r, uft);
	    type nt = ultimate_type(uft); // FI: get rid of the dimensions
	    cell nsink = make_anywhere_points_to_cell(nt);
	    points_to pt = make_points_to(nsource, nsink, 
					  make_approximation_may(),
					  make_descriptor_none());
	    pts = add_arc_to_pt_map(pt, pts);
	    sinks = CONS(CELL, nsink, NIL);
	  }
	}
      }
    }
    else if(array_type_p(vt)) {
      variable uftv = type_variable(vt);
      basic uftb = variable_basic(uftv);
      if(basic_pointer_p(uftb)) {
	cell nsource = copy_cell(source); // FI: memory leak?
	reference r = cell_any_reference(nsource);
	reference_add_zero_subscripts(r, vt);
      }
    }
    else
      // FI: struct might be dereferenced?
      // FI: should this be tested when entering this function rather
      // than expecting that the caller is safe
      pips_internal_error("Unexpected dereferenced type.\n");
  }
  else if(type_area_p(vt)) {
    // FI: this should be removed using type_overloaded for the
    // "untyped" anywhere
    entity e = entity_anywhere_locations();
    reference r = make_reference(e, NIL);
    cell c = make_cell_reference(r);
    sinks = CONS(CELL, c, NIL);
    points_to pt = make_points_to(copy_cell(source), c,
				  make_approximation_may(),
				  make_descriptor_none());
    pts = add_arc_to_pt_map(pt, pts);
  }
  else {
    pips_internal_error("Unexpected dereferenced type.\n");
  }
  return sinks;
}

/* For debugging */
void print_points_to_path(list p)
{
  if(ENDP(p))
    fprintf(stderr, "p is empty.\n");
  else {
    FOREACH(CELL, c, p) {
      if(c!=CELL(CAR(p)))
	fprintf(stderr, "->");
      print_points_to_cell(c);
    }
    fprintf(stderr, "\n");
  }
}

/* A type "t" is compatible with a cell "c" if any of the enclosing
 * cell "c'" of "c", including "c", is of type "t".
 *
 * For instance, "a.next" is included in "a". It is compatible with
 * both the type of "a" and the type of "a.next".
 */
bool type_compatible_with_points_to_cell_p(type t, cell c)
{
  cell nc = copy_cell(c);
  reference nr = cell_any_reference(nc);
  //list sl = reference_indices(nr);
  bool compatible_p = false;

  do {
    bool to_be_freed;
    type nct = cell_to_type(nc, &to_be_freed);
    if(concrete_array_pointer_type_equal_p(t, nct)) {
      compatible_p = true;
      if(to_be_freed) free_type(nct);
      break;
    }
    else if(ENDP(reference_indices(nr))) {
      if(to_be_freed) free_type(nct);
      break;
    }
    else {
      if(to_be_freed) free_type(nct);
      /* Remove the last subscript */
      expression l = EXPRESSION(CAR(gen_last(reference_indices(nr))));
      gen_remove(&reference_indices(nr), l);
    }
  } while(true);

  ifdebug(8) {
    bool to_be_freed;
    type ct = cell_to_type(nc, &to_be_freed);
    pips_debug(8, "Cell of type \"%s\" is %s included in cell of type \"%s\"\n",
	       type_to_full_string_definition(t),
	       compatible_p? "":"not",
	       type_to_full_string_definition(ct));
      if(to_be_freed) free_type(ct);
  }

  free_cell(nc);

  return compatible_p;
}

/* See if a super-cell of "c" exists witf type "t". A supercell is a
 * cell "nc" equals to cell "c" but with a shorter subscript list.
 *
 * This function is almost identical to the previous one.
 *
 * A new cell is allocated and returned.
 */
cell type_compatible_super_cell(type t, cell c)
{
  cell nc = copy_cell(c);
  reference nr = cell_any_reference(nc);
  //list sl = reference_indices(nr);
  bool compatible_p = false;

  do {
    bool to_be_freed;
    type nct = cell_to_type(nc, &to_be_freed);
    if(type_equal_p(t, nct)) {
      compatible_p = true;
      if(to_be_freed) free_type(nct);
      break;
    }
    else if(ENDP(reference_indices(nr))) {
      if(to_be_freed) free_type(nct);
      break;
    }
    else {
      if(to_be_freed) free_type(nct);
      /* Remove the last subscript */
      expression l = EXPRESSION(CAR(gen_last(reference_indices(nr))));
      gen_remove(&reference_indices(nr), l);
    }
  } while(true);

  ifdebug(8) {
    bool to_be_freed;
    type ct = cell_to_type(nc, &to_be_freed);
    pips_debug(8, "Cell of type \"%s\" is %s included in cell of type \"%s\"\n",
	       type_to_full_string_definition(t),
	       type_to_full_string_definition(ct),
	       compatible_p? "":"not");
    if(to_be_freed) free_type(ct);
    if(compatible_p) {
      pips_debug(8, "Type compatible cell \"");
      print_points_to_cell(nc);
      fprintf(stderr, "\"\n.");
    }
  }

  return nc;
}


/* Find the "k"-th node of type "t" in list "p". Beware of cycles? No
 * reason since "p" is bounded... The problem must be addressed when
 * "p" is built.
 *
 * An issue with "t": the nodes are references and they carry multiple
 * types, one for each number of subscripts or fields they have. So
 * for instance, s1 and s1.next denote the same location.
 */
cell find_kth_points_to_node_in_points_to_path(list p, type t, int k)
{
  int count = 0;
  cell kc = cell_undefined;
  FOREACH(CELL, c, p) {
    // bool to_be_freed;
    // type ct = cell_to_type(c, &to_be_freed);
    // if(type_equal_p(t, ct)) {
    if(type_compatible_with_points_to_cell_p(t, c)) {
      count++;
      if(count==k) {
	kc = type_compatible_super_cell(t,c);
	break;
      }
    }
  }
  ifdebug(8) {
    pips_debug(8, "Could not find %d nodes of type \"%s\" in path \"", k,
	       type_to_full_string_definition(t));
    print_points_to_path(p);
    fprintf(stderr, "\"\n");
  }
  return kc;
}

bool node_in_points_to_path_p(cell n, list p)
{
  bool in_path_p = false;
  FOREACH(CELL, c, p) {
    if(cell_equal_p(c, n)) {
      in_path_p = true;
      break;
    }
  }
  return in_path_p;
}

/* "p" is a points-to path ending with a cell that points towards a
 * new cell ot type "t". To avoid creating infinite/unbounded path, no
 * more than k nodes of type "t" can be present in path "p". If k are
 * found, a cycle is created to represent longer paths. The
 * corresponding arc is returned. If the creation condition is not
 * met, do not create a new arc.
 */
points_to points_to_path_to_k_limited_points_to_path(list p,
						     int k,
						     type t,
						     bool array_p,
						     pt_map in)
{
  pips_assert("p contains at least one element", !ENDP(p));

  points_to pt = points_to_undefined;
  cell c = CELL(CAR(p));
  list sources = sink_to_sources(c, points_to_graph_set(in), false); // No duplication of cells

  if(ENDP(sources)) {
    /* The current path cannot be made any longer */

    /* Find the k-th node of type "t" if it exists */
    cell kc = find_kth_points_to_node_in_points_to_path(p, t, k);
    if(!cell_undefined_p(kc)) {
      // cell nkc = copy_cell(kc);
      // The above function should return a freshly allocated cell or
      // a cell_undefined
      cell nkc = kc;
      cell source = copy_cell(CELL(CAR(gen_last(p))));
      pt = make_points_to(source, nkc, make_approximation_may(),
			  make_descriptor_none());
    }
  }
  else {
    FOREACH(CELL, source, sources) {
      /* Skip sources that are already in the path "p" so as to avoid
	 infinite path due to cycles in points-to graph "in". */
      if(node_in_points_to_path_p(source, p)) {
	; // Could be useful for debugging
      }
      else {
	list np = CONS(CELL, source, p); // make the path longer
	// And recurse
	pt = points_to_path_to_k_limited_points_to_path(np, k, t, array_p, in);
	// And restore p
	CDR(np) = NIL;
	gen_free_list(np);
	if(!points_to_undefined_p(pt)) // Stop as soon as an arc has been created; FI->AM/FC: may not be correct...
	  break;
      }
    }
  }
  return pt;
}


/* Create a new node "sink" of type "t" and a new arc "pt" starting
 * from node "source", if no path starting from any node and ending in
 * "source", built with arcs in the points-to set "in", contains more
 * than k nodes of type "t" (the type of the sink). If k nodes of type
 * "t" are already in the path, create a new arc "pt" between the
 * "source" and the k-th node in the path.
 *
 * Parameter "array_p" indicates if the source is an array or a
 * scalar. Different models can be chosen. For instance, Beatrice
 * Creusillet wants to have an array as target and obtain something
 * like argv[*]->_argv_1[*] although argv[*]->_argv-1 might also be a
 * correct model if _argv_1 is an abstract location representing lots
 * of different physical locations.
 *
 * Parameter k is defined by a property.
 *
 * FI: not to clear about what is going to happen when "source" is the
 * final node of several paths.
 *
 * Also, beware of circular paths.
 *
 * Efficiency is not yet a goal...
 */
points_to create_k_limited_stub_points_to(cell source, type t, bool array_p, pt_map in)
{
  int k = get_int_property("POINTS_TO_PATH_LIMIT");
  pips_assert("k is greater than one", k>=1);
  points_to pt = points_to_undefined;

  // FI: the vertex with an excessive out-degree does not have to be
  // source and is not source in list05 case... The test below is useless

  // We should check the out-degree of each source in "in" and see if
  // any is beyond limit.

  //list sink_l = points_to_source_to_sinks(source, in, false);
  //int od = (int) gen_length(sink_l);
  //string mod_cell_name = string_undefined; // maximum out-degree cell
  //int od = maximal_out_degree_of_points_to_graph(&mod_cell_name, in);
  // list sink_l = points_to_source_to_sinks(mod_cell_name, in, false);
  //list sink_l = points_to_source_name_to_sinks(mod_cell_name, in, false);
  if(false /*&& od>=odl*/ ) {
    // FI: not too sure about argument "true"
    //cell mod_cell = points_to_source_name_to_source_cell(mod_cell_name, in, true);
    //pt = fuse_points_to_sink_cells(mod_cell, sink_l, in);
    ;
  }
  else {
    list p = CONS(CELL, source, NIL); // points-to path...

    // FI: not to sure about he possible memory leaks...
    pt = points_to_path_to_k_limited_points_to_path(p, k, t, array_p, in);

    /* No cycle could be created, the paths can safely be made longer. */
    if(points_to_undefined_p(pt)) {
      // exact or not?
      // FI: the points-to towards NULL will be added later by a caller...
      bool exact_p = !get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
      pt = create_stub_points_to(source, t, exact_p);
    }
    gen_free_list(p);
  }
  return pt;
}

/* Build a list of possible cell sources for cell "sink" in points-to
 * graph "pts". If fresh_p is set, allocate new cells, if not just
 * build the spine of the list.
 */
list sink_to_sources(cell sink, set pts, bool fresh_p)
{
  list sources = NIL;

  // FI: This is a short-term short cut
  // The & operator can be applied to anything
  // We should start with all subscripts and then get rid of subscript
  // one by one and each time add a new source

  /* Get rid of the constant subscripts since they are not direclty
     part of the points-to scheme on the sink side */
  entity v = reference_variable(cell_any_reference(sink));
  reference nr = make_reference(v, NIL);
  cell nsink = make_cell_reference(nr);

  /* 1. Try to find the source in the points-to information */
  SET_FOREACH(points_to, pt, pts) {
    // FI: a more flexible test is needed as the sink cell may be
    // either a, or a[0] or a[*] or a[*][*] or...
    if(cell_equal_p(nsink, points_to_sink(pt))) {
      cell sc = fresh_p? copy_cell(points_to_source(pt))
	: points_to_source(pt);
      sources = CONS(CELL, sc, sources);
    }
  }
  free_cell(nsink);
  return sources;
}

list stub_source_to_sinks(cell source, pt_map pts, bool fresh_p)
{
  list sinks = NIL;
  reference r = cell_any_reference(source);
  list sl = reference_indices(r);
  if(ENDP(sl))
    sinks = scalar_stub_source_to_sinks(source, pts, fresh_p);
  else
    sinks = array_stub_source_to_sinks(source, pts, fresh_p);
  return sinks;
}

list scalar_stub_source_to_sinks(cell source, pt_map pts, bool fresh_p)
{
  list sinks = generic_stub_source_to_sinks(source, pts, false, fresh_p);
  return sinks;
}

list array_stub_source_to_sinks(cell source, pt_map pts, bool fresh_p)
{
  list sinks = generic_stub_source_to_sinks(source, pts, true, fresh_p);
  return sinks;
}

list generic_stub_source_to_sinks(cell source, pt_map pts, bool array_p, bool fresh_p)
{
  list sinks = NIL;
  bool null_initialization_p =
    get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
  //type ost = ultimate_type(entity_type(v));
  bool to_be_freed; // FI: memory leak for the time being
  type rt = cell_to_type(source, &to_be_freed); // reference type
  if(pointer_type_p(rt)) {
    // bool array_p = array_type_p(rt); FI: always false if pointer_type_p(rt)
    type nst = type_to_pointed_type(rt);
    points_to pt = create_k_limited_stub_points_to(source, nst, array_p, pts);
    pts = add_arc_to_pt_map(pt, pts);
    add_arc_to_points_to_context(copy_points_to(pt));
    sinks = source_to_sinks(source, pts, fresh_p);
    if(null_initialization_p) {
      // FI: I'm lost here... both with the meaning of null
      // initialization_p and with the insertion of a
      // corresponding arc in "pts"
      list ls = null_to_sinks(source, pts);
      sinks = gen_nconc(ls, sinks);	 
    }
  }
  else if(struct_type_p(rt)) {
    // FI FI FI - to be really programmed with the field type
    // FI->AM: I am really confused about what I am doing here...
    variable vrt = type_variable(rt);
    basic b = variable_basic(vrt);
    entity se = basic_derived(b);
    type st = entity_type(se);
    pips_assert("se is an internal struct", type_struct_p(st));
    list fl = type_struct(st);
    FOREACH(ENTITY, f, fl) {
      type ft = entity_type(f);
      type uft = ultimate_type(ft);
      bool array_p = array_type_p(ft); // not consistent with ultimate_type()
      points_to pt = create_k_limited_stub_points_to(source, uft, array_p, pts);
      pts = add_arc_to_pt_map(pt, pts);
      add_arc_to_points_to_context(copy_points_to(pt));
      sinks = source_to_sinks(source, pts, fresh_p);
    }
  }
  else if(array_type_p(rt)) {
    if(array_of_pointers_type_p(rt)) {
      cell ns = copy_cell(source);
      points_to_cell_add_unbounded_subscripts(ns);
      basic nsb = copy_basic(variable_basic(type_variable(rt)));
      type nst = make_type_variable(make_variable(nsb, NIL, NIL));
      type nspt = type_to_pointed_type(nst);
      bool array_p = true;
      points_to pt = create_k_limited_stub_points_to(ns, nspt, array_p, pts);
      pts = add_arc_to_pt_map(pt, pts);
      add_arc_to_points_to_context(copy_points_to(pt));
      // sinks = source_to_sinks(source, pts, false); should be ns instead of source
      points_to_cell_add_zero_subscripts(source);
      sinks = source_to_sinks(source, pts, fresh_p);
      //sinks = source_to_sinks(ns, pts, false);
      //sinks = CONS(CELL, copy_cell(points_to_sink(pt)), NIL);
      // FI: this is usually dealt with at the source_to_sinks() level...
      list nsinks = points_to_cell_null_initialization(ns, pts);
      sinks = gen_nconc(sinks, nsinks);
    }
    else {
      reference r = cell_any_reference(source);
      entity v = reference_variable(r);
      printf("Entity \"%s\"\n", entity_local_name(v));
      pips_internal_error("Not implemented yet.\n");
    }
  }
  else {
    /* The source type cannot contain a pointer field: for instance,
       int or char */
    fprintf(stderr, "Type of source: \"");
    dprint((expression)rt);
    fprintf(stderr, "\"\n");
    pips_internal_error("Type of source is incompatible with a source\n");
  }
  return sinks;
}

/* If the subscripts of the effective cell source "ec" are more
 * precise than the subscripts of the cell "fc" found in the points-to
 * set, update the subscript of the sink cell "sc" accordingly.
 *
 * For instance, if "ec==q[0]" and "fc=q[*]" and "sc=_q_1[*]",
 * transform "sc" into "_q_1[0]".
 */
static void refine_points_to_cell_subscripts(cell sc, cell ec, cell fc)
{
  reference rsc = cell_any_reference(sc);
  reference rec = cell_any_reference(ec);
  reference rfc = cell_any_reference(fc);
  list slrsc = reference_indices(rsc);

  if(!ENDP(slrsc)) {
    list slrec = reference_indices(rec);
    list slrfc = reference_indices(rfc);
    list cslrsc = slrsc;
    list cslrec = slrec;
    list cslrfc = slrfc;
    for( ; !ENDP(cslrsc) && !ENDP(cslrec) && !ENDP(cslrfc);
	 POP(cslrsc), POP(cslrec), POP(cslrfc)) {
      expression ecs = EXPRESSION(CAR(cslrec));
      expression fcs = EXPRESSION(CAR(cslrfc));
      // FI: a less crude test would save some work, but this is not
      // the point right now...
      if(unbounded_expression_p(fcs)) {
	free_expression(EXPRESSION(CAR(cslrsc)));
	EXPRESSION_(CAR(cslrsc)) = copy_expression(ecs);
      }
    }
  }
}

/* If required according to the property, create a new arc from cell
 * "c" to "null". Cell "c" is absorbed not by the points_to created and
 * added to set "pts".
 */
list points_to_cell_null_initialization(cell c, pt_map pts)
{
  list sinks = NIL;
  bool null_initialization_p =
    get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
  if(null_initialization_p) {
    sinks = null_to_sinks(c, pts);
  }
  return sinks;
}

list nowhere_source_to_sinks(cell source, pt_map pts)
{
  list sinks = NIL;
  bool uninitialized_dereferencing_p =
    get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING");

  if(uninitialized_dereferencing_p) {
    reference r = cell_any_reference(source);
    entity v = reference_variable(r);
    pips_user_warning("Possibly undefined pointer \"%s\" is dereferenced.\n",
		      entity_local_name(v));
    sinks = anywhere_source_to_sinks(source, pts);
  }

  return sinks;
}

list null_source_to_sinks(cell source, pt_map pts)
{
  list sinks = NIL;
  bool null_dereferencing_p =
    get_bool_property("POINTS_TO_NULL_POINTER_DEREFERENCING");

  if(null_dereferencing_p) {
    reference r = cell_any_reference(source);
    entity v = reference_variable(r);
    pips_user_warning("Possibly null pointer \"%s\" is dereferenced.\n",
		      entity_local_name(v));
    sinks = anywhere_source_to_sinks(source, pts);
  }

  return sinks;
}

/* Creation of a stub for a formal parameter or for a reference based
 * on a formal parameter. The formal parameter may be a pointer, an
 * array of something or a struct of something and so on recursively.
 *
 * New dimensions may have to be added to the sink type if the source
 * entity type is an array or if the types are assumed not strict for
 * pointer arithmetic. This is a general issue for stub generation and
 * dealt with at a lower level.
 *
 * Because references must be considered, it is not clear that formal
 * parameters must be handled differently from stubs or global
 * variables. The initial decision was made, I believe, because they
 * were assumed references in a very simple way, for instance as
 * simple direct references.
 *
 * Test cases: argv03.c
 */
list formal_source_to_sinks(cell source, pt_map pts, bool fresh_p)
{
  list sinks = NIL;

  bool null_initialization_p =
    get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
  // bool strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");

  reference r = cell_any_reference(source);
  entity v = reference_variable(r);
  type vt = compute_basic_concrete_type(entity_type(v));
  bool to_be_freed;
  type source_t =
    compute_basic_concrete_type(points_to_cell_to_type(source, &to_be_freed));
  pips_assert("The source type is a pointer type", C_pointer_type_p(source_t));
  type st = compute_basic_concrete_type(type_to_pointed_type(source_t));

  // FI: the type retrieval must be improved for arrays & Co
  // FI: This is not going to work with typedefs...
  // FI: You need array_p to depend on the dimensionality of the
  // reference as it may have arrays at several level, intertwinned with
  // structs.
  bool array_p = array_type_p(vt);
  // Should be points_to_cell_dimension(), counting the number of
  // numerical or unbounded dimensions.

  points_to pt = create_k_limited_stub_points_to(source, st, array_p, pts);

  if(null_initialization_p) {
    free_approximation(points_to_approximation(pt));
    points_to_approximation(pt) = make_approximation_may();
  }
  pts = add_arc_to_pt_map(pt, pts);
  add_arc_to_points_to_context(copy_points_to(pt));
  sinks = source_to_sinks(source, pts, fresh_p);
  if(null_initialization_p){
    list ls = null_to_sinks(source, pts);
    sinks = gen_nconc(ls, sinks);
  }

  return sinks;
}

list global_source_to_sinks(cell source, pt_map pts, bool fresh_p)
{
  bool null_initialization_p =
    get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
  reference r = cell_any_reference(source);
  entity v = reference_variable(r);
  type t = entity_type(v);
  list sinks = NIL;
  type ist = type_to_pointed_type(ultimate_type(entity_type(v)));
  type st = type_undefined;
  points_to pt = points_to_undefined;

  bool strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
  if(scalar_type_p(ist) && !strict_p) {
    /* Add an implicit dimension for pointer arithmetic */
    st = type_to_array_type(ist);
  }
  else
    st = copy_type(ist);

  if(const_variable_p(v)) {
    expression init = variable_initial_expression(v);
    sinks = expression_to_points_to_sinks(init, pts);
    free_expression(init);
    /* Add these new arcs to the context */
    bool exact_p = gen_length(sinks)==1;
    FOREACH(CELL, sink, sinks) {
      cell nsource = copy_cell(source);
      cell nsink = copy_cell(sink);
      approximation na = exact_p? make_approximation_exact():
	make_approximation_may();
      points_to npt = make_points_to(nsource, nsink, na,
				     make_descriptor_none());
      pts = add_arc_to_pt_map(npt, pts);
      add_arc_to_points_to_context(copy_points_to(npt));
    }
  }
  else {
    /* Do we have to ignore an initialization? */
    value val = entity_initial(v);
    if(!value_unknown_p(val))
      pips_user_warning("Initialization of global variable \"%s\" is ignored "
			"because the \"const\" qualifier is not used.\n",

			entity_user_name(v));
    //bool array_p = array_type_p(st);
    bool array_p = array_type_p(t); // array_p is related to the source
    pt = create_k_limited_stub_points_to(source, st, array_p, pts);

    // FI: cut-and-pasted from line 945
    if(null_initialization_p) {
      free_approximation(points_to_approximation(pt));
      points_to_approximation(pt) = make_approximation_may();
    }
    pts = add_arc_to_pt_map(pt, pts);
    add_arc_to_points_to_context(copy_points_to(pt));
    sinks = source_to_sinks(source, pts, fresh_p);
    if(null_initialization_p){
      list ls = null_to_sinks(source, pts);
      sinks = gen_nconc(ls, sinks);
    }
  }

  return sinks;
}

 list points_to_reference_to_translation(reference n_r, list sl, pt_map ptm, bool fresh_p)
 {
   list translations = NIL;
   cell n_c = make_cell_reference(n_r); // FI: memory leak
   list atl = points_to_source_to_sinks(n_c, ptm, fresh_p); // Address Translation List?

   if(ENDP(atl)) {
     if(ENDP(sl)) {
       pips_internal_error("Reference \"n_r\" cannot be translated with \"ptm\".\n");
     }
     else {
       /* Try to use an extra subscript */
       expression ns = EXPRESSION(CAR(sl));
       reference_indices(n_r) = gen_nconc(reference_indices(n_r),
					  CONS(EXPRESSION, copy_expression(ns), NIL));
       translations = points_to_reference_to_translation(n_r, CDR(sl), ptm, fresh_p);
     }
   }
   else {
     /* We've got one translation at least */
     FOREACH(CELL, c, atl) {
       /* Add the next subscripts */
       reference c_r = cell_any_reference(c);
       reference_indices(c_r) = gen_nconc(reference_indices(c_r),
					  gen_full_copy_list(sl));
       translations = atl; // FI: we may not need to variables...
     }
   }
   return translations;
 }
/* Use "ptm" as a translation map
 *
 * Must be similar to a function written by Beatrice to evaluate a
 * complex reference according to points-to information. In her case,
 * it is not a translation, but an evaluation of the possibly many
 * dereferencements contained in the reference.
 *
 * Try to translate a prefix of the source reference and substitue it
 * when a translation is found. No need to translate further down,
 * unike Beatrice's function.
 *
 * fresh_p might be useless because new cells always must be generated.
 */
list points_to_source_to_translations(cell source, pt_map ptm, bool fresh_p)
{
  list translations = NIL;
  reference r = cell_any_reference(source);
  entity v = reference_variable(r);
  list sl = reference_indices(r);
  reference n_r = make_reference(v, NIL);
  translations = points_to_reference_to_translation(n_r, sl, ptm, fresh_p);
  return translations;
}

/* Build the sinks of source "source" according to the points-to
 * graphs. If "source" is not found in the graph, return an empty list
 * "sinks". If "fresh_p", allocate copies. If not, return pointers to
 * the destination vertices in "ptm".
 *
 * It is not clear how much the abstract address lattice must be used
 * to retrieve sinks... If source = a[34], clearly a[*] is an OK
 * equivalent source if a[34] is not a vertex of "ptm".
 *
 * If !strict_p, "a[34]" is considered a source for "a[*]".
 */
list generic_points_to_source_to_sinks(cell source, pt_map ptm,
				       bool fresh_p, bool strict_p)
{
  list sinks = NIL;
  set pts = points_to_graph_set(ptm);

  /* 1. See if cell "source" is the starting vertex of a points-to arc. */
  SET_FOREACH( points_to, pt, pts) {
    if(cell_equal_p(source, points_to_source(pt))) {
      cell sc = fresh_p? copy_cell(points_to_sink(pt)) : points_to_sink(pt);
      sinks = CONS(CELL, sc, sinks);
    }
  }

  /* 2. Much harder... See if source is contained in one of the many
     abstract sources. Step 1 is subsumed by Step 2... but is much faster.  */
  if(ENDP(sinks)) {
    SET_FOREACH(points_to, pt, pts) {
      if(cell_included_p(source, points_to_source(pt))) {
	// FI: memory leak forced because of refine_points_to_cell_subscripts
	cell sc = (true||fresh_p)? copy_cell(points_to_sink(pt)) : points_to_sink(pt);
	// FI->AM: if "source" is stricly included in
	// "points_to_source(pt)", the subscript expression of sc
	// might have to be cleaned up
	//
	// Which implies to allocate a new copy of sc no matter what
	// fresh_p prescribes...
	refine_points_to_cell_subscripts(sc, source, points_to_source(pt));
	sinks = CONS(CELL, sc, sinks);
      }
    }
  }

  /* 3. Much harder... See if source contains one of the many
     abstract sources.  */
  if(ENDP(sinks) && !strict_p) {
    SET_FOREACH(points_to, pt, pts) {
      if(cell_included_p(points_to_source(pt), source)) {
	// FI: memory leak forced because of refine_points_to_cell_subscripts
	cell sc = (true||fresh_p)? copy_cell(points_to_sink(pt)) : points_to_sink(pt);
	// FI->AM: if "source" is stricly included in
	// "points_to_source(pt)", the subscript expression of sc
	// might have to be cleaned up
	//
	// Which implies to allocate a new copy of sc no matter what
	// fresh_p prescribes...
	refine_points_to_cell_subscripts(sc, source, points_to_source(pt));
	sinks = CONS(CELL, sc, sinks);
      }
    }
  }

  return sinks;
}

/* Build the sinks of source "source" according to the points-to
 * graphs. If "source" is not found in the graph, return an empty list
 * "sinks". If "fresh_p", allocate copies. If not, return pointers to
 * the destination vertices in "ptm".
 *
 * It is not clear how much the abstract address lattice must be used
 * to retrieve sinks... If source = a[34], clearly a[*] is an OK
 * equivalent source if a[34] is not a vertex of "ptm".
 */
list points_to_source_to_sinks(cell source, pt_map ptm, bool fresh_p)
{
  return generic_points_to_source_to_sinks(source, ptm, fresh_p, true);
}

/* May not retrieve all sinks of the source.
 *
 * This happens with arrays of pointers. See EffectsWithPointers/call22.c
 */
list points_to_source_to_some_sinks(cell source, pt_map ptm, bool fresh_p)
{
  return generic_points_to_source_to_sinks(source, ptm, fresh_p, false);
}

/* Build the sources of sink "sink" according to the points-to
 * graphs. If "sink" is not found in the graph, return an empty list
 * "sources". If "fresh_p", allocate copies. If not, return pointers to
 * the destination vertices in "ptm".
 *
 * It is not clear how much the abstract address lattice must be used
 * to retrieve sources... If source = a[34], clearly a[*] is an OK
 * equivalent source if a[34] is not a vertex of "ptm".
 */
list points_to_sink_to_sources(cell sink, pt_map ptm, bool fresh_p)
{
  list sources = NIL;
  set pts = points_to_graph_set(ptm);

  /* 1. See if cell "sink" is the destination vertex of a points-to arc. */
  SET_FOREACH( points_to, pt, pts) {
    if(cell_equal_p(sink, points_to_sink(pt))) {
      cell sc = fresh_p? copy_cell(points_to_source(pt)) : points_to_source(pt);
      sources = CONS(CELL, sc, sources);
    }
  }


  /* 2. Much harder... See if sink is contained in one of the many
     abstract sinks or if its address can be obtained from the address
     of another sink cell thanks to pointer arithmetic or
     indexing. Step 1 is subsumed by Step 2... but is much faster.  */
  if(ENDP(sources)) {
    SET_FOREACH(points_to, pt, pts) {
      if(cell_included_p(sink, points_to_sink(pt))
	 /* FI: I am not sure that using pointer arithmetics to
	    declare equivalence is a good idea. */
	 || cell_equivalent_p(sink, points_to_sink(pt))) {
	// FI: memory leak forced because of refine_points_to_cell_subscripts
	cell sc = (true||fresh_p)? copy_cell(points_to_source(pt)) : points_to_source(pt);
	// FI: I do not remember what this is for...
	refine_points_to_cell_subscripts(sc, sink, points_to_sink(pt));
	sources = CONS(CELL, sc, sources);
      }
    }
  }

  return sources;
}

/* Return the points-to "fpt" ending in cell "sink" if it
   exists. Return points-to_undefined otherwise. */
points_to points_to_sink_to_points_to(cell sink, pt_map ptm)
{
  points_to fpt = points_to_undefined;;
  set pts = points_to_graph_set(ptm);

  /* 1. See if cell "sink" is the destination vertex of a points-to arc. */
  SET_FOREACH( points_to, pt, pts) {
    if(cell_equal_p(sink, points_to_sink(pt))) {
      fpt = pt;
      break;
    }
  }
  return fpt;
}

/* Use "sn" as a source name to derive a list of sink cells according
 * to the points-to graph ptm.
 *
 * Allocate copies of the sink cells if "fresh_p".
 */
list points_to_source_name_to_sinks(string sn, pt_map ptm, bool fresh_p)
{
  list sinks = NIL;
  set pts = points_to_graph_set(ptm);

  SET_FOREACH( points_to, pt, pts) {
    cell c = points_to_source(pt);
    string cn = points_to_cell_name(c);
    if(strcmp(sn, cn)==0) {
      cell sc = fresh_p? copy_cell(points_to_sink(pt)) : points_to_sink(pt);
      sinks = CONS(CELL, sc, sinks);
    }
    free(cn);
  }

  return sinks;
}

cell points_to_source_name_to_source_cell(string sn, pt_map ptm, bool fresh_p)
{
  cell rc = cell_undefined;
  set pts = points_to_graph_set(ptm);

  SET_FOREACH(points_to, pt, pts) {
    cell c = points_to_source(pt);
    string cn = points_to_cell_name(c);
    if(strcmp(sn, cn)==0) {
      rc = fresh_p? copy_cell(c) : c;
      break;
    }
    free(cn);
  }

  return rc;
}

/* Build the union of the sinks of cells in "sources" according to the
 * points-to graphs "ptm". Allocate new cells if "fresh_p". No
 * specific order in the returned list.
 */
list points_to_sources_to_sinks(list sources, pt_map ptm, bool fresh_p)
{
  list sinks = NIL;
  FOREACH(CELL, source, sources) {
    list subl = points_to_source_to_sinks(source, ptm, fresh_p);
    sinks = gen_nconc(sinks, subl); // to ease debugging... Could be CONS
  }

  return sinks;
}

/* Build the list of arcs whose source is "source" according to the points-to
 * graphs "ptm". If "source" is not found in the graph, return an empty list
 * "sinks". If "fresh_p", allocate copies. If not, return pointers to
 * the arcs in "ptm".
 *
 * It is not clear how much the abstract address lattice must be used
 * to retrieve sinks... If source = a[34], clearly a[*] is an OK
 * equivalent source if a[34] is not a vertex of "ptm". Currently, we
 * assume that the origin vertex must be exactly "source".
 */
list points_to_source_to_arcs(cell source, pt_map ptm, bool fresh_p)
{
  list arcs = NIL;
  set pts = points_to_graph_set(ptm);

  /* See when the cell "source" is the starting vertex of a points-to arc. */
  SET_FOREACH( points_to, pt, pts) {
    if(cell_equal_p(source, points_to_source(pt))) {
      points_to pta = fresh_p? copy_points_to(pt) : pt;
      arcs = CONS(POINTS_TO, pta, arcs);
    }
  }

  return arcs;
}


/* Return a list of cells, "sinks", that are sink for some arc whose
 * source is "source" or related to "source" in set "pts". If no such
 * arc is found, add new points-to stubs and new arcs in "pts" when
 * global, formal or virtual variables are used in "source". Manage
 * fix point detection to avoid creating an infinite number of such
 * points-to stubs when recursive data structures are accessed in
 * loops.
 *
 * If "fresh_p" is set to true, no sharing is created between list
 * "sinks" and reference "source" or points-to set "pts". Else, the
 * cells in list "sinks" are the cells in arcs of the points-to set.
 *
 * FI: I am not sure the above paragraph is properly implemented.
 *
 * This function is based on several other simpler functions:
 *
 * * points_to_source_to_sinks()
 *
 * * anywhere_source_to_sinks(), nowhere_source_to_sinks() and
 *   null_source_to_sinks()
 *
 * * formal_source_to_sinks()
 *
 * * global_source_to_sinks()
 *
 * * stub_source_to_sinks()
 *
 * This function should never return an empty list. The caller should
 * handle it as a bug in the analyzed code.
 *
 * Function added by FI. It is recursive via...
 */
list source_to_sinks(cell source, pt_map pts, bool fresh_p)
{
  list sinks = NIL;
  bool to_be_freed;
  type source_t = points_to_cell_to_type(source, & to_be_freed);
  type c_source_t = compute_basic_concrete_type(source_t);
  bool ok_p = C_pointer_type_p(c_source_t)
    || overloaded_type_p(c_source_t) // might be a pointer
    || null_cell_p(source);
  if(to_be_freed) free_type(source_t);

  /* Can we expect a sink? */
  if(!ok_p) {
    // Likely typing error in source code
    entity v = reference_variable(cell_any_reference(source));
    pips_user_warning("Typing error in a pointer assignment or a dereferencing with \"%s\" at line %d.\n",
		      entity_user_name(v),
		      points_to_context_statement_line_number());
    // Return an empty list
  }
  else if(nowhere_cell_p(source)) {
    sinks = nowhere_source_to_sinks(source, pts);
  }
  else if(anywhere_cell_p(source) || cell_typed_anywhere_locations_p(source)) {
    sinks = anywhere_source_to_sinks(source, pts);
  }
  else if(null_pointer_value_cell_p(source)) {
    sinks = null_source_to_sinks(source, pts);
  }
  else {
    /* 0. Is the source a pointer? You would expect a yes, but C
       pointer arithmetics requires some strange typing. We assume it
       is an array of pointers. */
    bool to_be_freed;
    type ct = points_to_cell_to_type(source, &to_be_freed);
    if(array_type_p(ct)) {
      basic ctb = variable_basic(type_variable(ct));
      // FI->AM: I am not happy at all with his
      if(basic_pointer_p(ctb)) {
	;
      }
      else {
	cell sc = copy_cell(source);
	sinks = CONS(CELL, sc, sinks);
      }
    }
    if(to_be_freed) free_type(ct);

    /* 1. Try to find the source in the points-to information */
    if(ENDP(sinks))
      sinks = points_to_source_to_sinks(source, pts, fresh_p);

    /* 2. If the previous step has failed, build a new sink if the
       source is a formal parameter, a global variable, a C file local
       global variable (static) or a stub. */
    if(ENDP(sinks)) {
      reference r = cell_any_reference(source);
      entity v = reference_variable(r);
      if(formal_parameter_p(v)) {
	sinks = formal_source_to_sinks(source, pts, fresh_p);
      }
      else if(top_level_entity_p(v) || static_global_variable_p(v)) {
	sinks = global_source_to_sinks(source, pts, fresh_p);
      }
      else if(entity_stub_sink_p(v)) {
	sinks = stub_source_to_sinks(source, pts, fresh_p);
      }
      else if(entity_typed_anywhere_locations_p(v)) {
	pips_internal_error("This case should have been handled above.\n");
      }
      if(ENDP(sinks)) {
	/* A bug somewhere up... */
	reference r = cell_any_reference(source);
	print_reference(r);
	pips_user_warning("\nUninitialized or null pointer dereferenced: "
			  "Sink missing for a source based on \"%s\".\n"
			  "Update points-to property POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING and/or POINTS_TO_UNINITIALIZED_NULL_DEREFERENCING according to needs.\n",
			  entity_user_name(v));
	clear_pt_map(pts);
	points_to_graph_bottom(pts) = true;
	// FI: it is not a pips error but a user error (in theory)
	// pips_internal_error("Dereferencing of an unitialized pointer.\n");
      }
    }
  }
  // FI: use gen_nreverse() to simplify debbugging? Not meaningful
  // with SET_FOREACH
  return sinks;
}

list extended_source_to_sinks(cell sc, pt_map in)
{
  list sinks = NIL;
  bool null_dereferencing_p
    = get_bool_property("POINTS_TO_NULL_POINTER_DEREFERENCING");
  bool nowhere_dereferencing_p
    = get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING");
  if( (null_dereferencing_p || !null_cell_p(sc))
      && (nowhere_dereferencing_p || !nowhere_cell_p(sc))) {
    /* Do not create sharing between elements of "in" and
       elements of "sinks". */
    cell nsc = copy_cell(sc);
    list starpointed = source_to_sinks(nsc, in, true);
    free_cell(nsc);

    if(ENDP(starpointed)) {
      reference sr = cell_any_reference(sc);
      entity sv = reference_variable(sr);
      string words_to_string(list);
      pips_internal_error("No pointed location for variable \"%s\" and reference \"%s\"\n",
			  entity_user_name(sv),
			  words_to_string(words_reference(sr, NIL)));
    }
    sinks = gen_nconc(sinks, starpointed);
  }
  // FI: I'd like a few else clauses to remove arcs that
  // cannot exist if the code is correct. E.g. p->i, p->NULL
  // if card(cl)==1, remove arc(c->sc)?
  return sinks;
}

list extended_sources_to_sinks(list pointed, pt_map in)
{
  list sinks = NIL;
  /* Dereference the pointer(s) to find the sinks, memory(memory(p)) */
  FOREACH(CELL, sc, pointed) {
    list starpointed = extended_source_to_sinks(sc, in);
    sinks = gen_nconc(sinks, starpointed);
  }
  return sinks;
}

/* Return all cells in points-to set "pts" who source is based on entity "e". 
 *
 * Similar to points_to_source_to_sinks, but much easier and shorter.
 */
list variable_to_sinks(entity e, pt_map ptm, bool fresh_p)
{
  list sinks = NIL;
  set pts = points_to_graph_set(ptm);
  SET_FOREACH( points_to, pt, pts) {
    cell source = points_to_source(pt);
    reference sr = cell_any_reference(source);
    entity v = reference_variable(sr);
    if(e==v) {
      cell sc = fresh_p? copy_cell(points_to_sink(pt)) : points_to_sink(pt);
      sinks = CONS(CELL, sc, sinks);
    }
  }
  return sinks;

}

/* Create a list of null sinks and add a new null points-to relation to pts.
   pts is modified by side effect.
*/
list null_to_sinks(cell source, pt_map ptm)
{
  cell nsource = copy_cell(source);
  cell nsink = make_null_pointer_value_cell();
  points_to npt = make_points_to(nsource, nsink,
				 make_approximation_may(),
				 make_descriptor_none());
  ptm = add_arc_to_pt_map(npt, ptm);
  add_arc_to_points_to_context(copy_points_to(npt));
  list sinks = CONS(CELL, copy_cell(nsink), NIL);
  return sinks;
}

/* Same as source_to_sinks, but for a list of cells. */
list sources_to_sinks(list sources, pt_map ptm, bool fresh_p)
{
  list sinks = NIL;
  FOREACH(CELL, c, sources) {
    list cl =  source_to_sinks(c, ptm, fresh_p);
    sinks = gen_nconc(sinks, cl);
  }
  return sinks;
}

list reference_to_sinks(reference r, pt_map in, bool fresh_p)
{
  cell source = make_cell_reference(copy_reference(r));
  list sinks = source_to_sinks(source, in, fresh_p);
  free_cell(source);
  return sinks;
}

/* Merge two points-to sets
 *
 * This function is required to compute the points-to set resulting of
 * an if control statements.
 *
 * A new set is allocated but it reuses the elements of "s1" and "s2".
 */
set merge_points_to_set(set s1, set s2) {
  set Merge_set = set_generic_make(set_private, points_to_equal_p,
				   points_to_rank);

  pips_assert("Consistent set s1", consistent_points_to_set(s1));
  pips_assert("Consistent set s2", consistent_points_to_set(s2));

  if(set_empty_p(s1))
    set_assign(Merge_set, s2);
  else if(set_empty_p(s2)) 
    set_assign(Merge_set, s1);
  else {
    set Definite_set = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
    set Possible_set = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
    set Intersection_set = set_generic_make(set_private, points_to_equal_p,
					    points_to_rank);
    set Union_set = set_generic_make(set_private, points_to_equal_p,
				     points_to_rank);

    Intersection_set = set_intersection(Intersection_set, s1, s2);
    Union_set = set_union(Union_set, s1, s2);

    SET_FOREACH ( points_to, i, Intersection_set ) {
      if ( approximation_exact_p(points_to_approximation(i)) 
	   || approximation_must_p(points_to_approximation(i)) )
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
  }

  pips_assert("Consistent merged set", consistent_points_to_set(Merge_set));

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

bool points_to_in_list_p(points_to pt, const list lx)
{
  list l = (list) lx;
  for (; !ENDP(l); POP(l))
    if (points_to_equal_p(POINTS_TO(CAR(l)), pt)) return true; /* found! */

  return false; /* else no found */
}

/*  */
bool points_to_compare_cell(cell c1, cell c2)
{
  if(c1==c2)
    return true;

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

/*  */
bool points_to_compare_ptr_cell(const void * vcel1, const void * vcel2)
{
  int i = 0;
  cell c1 = *((cell *)vcel1);
  cell c2 = *((cell *)vcel2);
  reference r1 = cell_to_reference(c1);
  reference r2 = cell_to_reference(c2);
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  list sl1 = NIL, sl2 = NIL;
  extern const char* entity_minimal_user_name(entity);
  string n1 =   entity_abstract_location_p(v1)?
    (string) entity_local_name(v1) :  (string) entity_minimal_user_name(v1);
  string n2 =   entity_abstract_location_p(v2)?
    (string) entity_local_name(v2) : (string) entity_minimal_user_name(v2);
  i = strcmp(n1, n2);
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

  return i;
}


/* Order the two points-to relations according to the alphabetical
 * order of the underlying variables. Return -1, 0, or 1.
 */
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

/* make sure that set "s" does not contain redundant or contradictory
   elements */
bool consistent_points_to_set(set s)
{
  bool consistent_p = true;

  /* Check the consistency of each arc */
  SET_FOREACH(points_to, a, s) {
    consistent_p = consistent_p && points_to_consistent_p(a);
  }

  /* Check the validity of the approximations */
  SET_FOREACH(points_to, pt1, s) {
    approximation a1 = points_to_approximation(pt1);
    SET_FOREACH(points_to, pt2, s) {
      if(pt1!=pt2) {
	//same source
	cell c1 = points_to_source(pt1);
	cell c2 = points_to_source(pt2);
	bool cmp1 = locations_equal_p(c1,c2);

	if(cmp1 && approximation_exact_p(a1)) {
	  fprintf(stderr,
		  "Contradictory points-to arcs: incompatible approximation\n");
	  print_points_to(pt1);
	  print_points_to(pt2);
	  consistent_p = false;
	}

	// same sink
	cell c3 = points_to_sink(pt1);
	cell c4 = points_to_sink(pt2);
	bool cmp2 = locations_equal_p(c3,c4);
	if(cmp1&&cmp2) {
	  // same approximation
	  approximation a1 = points_to_approximation(pt1);
	  approximation a2 = points_to_approximation(pt2);
	  // FI: must is forgotten...
	  //bool cmp3 = (approximation_exact_p(a1) && approximation_exact_p(a2))
	  //  || ( approximation_may_p(a1) && approximation_may_p(a2));
	  // FI: should we identify "exact" and "must"?
	  bool cmp3 = (approximation_tag(a1)==approximation_tag(a2));
	  if(cmp3) {
	    fprintf(stderr, "Redundant points-to arc:\n");
	    print_points_to(pt1);
	    consistent_p = false;
	  }
	  else {
	    fprintf(stderr, "Contradictory points-to arcs: incompatible approximation\n");
	    print_points_to(pt1);
	    print_points_to(pt2);
	    consistent_p = false;
	  }
	}
      }
    }
  }

  /* Make sure that the element of set "s" belong to "s" (issue with
   * side effects performed on subscript expressions).
   */
  SET_FOREACH(points_to, pt, s) {
    if(!set_belong_p(s,pt)) {
      fprintf(stderr, "Points-to %p ", pt);
      print_points_to(pt);
      fprintf(stderr, " is in set s but does not belong to it!\n");
      consistent_p = false;
    }
  }

  /* Check that no sharing exists between arcs at the cell and
     reference levels */
  if(consistent_p) {
    consistent_p = !points_to_set_sharing_p(s);
    if(!consistent_p)
      fprintf(stderr, "Sharing detected\n");
  }
  return consistent_p;
}

bool points_to_set_sharing_p(set s)
{
  bool sharing_p = false;

  SET_FOREACH(points_to, pt1, s) {
    cell source_1 = points_to_source(pt1);
    cell sink_1 = points_to_sink(pt1);
    reference source_r1 = cell_any_reference(source_1);
    reference sink_r1 = cell_any_reference(sink_1);

    bool found_p = false;
    SET_FOREACH(points_to, pt2, s) {
      if(pt1==pt2) 
	found_p = true;
      if(found_p && pt1!=pt2) {
	cell source_2 = points_to_source(pt2);
	cell sink_2 = points_to_sink(pt2);
	reference source_r2 = cell_any_reference(source_2);
	reference sink_r2 = cell_any_reference(sink_2);

	bool new_sharing_p = false;

	/* Sharing of cells */
	if(source_1==source_2) {
	  new_sharing_p = true;
	  fprintf(stderr, "Sharing between source_1 and source_2.\n");
	}
	else if(source_1==sink_2) {
	  new_sharing_p = true;
	  fprintf(stderr, "Sharing between source_1 and sink_2.\n");
	}
	else if(sink_1==source_2) {
	  new_sharing_p = true;
	  fprintf(stderr, "Sharing between sink_1 and source_2.\n");
	}
	else if(sink_1==sink_2) {
	  new_sharing_p = true;
	  fprintf(stderr, "Sharing between sink_1 and sink_2.\n");
	}

	if(!new_sharing_p) {
	  /* Sharing of references */
	  if(source_r1==source_r2) {
	    new_sharing_p = true;
	    fprintf(stderr, "Sharing between source_r1 and source_r2.\n");
	  }
	  else if(source_r1==sink_r2) {
	    new_sharing_p = true;
	    fprintf(stderr, "Sharing between source_r1 and sink_r2.\n");
	  }
	  else if(sink_r1==source_r2) {
	    new_sharing_p = true;
	    fprintf(stderr, "Sharing between sink_r1 and source_r2.\n");
	  }
	  else if(sink_r1==sink_r2) {
	    new_sharing_p = true;
	    fprintf(stderr, "Sharing between sink_r1 and sinke_r2.\n");
	  }
	}
	if(new_sharing_p) {
	  fprintf(stderr, "For pt1 ");
	  dump_points_to(pt1);
	  fprintf(stderr, "\nand pt2 ");
	  dump_points_to(pt2);
	}
	sharing_p = sharing_p || new_sharing_p;
      }
    }
  }
  return sharing_p;
}

/* because of points-to set implementation, you cannot change
 * approximations by side effects.
 */
void upgrade_approximations_in_points_to_set(pt_map ptm)
{
  set pts = points_to_graph_set(ptm);
  SET_FOREACH(points_to, pt, pts) {
    approximation a = points_to_approximation(pt);
    if(!approximation_exact_p(a)) {
      cell source = points_to_source(pt);
      if(!cell_abstract_location_p(source) // Represents may locations
	 && !stub_points_to_cell_p(source)) { // May not exist...
	list sinks = source_to_sinks(source, ptm, false);
	if(gen_length(sinks)==1) {
	  cell sink = points_to_sink(pt);
	  if(!cell_abstract_location_p(sink)) {
	    points_to npt = make_points_to(copy_cell(source),
					   copy_cell(sink),
					   make_approximation_exact(),
					   make_descriptor_none());
	    remove_arc_from_pt_map(pt, ptm);
	    (void) add_arc_to_pt_map(npt, ptm);
	  }
	}
	gen_free_list(sinks);
      }
    }
  }
}

void remove_points_to_arcs(cell source, cell sink, pt_map pt)
{
  points_to a = make_points_to(copy_cell(source), copy_cell(sink),
			       make_approximation_may(),
			       make_descriptor_none());
  remove_arc_from_pt_map(a, pt);
  free_points_to(a);

  a = make_points_to(copy_cell(source), copy_cell(sink),
			       make_approximation_exact(),
			       make_descriptor_none());
  remove_arc_from_pt_map(a, pt);
  free_points_to(a);
}



/* Compute A = A inter B: complexity in O(n2) */
void
points_to_cell_list_and(list * a, const list b)
{
  if (ENDP(*a))
    return ;
  if (!points_to_cell_in_list_p(CELL(CAR(*a)),b)) {
    /* This element of a is not in list b: delete it: */
    cons *aux = *a;

    *a = CDR(*a);
    free(aux);
    points_to_cell_list_and(a, b);
  }
  else
    points_to_cell_list_and(&CDR(*a), b);
}
/* Free several sets in one call. Useful when many sets are used
   simultaneously. */
void free_points_to_graph_sets(points_to_graph s,...)
{
  va_list args;

  /* Analyze in args the variadic arguments that may be after t: */
  va_start(args, s);
  /* Since a variadic function in C must have at least 1 non variadic
     argument (here the s), just skew the varargs analysis: */
  do {
    free_points_to_graph(s);
    /* Get the next argument */
    //s = va_arg(args, points_to_graph);
    s = va_arg(args, pt_map);
  } while(s!=NULL);
  /* Release the variadic analysis */
  va_end(args);
}

/* FI: I add functions dealing with points_to_graph variable, i.e. pt_map */

pt_map graph_assign_list(pt_map ptm, list l)
{
  bool b = points_to_graph_bottom(ptm);
  if(b) {
    // FI: I am in trouble here; what should be the semantics?
    pips_debug(1, "Impossible initialization of a bottom graph\n");
    pips_internal_error("Mismanaged points-to graph\n");
  }
  else
    set_assign_list(points_to_graph_set(ptm), l);
  return ptm;
}

pt_map merge_points_to_graphs(pt_map s1, pt_map s2)
{
  set merged = merge_points_to_set(points_to_graph_set(s1),
				   points_to_graph_set(s2));
  pt_map pt_merged = new_pt_map();
  points_to_graph_set(pt_merged) = merged;
  if(points_to_graph_bottom(s1) && points_to_graph_bottom(s2))
    points_to_graph_bottom(pt_merged) = true;
  return pt_merged;
}

pt_map points_to_graph_assign(pt_map out, pt_map in)
{
  points_to_graph_set(out) = set_assign(points_to_graph_set(out),
					points_to_graph_set(in));
  return out;
}

/* All vertices in "sink_l" are assumed to be sinks of vertex "source"
 * in points-to graph "in".
 *
 * These vertices must be replaced by a unique vertex, their minimum
 * upper bound in the abstract address lattice. And their own
 * out-going arcs must also be rearranged.
 *
 * Clearly, some abstract addresses high in the lattice should be
 * allowed large out-degree numbers.
 *
 * A newly allocated points-to arc is returned. It could be integrated
 * directly in "in", but the integration is performed by a caller.
 */
points_to fuse_points_to_sink_cells(cell source, list sink_l, pt_map in)
{
  pt_map out = in;

  pips_assert("\"in\" is consistent", consistent_pt_map_p(in));

  /* Find the minimal upper bound of "sink_l" */
  cell mupc = points_to_cells_minimal_upper_bound(sink_l);

  /* Compute the sinks of the vertex "mupc" as the union of the sinks
   * of cells in "sink_l" and add the corresponding arcs to "out".
   */
  list iptl = points_to_sources_to_sinks(sink_l, in, true); // indirect points-to
  FOREACH(CELL, sink, iptl) {
    cell mupcc = copy_cell(mupc);
    points_to pta = make_points_to(mupcc, sink,
				  make_approximation_may(),
				  make_descriptor_none());
    add_arc_to_pt_map(pta, in);
  }
  gen_free_list(iptl);

  /* Find the incoming arcs on cells of "sink_l" and replace them by arcs
     towards copies of mupc. */
  FOREACH(CELL, sink, sink_l) {
    if(!null_cell_p(sink) && !nowhere_cell_p(sink)) {
      /* Finds it sources */
      SET_FOREACH(points_to, pt, points_to_graph_set(in)) {
	cell oc = points_to_source(pt);
	cell dc = points_to_sink(pt);
	if(points_to_cell_equal_p(dc, sink)) {
	  points_to npt = make_points_to(copy_cell(oc), copy_cell(mupc),
					 make_approximation_may(),
					 make_descriptor_none());
	  add_arc_to_pt_map(npt, in);
	  remove_arc_from_pt_map(pt, in);
	}
      }
    }
  }
  // pips_internal_error("not implemented yet.\n");

  /* Find the set of points-to arcs to destroy and remove them from
   * the points-to graph "in".
   */
  list ptal = points_to_source_to_arcs(source, in, false);
  FOREACH(POINTS_TO, pta, ptal) {
    cell sink = points_to_sink(pta);
    if(!null_cell_p(sink) && !nowhere_cell_p(sink))
      if(!points_to_cell_equal_p(sink, mupc))
	remove_arc_from_pt_map(pta, in);
  }
  gen_free_list(ptal);

  /* Create an arc from "source" to "mupc" */
  points_to pta = make_points_to(source, mupc,
				 make_approximation_may(),
				 make_descriptor_none());
  // add_arc_to_pt_map(pta, in); Might be done by the calller?

  pips_assert("\"out\" is consistent", consistent_pt_map_p(out));

  return pta;
}

/* returns the cell vertex "mod_cell" with the maximal out_degree in
 * graph "in", and its out-degree.
 *
 * When several cells have the same maximal out-degree, return any of
 * them.
 */
int maximal_out_degree_of_points_to_graph(string * mod_cell, pt_map in)
{
  hash_table cell_out_degree = hash_table_make(hash_string, 0);

  SET_FOREACH(points_to, pt, points_to_graph_set(in)) {
    cell source = points_to_source(pt);
    string name = points_to_cell_name(source);
    long long int i =
      (long long int) hash_get(cell_out_degree, (void *) name);
    if(i== (long long int) HASH_UNDEFINED_VALUE) {
      i = 1;
      hash_put(cell_out_degree, (void *) name, (void *) i);
    }
    else {
      i++;
      hash_update(cell_out_degree, (void *) name, (void *) i);
    }
  }

  long long int m = 0;
  HASH_MAP(k, v, {
      if((long long int) v > m) {
	m = (long long int) v;
	*mod_cell = strdup((string) k);
      }
    }, cell_out_degree);
  hash_table_free(cell_out_degree);
  return (int) m;
}

/* For the time being, control the out-degree of the vertices in
 * points-to graph "ptg" and fuse the vertex with the maximal
 * out-degree to reduce it if it is greater than an expected limit.
 *
 * Points-to graph "ptg" i modified by side-effects and returned.
 */
pt_map normalize_points_to_graph(pt_map ptg)
{
  int odl = get_int_property("POINTS_TO_OUT_DEGREE_LIMIT");
  int sl = get_int_property("POINTS_TO_SUBSCRIPT_LIMIT");

  /* The out-degree limit must take the subscript limit sl into
     account as well as possible NULL and NOWHERE values (+2). The
     unbounded susbcript must also be added because it does not
     necessarily subsume all integer subscripts (+1). The subscript
     limit will kick in anyway later. Subscripts are limited to the
     range [-sl,sl], which contains 2*sl+1 values. */
  if(odl<2*sl+1+2) odl = 2*sl+1+2+1;

  pips_assert("odl is greater than one", odl>=1);
  string mod_cell_name = string_undefined; // maximum out-degree cell
  int od = maximal_out_degree_of_points_to_graph(&mod_cell_name, ptg);
  if(od>odl) {
    ifdebug(1) {
      pips_debug(1, "Normalization takes place for graph \"ptg\" with \"od\"=%d and \"odl\"=%d:\n", od, odl);
      print_points_to_set("Loop points-to set ptg:\n",
			      points_to_graph_set(ptg));
    }
    // FI: not too sure about argument "true"
    cell mod_cell = points_to_source_name_to_source_cell(mod_cell_name, ptg, true);
    if(cell_undefined_p(mod_cell))
      pips_internal_error("Inconsistent result for ptg.\n");
    list sink_l = points_to_source_name_to_sinks(mod_cell_name, ptg, false);
    points_to pt = fuse_points_to_sink_cells(mod_cell, sink_l, ptg);
    add_arc_to_pt_map(pt, ptg);
    ifdebug(1) {
      pips_debug(1, "After normalization, \"ptg\":\n");
      print_points_to_set("Loop points-to set ptg:\n",
			      points_to_graph_set(ptg));
    }
  }
  return ptg;
}

/* Remove arcs in points-to graph "ptg" when they start from a stub
 * cell that is not reachable.
 *
 * Points-to graph "ptg" i modified by side-effects and returned.
 *
 * This clean-up should be performed each time a projection is
 * performed, and even potentially, each time an arc is removed.
 *
 * Note: see also freed_pointer_to_points_to() for a recursive
 * implementation of the arc elimination. The current clean-up is
 * *not* recursive. This function should be called repeatedly till the
 * results converge to a fix point...
 */
pt_map remove_unreachable_vertices_in_points_to_graph(pt_map ptg)
{
  set ptg_s = points_to_graph_set(ptg);
  list ual = NIL; // unreachable arc list

  pips_assert("pts is consistent before unreachable arc removal",
	      consistent_pt_map_p(ptg));

  /* Find arcs whose origin vertex is an unreachable stub. */
  SET_FOREACH(points_to, pt, ptg_s) {
    cell source = points_to_source(pt);
    reference r = cell_any_reference(source);
    entity e = reference_variable(r);
    if(entity_stub_sink_p(e)) {
      // list S = points_to_source_to_sinks(source, ptg);
      list S = points_to_sink_to_sources(source, ptg, false);
      if(ENDP(S))
	ual = CONS(POINTS_TO, pt, ual);
      gen_free_list(S);
    }
  }

  /* Remove arcs in ual. */
  FOREACH(POINTS_TO, pt, ual) {
    remove_arc_from_pt_map(pt, ptg);
  }

  //gen_full_free_list(ual);

  pips_assert("pts is consistent after unreachable arc removal",
	      consistent_pt_map_p(ptg));
  
  return ptg;
}

bool consistent_points_to_graph_p(points_to_graph ptg)
{
  bool consistent_p;
    set ptg_s = points_to_graph_set(ptg);
  if(points_to_graph_bottom(ptg)) {
    consistent_p = set_empty_p(ptg_s);
    if(!consistent_p)
      pips_internal_error("Bottom graph is not empty.\n");
  }
  else {
    consistent_p = consistent_points_to_set(ptg_s);
  }
  return consistent_p;
}

/* You know that null and undefined cells in "*pL" are impossible
 * because of the operation that is going to be performed on
 * it. Remove the corresponding arcs in points-to graph "in". Remove
 * the corresponding cells from "*pL".
 *
 * The search uses pointers. So "*pL" must contain sink cells of arcs of
 * "in".
 */
void remove_impossible_arcs_to_null(list * pL, pt_map in)
{
  list fl = NIL;
  bool nowhere_ok_p =
    get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING");
  FOREACH(CELL, pc, *pL) {
    if(null_cell_p(pc) || (nowhere_cell_p(pc) && !nowhere_ok_p)) {
      points_to pt = points_to_sink_to_points_to(pc, in);
      if(points_to_undefined_p(pt))
	pips_internal_error("NULL, returned as a source for an expression, "
			    "does not appear in the points-to graph.\n");
      remove_arc_from_pt_map(pt, in);
      fl = CONS(CELL, pc, fl);
    }
  }
  gen_list_and_not(pL, fl);
  gen_free_list(fl);
}

/* Check if points-to arc "spt" belongs to points-to set "pts". */
bool arc_in_points_to_set_p(points_to spt, set pts)
{
  bool in_p = false;
  SET_FOREACH(points_to, pt, pts) {
    if(points_to_equal_p(spt, pt)) {
      in_p = true;
      break;
    }
  }
  return in_p;
}
