#include <stdlib.h>
#include <stdio.h>
/* For strdup: */
// FI: already defined elsewhere
// #define _GNU_SOURCE
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "pipsdbm.h"
#include "resources.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

/* Points-to interprocedural analysis is inspired by the work of
 * Wilson[95]. The documentation of this phase is a chapter of Amira
 * Mensi thesis.
 */




/* obsolete function*/
/* For each stub cell e  find its formal corresponding parameter and add to it a dereferencing dimension */
cell formal_access_paths(cell e, list args, __attribute__ ((__unused__)) set pt_in)
{
  int i;
  cell f_a = cell_undefined;
  reference r1 = cell_any_reference(e);
  reference r = reference_undefined;
  entity v1 = reference_variable(r1);
  const char * en1 = entity_local_name(v1);
  list sl = NIL; 
  bool change_p = false ;
  while(!ENDP(args) && !change_p) {
    cell c = CELL(CAR(args));
    reference r2 = cell_to_reference(c);
    entity v2 = reference_variable(r2);
    const char * en2 = entity_local_name(v2);
    char *cmp = strstr(en1, en2);
    r = copy_reference(r2);
    for(i = 0; cmp != NULL && cmp[i]!= '\0' ; i++) {
      if(cmp[i]== '_') 	{
	expression s =  int_to_expression(0);
	sl = CONS(EXPRESSION, s, sl);
	change_p = true;
      }
    }
    if(change_p) {
      reference_indices(r) = gen_nconc( sl, reference_indices(r));
      f_a = make_cell_reference(r);  
      break ;
    }
    args = CDR(args);
  }
 
  if(cell_undefined_p(f_a)) 
    pips_user_error("Formal acces paths undefined \n");

  return f_a;
}


/* obsolete function*/
/* returns the prefix parameter of the stub c */
cell points_to_stub_prefix(cell s, list params)
{
  cell c = cell_undefined;
  reference r1 = cell_any_reference(s);
  entity v1 = reference_variable(r1);
  const char * en1 = entity_local_name(v1);
  while(!ENDP(params)) {
    c = CELL(CAR(params));
    reference r2 = cell_to_reference(c);
    entity v2 = reference_variable(r2);
    const char * en2 = entity_local_name(v2);
    char *cmp = strstr(en1, en2);
    if(cmp!=NULL)
      break ;
    params = CDR(params);
  }
  return c;

}

/* obsolete function*/
list points_to_backword_cell_translation(cell c, list params, set pt_in, set pt_binded)
{
  int i;
  list inds1 = NIL, inds2 = NIL;
  list inds = reference_indices(cell_any_reference(c));
  list tl = sink_to_sources(c, pt_in, true);
  // FI: impedance problem and memory leak
  points_to_graph pt_binded_g = make_points_to_graph(false, pt_binded);

  FOREACH(cell, s , tl) {
    if(stub_points_to_cell_p(s)) {
      cell f_a = formal_access_paths(s, params, pt_in);
      inds1 = reference_indices(cell_any_reference(f_a));
      inds2 = reference_indices(cell_any_reference(s));
      }
      cell pr = points_to_stub_prefix(s, params);
      tl = source_to_sinks(pr, pt_binded_g, true);
      for(i = 1; i <= (int) gen_length(inds1); i++) {
	list tmp = gen_full_copy_list(tl);
	FOREACH(cell, sk, tmp) {
	  tl = source_to_sinks(sk, pt_binded_g, true);
	}
	gen_free_list(tmp);
      }
      FOREACH(cell, st, tl) {
	reference r = cell_to_reference(st);
	reference_indices_(r) = gen_nconc(reference_indices_(r), inds2);
      }
  }
 
  if(!ENDP(inds)) {
    FOREACH(cell, t, tl) {
      reference r = cell_to_reference(t);
      reference_indices(r) = gen_nconc(reference_indices(r), inds);
    }
  }
  return tl;
}

/* obsolete function*/
/* Evalute c using points-to relation already computed */ 
list actual_access_paths(cell c, set pt_binded)
{
  bool exact_p = false;
  set_methods_for_proper_simple_effects();
  list l_in = set_to_sorted_list(pt_binded,
				 (int(*)(const void*, const void*))
				 points_to_compare_location);
  list l = eval_cell_with_points_to(c, l_in, &exact_p);
  generic_effects_reset_all_methods();
  return l;
}


/* obsolete function*/
/* Translate each element of E into the caller's scope */
list caller_addresses(cell c, list args, set pt_in, set pt_binded)
{
  
  cell c_f = formal_access_paths(c, args, pt_in);
  list a_p = actual_access_paths(c_f, pt_binded);
  if(ENDP(a_p))
    a_p = points_to_cell_translation(c,args, pt_in, pt_binded);
  return a_p;
}

/* obsolete function*/
list points_to_cell_translation(cell sink, list args, set pt_in, set pt_binded)
{
  list l = NIL, ca = NIL;
  list sources =  sink_to_sources(sink, pt_binded, true);
  FOREACH(cell, c, sources) {
    ca = gen_nconc(caller_addresses(c, args, pt_in, pt_binded), ca);
  }
  FOREACH(cell, s, ca) {
    l = gen_nconc(sink_to_sources(s, pt_binded, true), l);
  }
  return l;
}


/* returns all the element of E, the set of stubs created when the callee is analyzed.
 *
 * E = {e in pt_in U pt_out|entity_stub_sink_p(e)}
 *
 * FI->AM: a[*][*] or p[next] really are elements of set E?
 */
list stubs_list(set pt_in, set pt_out)
{
  list sli = points_to_set_to_stub_cell_list(pt_in, NIL);
  list slo = points_to_set_to_stub_cell_list(pt_out, sli);
  return slo;
}

/* Check compatibility of points-to set "pt_in" of the callee and
 * "pt_binded" of the call site in the caller.
 *
 * Parameter "stubs" is the set E in the intraprocedural and
 * interprocedural analysis chapters of Amira Mensi's PhD
 * dissertation. The list "stubs" contains all the stub references
 * generated when the callee is analyzed.
 *
 * Parameter "args" is a list of cells. Each cell is a reference to a
 * formal parameter of the callee.
 */
bool sets_binded_and_in_compatible_p(list stubs,
				     list args,
				     set pt_binded __attribute__ ((unused)),
				     set pt_in __attribute__ ((unused)),
				     set pt_out __attribute__ ((unused)),
				     set translation)
{
  bool compatible_p = true;
  // FI->AM: this set does not seem to be used
  // set io = new_simple_pt_map();
  set bm = new_simple_pt_map();
  list tmp = gen_full_copy_list(stubs);
  gen_sort_list(args, (gen_cmp_func_t) points_to_compare_ptr_cell );
  
  // io = set_union(io, pt_in, pt_out);

  /* "bm" is a mapping from the formal context to the calling
   * context. It includes the mapping from formal arguments to the
   * calling context.
   *
   * FI: how can "bm" be computed if "binded" and "in" are incompatible?
   */
  // bm = points_to_binding(args, pt_in, pt_binded);
  bm = translation;

  while(!ENDP(stubs) && compatible_p) {
    cell st = CELL(CAR(stubs));
    bool to_be_freed;
    type st_t = points_to_cell_to_type(st, &to_be_freed);
    if(pointer_type_p(st_t)) {
      bool found_p = false; // the comparison is symetrical
      FOREACH(CELL, c, tmp) {
	if(points_to_cell_equal_p(c, st))
	  found_p = true;
	if(found_p && !points_to_cell_equal_p(c, st)) {
	  bool to_be_freed_2;
	  type c_t = points_to_cell_to_type(c, &to_be_freed_2);
	  if(pointer_type_p(c_t)) {
	    // FI: impedance issue + memory leak
	    points_to_graph bm_g = make_points_to_graph(false, bm);
	    list act1 = source_to_sinks(c, bm_g, false);
	    list act2 = source_to_sinks(st, bm_g, false);
	    /* Compute the intersection of the abstract value lists of
	       the two stubs in the actual context. */
	    points_to_cell_list_and(&act1, act2);

	    if(!ENDP(act1)) {
	      /* We are still OK if the common element(s) are NULL,
		 NOWHERE or even maybe a general heap abstract
		 location. */
	      //entity ne = entity_null_locations();
	      //reference nr = make_reference(ne, NIL);
	      //cell nc = make_cell_reference(nr);
	      //cell nc = make_null_pointer_value_cell();
	      //entity he =  entity_all_heap_locations();
	      //reference hr = make_reference(he, NIL);
	      //cell hc = make_cell_reference(hr);
	  
	      //if((int) gen_length(act1) == 1
	      // && (!points_to_cell_in_list_p(nc, act1)
	      //     || !points_to_cell_in_list_p(hc, act1)))
	      // FI: I'm not convinced this test is correct
	      // I guess we should remove all possibly shared targets
	      // and then test for emptiness
	      if((int) gen_length(act1) == 1) {
		cell act1_c = CELL(CAR(act1));
		if(!null_cell_p(act1_c)
		   && !all_heap_locations_cell_p(act1_c)) {
		  compatible_p = false;
		  break;
		}
	      }
	    }
	    gen_free_list(act1);
	    gen_free_list(act2);
	  }
	  if(to_be_freed_2) free_type(c_t);
	}
      }
    }
    if(to_be_freed) free_type(st_t);
    stubs = CDR(stubs);
  }
  gen_full_free_list(tmp);
  // set_free(io);

  return compatible_p;
}
