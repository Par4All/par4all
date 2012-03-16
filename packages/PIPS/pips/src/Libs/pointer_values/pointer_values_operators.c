/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2010 HPC Project

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "effects.h"
#include "effects-util.h"
#include "text-util.h"
#include "effects-simple.h"
#include "effects-generic.h"
#include "misc.h"

#include "pointer_values.h"


/*
 @brief makes a pointer value cell_relation from the two input effects; the interpretation
        of the second effect cell is given by ci. The cells and potential descriptors are copied
        so that no sharing is introduced.
 @param lhs_eff effect gives the first cell of the returned pointer value cell_relation
 @param rhs_eff effect gives the second cell of the returned pointer value cell_relation
 @param ci gives the interpretation of rhs_eff cell in the returned cell_relation (either value_of or address_of).

 @return a cell_relation representing a pointer value relation.
 */
list make_simple_pv_from_simple_effects(effect lhs_eff, effect rhs_eff, cell_interpretation ci, list l_in)
{
  cell_relation pv;
  list l_pv = NIL;

  pips_debug(1,"begin for %s cell_interpretation and effects :\n", cell_interpretation_value_of_p(ci)? "value_of" : "address_of");
  pips_debug_effect(2, "lhs_eff:", lhs_eff);
  pips_debug_effect(2, "rhs_eff:", rhs_eff);

  tag lhs_t = effect_approximation_tag(lhs_eff);
  tag rhs_t = effect_approximation_tag(rhs_eff);
  tag t = approximation_and(lhs_t, rhs_t);

  pips_debug(5,"approximation before converting to store independent cells: %s\n",
	     t == is_approximation_exact ? "must": "may");

  if (t == is_approximation_exact) t = is_approximation_exact;

  cell lhs_c = make_cell(is_cell_reference, copy_reference(effect_any_reference(lhs_eff)));


  cell rhs_c = effect_cell(rhs_eff);
  bool changed_rhs_p = false;
  bool changed_lhs_p = false;

  if (undefined_pointer_value_cell_p(rhs_c))
    rhs_c = make_undefined_pointer_value_cell();
  else
    {
      rhs_c = make_cell(is_cell_reference, copy_reference(effect_any_reference(rhs_eff)));
      lhs_c = simple_cell_to_store_independent_cell(lhs_c, &changed_lhs_p);
      rhs_c = simple_cell_to_store_independent_cell(rhs_c, &changed_rhs_p);
    }

  if (changed_lhs_p || changed_rhs_p)
    {
      pips_debug(5, "approximation set to may after change to store independent cell\n");
      t = is_approximation_may;
    }

  if (cell_interpretation_value_of_p(ci))
    pv = make_value_of_pointer_value(lhs_c,
				     rhs_c,
				     t,
				     make_descriptor_none());
  else
    pv = make_address_of_pointer_value(lhs_c,
				       rhs_c,
				       t,
				       make_descriptor_none());

  bool exact_preceding_p;
  if (simple_cell_reference_preceding_p(cell_reference(lhs_c), descriptor_undefined,
					cell_reference(rhs_c), descriptor_undefined,
					transformer_undefined,
					true,
					& exact_preceding_p))
    {
      list l_remnants = NIL;
      cell_relation exact_old_pv = cell_relation_undefined;
      list l_old_values = NIL;

      pips_debug(4, "lhs path is a predecessor of rhs path, looking for an exact old value for rhs_eff\n");

      l_old_values = effect_find_equivalent_pointer_values(lhs_eff, l_in,
							   &exact_old_pv,
							   &l_remnants);
      gen_free_list(l_remnants);
      if (!cell_relation_undefined_p(exact_old_pv))
	{
	  cell_relation new_pv = simple_pv_translate(pv, false, exact_old_pv);
	  l_pv = CONS(CELL_RELATION, new_pv, l_pv);
	}
      else
	{
	  FOREACH(CELL_RELATION, old_pv, l_old_values)
	    {
	      cell_relation new_pv = simple_pv_translate(pv, false, old_pv);
	      l_pv = CONS(CELL_RELATION, new_pv, l_pv);
	    }
	}
      free_cell_relation(pv);
      gen_free_list(l_old_values);
    }
  else
    l_pv = CONS(CELL_RELATION, pv, l_pv);

  pips_debug_pvs(2, "generating:", l_pv);
  pips_debug(1,"end\n");
  return l_pv;
}

/**
   @brief eliminate the cells of l_kill from l_in
   @param l_kill is the list of effects describing the cells to eliminated from l_in
   @param l_in is the input list of pointer_values.
   @param ctxt is a pointer on the pointer values analysis contex.

   @return a list of newly allocated pointer_values
 */
list kill_pointer_values(list /* of cell_relations */ l_in,
			 list /* of effects */ l_kill,
			 pv_context * ctxt)
{
  list l_res = NIL;
  pips_debug_pvs(5, "l_in =", l_in);
  pips_debug_effects(5, "l_kill =", l_kill);

  if (ENDP(l_kill))
    {
      l_res = gen_full_copy_list(l_in);
    }
  else
    {
      l_res = l_in;
      FOREACH(EFFECT, eff_kill, l_kill)
	{
	  list l_cur = kill_pointer_value(eff_kill, l_res, ctxt);
	  if (l_res != l_in) gen_full_free_list(l_res);
	  l_res = l_cur;
	}
    }

  pips_debug_pvs(5, "returning :", l_res);
  return l_res;
}


/**
   @brief eliminate the cell of eff_kill from l_in
   @param eff_kill is the effect describing the cell to be killed
   @param l_in is the input list of pointer_values.
   @param ctxt is a pointer on the pointer values analysis contex.

   @return a list of newly allocated pointer_values
*/
/* Not yet very generic: either should be made generic or a specific version made
   for convex pointer values/effects.  */
list kill_pointer_value(effect eff_kill, list /* of cell_relations */ l_in,
			pv_context * ctxt)
{
  list l_out = NIL;

  pips_debug_pvs(1, "begin with l_in:", l_in);
  pips_debug_effect(1, "and eff_kill:", eff_kill);


  if (anywhere_effect_p(eff_kill))
    {
      /* all pointers may be killed */
      pips_debug(5, "anywhere case \n");

      FOREACH(CELL_RELATION, pv_in, l_in)
	{
	  cell_relation pv_out = copy_cell_relation(pv_in);
	  cell_relation_approximation_tag(pv_out) = is_approximation_may;
	  l_out = CONS(CELL_RELATION, pv_out, l_out);
	}
    }
  else
    {
      /* eff_kill characteristics */
      cell cell_kill = effect_cell(eff_kill);
      tag app_kill = effect_approximation_tag(eff_kill);
      reference ref_kill = effect_any_reference(eff_kill);
      entity e_kill = reference_variable(ref_kill);
      list ind_kill = reference_indices(ref_kill);
      size_t nb_ind_kill = gen_length(ind_kill);
      /******/

      /* using old_values, take into account the impact of eff_kill on
	 l_in pointer values second cells which must be expressed in terms of
	 unchanged paths.
      */
      list l_remnants = NIL;
      cell_relation exact_old_pv = cell_relation_undefined;
      list l_old_values = NIL;

      pips_debug(4, "begin, looking for an exact old value for eff_orig\n");

      l_old_values = effect_find_equivalent_pointer_values(eff_kill, l_in,
							   &exact_old_pv,
							   &l_remnants);
      pips_debug_pvs(3, "l_old_values:", l_old_values);
      pips_debug_pvs(3, "l_remnants:", l_remnants);
      pips_debug_pv(3, "exact_old_pv:", exact_old_pv);

      list l_keep = NIL;
      FOREACH(CELL_RELATION, pv_in, l_remnants)
	{
	  bool first_p = false; /* should we translate the first or the second pv_in cell */
	  bool to_be_translated = false; /* must it be translated ? */
	  bool exact_preceding_test = true;

	  /* pv_in first cell characteristics */
	  cell cell_in_1 = cell_relation_first_cell(pv_in);
	  reference ref_in_1 = cell_reference(cell_in_1);
	  entity e_in_1 = reference_variable(ref_in_1);
	  list ind_in_1 = reference_indices(ref_in_1);
	  size_t nb_ind_in_1 = gen_length(ind_in_1);
	  /******/

	  /* pv_in second cell characteristics */
	  cell cell_in_2 = cell_relation_second_cell(pv_in);
	  reference ref_in_2 = cell_reference(cell_in_2);
	  entity e_in_2 = reference_variable(ref_in_2);
	  list ind_in_2 = reference_indices(ref_in_2);
	  size_t nb_ind_in_2 = gen_length(ind_in_2);
	  /******/

	  pips_debug_pv(3, "considering pv_in:", pv_in);

	  if (same_entity_p(e_kill, e_in_2) && nb_ind_kill <= nb_ind_in_2)
	    {
	      if (cell_relation_second_address_of_p(pv_in) && nb_ind_kill == nb_ind_in_2)
		{
		  /* pointer value relation is still valid */
		  pips_debug(3, "address_of case, and nb_ind_in == nb_ind_kill_2\n");
		  to_be_translated = false;
		}
	      else
		{
		  pips_debug(3, "second cell is candidate for translation\n");
		  first_p = false;
		  bool inclusion_test_exact_p = false;
		  if ( (nb_ind_kill == nb_ind_in_2 &&
			cell_inclusion_p(cell_in_2, cell_kill, &inclusion_test_exact_p))
		       ||
		       (nb_ind_kill < nb_ind_in_2 &&
			simple_cell_reference_preceding_p(ref_kill, descriptor_undefined,
							  ref_in_2, descriptor_undefined,
							  transformer_undefined,
							  true,
							  &exact_preceding_test)))
		    to_be_translated = true;
		  else to_be_translated = false;
		}
	    }
	  else if (same_entity_p(e_kill, e_in_1) && nb_ind_kill <= nb_ind_in_1)
	    {
	      pips_debug(3, "first cell is candidate for translation\n");
	      first_p = true;
	      bool inclusion_test_exact_p = false;
	      if ( (nb_ind_kill == nb_ind_in_1 &&
		    cell_inclusion_p(cell_in_1, cell_kill, &inclusion_test_exact_p) )
		   ||
		   (nb_ind_kill < nb_ind_in_1 &&
		    simple_cell_reference_preceding_p(ref_kill, descriptor_undefined,
						      ref_in_1, descriptor_undefined,
						      transformer_undefined,
						      true,
						      &exact_preceding_test)))
		to_be_translated = true;
	      else to_be_translated = false;
	    }
	  else
	    {
	      to_be_translated = false;
	    }

	  if (to_be_translated)
	    {
	      pips_debug(3, "%s cell must be translated\n", first_p ? "first" : "second");

	      /* This should be made generic */

	      /* we must translate ref_in using the old_values */
	      /* if there is an exact candidate, it is ranked first
		 and we can use it */
	      if (exact_old_pv != cell_relation_undefined)
		{
		  cell_relation new_pv = simple_pv_translate(pv_in, first_p, exact_old_pv);
		  pips_debug_pv(3, "translated to:", new_pv);
		  l_out = CONS(CELL_RELATION, new_pv, l_out);
		}

	      else /* generate a new pv for each element of old_values */
		{
		  FOREACH(CELL_RELATION, old_pv, l_old_values)
		    {
		      cell_relation new_pv = simple_pv_translate(pv_in, first_p, old_pv);
		      pips_debug_pv(3, "translated to:", new_pv);
		      l_out = CONS(CELL_RELATION, new_pv, l_out);

		    } /*  FOREACH(CELL_RELATION, old_pv, l_old_values) */
		} /* else branch of if (exact_first_pv) */
	    } /*  if (to_be_translated) */
	  else
	    {
	      pips_debug(3, "non matching case, keep as is\n");
	      l_keep = CONS(CELL_RELATION, copy_cell_relation(pv_in), l_keep);
	    }
	} /* FOREACH (CELL_RELATION, pv_in, l_remnants) */

      list l_tmp = (*ctxt->pvs_must_union_func)(l_out, l_keep);
      //gen_full_free_list(l_out);
      //gen_full_free_list(l_keep);
      l_out = l_tmp;

      /* Second, take into account the impact of eff_kill on l_old_values relations.
	 We only have to keep those which are not completely killed by kill_eff, and
	 set their approximation to may (this is also true for exact_old_pv)
      */

      /* first use exact_old_pv to translate exact old_values */
      cell_relation tmp_old_pv = cell_relation_undefined;
      if (!cell_relation_undefined_p(exact_old_pv))
	{
	  pips_debug(3, "handling exact_old_pv\n");
	  reference ref_old =
	    cell_reference(cell_relation_second_cell(exact_old_pv));
	  if(same_entity_p(reference_variable(ref_old),e_kill))
	    {
	      pips_debug(3, "exact_old_pv is inverted -> translate\n");
	      tmp_old_pv = make_value_of_pointer_value
		(copy_cell(cell_relation_second_cell(exact_old_pv)),
		 copy_cell(cell_relation_first_cell(exact_old_pv)),
		 cell_relation_approximation_tag(exact_old_pv),
		 make_descriptor_none());
	    }
	  else
	    {
	      pips_debug(3, "exact_old_pv is not inverted\n");
	      tmp_old_pv = copy_cell_relation(exact_old_pv);
	    }
	}

      /* Then the other old_values */
      pips_debug(3, "dealing with old values\n");
      FOREACH(CELL_RELATION, pv_old, l_old_values)
	{
	  pips_debug_pv(3, "dealing with pv_old:", pv_old);
	  /* we already know that there may be a non-empty
	     intersection with cell_kill */
	  if (app_kill == is_approximation_may)
	    {
	      pips_debug(3, "may kill, just change the approximation\n");
	      cell_relation pv_out = copy_cell_relation(pv_old);
	      cell_relation_approximation_tag(pv_out) = is_approximation_may;
	      l_out = CONS(CELL_RELATION, pv_out, l_out);
	    }
	  else /* some more work is necessary */
	    {
	      cell first_cell_old = cell_relation_first_cell(pv_old);
	      bool exact_inclusion_p = false;
	      bool inclusion_p = cell_inclusion_p(first_cell_old, cell_kill,
						  &exact_inclusion_p);

	      if (inclusion_p && exact_inclusion_p)
		{
		  pips_debug(3, "first_cell_old exactly included in cell_kill"
			     " -> pv_old is translated or killed\n");
		  if(!cell_relation_undefined_p(tmp_old_pv)
		     && !(cell_relation_second_address_of_p(pv_old)
			  && (abstract_pointer_value_cell_p(cell_relation_second_cell(tmp_old_pv))
			      || cell_relation_second_address_of_p(tmp_old_pv))))
		    {
		      cell_relation new_pv = simple_pv_translate(tmp_old_pv,
								 true,
								 pv_old);
		      pips_debug_pv(3, "translating to:", new_pv);
		      l_out = CONS(CELL_RELATION, new_pv, l_out);
		    }
		}
	      else
		{
		  cell second_cell_old = cell_relation_second_cell(pv_old);
		  bool exact_inclusion_p = false;
		  bool inclusion_p = cell_inclusion_p(second_cell_old, cell_kill,
						      &exact_inclusion_p);
		  if (inclusion_p && exact_inclusion_p)
		    {
		      pips_debug(3, "second_cell_old exactly included in "
				 "cell_kill -> pv_old is translated or killed\n");

		      if(!cell_relation_undefined_p(tmp_old_pv))
			{
			  cell_relation new_pv = simple_pv_translate(tmp_old_pv,
								     true,
								     pv_old);
			  pips_debug_pv(3, "translating to:", new_pv);
			  l_out = CONS(CELL_RELATION, new_pv, l_out);
			}
		    }
		  else
		    {
		      pips_debug(3, "may be included case"
				 " -> keep with may approximation\n");
		      /* some more precise work could be done here by
			 computing the difference between the pv_old
			 first cell and cell_kill. I don't know if it
			 would be really useful.  So let us avoid too
			 complex things for the moment.
		      */
		      cell_relation pv_out = copy_cell_relation(pv_old);
		      cell_relation_approximation_tag(pv_out) =
			is_approximation_may;
		      l_out = CONS(CELL_RELATION, pv_out, l_out);
		    }
		}
	    }
	}
      gen_free_list(l_old_values);

      pips_debug(3, "dealing with exact_old_pv\n");
      if (!cell_relation_undefined_p(tmp_old_pv))
	{
	  cell first_cell_old = cell_relation_first_cell(tmp_old_pv);
	  bool exact_inclusion_p = false;
	  if (app_kill == is_approximation_may)
	    {
	      pips_debug(3,
			 "may kill, keep exact_old_pv with may approximation\n");
	      cell_relation pv_out = copy_cell_relation(exact_old_pv);
	      cell_relation_approximation_tag(pv_out) = is_approximation_may;
	      l_out = CONS(CELL_RELATION, pv_out, l_out);
	    }
	  else if (cell_inclusion_p(first_cell_old, cell_kill, &exact_inclusion_p)
		   && exact_inclusion_p)
	    {
	      pips_debug(3, "first cell of exact_old_pv exactly included in cell_kill "
			 "-> exact_old_pv is killed\n");
	    }
	  else
	    {
	      pips_debug(3, "may be included case"
			 " -> keep with may approximation\n");
	      /* some more precise work could be done here by
		 computing the difference between the pv_old
		 first cell and cell_kill. I don't know if it
		 would be really useful.  So let us avoid too
		 complex things for the moment.
	      */
	      cell_relation pv_out = copy_cell_relation(tmp_old_pv);
	      cell_relation_approximation_tag(pv_out) =
		is_approximation_may;
	      l_out = CONS(CELL_RELATION, pv_out, l_out);
	    }
	  free_cell_relation(tmp_old_pv);
	}

    }
  pips_debug_pvs(1, "returning:", l_out);

  return l_out;
}

/**
   @param pv_in a the input pointer_value relation
   @param in_first_p is true (false) if the first (second) cell of pv_in has to be translated
   @param pv_old is the cell relation that gives the value of a prefix path of the
          first (second) cell of pv_in
   @result a newly allocated pointer_value relation in which the first (second) has been translated.
 */
cell_relation simple_pv_translate(cell_relation pv_in, bool in_first_p, cell_relation pv_old)
{
  cell_relation pv_new;

  pips_debug_pv(5, "pv_in =", pv_in);
  pips_debug(5, "translating %s cell\n", in_first_p? "first": "second");
  pips_debug_pv(5, "pv_old =", pv_old);

  /* pv_in first or second cell characteristics */
  cell cell_in = in_first_p ? cell_relation_first_cell(pv_in) : cell_relation_second_cell(pv_in);
  reference ref_in = cell_reference(cell_in);
  entity e_in = reference_variable(ref_in);
  list ind_in = reference_indices(ref_in);
  size_t nb_ind_in = gen_length(ind_in);
  /******/

  /* pv_old characteristics */
  reference ref_old_1 =
    cell_reference(cell_relation_first_cell(pv_old));
  list ind_old_1 = reference_indices(ref_old_1);
  size_t nb_ind_old_1 = gen_length(ind_old_1);

  reference ref_old_2 =
    cell_reference(cell_relation_second_cell(pv_old));
  list ind_old_2 = reference_indices(ref_old_2);
  size_t nb_ind_old_2 = gen_length(ind_old_2);
  bool anywhere_old_p = cell_relation_second_address_of_p(pv_old)
    && entity_all_locations_p(reference_variable(ref_old_2)) ;
  /******/

  bool old_first_p = same_entity_p(reference_variable(ref_old_1), e_in); /* is the first cell of pv_old the prefix of ref_in? */

  //reference prefix_ref = old_first_p ? ref_old_1 : ref_old_2;
  reference target_ref = old_first_p ? ref_old_2 : ref_old_1;

  reference ref;
  descriptor d;
  bool exact_translation_p;
  int nb_common_indices;
  bool address_of_ref = false;

  if (old_first_p && anywhere_old_p)
    {
      cell c1 =  in_first_p ? copy_cell(cell_relation_second_cell(pv_in)) :
	copy_cell(cell_relation_first_cell(pv_in));
      cell c2 = copy_cell(cell_relation_second_cell(pv_old));
      pv_new = make_address_of_pointer_value(c1, c2,
					     is_approximation_may, make_descriptor_none());
    }
  else
    {
      if ( (!old_first_p) && cell_relation_second_address_of_p(pv_old))
	{
	  /* act as if there were a [0] indice at the end of ref_old_1 */
	  nb_common_indices = (int) nb_ind_old_1 + 1;

	  simple_cell_reference_with_value_of_cell_reference_translation
	    (ref_in, descriptor_undefined, /* not generic here */
	     target_ref, descriptor_undefined, /* not generic here */
	     nb_common_indices,
	     &ref, &d, &exact_translation_p);

	}
      else
	{
	  nb_common_indices = old_first_p ? (int) nb_ind_old_1 : (int) nb_ind_old_2;

	  if (cell_relation_second_address_of_p(pv_old))
	    {
	      if (nb_ind_in == 0)
		{
		  ref = copy_reference(target_ref);
		  exact_translation_p = true;
		  address_of_ref = true;
		}
	      else
		simple_cell_reference_with_address_of_cell_reference_translation
		  (ref_in, descriptor_undefined, /* not generic here */
		   target_ref, descriptor_undefined, /* not generic here */
		   nb_common_indices,
		   &ref, &d, &exact_translation_p);
	    }
	  else
	    simple_cell_reference_with_value_of_cell_reference_translation
	      (ref_in, descriptor_undefined, /* not generic here */
	       target_ref, descriptor_undefined, /* not generic here */
	       nb_common_indices,
	       &ref, &d, &exact_translation_p);
	}
      pips_debug(5, "ref after translation %s\n",
		 words_to_string(words_reference(ref, NIL)));

      tag new_t = (cell_relation_may_p(pv_in) || cell_relation_may_p(pv_old) || !exact_translation_p)
	? is_approximation_may : is_approximation_exact;

      if (in_first_p)
	{
	  if(cell_relation_second_value_of_p(pv_in))
	    {
	      if (!address_of_ref)
		pv_new = make_value_of_pointer_value(make_cell_reference(ref),
						     copy_cell(cell_relation_second_cell(pv_in)),
						     new_t, make_descriptor_none());
	      else
		pv_new = make_address_of_pointer_value(copy_cell(cell_relation_second_cell(pv_in)),
						       make_cell_reference(ref),
						       new_t, make_descriptor_none());
	    }
	  else
	    {
	      pips_assert("pointer values do not have two address of cells\n", !address_of_ref);
	      pv_new = make_address_of_pointer_value(make_cell_reference(ref),
						     copy_cell(cell_relation_second_cell(pv_in)),
						     new_t, make_descriptor_none());
	    }
	}
      else
	{
	  if(cell_relation_second_value_of_p(pv_in) && !address_of_ref )
	    pv_new = make_value_of_pointer_value(copy_cell(cell_relation_first_cell(pv_in)),
						 make_cell_reference(ref),
						 new_t, make_descriptor_none());
	  else
	    pv_new = make_address_of_pointer_value(copy_cell(cell_relation_first_cell(pv_in)),
						   make_cell_reference(ref),
						   new_t, make_descriptor_none());
	}
    }
  return pv_new;
}

/* This one is not generic at all and it's name should reflect the fact
   that it only concerns simple cells. The API should evolve as well: descriptors
   should be added or effects passed as arguments instead of cells.
*/
bool cell_inclusion_p(cell c1, cell c2, bool * exact_inclusion_test_p)
{
  bool res = true;
  *exact_inclusion_test_p = true;

  if (cell_gap_p(c1) || cell_gap_p(c2))
    pips_internal_error("gap case not handled yet ");

  reference r1 = cell_reference_p(c1)
    ? cell_reference(c1) : preference_reference(cell_preference(c1));
  entity e1 = reference_variable(r1);
  reference r2 = cell_reference_p(c2)
    ? cell_reference(c2) : preference_reference(cell_preference(c2));
  entity e2 = reference_variable(r2);
  pips_debug(8, "begin for r1 = %s, and r2 = %s\n",
	     words_to_string(words_reference(r1, NIL)),
	     words_to_string(words_reference(r2, NIL)));

  /* only handle all_locations cells for the moment */
  if (entity_all_locations_p(e1))
    {
      if (entity_all_locations_p(e2))
	{
	  pips_debug(8, "case 1\n");
	  *exact_inclusion_test_p = true;
	  res = true;
	}
      else
	{
	  pips_debug(8, "case 2\n");
	  *exact_inclusion_test_p = true;
	  res = false;
	}
    }
  else
    {
      if (entity_all_locations_p(e2)) /* we cannot have entity_all_locations_p(e1) here */
	{
	  pips_debug(8, "case 3\n");
	  *exact_inclusion_test_p = true;
	  res = true;
	}
      else if (same_entity_p(e1, e2))
	{
	  list inds1 = reference_indices(r1);
	  list inds2 = reference_indices(r2);

	  pips_debug(8, "case 4, same entities: %s\n", entity_name(e1));

	  if (gen_length(inds1) == gen_length(inds2))
	    {
	      pips_debug(8, "same number of dimensions\n");
	      for(;!ENDP(inds1) && res == true; POP(inds1), POP(inds2))
		{
		  expression exp1 = EXPRESSION(CAR(inds1));
		  expression exp2 = EXPRESSION(CAR(inds2));

		  if (unbounded_expression_p(exp1))
		    {
		      pips_debug(8, "case 4.1\n");
		      if (!unbounded_expression_p(exp2))
			{
			  pips_debug(8, "case 4.2\n");
			  res = false;
			  *exact_inclusion_test_p = true;
			}
		    }
		  else if (!unbounded_expression_p(exp2) && !expression_equal_p(exp1, exp2) )
		    {
		      pips_debug(8, "case 4.3\n");
		      res = false;
		      *exact_inclusion_test_p = true;
		    }
		}
	    }
	  else
	    {
	      pips_debug(8, "not same number of dimensions\n");
	      *exact_inclusion_test_p = true;
	      res = false;
	    }
	}
      else
	{
	  pips_debug(8, "case 5, not same entities: %s\n", entity_name(e1));

	  *exact_inclusion_test_p = true;
	  res = false;
	}
    }
  pips_debug(8, "returning %s (%s)\n", res? "true":"false", *exact_inclusion_test_p ? "exact": "non exact");

  return res;
}

/* This one is not generic at all and it's name should reflect the fact
   that it only concerns simple cells. The API should evolve as well: descriptors
   should be added or effects passed as arguments instead of cells.
*/
bool cell_intersection_p(cell c1, cell c2, bool * intersection_test_exact_p)
{

  bool res = true;
  *intersection_test_exact_p = true;

  if (cell_gap_p(c1) || cell_gap_p(c2))
    pips_internal_error("gap case not handled yet ");

  reference r1 = cell_reference_p(c1) ? cell_reference(c1) :
    preference_reference(cell_preference(c1));
  entity e1 = reference_variable(r1);
  reference r2 = cell_reference_p(c2) ? cell_reference(c2) :
    preference_reference(cell_preference(c2));
  entity e2 = reference_variable(r2);

  /* only handle all_locations cells for the moment */
  if (entity_all_locations_p(e1))
    {
      *intersection_test_exact_p = true;
      res = true;
    }
  else if (entity_all_locations_p(e2))
	{
	  *intersection_test_exact_p = true;
	  res = true;
	}
  else if (same_entity_p(e1, e2))
    {
      list inds1 = reference_indices(r1);
      list inds2 = reference_indices(r2);

      if (gen_length(inds1) == gen_length(inds2))
	{
	  for(;!ENDP(inds1) && res == true; POP(inds1), POP(inds2))
	    {
	      expression exp1 = EXPRESSION(CAR(inds1));
	      expression exp2 = EXPRESSION(CAR(inds2));

	      if (!unbounded_expression_p(exp1)
		  && !unbounded_expression_p(exp2) &&
		  !expression_equal_p(exp1, exp2) )
		{
		  res = false;
		  *intersection_test_exact_p = true;
		}
	    }
	}
      else
	{
	  *intersection_test_exact_p = true;
	  res = false;
	}
    }
  else
    {
      *intersection_test_exact_p = true;
      res = false;
    }

  return res;
}


/**
  @input eff is an input effect describing a memory path
  @return a list of effects corresponding to effects on eff cell prefix pointer paths
*/
/* This one could be made generic */
list simple_effect_intermediary_pointer_paths_effect(effect eff)
{
  pips_debug_effect(5, "input effect :", eff);
  list l_res = NIL;
  reference ref = effect_any_reference(eff);
  entity e = reference_variable(ref);
  list ref_inds = reference_indices(ref);
  reference tmp_ref = make_reference(e, NIL);
  type t = entity_basic_concrete_type(e);
  bool finished = false;

  if (entity_all_locations_p(e))
    return CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), NIL);

  while (!finished && !ENDP(ref_inds))
    {
      switch (type_tag(t))
	{

	case is_type_variable:
	  {
	    pips_debug(5," variable case\n");
	    basic b = variable_basic(type_variable(t));
	    size_t nb_dim = gen_length(variable_dimensions(type_variable(t)));

	    /* add to tmp_ref as many indices from ref as nb_dim */
	    for(size_t i = 0; i< nb_dim; i++, POP(ref_inds))
	      {
		reference_indices(tmp_ref) =
		  gen_nconc(reference_indices(tmp_ref),
			    CONS(EXPRESSION,
				 copy_expression(EXPRESSION(CAR(ref_inds))),
				 NIL));
	      }

	    if (basic_pointer_p(b))
	      {
		pips_debug(5," pointer basic\n");
		if (!ENDP(ref_inds))
		  {
		    pips_debug(5,"and ref_inds is not empty\n");
		    effect tmp_eff =
		      make_effect(make_cell_reference(copy_reference(tmp_ref)),
				  copy_action(effect_action(eff)),
				  copy_approximation(effect_approximation(eff)),
				  make_descriptor_none());
		    l_res = CONS(EFFECT, tmp_eff, l_res);
		    reference_indices(tmp_ref) =
		      gen_nconc(reference_indices(tmp_ref),
				CONS(EXPRESSION,
				     copy_expression(EXPRESSION(CAR(ref_inds))),
				     NIL));
		    POP(ref_inds);

		    type new_t = copy_type(basic_pointer(b));
		    /* free_type(t);*/
		    t = new_t;
		  }
		else
		  finished = true;
	      }
	    else if (basic_derived_p(b))
	      {
		pips_debug(5,"derived basic\n");
		type new_t = entity_basic_concrete_type(basic_derived(b));
		t = new_t;
	      }
	    else
	      finished = true;
	  }
	  break;
	case is_type_struct:
	case is_type_union:
	case is_type_enum:
	  {
	    pips_debug(5,"struct union or enum type\n");

	    /* add next index */
	    expression field_exp = EXPRESSION(CAR(ref_inds));
	    reference_indices(tmp_ref) =
	      gen_nconc(reference_indices(tmp_ref),
			CONS(EXPRESSION,
			     copy_expression(field_exp),
			     NIL));
	    POP(ref_inds);
	    entity field_ent = expression_to_entity(field_exp);
	    pips_assert("expression is a field entity\n", !entity_undefined_p(field_ent));
	    type new_t = entity_basic_concrete_type(field_ent);
	    t = new_t;
	  }
	  break;
	default:
	    pips_internal_error("unexpected type tag");

	}
    }
  return l_res;
}

/**
   @brief find pointer_values in l_in which give (possible or exact) paths
          equivalent to eff.
   @param eff is the considered input path.
   @param l_in is the input pointer values list.
   @param exact_aliased_pv gives an exact equivalent path found in l_in if it exists.
   @param l_in_remnants contains the elemnts of l_in which are neither
          exact_aliased_pv nor in the returned list.

   @return a list of elements of l_in which give (possible or exact) paths
           equivalent to eff, excluding exact_aliased_pv if one exact equivalent
           path can be found in l_in.
 */
list effect_find_equivalent_pointer_values(effect eff, list l_in,
					   cell_relation * exact_aliased_pv,
					   list * l_in_remnants)
{

  pips_debug_pvs(1,"begin, l_in =", l_in);
  pips_debug_effect(1, "and eff:", eff);

  /* eff characteristics */
  cell eff_cell = effect_cell(eff);
  //reference ref = effect_any_reference(eff);
  //list ind = reference_indices(ref);
  /******/

 /* first, search for the (exact/possible) values of eff cell in l_in */
  /* we search for the cell_relations where ref appears
     as a first cell, or the exact value_of pointer_values where ref appears as
     a second cell. If an exact value_of relation is found, it is retained in
     exact_aliased_pv
  */
  *l_in_remnants = NIL;
  *exact_aliased_pv = cell_relation_undefined;
  list l_res = NIL;

  FOREACH(CELL_RELATION, pv_in, l_in)
    {
      cell first_cell_in = cell_relation_first_cell(pv_in);
      cell second_cell_in = cell_relation_second_cell(pv_in);
      bool intersection_test_exact_p = false;
      bool inclusion_test_exact_p = true;

      pips_debug_pv(4, "considering:", pv_in);
      if (cell_intersection_p(eff_cell, first_cell_in,
			      &intersection_test_exact_p))
	{
	  pips_debug(4, "non empty intersection with first cell (%sexact)\n",
		     intersection_test_exact_p? "": "non ");
	  if (cell_relation_exact_p(pv_in)
	      && intersection_test_exact_p
	      && cell_inclusion_p(eff_cell, first_cell_in,
				  &inclusion_test_exact_p)
	      && inclusion_test_exact_p)
	    {
	      if (cell_relation_undefined_p(*exact_aliased_pv))
		{
		  pips_debug(4, "exact value candidate found\n");
		  *exact_aliased_pv = pv_in;
		}
	      else if ((cell_relation_second_address_of_p(*exact_aliased_pv)
			&& cell_relation_second_value_of_p(pv_in))
		       || null_pointer_value_cell_p(cell_relation_second_cell(pv_in))
		       || undefined_pointer_value_cell_p(cell_relation_second_cell(pv_in)))
		{
		  pips_debug(4, "better exact value candidate found\n");
		  l_res = CONS(CELL_RELATION, *exact_aliased_pv, l_res);
		  *exact_aliased_pv = pv_in;
		}
	      else
		{
		  pips_debug(4, "not kept as exact candidate\n");
		  l_res = CONS(CELL_RELATION, pv_in, l_res);
		}
	    }
	  else
	    {
	      pips_debug(5, "potentially non exact value candidate found\n");
	      l_res = CONS(CELL_RELATION, pv_in, l_res);
	    }
	}
      else if(cell_relation_second_value_of_p(pv_in)
	      && cell_intersection_p(eff_cell, second_cell_in,
				     &intersection_test_exact_p))
	{
	  pips_debug(4, "non empty intersection with second value_of cell "
		     "(%sexact)\n", intersection_test_exact_p? "": "non ");
	  if(cell_relation_exact_p(pv_in)
	      && intersection_test_exact_p
	      && cell_inclusion_p(eff_cell, second_cell_in,
				  &inclusion_test_exact_p)
	      && inclusion_test_exact_p)
	    {
	      if (cell_relation_undefined_p(*exact_aliased_pv))
		{
		  pips_debug(4, "exact value candidate found\n");
		  *exact_aliased_pv = pv_in;
		}
	       else if (cell_relation_second_address_of_p(*exact_aliased_pv))
		{
		  pips_debug(4, "better exact value candidate found\n");
		  l_res = CONS(CELL_RELATION, *exact_aliased_pv, l_res);
		  *exact_aliased_pv = pv_in;
		}
	      else
		{
		  pips_debug(4, "not kept as exact candidate\n");
		  l_res = CONS(CELL_RELATION, pv_in, l_res);
		}
	    }
	  else
	    {
	      pips_debug(5, "potentially non exact value candidate found\n");
	      l_res = CONS(CELL_RELATION, pv_in, l_res);
	    }
	}
      else
	{
	  pips_debug(4, "remnant\n");
	  *l_in_remnants = CONS(CELL_RELATION, pv_in, *l_in_remnants);
	}
    }
  pips_debug_pvs(3, "l_in_remnants:", *l_in_remnants);
  pips_debug_pvs(3, "l_res:", l_res);
  pips_debug_pv(3, "*exact_aliased_pv:", *exact_aliased_pv);

  return l_res;
}

/**
   @brief find all paths equivalent to eff cell in l_pv by performing a transitive closure
   @param eff is the input effect
   @param l_pv is the list of current pointer_values relations
   @param ctxt is the pv analysis context
   @return a list of effects whose cells are equivalent to eff_kill cell according to l_pv.
           Their approximation does not depend on the approximation of the input effect,
	   but only on the exactness of the finding process.

 */
list effect_find_aliased_paths_with_pointer_values(effect eff, list l_pv, pv_context *ctxt)
{
  list l_res = NIL;
  list l_remnants = l_pv;
  reference eff_ref = effect_any_reference(eff);
  bool anywhere_p = false;

  pips_debug_effect(5, "begin with eff:", eff);
  pips_debug_pvs(5, "and l_pv:", l_pv);

  if (anywhere_effect_p(eff)) /* should be turned into entity_abstract_location_p */
    {
      pips_debug(5, "anywhere case\n");
      return (NIL);
    }
  else
    {
      /* first we must find in eff_kill intermediary paths to pointers */
      /* not generic here */
      list l_intermediary = simple_effect_intermediary_pointer_paths_effect(eff);
      pips_debug_effects(5, "intermediary paths to eff:", l_intermediary);

      /* and find if this gives equivalent paths in l_pv */
      FOREACH(EFFECT, eff_intermediary, l_intermediary)
	{
	  pips_debug_effect(5, "considering intermediary path:", eff_intermediary);
	  list tmp_l_remnants = NIL;
	  cell_relation pv_exact = cell_relation_undefined;
	  list l_equiv = effect_find_equivalent_pointer_values(eff_intermediary,
							       l_remnants,
							       &pv_exact,
							       &tmp_l_remnants);
	  if (!cell_relation_undefined_p(pv_exact))
	    {
	     l_equiv = CONS(CELL_RELATION, pv_exact, l_equiv);
	    }
	  l_remnants = tmp_l_remnants;
	  pips_debug_pvs(5, "list of equivalent pvs \n", l_equiv);

	  reference ref_intermediary = effect_any_reference(eff_intermediary);
	  entity ent_intermediary = reference_variable(ref_intermediary);
	  //descriptor d_intermediary = effect_descriptor(eff_intermediary);
	  int nb_common_indices =
	    (int) gen_length(reference_indices(ref_intermediary));

	  FOREACH(CELL_RELATION, pv_equiv, l_equiv)
	    {
	      reference ref;
	      descriptor d;
	      bool exact_translation_p;
	      cell c1 = cell_relation_first_cell(pv_equiv);
	      cell c2 = cell_relation_second_cell(pv_equiv);

	      pips_debug_pv(5, "translating eff using pv: \n", pv_equiv);

	      if (undefined_pointer_value_cell_p(c1)
		  || undefined_pointer_value_cell_p(c2)
		  || null_pointer_value_cell_p(c1)
		  || null_pointer_value_cell_p(c2))
		{
		  if (cell_relation_exact_p(pv_equiv))
		    pips_user_error("\n\tdereferencing an undefined or null pointer (%s)\n",
				    words_to_string(effect_words_reference(effect_any_reference(eff))));
		  else
		    {
		      pips_debug(5,"potential dereferencement of an undefined or null pointer\n");
		    }
		}
	      else
		{
		  /* this is valid only if the first value_of corresponds
		     to eff_intermediary */
		  reference pv_equiv_first_ref = cell_reference(c1);
		  reference pv_equiv_second_ref = cell_reference(c2);

		  if (same_entity_p(ent_intermediary, reference_variable(pv_equiv_first_ref))
		      && (gen_length(reference_indices(ref_intermediary))
			  == gen_length(reference_indices(pv_equiv_first_ref))))
		    {
		      /* use second cell as equivalent value for intermediary path */
		      if (cell_relation_second_value_of_p(pv_equiv))
			{
			  (*ctxt->cell_reference_with_value_of_cell_reference_translation_func)
			    (eff_ref, descriptor_undefined, /* not generic here */
			     pv_equiv_second_ref,
			     descriptor_undefined, /* not generic here */
			     nb_common_indices,
			     &ref, &d, &exact_translation_p);
			}
		      else /* cell_relation_second_address_of_p is true */
			{
			  (*ctxt->cell_reference_with_address_of_cell_reference_translation_func)
			    (eff_ref, descriptor_undefined, /* not generic here */
			     pv_equiv_second_ref,
			     descriptor_undefined, /* not generic here */
			     nb_common_indices,
			     &ref, &d, &exact_translation_p);
			}
		    }
		  else /* use first cell as equivalent value for intermediary path  */
		    {
		      pips_assert("pv_equiv must be value_of here\n",
				  cell_relation_second_value_of_p(pv_equiv));

		      (*ctxt->cell_reference_with_value_of_cell_reference_translation_func)
			(eff_ref, descriptor_undefined, /* not generic here */
			 pv_equiv_first_ref,
			 descriptor_undefined, /* not generic here */
			 nb_common_indices,
			 &ref, &d, &exact_translation_p);
		    }
		  exact_translation_p = exact_translation_p && cell_relation_exact_p(pv_equiv);

		  effect eff_alias = make_effect(make_cell_reference(ref),
						 copy_action(effect_action(eff_intermediary)),
						 exact_translation_p ?
						 make_approximation_exact()
						 : make_approximation_may(), make_descriptor_none());
		  pips_debug_effect(5, "resulting effect \n", eff_alias);
		  if (anywhere_effect_p(eff_alias))
		    {
		      gen_full_free_list(l_res);
		      l_res = CONS(EFFECT, eff_alias, NIL);
		      anywhere_p = true;
		    }
		  else
		    {
		      l_res = CONS(EFFECT, eff_alias, l_res);
		    }
		}
	    } /* FOREACH */
	}

      if (!anywhere_p)
	{
	  pips_debug_effects(5, "l_res after first phase : \n", l_res);

	  /* Then we must find  if there are address_of second cells
	     which are preceding paths of eff path
	     in which case they must be used to generate other aliased paths
	  */
	  list l_remnants_2 = NIL;
	  FOREACH(CELL_RELATION, pv_remnant, l_remnants)
	    {
	      reference pv_remnant_second_ref =
		cell_reference(cell_relation_second_cell(pv_remnant));
	      bool exact_preceding_test = true;

	      pips_debug_pv(5, "considering pv: \n", pv_remnant);

	      if (cell_relation_second_address_of_p(pv_remnant)
		  && same_entity_p(reference_variable(eff_ref),
				   reference_variable(pv_remnant_second_ref))
		  && (gen_length(reference_indices(eff_ref))
		      >= gen_length(reference_indices(pv_remnant_second_ref)))
		  && simple_cell_reference_preceding_p(pv_remnant_second_ref, descriptor_undefined,
						       eff_ref, descriptor_undefined,
						       transformer_undefined,
						       true,
						       &exact_preceding_test))
		{
		  reference ref;
		  descriptor d;
		  bool exact_translation_p;

		  pips_debug(5, "good candidate (%sexact)\n",exact_preceding_test? "":"non ");
		  /* for the translation, add a dereferencing_dimension to pv_remnant_first_cell */
		  reference new_ref = copy_reference
		    (cell_reference(cell_relation_first_cell(pv_remnant)));
		  int nb_common_indices = (int) gen_length(reference_indices(pv_remnant_second_ref));
		  /* not generic here */
		  reference_indices(new_ref) = gen_nconc(reference_indices(new_ref),
							 CONS(EXPRESSION,
							      int_to_expression(0),
							      NIL));

		  (*ctxt->cell_reference_with_value_of_cell_reference_translation_func)
		    (eff_ref, descriptor_undefined, /* not generic here */
		     new_ref,
		     descriptor_undefined, /* not generic here */
		     nb_common_indices,
		     &ref, &d, &exact_translation_p);

		  exact_translation_p = exact_translation_p && cell_relation_exact_p(pv_remnant);

		  effect eff_alias = make_effect(make_cell_reference(ref),
						 make_action_write_memory(),
						 exact_translation_p && exact_preceding_test ?
						 make_approximation_exact()
						 : make_approximation_may(), make_descriptor_none());
		  free_reference(new_ref);
		  pips_debug_effect(5, "resulting effect \n", eff_alias);
		  l_res = CONS(EFFECT, eff_alias, l_res);

		}
	      else
		{
		  l_remnants_2 = CONS(CELL_RELATION, pv_remnant, l_remnants_2);
		}
	    } /* FOREACH */

	  l_remnants = l_remnants_2;
	} /* if (!anywhere_p)*/
      if (!ENDP(l_remnants))
	{
	  pips_debug(5, "recursing to find aliases to aliased effect...\n");
	  pips_debug_effects(5, "l_res before recursing : \n", l_res);
	  list l_recurs = NIL;
	  FOREACH(EFFECT, eff_alias, l_res)
	    {
	      l_recurs = gen_nconc(l_recurs,
				   effect_find_aliased_paths_with_pointer_values(eff_alias,
										 l_remnants,
										 ctxt));
	    }
	  l_res = gen_nconc(l_recurs, l_res);
	}
    } /* else branche of if (anywhere_effect_p(eff))*/

  pips_debug_effects(5, "returning : \n", l_res);
  return l_res;
}


void pointer_values_remove_var(entity e, bool may_p, list l_in,
			       pv_results *pv_res, pv_context *ctxt)
{
  pips_debug(5, "begin for entity %s\n", entity_name(e));
  pips_debug_pvs(5, "input l_in\n", l_in);

  /* possibly assign an undefined value to pointers reachable
     from e without dereferencements) */
  expression exp = entity_to_expression(e);
  assignment_to_post_pv(exp, may_p,
			expression_undefined,
			false, l_in, pv_res, ctxt);
  l_in = pv_res->l_out;
  free_expression(exp);

  /* Then replace all occurrences of e by an
     undefined value if it's not a may kill */
  if (!may_p)
    {
      list l_out = NIL;
      FOREACH(CELL_RELATION, pv, l_in)
	{
	  pips_debug_pv(5, "considering pv:", pv);
	  cell_relation new_pv = copy_cell_relation(pv);
	  cell c1 = cell_relation_first_cell(new_pv);
	  entity e1 = reference_variable(cell_reference(c1));

	  cell c2 = cell_relation_second_cell(new_pv);
	  entity e2 = reference_variable(cell_reference(c2));
	  bool keep = true;

	  if (same_entity_p(e1, e))
	    {
	      if (!undefined_pointer_value_cell_p(c2))
		{
		  free_cell(c1);
		  cell_relation_first_cell(new_pv)
		    = make_undefined_pointer_value_cell();
		}
	      else
		keep = false;
	    }
	  else if (same_entity_p(e2, e))
	    {
	      if (!undefined_pointer_value_cell_p(c1))
		{
		  free_cell(c2);
		  cell_relation_second_cell(new_pv)
		    = make_undefined_pointer_value_cell();
		  cell_relation_second_interpretation_tag(new_pv)
		    = is_cell_interpretation_value_of;
		}
	      else keep = false;
	    }
	  if (keep)
	    l_out = CONS(CELL_RELATION, new_pv, l_out);
	  else
	    free_cell_relation(new_pv);
	}
      gen_full_free_list(l_in);
      pv_res->l_out = l_out;
    }
  pips_debug_pvs(5, "end with pv_res->l_out = \n", pv_res->l_out );
}



/*
   @brief change each element of simple pointer values input list into a store independent pointer value.

   @param l_pv is the input list of simple pointer values
   @param t is unused, but is here for homogeneity purposes
*/
cell_relation simple_pv_composition_with_transformer(cell_relation pv, transformer  __attribute__ ((unused)) t)
{
  cell c1 = cell_relation_first_cell(pv);
  cell c2 = cell_relation_second_cell(pv);

  bool b1, b2;

  cell_relation_first_cell(pv) = simple_cell_to_store_independent_cell(c1, &b1);
  cell_relation_second_cell(pv) = simple_cell_to_store_independent_cell(c2, &b2);

  if (cell_relation_exact_p(pv) && (b1 || b2))
    cell_relation_approximation_tag(pv) = is_approximation_may;
  return pv;
}

/*
   @brief report the impact of store modification modelized by the input transformer onto the input list of pointer values

   @param l_pv is the input list of pointer values
   @param t is the transfomer that modelizes the store modification
   @param ctxt is a pointer on the pointer value analysis context holder.
*/
list pvs_composition_with_transformer(list l_pv, transformer t, pv_context * ctxt)
{
  FOREACH(CELL_RELATION, pv, l_pv)
    {
     pv = (*ctxt->pv_composition_with_transformer_func)(pv, t);
    }

  return l_pv;
}



list cell_relation_to_list(cell_relation cr)
{
  return CONS(CELL_RELATION, cr, NIL);
}

list cell_relation_to_may_list(cell_relation cr)
{
  cell_relation_approximation_tag(cr) = is_approximation_may;
  return CONS(CELL_RELATION, cr, NIL);
}

list simple_pv_must_union(cell_relation pv1, cell_relation pv2)
{
  pips_debug_pv(5, "pv1 =\n", pv1);
  pips_debug_pv(5, "pv2 =\n", pv2);

  cell c_first_1 = cell_relation_first_cell(pv1);
  cell c_second_1 = cell_relation_second_cell(pv1);

  cell c_first_2 = cell_relation_first_cell(pv2);
  cell c_second_2 = cell_relation_second_cell(pv2);

  cell_relation pv = cell_relation_undefined;
  if (entity_all_locations_p(cell_entity(c_second_1)))
    {
      pips_debug(5, "pv1 second cell is anywhere\n");
      pv = make_address_of_pointer_value(copy_cell(c_first_1),
					 copy_cell(c_second_1),
					 cell_relation_approximation_tag(pv1),
					 make_descriptor_none());
    }
  else if (entity_all_locations_p(cell_entity(c_second_2)))
    {
      pips_debug(5, "pv2 second cell is anywhere\n");
      pv = make_address_of_pointer_value(copy_cell(c_first_2),
					 copy_cell(c_second_2),
					 cell_relation_approximation_tag(pv2),
					 make_descriptor_none());
    }
  else
    {
      pips_debug(5, "general case\n");
      if ((cell_compare(&c_first_1, &c_first_2) == 0
	   && cell_compare(&c_second_1, &c_second_2) == 0)
	  || (cell_compare(&c_first_1, &c_second_2) == 0
	      && cell_compare(&c_second_1, &c_first_2) == 0)
	  )
	{

	  tag t1 = cell_relation_approximation_tag(pv1);
	  tag t2 = cell_relation_approximation_tag(pv2);
	  tag t;

	  if (t1 == t2) t = t1;
	  else t = is_approximation_exact;

	  pv = copy_cell_relation(pv1);
	  cell_relation_approximation_tag(pv) = t;
	}
     else
       {

	  // first cells are equal, but not second cells indices
	  // generate a pv with an unbounded dimension wherever dimensions
	  // are not equal
	  pv = copy_cell_relation(pv1);
	  cell_relation_approximation_tag(pv) = is_approximation_may;

	  cell c_second_pv = cell_relation_second_cell(pv);
	  list l_ind_c_second_pv = reference_indices(cell_any_reference(c_second_pv));
	  list l_ind_c_second_2 = reference_indices(cell_any_reference(c_second_2));

	  for(; !ENDP(l_ind_c_second_pv); POP(l_ind_c_second_pv), POP(l_ind_c_second_2))
	    {
	      expression ind_pv = EXPRESSION(CAR(l_ind_c_second_pv));
	      expression ind_2 = EXPRESSION(CAR(l_ind_c_second_2));

	      if (!expression_equal_p(ind_pv, ind_2))
		{
		  EXPRESSION_(CAR(l_ind_c_second_pv)) = make_unbounded_expression();
		}
	    }

       }
    }
  pips_debug_pv(5, "pv =\n", pv);
  list l_res = CONS(CELL_RELATION, pv, NIL);
  pips_debug_pvs(5, "returning:\n", l_res);
  return l_res;
}

list simple_pv_may_union(cell_relation pv1, cell_relation pv2)
{
  cell c_first_1 = cell_relation_first_cell(pv1);
  cell c_second_1 = cell_relation_second_cell(pv1);

  cell c_first_2 = cell_relation_first_cell(pv2);
  cell c_second_2 = cell_relation_second_cell(pv2);

  cell_relation pv;
  if (entity_all_locations_p(cell_entity(c_second_1)))
    {
      pips_debug(5, "pv1 second cell is anywhere\n");
      pv = make_address_of_pointer_value(copy_cell(c_first_1),
					 copy_cell(c_second_1),
					 cell_relation_approximation_tag(pv1),
					 make_descriptor_none());
    }
  else if (entity_all_locations_p(cell_entity(c_second_2)))
    {
      pips_debug(5, "pv2 second cell is anywhere\n");
      pv = make_address_of_pointer_value(copy_cell(c_first_2),
					 copy_cell(c_second_2),
					 cell_relation_approximation_tag(pv2),
					 make_descriptor_none());
    }
  else
    {
      if ((cell_compare(&c_first_1, &c_first_2) == 0
	   && cell_compare(&c_second_1, &c_second_2) == 0)
	  || (cell_compare(&c_first_1, &c_second_2) == 0
	      && cell_compare(&c_second_1, &c_first_2) == 0)
	  )
	{

	  tag t1 = cell_relation_approximation_tag(pv1);
	  tag t2 = cell_relation_approximation_tag(pv2);
	  tag t;

	  if (t1 == t2) t = t1;
	  else t = is_approximation_may;

	  pv = copy_cell_relation(pv1);
	  cell_relation_approximation_tag(pv) = t;
	}
      else
	{
	  // first cells are equal, but not second cells indices
	  // generate a pv with an unbounded dimension wherever dimensions
	  // are not equal
	  pv = copy_cell_relation(pv1);
	  cell_relation_approximation_tag(pv) = is_approximation_may;

	  cell c_second_pv = cell_relation_second_cell(pv);
	  list l_ind_c_second_pv = reference_indices(cell_any_reference(c_second_pv));
	  list l_ind_c_second_2 = reference_indices(cell_any_reference(c_second_2));
	  
	  for(; !ENDP(l_ind_c_second_pv); POP(l_ind_c_second_pv), POP(l_ind_c_second_2))
	    {
	      expression ind_pv = EXPRESSION(CAR(l_ind_c_second_pv));
	      expression ind_2 = EXPRESSION(CAR(l_ind_c_second_2));

	      if (!expression_equal_p(ind_pv, ind_2))
		{
		  EXPRESSION_(CAR(l_ind_c_second_pv)) = make_unbounded_expression();
		}
	    }

	}
    }
  list l_res = CONS(CELL_RELATION, pv, NIL);
  pips_debug_pvs(5, "returning:\n", l_res);
  return l_res;
}

bool pvs_union_combinable_p(cell_relation pv1, cell_relation pv2)
{
  bool undef1 = cell_relation_undefined_p(pv1);
  bool undef2 = cell_relation_undefined_p(pv2);

  pips_assert("error: there should be no undefined cell_relations in lists\n", !(undef1 && undef2));

  if (undef1 || undef2) return true;
  if (pv_cells_mergeable_p(pv1, pv2)) return true;


  cell c_first_1 = cell_relation_first_cell(pv1);
  cell c_second_1 = cell_relation_second_cell(pv1);

  cell c_first_2 = cell_relation_first_cell(pv2);
  cell c_second_2 = cell_relation_second_cell(pv2);

  if (entity_all_locations_p(cell_entity(c_second_1))
      && ! cell_relation_second_value_of_p(pv2))
    {
      int n_first_first = cell_compare(&c_first_1, &c_first_2);
      if (n_first_first == 0) return true;
      int n_first_second = cell_compare(&c_first_1, &c_second_2);
      if (n_first_second == 0) return true;
    }
  if (entity_all_locations_p(cell_entity(c_second_2))
      && ! cell_relation_second_value_of_p(pv1))
    {
      int n_first_first = cell_compare(&c_first_1, &c_first_2);
      if (n_first_first == 0) return true;
      int n_second_first = cell_compare(&c_second_1, &c_first_2);
      if (n_second_first == 0) return true;
    }

  return false;

}


/*
  @brief computes the union of two simple pointer_values list
  @param l_pv1 is the first list of pointer_values
  @param l_pv2 is the second list of pointer_values
  @return a new list of pointer values
 */
list simple_pvs_must_union(list l_pv1, list l_pv2)
{

  list l_res = cell_relations_generic_binary_op(
    l_pv1,
    l_pv2,
    pvs_union_combinable_p,
    simple_pv_must_union,
    cell_relation_to_list,
    cell_relation_to_list, simple_pvs_must_union);
  return l_res;
}

/*
  @brief computes the may union of two simple pointer_values list
  @param l_pv1 is the first list of pointer_values
  @param l_pv2 is the second list of pointer_values
  @return a new list of pointer values
 */
list simple_pvs_may_union(list l_pv1, list l_pv2)
{

  list l_res = cell_relations_generic_binary_op(
    l_pv1,
    l_pv2,
    pvs_union_combinable_p,
    simple_pv_may_union,
    cell_relation_to_may_list,
    cell_relation_to_may_list, simple_pvs_must_union);
  return l_res;
}

bool simple_pvs_syntactically_equal_p(list l_pv1, list l_pv2)
{
  bool result = true;

  if (gen_length(l_pv1) != gen_length(l_pv2))
    result = false;

  if (result)
    {
      /* first sort lists */
      gen_sort_list(l_pv1, (gen_cmp_func_t) pointer_value_compare);
      gen_sort_list(l_pv2, (gen_cmp_func_t) pointer_value_compare);

      /* then compare members syntactically */
      while(result && !ENDP(l_pv1))
	{
	  cell_relation pv1 = CELL_RELATION(CAR(l_pv1));
	  cell_relation pv2 = CELL_RELATION(CAR(l_pv2));

	  result = pv_cells_syntactically_equal_p(pv1, pv2);
	  POP(l_pv1);
	  POP(l_pv2);
	}
    }
  return result;
}
