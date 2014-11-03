/*

  $Id$

  Copyright 1989-2014 MINES ParisTech
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
      FOREACH(CELL_RELATION, old_pv, l_old_values) {
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
     * l_in pointer values second cells which must be expressed in terms of
     * unchanged paths.
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
              simple_cells_inclusion_p(cell_in_2, descriptor_undefined,
                  cell_kill, descriptor_undefined, &inclusion_test_exact_p))
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
            simple_cells_inclusion_p(cell_in_1, descriptor_undefined,
                cell_kill, descriptor_undefined,
                &inclusion_test_exact_p) )
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
         * and we can use it */
        if (exact_old_pv != cell_relation_undefined)
        {
          cell_relation new_pv = simple_pv_translate(pv_in, first_p, exact_old_pv);
          pips_debug_pv(3, "translated to:", new_pv);
          l_out = CONS(CELL_RELATION, new_pv, l_out);
        }
        else /* generate a new pv for each element of old_values */
        {
          FOREACH(CELL_RELATION, old_pv, l_old_values) {
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
     * We only have to keep those which are not completely killed by kill_eff, and
     * set their approximation to may (this is also true for exact_old_pv)
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
       * intersection with cell_kill */
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
        bool inclusion_p = simple_cells_inclusion_p(first_cell_old, descriptor_undefined,
            cell_kill, descriptor_undefined,
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
          bool inclusion_p = simple_cells_inclusion_p(second_cell_old, descriptor_undefined,
              cell_kill, descriptor_undefined,
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
             * computing the difference between the pv_old
             * first cell and cell_kill. I don't know if it
             * would be really useful.  So let us avoid too
             * complex things for the moment.
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
      else if (simple_cells_inclusion_p(first_cell_old, descriptor_undefined,
          cell_kill, descriptor_undefined,
          &exact_inclusion_p)
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
         * computing the difference between the pv_old
         * first cell and cell_kill. I don't know if it
         * would be really useful.  So let us avoid too
         * complex things for the moment.
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
  return generic_effect_find_equivalent_simple_pointer_values(eff, l_in, exact_aliased_pv, l_in_remnants,
							      simple_cells_intersection_p,
							      simple_cells_inclusion_p,
							      simple_cell_to_simple_cell_conversion);
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
  bool exact_p;
  return generic_effect_find_aliases_with_simple_pointer_values(eff, l_pv, &exact_p, transformer_undefined,
								simple_cell_preceding_p,
								simple_cell_with_address_of_cell_translation,
								simple_cell_with_value_of_cell_translation,
								simple_cells_intersection_p,
								simple_cells_inclusion_p,
								simple_cell_to_simple_cell_conversion);
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
