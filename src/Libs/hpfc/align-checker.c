/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
/* Align Checker
 *
 * try to decide whether two references are aligned or not.
 * if not, gives back the alignement shift vector of the
 * second reference to the first one.
 *
 * this should use use-def chains, vectors and so on
 *
 * Fabien Coelho  August 93
 */

#include "defines-local.h"
#include "access_description.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

GENERIC_STATIC_OBJECT(extern, hpfc_current_statement, statement)

#define REGION_TYPE EFFECT_TYPE

static bool 
write_on_entity_p(
    entity e)
{
    MAP(REGION, r,
	if (region_entity(r)==e && region_write_p(r)) return true,
	load_statement_local_regions(get_hpfc_current_statement()));

    return false;
}

/* true is the expression is locally constant, that is in the whole loop nest,
 * the reference is not written. ??? not very portable thru pips...
 */
bool 
local_integer_constant_expression(
    expression e)
{
    syntax s = expression_syntax(e);

    if ((syntax_reference_p(s)) &&
	(normalized_linear_p(expression_normalized(e))))
    {
	entity ent = reference_variable(syntax_reference(s));
	if (write_on_entity_p(ent)) return false;
    }
    
    return true;
}

/* true if the expression is shift function
 * of a loop nest index.
 */
static bool shift_expression_of_loop_index_p(e, pe, pi)
expression e;
entity *pe;
int *pi;
{
    normalized n = expression_normalized(e);

    pips_assert("normalized", !normalized_undefined_p(n));

    switch (normalized_tag(n))
    {
    case is_normalized_complex:
	return(false);
	break;
    case is_normalized_linear:
    {
	Pvecteur v = normalized_linear(n), vp = vect_del_var(v, TCST);
	int s = vect_size(vp);
	bool result;

	if (s!=1) return false;

	result = entity_loop_index_p((entity)var_of(vp)) && 
	    value_one_p(val_of(vp));

	if (result) 
	{
	    Value vi = vect_coeff(TCST, v);
	    *pe = (entity) var_of(vp);
	    *pi = VALUE_TO_INT(vi);
	}

	vect_rm(vp);

	return result;
	break;
    }
    default:
	pips_internal_error("unexpected normalized tag");
	break;
    }
    
    return false; /* just to avoid a gcc warning */
}

/* true if the expression is an affine function
 * of a loop nest index.
 */
static bool affine_expression_of_loop_index_p(e, pe, pi1, pi2)
expression e;
entity *pe;
int *pi1, *pi2;
{
    normalized n = expression_normalized(e);

    ifdebug(6)
    {
	fprintf(stderr, "[affine_expression_of_loop_index_p]\nexpression:\n");
	print_expression(e);
    }
	    
    pips_assert("normalized", !normalized_undefined_p(n));

    switch (normalized_tag(n))
    {
    case is_normalized_complex:
	return(false);
	break;
    case is_normalized_linear:
    {
	Pvecteur v = (Pvecteur) normalized_linear(n),
	    vp = vect_del_var(v, TCST);
	int s = vect_size(vp);
	bool result;

	if (s!=1) return(false);

	result = (entity_loop_index_p((entity)var_of(vp)) && 
		  value_notone_p(val_of(vp)));

	if (result) 
	{
	    Value v1 = vect_coeff(TCST, v);
	    *pe = (entity) var_of(vp);
	    *pi1 = VALUE_TO_INT(v1);
	    *pi2 = VALUE_TO_INT(val_of(vp));
	}

	vect_rm(vp);

	return result;
	break;
    }
    default:
	pips_internal_error("unexpected normalized tag");
	break;
    }
    
    return(false); /* just to avoid a gcc warning */
}

/* computes the shift vector that links the two references,
 * true if every thing is ok, i.e. the vector is ok for the
 * distributed dimensions...
 * The vector considered here is just a list of integers.
 * conditions: same template
 */
bool 
align_check(
    reference r1,
    reference r2,
    list *plvect,
    list *plkind)
{
    int procdim, i, ne2dim;
    entity e1, e2;
    align a1, a2;
    bool ok = true;
    list li1, li2;

    e1 = reference_variable(r1),
    e2 = reference_variable(r2);
    a1 = load_hpf_alignment(e1),
    a2 = load_hpf_alignment(e2);
    li1 = reference_indices(r1),
    li2 = reference_indices(r2);
    *plvect = NIL;
    *plkind = NIL;

    pips_debug(7, "with references to %s[DIM=%zd] and %s[DIM=%zd]\n",
	  entity_name(e1), gen_length(li1), entity_name(e2), gen_length(li2));

    ne2dim = NumberOfDimension(e2);
    
    if (!conformant_templates_p(align_template(a1),align_template(a2)))
    {
	pips_debug(5, "different templates\n");

	for (i=1 ; i<=ne2dim ; i++)
	    if (ith_dim_distributed_p(e2, i, &procdim))
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
                *plvect = gen_nconc(*plvect, CONS(PVECTOR,
						  (VECTOR) VECTEUR_NUL, NIL));
	    }
	    else
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_star, NIL));
                *plvect = gen_nconc(*plvect, CONS(PVECTOR,
						  (VECTOR) VECTEUR_NUL, NIL));
	    }
	
	return ok;
    }

    ifdebug(8)
	 {
	     fprintf(stderr, "[align_check] references are: ");
	     print_reference(r1);
	     fprintf(stderr, " and ");
	     print_reference(r2);
	     fprintf(stderr, "\n");
	 }

    for (i=1 ; i<=ne2dim ; i++)
    {
	expression indice1, indice2; 
	int 
	    affr1 = 1, /* default value, used in the shift case ! */
	    affr2 = 1,
	    shft1, shft2;
	entity
	    index1 = NULL, /* default value, used in the const case ! */
	    index2 = NULL;
	bool
	    baffin1, baffin2,
	    bshift1, bshift2,
	    bconst1, bconst2,
	    blcnst2=0;

	pips_debug(8, "considering dimension %d of %s\n", i, entity_name(e2));

	indice2 = EXPRESSION(gen_nth(i-1, li2));

	baffin2 = affine_expression_of_loop_index_p(indice2, 
						    &index2, &shft2, &affr2);
	bshift2 = shift_expression_of_loop_index_p(indice2, &index2, &shft2);
	bconst2 = hpfc_integer_constant_expression_p(indice2, &shft2);
	if (!(baffin2 || bshift2 || bconst2))
	    blcnst2 = local_integer_constant_expression(indice2);
	
	if (ith_dim_distributed_p(e2, i, &procdim))
	{
	    alignment
                a2 = FindArrayDimAlignmentOfArray(e2, i),
                a1 = FindTemplateDimAlignmentOfArray(e1,
				 alignment_templatedim(a2));
	    int
                dim1=((a1!=alignment_undefined)?(alignment_arraydim(a1)):(-1)),
		rate1, rate2, 
		cnst1, cnst2;

            rate2   = HpfcExpressionToInt(alignment_rate(a2));
	    cnst2   = HpfcExpressionToInt(alignment_constant(a2));

	    if ((dim1<0) || /* replication, say it is not aligned */
		((dim1==0) && (!bconst2))) /* replicated reference...*/
            {
		pips_debug(8, "%s[DIM=%d] not aligned aff %d shi %d cst %d\n",
			   entity_name(e2), i, baffin2, bshift2, bconst2);

                *plkind = gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
                *plvect = gen_nconc(*plvect, CONS(PVECTOR, 
						  (VECTOR) VECTEUR_NUL, NIL));
            }
	    else
	    if ((dim1==0) && (bconst2))
	    {
		/* a(i) -> t(i,1), b(i,j)->t(i,j), should detect 
		 * do i a(i) = b(i,1) as aligned...
		 *
		 * ??? this could be managed later in this function...
		 */

		int t1 = HpfcExpressionToInt(alignment_constant(a1)),
		    t2 = rate2*shft2+cnst2;
		
		/* should be ok, even if a delta is induced, if it stays
		 * on the same processor along this dimension. The case
		 * should also be managed downward, when playing with 
		 * local updates. 
		 * ??? maybe there is also a consequence on the 
		 * send/receive symetry, in order to put the message where it
		 * should be...
		 */

		/* it is checked later whether the data are on the same 
		 * processor or not.
		 */

		*plkind = gen_nconc(*plkind, CONS(INT, aligned_constant, NIL));
		*plvect = gen_nconc(*plvect,
				    CONS(PVECTOR, (VECTOR)
		   vect_add(vect_new(TEMPLATEV, int_to_value(t2)),
			    vect_new(DELTAV, int_to_value(t2-t1))),
					 NIL));
	    }
	    else
	    {
		indice1 = EXPRESSION(gen_nth(dim1-1, li1));

		rate1   = HpfcExpressionToInt(alignment_rate(a1));
		cnst1   = HpfcExpressionToInt(alignment_constant(a1));

		baffin1 = affine_expression_of_loop_index_p
		    (indice1, &index1, &shft1, &affr1);
		bshift1 = shift_expression_of_loop_index_p
		    (indice1, &index1, &shft1);
		bconst1 = hpfc_integer_constant_expression_p
		    (indice1, &shft1);          

		/* now we have everything to check whether it is aligned...
		 * and to compute the shift necessary to the overlap analysis.
		 */

		if (!((baffin1 || bshift1 || bconst1) &&
		      (baffin2 || bshift2 || bconst2)))
		{
		    *plkind = gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
		    *plvect = gen_nconc(*plvect, CONS(PVECTOR,(VECTOR) 
						      VECTEUR_NUL, NIL));
		}
		else /* something is true! */
		if (baffin1 || baffin2) /* just check whether aligned or not */
		{
		    int
			r1 = ((bconst1)?(0):(rate1*affr1)),
			r2 = ((bconst2)?(0):(rate2*affr2)),
			c1 = rate1*shft1+cnst1,
			c2 = rate2*shft2+cnst2;

		    /* even in the constant case ! not aligned */
		    if ((index1!=index2) ||
			(r1!=r2) || (c1!=c2))
		    {
			*plkind = 
			    gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
			*plvect = 
			    gen_nconc(*plvect, CONS(PVECTOR, 
					  (VECTOR) VECTEUR_NUL, NIL));
		    }    
		    else /* aligned... ??? 
			  * bug if not 1: later on, because decl shift
			  */
		    {
			*plkind = 
			    gen_nconc(*plkind, CONS(INT, aligned_star, NIL));
			*plvect = 
			    gen_nconc(*plvect, CONS(PVECTOR,
					 (VECTOR) VECTEUR_NUL, NIL));
		    }
		}
		else /* 4 cases study with bconst and bshift, plus the rates */
		if (bconst1 && bconst2)
		{
		    int
			t1 = (rate1*shft1+cnst1),
			t2 = (rate2*shft2+cnst2),
			d  = (t2-t1);

		    /* what should be the relevent information(s) ??? 
		     * not only d, but shift2 and t2 may be usefull either
		     * ??? what about rates!=1 ? decl shift problem added.
		     */

		    *plkind = 
			gen_nconc(*plkind, CONS(INT, aligned_constant, NIL));
		    *plvect = 
			gen_nconc(*plvect, 
				  CONS(PVECTOR, (VECTOR)
				       vect_make(VECTEUR_NUL,
						 DELTAV, int_to_value(d),
						 TEMPLATEV, int_to_value(t2),
						 TCST, int_to_value(shft2)),
				       NIL));
		}
		else /* say not aligned... */
		if ((rate1!=1) || (rate2!=1) || (index1!=index2)) 
		{
		    *plkind = gen_nconc(*plkind, 
					CONS(INT, not_aligned, NIL));
		    *plvect = gen_nconc(*plvect,
				CONS(PVECTOR, (VECTOR) VECTEUR_NUL, NIL));
		}
		else
		if (bshift1 && bshift2)
		{
		    /* ??? decl shift problem may occur here as well */
		    int
			tc1   = (shft1+cnst1),
			tc2   = (shft2+cnst2),
			shift = (tc2-tc1);

		    *plkind = gen_nconc(*plkind, CONS(INT, aligned_shift, NIL));
		    *plvect = gen_nconc
			(*plvect, 
			 CONS(PVECTOR, (VECTOR)
			      vect_make(VECTEUR_NUL,
					TSHIFTV, int_to_value(shift),
					(Variable) index1, VALUE_ONE,
					TCST, int_to_value(shft2)),
			      NIL));
		}
		else /* ??? I should do something with blcnst2? */
		{
		    /* well, say not aligned, but one to many or many to one
		     * communications may be used...
		     */
		    *plkind = 
			gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
		    *plvect = 
			gen_nconc(*plvect,
				  CONS(PVECTOR, (VECTOR)  VECTEUR_NUL, NIL));
		}			  	 
	    }
	}
	else /* ith dimension of e2 is not distributed */
	{
	    if (!(baffin2 || bshift2 || bconst2 || blcnst2))
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_star, NIL));
		*plvect = gen_nconc(*plvect,
				    CONS(PVECTOR, (VECTOR) VECTEUR_NUL, NIL));
	    }
	    else
	    if (baffin2) /* may be used to generate RSDs */
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_affine, NIL));
		*plvect = 
		    gen_nconc(*plvect, 
		       CONS(PVECTOR, (VECTOR)
			    vect_make(VECTEUR_NUL,
				      (Variable) index2, int_to_value(affr2),
				      TCST, int_to_value(shft2)),
			    NIL));
	    }
	    else
	    if (bshift2)
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_shift, NIL));
		*plvect = 
		    gen_nconc(*plvect, 
			      CONS(PVECTOR, (VECTOR)
				   vect_make(VECTEUR_NUL,
					     (Variable) index2, VALUE_ONE,
					     TCST, int_to_value(shft2)),
				   NIL));
	    }
	    else
	    if (bconst2)
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_constant, NIL));
		*plvect = gen_nconc(*plvect, CONS(PVECTOR, (VECTOR)
			     vect_new(TCST, int_to_value(shft2)),
						  NIL));
	    }
	    /* else the local constant should be detected, for onde24... */
	    else
	    if (blcnst2)
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_form_cst, NIL));
		*plvect = 
		    gen_nconc(*plvect, 
			      CONS(PVECTOR,  (VECTOR)
		           normalized_linear(expression_normalized(indice2)),
				   NIL));
	    }
	}
    }

    ifdebug(8)
    {
	list l;
	fprintf(stderr, "[align_check] returning: ");
	for (l=*plkind; l; POP(l))
	    fprintf(stderr, "%"PRIdPTR", ", INT(CAR(l)));
	fprintf(stderr, "\n");
    }

    return ok;
}

bool hpfc_integer_constant_expression_p(e, pi)
expression e;
int *pi;
{
    normalized n = expression_normalized(e);

    if (normalized_undefined_p(n)) 
    {
	n = NORMALIZE_EXPRESSION(e);
    }

    switch (normalized_tag(n))
    {
    case is_normalized_complex:
	return(false);
	break;
    case is_normalized_linear:
    {
	Pvecteur  v = (Pvecteur) normalized_linear(n),
	    vp = vect_del_var(v, TCST);
	int s = vect_size(vp);
	bool result = (s==0);

	vect_rm(vp);

	if (result)
	{
	    Value vi = vect_coeff(TCST, v);
	    *pi = VALUE_TO_INT(vi);
	}
	
	return(result);
	break;
    }
    default:
	pips_internal_error("unexpected normalized tag");
	break;
    }
    
    return(false); /* just to avoid a gcc warning */
}
