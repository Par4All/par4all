/*
 * Align Checker
 *
 * try to decide whether two references are aligned or not.
 * if not, gives back the alignement shift vector of the
 * second reference to the first one.
 *
 * this should use use-def chains, vectors and so on
 *
 * Fabien Coelho  August 93
 */


#include <stdio.h>
#include <values.h>

extern int fprintf();

#include "genC.h"

#include "ri.h"
#include "hpf.h"
#include "hpf_private.h"
#include "message.h"

#include "ri-util.h"
#include "misc.h"
#include "effects.h"
#include "hpfc.h"
#include "defines-local.h"
#include "access_description.h"

/*
 * bool align_check(r1, r2, plint, pkind, pindx)
 *
 * computes the shift vector that links the two references,
 * TRUE if every thing is ok, i.e. the vector is ok for the
 * distributed dimensions...
 * The vector considered here is just a list of integers.
 *
 * conditions:
 * - same template
 */
bool align_check(r1, r2, plvect, plkind)
reference r1, r2;
list *plvect, *plkind;
{
    int 
	procdim,
	i;
    entity
	e1 = reference_variable(r1),
	e2 = reference_variable(r2);
    align
	a1 = load_entity_align(e1),
	a2 = load_entity_align(e2);
    bool
	ok = TRUE;
    list
	li1 = reference_indices(r1),
	li2 = reference_indices(r2);

    *plvect = NULL;
    *plkind = NULL;

    debug(7, "align_check",
	  "with references to %s, %d indices and %s, %d indices\n",
	  entity_name(e1), gen_length(li1), entity_name(e2), gen_length(li2));

    if (align_template(a1)!=align_template(a2))
    {
	for (i=1 ; i<=NumberOfDimension(e2) ; i++)
	    if (ith_dim_distributed_p(e2, i, &procdim))
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
                *plvect = gen_nconc(*plvect, CONS(PVECTOR, VECTEUR_NUL, NIL));
	    }
	    else
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_star, NIL));
                *plvect = gen_nconc(*plvect, CONS(PVECTOR, VECTEUR_NUL, NIL));
	    }
	
	return(ok);
    }

    ifdebug(8)
	 {
	     fprintf(stderr, "[align_check] references are: ");
	     print_reference(r1);
	     fprintf(stderr, " and ");
	     print_reference(r2);
	     fprintf(stderr, "\n");
	 }

    for (i=1 ; i<=NumberOfDimension(e2) ; i++)
    {
	expression
	    indice1,
	    indice2; 
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

	debug(8, "align_check",
	      "considering dimension %d of %s\n",
	      i, entity_name(e2));

	indice2 = nth_expression(i-1, li2);

	baffin2 = affine_expression_of_loop_index_p(indice2, &index2, &shft2, &affr2);
	bshift2 = shift_expression_of_loop_index_p(indice2, &index2, &shft2);
	bconst2 = hpfc_integer_constant_expression_p(indice2, &shft2);
	if (!(baffin2 || bshift2 || bconst2))
	    blcnst2 = local_integer_constant_expression(indice2);
	
	if (ith_dim_distributed_p(e2, i, &procdim))
	{
	    alignment
                a2 = FindArrayDimAlignmentOfArray(e2, i),
                a1 = FindTemplateDimAlignmentOfArray(e1, alignment_templatedim(a2));
	    int
                dim1  = ((a1!=alignment_undefined)?(alignment_arraydim(a1)):(-1)),
		rate1, rate2, 
		cnst1, cnst2;

            rate2   = HpfcExpressionToInt(alignment_rate(a2));
	    cnst2   = HpfcExpressionToInt(alignment_constant(a2));

	    if ((dim1<0) || /* replication, say it is not aligned */
		((dim1==0) && (!bconst2))) /* replicated reference...*/
            {
		debug(8, "align_check",
		      "%s dim %d not aligned aff %d shift %d const %d\n",
		      entity_name(e2), i, baffin2, bshift2, bconst2);

                *plkind = gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
                *plvect = gen_nconc(*plvect, CONS(PVECTOR, VECTEUR_NUL, NIL));
            }
	    else
	    if ((dim1==0) && (bconst2))
	    {
		/*
		 * a(i) -> t(i,1), b(i,j)->t(i,j), should detect 
		 * do i a(i) = b(i,1) as aligned...
		 *
		 * ??? this could be managed later in this function...
		 */

		int
		    t1 = HpfcExpressionToInt(alignment_constant(a1)),
		    t2 = rate2*shft2+cnst2;
		
		/*
		 * should be ok, even if a delta is induced, if it stays
		 * on the same processor along this dimension. The case
		 * should also be managed downward, when playing with 
		 * local updates. 
		 * ??? maybe there is also a consequence on the 
		 * send/receive symetry, in order to put the message where it
		 * should be...
		 */

		/*
		 * it is checked later whether the data are on the same 
		 * processor or not.
		 */

		*plkind = gen_nconc(*plkind, CONS(INT, aligned_constant, NIL));
		*plvect = gen_nconc(*plvect, CONS(PVECTOR, 
						  vect_add(vect_new(TEMPLATEV, t2),
							   vect_new(DELTAV, t2-t1)),
						  NIL));
	    }
	    else
	    {
		indice1 = nth_expression(dim1-1, li1);

		rate1   = HpfcExpressionToInt(alignment_rate(a1));
		cnst1   = HpfcExpressionToInt(alignment_constant(a1));

		baffin1 = affine_expression_of_loop_index_p
		    (indice1, &index1, &shft1, &affr1);
		bshift1 = shift_expression_of_loop_index_p
		    (indice1, &index1, &shft1);
		bconst1 = hpfc_integer_constant_expression_p
		    (indice1, &shft1);          

		/*
		 * now we have everything to check whether it is aligned or not,
		 * and to compute the shift necessary to the overlap analysis.
		 */

		if (!((baffin1 || bshift1 || bconst1) &&
		      (baffin2 || bshift2 || bconst2)))
		{
		    *plkind = gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
		    *plvect = gen_nconc(*plvect, CONS(PVECTOR, VECTEUR_NUL, NIL));
		}
		else /* something is true! */
		if (baffin1 || baffin2) /* just check whether aligned or not */
		{
		    int
			r1 = ((bconst1)?(0):(rate1*affr1)),
			r2 = ((bconst2)?(0):(rate2*affr2)),
			c1 = rate1*shft1+cnst1,
			c2 = rate2*shft2+cnst2;

		    if ((index1!=index2) ||/* even in the constant case ! not aligned */
			(r1!=r2) || (c1!=c2))
		    {
			*plkind = 
			    gen_nconc(*plkind, CONS(INT, not_aligned, NIL));
			*plvect = 
			    gen_nconc(*plvect, CONS(PVECTOR, VECTEUR_NUL, NIL));
		    }    
		    else /* aligned... ??? 
			  * bug if not 1: later on, because decl shift
			  */
		    {
			*plkind = 
			    gen_nconc(*plkind, CONS(INT, aligned_star, NIL));
			*plvect = 
			    gen_nconc(*plvect, CONS(PVECTOR, VECTEUR_NUL, NIL));
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
				  CONS(PVECTOR, 
				       vect_make(VECTEUR_NUL,
						 DELTAV, d,
						 TEMPLATEV, t2,
						 TCST, shft2),
				       NIL));
		}
		else /* say not aligned... */
		if ((rate1!=1) || (rate2!=1) || (index1!=index2)) 
		{
		    *plkind = gen_nconc(*plkind, 
					CONS(INT, not_aligned, NIL));
		    *plvect = gen_nconc(*plvect, 
					CONS(PVECTOR, VECTEUR_NUL, NIL));
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
		    *plvect = gen_nconc(*plvect, 
					CONS(PVECTOR,
					     vect_make(VECTEUR_NUL,
						       TSHIFTV, shift,
						       (Variable) index1, 1,
						       TCST, shft2),
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
			gen_nconc(*plvect, CONS(PVECTOR, VECTEUR_NUL, NIL));
		}			  	 
	    }
	}
	else /* ith dimension of e2 is not distributed */
	{
	    if (!(baffin2 || bshift2 || bconst2 || blcnst2))
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_star, NIL));
		*plvect = gen_nconc(*plvect, CONS(PVECTOR, VECTEUR_NUL, NIL));
	    }
	    else
	    if (baffin2) /* may be used to generate RSDs */
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_affine, NIL));
		*plvect = 
		    gen_nconc(*plvect, 
			      CONS(PVECTOR, 
				   vect_make(VECTEUR_NUL,
					     (Variable) index2, affr2,
					     TCST, shft2),
				   NIL));
	    }
	    else
	    if (bshift2)
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_shift, NIL));
		*plvect = 
		    gen_nconc(*plvect, 
			      CONS(PVECTOR, 
				   vect_make(VECTEUR_NUL,
					     (Variable) index2, 1,
					     TCST, shft2),
				   NIL));
	    }
	    else
	    if (bconst2)
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_constant, NIL));
		*plvect = gen_nconc(*plvect, CONS(PVECTOR, 
						  vect_new(TCST, shft2),
						  NIL));
	    }
	    /* else the local constant should be detected, for onde24... */
	    else
	    if (blcnst2)
	    {
		*plkind = gen_nconc(*plkind, CONS(INT, local_form_cst, NIL));
		*plvect = 
		    gen_nconc(*plvect, 
			      CONS(PVECTOR, 
				   normalized_linear(expression_normalized(indice2)),
				   NIL));
	    }
	}
    }

    return(ok);
}


/*
 * bool shift_expression_of_loop_index_p(e, pe, pi)
 *
 * true if the expression is shift function
 * of a loop nest index.
 */
bool shift_expression_of_loop_index_p(e, pe, pi)
expression e;
entity *pe;
int *pi;
{
    normalized
	n = expression_normalized(e);

    pips_assert("shift_expression_of_loop_index_p",
		(!normalized_undefined_p(n)));

    switch (normalized_tag(n))
    {
    case is_normalized_complex:
	return(FALSE);
	break;
    case is_normalized_linear:
    {
	Pvecteur
	    v = (Pvecteur) normalized_linear(n),
	    vp = vect_del_var(v, TCST);
	int
	    s = vect_size(vp);
	bool
	    result;

	if (s!=1) return(FALSE);

	result = ((entity_loop_index_p((entity)(vp->var)) && ((vp->val)==1)));

	vect_rm(vp);
	if (result) 
	{
	    *pe = (entity) (vp->var);
	    *pi = vect_coeff(TCST, v);
	}
	return(result);
	break;
    }
    default:
	pips_error("shift_expression_of_loop_index_p",
		   "unexpected normalized tag\n");
	break;
    }
    
    return(FALSE); /* just to avoid a gcc warning */
}

/*
 * bool affine_expression_of_loop_index_p(e, pe, pi1, pi2)
 *
 * true if the expression is an affine function
 * of a loop nest index.
 */
bool affine_expression_of_loop_index_p(e, pe, pi1, pi2)
expression e;
entity *pe;
int *pi1, *pi2;
{
    normalized
	n = expression_normalized(e);

    ifdebug(6)
	{
	    fprintf(stderr, "[affine_expression_of_loop_index_p]\nexpression:\n");
	    print_expression(e);
	}
	    
    pips_assert("affine_expression_of_loop_index_p",
		(!normalized_undefined_p(n)));

    switch (normalized_tag(n))
    {
    case is_normalized_complex:
	return(FALSE);
	break;
    case is_normalized_linear:
    {
	Pvecteur
	    v = (Pvecteur) normalized_linear(n),
	    vp = vect_del_var(v, TCST);
	int
	    s = vect_size(vp);
	bool
	    result;

	if (s!=1) return(FALSE);

	result = (entity_loop_index_p((entity)(vp->var)) && ((vp->val)!=1));

	vect_rm(vp);
	if (result) 
	{
	    *pe = (entity) (vp->var);
	    *pi1 = vect_coeff(TCST, v);
	    *pi2 = (int) (vp->val);
	}
	return(result);
	break;
    }
    default:
	pips_error("affine_expression_of_loop_index_p",
		   "unexpected normalized tag\n");
	break;
    }
    
    return(FALSE); /* just to avoid a gcc warning */
}

/*
 * bool hpfc_integer_constant_expression_p(e, pi)
 *
 * 
 * 
 */
bool hpfc_integer_constant_expression_p(e, pi)
expression e;
int *pi;
{
    normalized 
	n = expression_normalized(e);

    if (normalized_undefined_p(n)) 
    {
	n = NormalizeExpression(e);
	expression_normalized(e) = n;
    }

    switch (normalized_tag(n))
    {
    case is_normalized_complex:
	return(FALSE);
	break;
    case is_normalized_linear:
    {
	Pvecteur
	    v = (Pvecteur) normalized_linear(n),
	    vp = vect_del_var(v, TCST);
	int
	    s = vect_size(vp);
	bool
	    result = (s==0);

	vect_rm(vp);

	if (result)
	    *pi = vect_coeff(TCST, v);
	
	return(result);
	break;
    }
    default:
	pips_error("hpfc_integer_constant_expression_p",
		   "unexpected normalized tag\n");
	break;
    }
    
    return(FALSE); /* just to avoid a gcc warning */
}

/*
 * expression nth_expression(n, l)
 *
 */
expression nth_expression(n, l)
int n;
list l;
{
    return((n==0)?(EXPRESSION(CAR(l))):(nth_expression(n-1, CDR(l))));
}

/*
 * bool local_integer_constant_expression(e)
 * 
 * true is the expression is locally constant, that is in the whole loop nest,
 * the reference is not written.
 */
bool local_integer_constant_expression(e)
expression e;
{
    bool
	result = FALSE;
    syntax
	s = expression_syntax(e);

    if ((syntax_reference_p(s)) &&
	(normalized_linear_p(expression_normalized(e))))
    {
	entity
	    ent = reference_variable(syntax_reference(s));

	result =
	    (!effects_write_entity_p(get_hpfc_current_statement_effects(), ent));

	debug(7, "local_integer_constant_expression",
	      "looking for effects on %s reference, result %d\n",
	      entity_name(ent), result);
    }
    
    return(result);
}

/*
 * hpfc current statement management
 * 
 * ??? could be put somewhere else..
 */

static statement 
    hpfc_current_statement = statement_undefined;

static list
    hpfc_current_statement_effects = NIL;

void set_hpfc_current_statement(stat)
statement stat;
{
    hpfc_current_statement = stat;
    hpfc_current_statement_effects = statement_to_effects(stat);
}

statement get_hpfc_current_statement()
{
    return(hpfc_current_statement);
}

list get_hpfc_current_statement_effects()
{
    return(hpfc_current_statement_effects);
}

void reset_hpfc_current_statement()
{
    hpfc_current_statement = statement_undefined;
    gen_free_list(hpfc_current_statement_effects);
    hpfc_current_statement_effects = NIL;
}
