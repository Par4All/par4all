 /* package sc*/

#include <stdio.h>
#include <string.h>
#include <malloc.h>
extern int fprintf();

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* 
 *
 * sort contrainte c, base b, 
 * relatively to sort_base, as defined by the switches.
 *
 * inner_first: innermost first
 * complex_first: the more complex the likely to be put earlier
 */

typedef struct
{
  Variable var;     /* the higher rank variable */
  Value    val;     /* the value of the coef of this variable */
  Value    cst;     /* the constant value for the system */
  int      rank;    /* the rank of the system */
  int      n_sort;  /* the number of indexes in the system (var included) */
  int      n_other; /* the number of parameters in the system (TCST included) */
  Pvecteur v;       /* the vector to deal with */
} sort_info, *Psort_info;


static Psort_info compute_sort_info(v, b, psi)
Pvecteur v;
Pbase b;
Psort_info psi;
{
    int 
	found = FALSE,
	i = 0,
	rank = 0,     /* rank of this variable, 0 means none */
	n_sort = 0,   /* number of sort variables (higher included) */
	n_other = 0;  /* number of other variables (TCST included) */
    Pvecteur 
	pv, pb;
    Variable 
	var, 
	higher_rank_var=VARIABLE_UNDEFINED; /* variable of higher rank */
    Value
	higher_rank_val=0;    /* value of this variable */

    for (pv=v;
	 pv!=NULL;
	 pv=pv->succ)
    {
	var = var_of(pv);

	for (found = FALSE, i=0, pb=b;
	     pb!=NULL && !found;
	     pb=pb->succ, i++)
	    if (var==var_of(pb)) found=TRUE;
	
	if (found)
	{
	    n_sort++;
	    if (i>rank)
		rank = i,
		higher_rank_var = var,
		higher_rank_val = val_of(pv);
	}
	else
	    n_other++;
    }
    
    psi->var = higher_rank_var,
    psi->val = higher_rank_val,
    psi->cst = vect_coeff(TCST, v),
    psi->rank = rank,
    psi->n_sort = n_sort,
    psi->n_other = n_other,
    psi->v = v;

    return(psi);
}

/*
 * a constraint is complex if: 
 *    |coef|!=1,
 *    the more sort variables,
 *    the more other variables,
 *    the higher/lowest the constant (min/max)
 * ??? could be improved/discussed...
 */
static float sort_info_to_value(psi, n_sort_vars, n_vars,
				inner_first, complex_first)
Psort_info psi;
int n_sort_vars, n_vars, inner_first, complex_first;
{
    float
	value_interval = (n_sort_vars+1)*(n_vars-n_sort_vars)+1,
	result = 0,
	order = 0,
	cst = 0;

    /* the sorting value is actually computed here.
     * ??? it could be improved by giving an higher complexity 
     * if the sorting variables are inner!
     */
    result = 2*value_interval*(inner_first ? psi->rank : n_sort_vars-psi->rank);

    if (psi->rank!=0 &&	abs(psi->val)==1) /* no integer division needed */
	result += complex_first ? 0 : value_interval;
    else
	result += complex_first ? value_interval : 0;

    order = psi->n_sort*(n_vars-n_sort_vars)+psi->n_other,
    result += (complex_first ? order : value_interval-order-1);

    cst = 1.0/(2+abs(psi->cst)),
    result += (psi->val>0 ? 
	       (complex_first ? 1-cst : cst) :
	       (complex_first ? cst : 1-cst));

    return(result);
}

Pcontrainte contrainte_sort_info(c, base, sort_base, 
				 inner_first, complex_first, info)
Pcontrainte c;
Pbase base, sort_base;
int inner_first, complex_first;
int info[][2];
{
    int
	i=0,
	nb_of_constraints = nb_elems_list(c),
	nb_of_variables = vect_size(base)+1,   /* TCST included */
	nb_of_sort_vars = vect_size(sort_base),
	*perm   = (int*) malloc(sizeof(int)*nb_of_constraints);    
    float
	*values = (float*) malloc(sizeof(float)*nb_of_constraints);
    Pcontrainte
	pc = CONTRAINTE_UNDEFINED,
	*tc = (Pcontrainte*) malloc(sizeof(Pcontrainte)*nb_of_constraints);
    sort_info si;
    
    if (nb_of_constraints==0)
    {
	free(tc), free(values), free(perm);
	return(c);
    }

    for (i=0; i<nb_of_sort_vars; i++)
	info[i][0]=0,
	info[i][1]=0;

    /*  each constraint is given its value for sorting
     */
    for (i=0, pc=c;
	 pc!=NULL;
	 i++, pc=pc->succ)
    {
	tc[i] = pc;
	values[i] = sort_info_to_value
	    (compute_sort_info(pc->vecteur, sort_base, &si),
	     nb_of_sort_vars,
	     nb_of_variables,
	     inner_first, 
	     complex_first);
	info[si.rank][(si.val>0) ? 1 : 0]++;
    }
    
    /*   now the table is sorted by decreasing order
     */
    merge_sort(nb_of_constraints, values, perm, TRUE);

    /*  the permutation given back by the sorting phase is used to
     *  generate again a list of constraints
     */
    for (i=0; i<nb_of_constraints-1; i++)
	tc[perm[i]]->succ = tc[perm[i+1]];

    tc[perm[nb_of_constraints-1]]->succ = NULL,
    c = tc[perm[0]];

    /*   clean!
     */
    free(tc), free(values), free(perm);

    return(c);
}

Pcontrainte contrainte_sort(c, base, sort_base, inner_first, complex_first)
Pcontrainte c;
Pbase base, sort_base;
int inner_first, complex_first;
{
    int 
	n = vect_size(sort_base)+1,
	(*info)[2];
    Pcontrainte r;

    info = malloc(sizeof(int)*2*n);

    r = contrainte_sort_info(c, base, sort_base, 
			     inner_first, complex_first, info);

    free(info);
    return(r);
}

/* Psysteme sc_build_triang_nredund(sc, ineg, b)
 * Psysteme sc;
 * Pcontrainte ineg;
 * Pbase b;
 *
 * this function builds a system from sc and ineg, by keeping
 * the contraints of ineg that are not redundant in sc.
 * the constraints are order according to the base b, to keep the
 * simplest if possible.
 * this function is designed for an efficient implementation of
 * the row_echelon algorithm.
 * the constraints are assumed to be on the same side of one variable!
 */
Psysteme sc_build_triang_nredund(sc, ineg, b)
Psysteme sc;
Pcontrainte ineg;
Pbase b;
{
    Pcontrainte
	c, ctmp, cprev,
	sorted = contrainte_sort(ineg, sc->base, b, TRUE, TRUE),
	killed = NULL;
    int
	i,
	n = nb_elems_list(ineg);

    /* rather low level, but I want to be sure that the constraints to 
     * be added are at the head of the list...
     */
    
    if (sorted==NULL) return(sc);	/* ??? or abort ? */

    for (ctmp=sorted; ctmp->succ!=NULL; ctmp=ctmp->succ);

    ctmp->succ=sc_inegalites(sc),
    sc_inegalites(sc)=sorted,
    sc_nbre_inegalites(sc)+=n;

    /* now the first constraints are those to be tested for redundancy
     */
    c=sorted, cprev=NULL, i=n; 
    while (n>1 && i>0)
    {
	contrainte_reverse(c);
	if (!sc_feasible_ofl(sc, TRUE))
	{
	    /* c must be removed */
	    n--, sc_nbre_inegalites(sc)--, i--, ctmp=c->succ;
	    c->succ=killed, killed=c;

	    if (cprev == NULL)
		sc_inegalites(sc)=ctmp;
	    else
		cprev->succ=ctmp;

	    c=ctmp;
	}   
	else { 
	    contrainte_reverse(c);
	    i--,
	    c=c->succ,
	    cprev=(cprev==NULL ? sc_inegalites(sc) : cprev->succ);
	}
    }

    /* clean
     */
    contraintes_free(killed);

    return(sc);
}

/* Psysteme sc_triang_elim_redond(Psysteme ps, Pbase base_index):
 * elimination des contraintes lineaires redondantes dans le systeme ps 
 * par test de faisabilite de contrainte inversee; cette fonction est
 * utilisee pour calculer des bornes de boucles, c'est pourquoi il peut
 * etre necessaire de garder des contraintes redondantes afin d'avoit
 * toujours au moins une borne inferieure et une borne superieure
 * pour chaque indice.
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme	    : Le systeme initial est modifie. Il est egal a NULL si 
 *		      le systeme initial est non faisable.
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme ps    : systeme lineaire 
 *
 *  Attention: pour chaque indice dans base_index, il doit rester au moins deux
 *             contraintes correspondantes (une positive et une negative).
 *             C'est la seule difference avec la fonction sc_elim_redond().
 *
 * Yi-Qing YANG
 *
 * Modifications:
 *  - add a normalization step for inequalities; if they are not normalized,
 *    i.e. if the GCD of the variable coefficients is not 1, the constant
 *    term of the inverted constraint should be carefully updated using
 *    the GCD?!? (Francois Irigoin, 30 October 1991)
 *  - the variables are sorted in order to get a deterministic result
 *    (FC 26 Sept 94)
 *  - the feasible overflow function is called, and only if the constraint
 *    is not the last one (performance bug of the previous version)
 *    (FC 27 Sept 94)
 *  - a warning is displayed if many inequalities are to be dealt with,
 *    instead of returning the system as is.
 *    (FC 28 Sept 94)
 *  - sc_normalize inserted, in place of many loop that were 
 *    doing nearly the same. (FC 29/09/94)
 */
/* extern char *entity_local_name(); */

Psysteme sc_sort_constraints(ps, base_index)
Psysteme ps;
Pbase base_index;
{
    int 
	n = vect_size(base_index)+1,
	(*info)[2];

    info = malloc(sizeof(int)*2*n);

    ps->inegalites = 
	contrainte_sort_info(ps->inegalites, 
			     ps->base, base_index,
			     TRUE, TRUE, info);

    free(info);
    return(ps);
}

Psysteme sc_triang_elim_redond(ps, base_index)
Psysteme ps;
Pbase base_index;
{
    Pcontrainte eq, eq1;
    int 
	level,
	n = vect_size(base_index)+1,
	(*info)[2];

    /*
       fprintf(stderr, "[sc_triang_elim_redond] input\n"); 
       sc_fprint(stderr, ps, entity_local_name);
       */

    ps = sc_normalize(ps);

    if (ps==NULL)
	return(NULL);

    if (ps->nb_ineq > NB_INEQ_MAX1)
	fprintf(stderr,
		"[sc_triang_elim_redond] warning, %d inequalities\n",
		ps->nb_ineq);

    /* ??? should be checked outside, before calling the function ? */    

    if (!sc_integer_feasibility_ofl_ctrl(ps, OFL_CTRL,TRUE))
    {	sc_rm(ps), ps=NULL;
	return(NULL);
    }

    info = malloc(sizeof(int)*2*n);

    ps->inegalites = 
	contrainte_sort_info(ps->inegalites, 
			     ps->base, base_index,
			     TRUE, TRUE, info);
    for (eq = ps->inegalites; eq != NULL; eq = eq1)
    {
	eq1 = eq->succ;
	level = level_contrainte(eq, base_index);

	/* only the variables that have more than one 
	 * constraints on a given size and that deal with 
	 * the variables of base_index are tested.
	 *
	 * an old comment suggested that keeping contraints on variables
	 * out of base_index would help find redundancy on the base_index
	 * contraints, but this should not be true anymore, since the
	 * variables are sorted... just help to deal with larger systems...
	 *
	 * FC 28/09/94
	 */
	if (level!=0 && info[abs(level)][level<0?0:1]>1)
	{
	    /* inversion du sens de l'inegalite par multiplication
	     * par -1 du coefficient de chaque variable
	     */
	    contrainte_reverse(eq);

	    /*
	       fprintf(stderr,"test redundant constraint:");
	       inegalite_fprint(stderr, eq, entity_local_name);
	       */

	    /* test de sc_faisabilite avec la nouvelle inegalite 
	     */
	    if (sc_integer_feasibility_ofl_ctrl(ps,OFL_CTRL, TRUE))
		/* restore the initial constraint */
		contrainte_reverse(eq);
	    else
	    {
		eq_set_vect_nul(eq),
		sc_elim_empty_constraints(ps,0),
		info[abs(level)][level<0?0:1]--;
	    }
	}
    }
    ps = sc_kill_db_eg(ps);

    /* fprintf(stderr, "[sc_triang_elim_redond] output\n");
       sc_fprint(stderr, ps, entity_local_name);  
       */

    free(info);
    return(ps);
}

/*   That is all
 */
