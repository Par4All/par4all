 /* package sc
  *
  * SCCS stuff:
  * $RCSfile: sc_triang_elim_redond.c,v $ ($Date: 1995/01/24 11:20:12 $, )
  * version $Revision$
  * got on %D%, %T%
  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include "assert.h"

extern int fprintf();

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"


/*   COMPARISON of CONSTRAINTS
 *
 *
 *
 *
 *
 */

static Pbase 
  rbase_for_compare  = BASE_NULLE, 
  others_for_compare = BASE_NULLE;

static void set_static_bases_for_compare(base, sort_base)
Pbase base, sort_base;
{
    assert(BASE_NULLE_P(rbase_for_compare) &&
	   BASE_NULLE_P(others_for_compare));

    /* the base is reversed! inner indexes first!!
     */
    rbase_for_compare  = base_normalize(base_reversal(sort_base));
    others_for_compare =  base_normalize(base_difference(base, sort_base));
}

static void reset_static_bases_for_compare()
{
    base_rm(rbase_for_compare),  rbase_for_compare=BASE_NULLE;
    base_rm(others_for_compare), others_for_compare=BASE_NULLE;
}

#define ADD_COST (1)
#define MUL_COST (1)
#define AFF_COST (1)
static int cost_of_constant_operations(v) 
Pvecteur v;
{
    int cost = AFF_COST;
    Pbase b;
    Value val;
    
    /*   constant
     */
    if (vect_coeff(TCST, v)!=0) cost += ADD_COST;

    /*   other variables
     */
    for (b=others_for_compare; b!=(Pvecteur)NULL; b=b->succ)
    {
	val = abs(vect_coeff(var_of(b), v));

	if (val!=0) cost += val==1 ? ADD_COST : (MUL_COST+ADD_COST) ;
    }

    return(cost);
}

/* for qsort, returns "is simpler than"
 *
 *    - : v1 < v2
 *    0 : v1==v2
 *    + : v1 > v2
 *
 * with the following criterion 
 *
 *  1/ ranks
 *  2/ coef of comparable ranks, +-1 or simpler...
 *  3/ 
 *
 * rational: 
 *  - loop sizes are assumed to be infinite
 *  - invariant code motion
 *  - induction variables recognized
 */
#define RESULT(e)\
{ \
      int result = (e);\
      fprintf(stderr, "[compare_the_constraints]\n");\
      vect_debug(v1); vect_debug(v2);\
      fprintf(stderr, "%s\n", result==0 ? "=" : result>0 ? ">" : "<"); \
      return(result);\
}

static int compare_the_constraints(pc1, pc2)
Pcontrainte *pc1, *pc2;
{
    Pvecteur
	v1 = (*pc1)->vecteur,
	v2 = (*pc2)->vecteur;
    int
	null_1, null_2, i, irank=0, cost_1, cost_2;
    Value 
	val_1, val_2, val;
    Pbase
	b;

    /*  for each inner first indexes,
     *  the first constraint with a null coeff while the other one is non null
     *  is the simplest.
     */
    for (i=1, b=rbase_for_compare; !BASE_NULLE_P(b); i++, b=b->succ)
    {
	null_1 = vect_coeff(var_of(b), v1)==0,
	null_2 = vect_coeff(var_of(b), v2)==0;

	if (null_1 ^ null_2) return(null_1-null_2);

	if (irank==0 && (!null_1||!null_2)) irank=i;      /* set the irank */
    }

    /*   no difference on the ranks, have a look at the idiv
     *   the greater the worse (should not be that)
     */

    b = search_i_element(rbase_for_compare, irank);

    val_1 = vect_coeff(var_of(b), v1);
    val_2 = vect_coeff(var_of(b), v2);

    if (val_1!=val_2) 
	return(val_1<0 && val_2<0 ? val_1-val_2 : val_2-val_1);

    val=val_1;
    
    /*   constant operations
     */
    cost_1 = cost_of_constant_operations(v1),
    cost_2 = cost_of_constant_operations(v2);

    if (cost_1!=cost_2) return(cost_2-cost_1);

    /*   compare the coefficients for the base
     */
    for (b=b->succ; !BASE_NULLE_P(b); b=b->succ)
    {
	val_1 = vect_coeff(var_of(b), v1),
	val_2 = vect_coeff(var_of(b), v2);
	
	if (val_1!=val_2) 
	    return(val_1<0 && val_2<0 ? val_1-val_2 : val_2-val_1);
    }

    /*   do it for the for the parameters
     */
    for (b=others_for_compare; !BASE_NULLE_P(b); b=b->succ)
    {
	val_1 = vect_coeff(var_of(b), v1),
	val_2 = vect_coeff(var_of(b), v2);
	
	if (val_1!=val_2) return(val_2-val_1);
    }
    
    /*   at last the constant
     */
    val_1 = vect_coeff(TCST, v1),
    val_2 = vect_coeff(TCST, v2);

    return(val>0 ? val_2-val_1 : val_1-val_2);
}

/* returns the highest rank pvector of v in b, of rank *prank
 */
Pvecteur highest_rank_pvector(v, b, prank)
Pvecteur v;
Pbase b;
int *prank;
{
    Pbase pb;
    Pvecteur pv, result=(Pvecteur) NULL;
    Variable var;
    int rank;

    for (*prank=-1, rank=1, pb=b;
	 !BASE_NULLE_P(pb);
	 pb=pb->succ, rank++)
    {
	var = var_of(pb);
	
	for (pv=v; pv!=NULL; pv=pv->succ)
	    if (var_of(pv)==var) 
	    {
		result=pv;
		*prank=rank;
		continue;
	    }
    }

    return(result);
}



/*  sorts the constraints according to the compare function,
 *  and set the number of constraints for each index of the sort base
 */

Pcontrainte constraints_sort_info(c, sort_base, compare, info)
Pcontrainte c;
Pbase sort_base;
int (*compare)();
int info[][2];
{
    Pcontrainte pc, *tc;
    Pvecteur phrank;
    int	i, rank,
	nb_of_sort_vars = vect_size(sort_base),
	nb_of_constraints = nb_elems_list(c);
Psysteme s;

    if (nb_of_constraints<=1) return(c);


   /* fprintf(stderr, "[constraints_sort_info] %d constraints\n", 
      nb_of_constraints); 
      s = sc_make(NULL, c);
      syst_debug(s);
      vect_debug(rbase_for_compare);
      vect_debug(others_for_compare);
      */

    tc   = (Pcontrainte*) malloc(sizeof(Pcontrainte)*nb_of_constraints);

    for (i=0; i<=nb_of_sort_vars; i++)
	info[i][0]=0, info[i][1]=0;

    /*   the constraints are put in the table
     *   and info is set.
     */
    for (i=0, pc=c; pc!=NULL; i++, pc=pc->succ)
    {
	tc[i] = pc;
	phrank = highest_rank_pvector(pc->vecteur, sort_base, &rank);
	info[rank==-1 ? 0 : rank][rank==-1 ? 0 : (val_of(phrank)>0 ? 1 : 0)]++;
    }
    
   qsort(tc, nb_of_constraints, sizeof(Pcontrainte), compare);
    

    /*  the list of constraints is generated again
     */
    for (i=0; i<nb_of_constraints-1; i++)
    {
	tc[i]->succ = tc[i+1];
    }
    tc[nb_of_constraints-1]->succ=NULL;
    c = tc[0];

    /*   clean!
     */
    free(tc);


    /* sc_inegalites(s)=c;
       syst_debug(s);*/

    return(c);
}

Pcontrainte constraints_sort_with_compare(c, sort_base, compare)
Pcontrainte c;
Pbase sort_base;
int (*compare)();
{
    int 
	n = vect_size(sort_base)+1,
	(*info)[2];

    info = malloc(sizeof(int)*2*n);

    c = constraints_sort_info(c, sort_base, compare, info);

    free(info);
    return(c);
}

static boolean complex_first_p;

static int contrainte_comparison(pc1, pc2)
Pcontrainte *pc1, *pc2;
{
    int 
	comp = compare_the_constraints(pc1, pc2);

/*
    int comp_2 = compare_the_constraints(pc2, pc1);
    assert((comp==0 && comp_2==0) || (comp*comp_2<0));
*/

    return(complex_first_p ? comp : -comp);
}

Pcontrainte contrainte_sort(c, base, sort_base, complex_first)
Pcontrainte c;
Pbase base, sort_base;
boolean complex_first;
{
    set_static_bases_for_compare(base, sort_base);
    complex_first_p=complex_first;

    c = constraints_sort_with_compare(c, sort_base, contrainte_comparison);

    reset_static_bases_for_compare();
    return(c);
}


Psysteme sc_sort_constraints(ps, base_index)
Psysteme ps;
Pbase base_index;
{
    ps->inegalites = 
	contrainte_sort(ps->inegalites, ps->base, base_index, TRUE);

    return(ps);
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
	sorted = contrainte_sort(ineg, sc->base, b, TRUE),
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



/* sort  contrainte c, base b, 
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


static Psort_info compute_sort_info_2levels(v, b, psi)
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
static float sort_info_to_value_2levels(psi, n_sort_vars, n_vars,
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

Pcontrainte contrainte_sort_info_2levels(c, base, sort_base, 
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
	values[i] = sort_info_to_value_2levels
	    (compute_sort_info_2levels(pc->vecteur, sort_base, &si),
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

Pcontrainte contrainte_sort_2levels(c, base, sort_base, inner_first, complex_first)
Pcontrainte c;
Pbase base, sort_base;
int inner_first, complex_first;
{
    int 
	n = vect_size(sort_base)+1,
	(*info)[2];
    Pcontrainte r;

    info = malloc(sizeof(int)*2*n);

    r = contrainte_sort_info_2levels(c, base, sort_base, 
			     inner_first, complex_first, info);

    free(info);
    return(r);
}

Psysteme sc_sort_constraints_2levels(ps, base_index)
Psysteme ps;
Pbase base_index;
{
    int 
	n = vect_size(base_index)+1,
	(*info)[2];

    info = malloc(sizeof(int)*2*n);

    ps->inegalites = 
	contrainte_sort_info_2levels(ps->inegalites, 
			     ps->base, base_index,
			     TRUE, TRUE, info);

    free(info);
    return(ps);
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

Psysteme sc_triang_elim_redond(ps, base_index)
Psysteme ps;
Pbase base_index;
{
    Pcontrainte eq, eq1;
    int 
	level,
	n = vect_size(base_index)+1,
	(*info)[2];

    ps = sc_normalize(ps);
   /* fprintf(stderr, "[sc_triang_elim_redond] input\n"); 
       sc_fprint(stderr, ps,variable_default_name); */
     
    if (ps==NULL)
	return(NULL);

  if (ps->nb_ineq > NB_INEQ_MAX1) 
	fprintf(stderr,
		"[sc_triang_elim_redond] warning, %d inequalities\n",
		ps->nb_ineq);

  
    if (!sc_integer_feasibility_ofl_ctrl(ps, OFL_CTRL,TRUE))
    {
	sc_rm(ps), ps=NULL;
	return(NULL);
    }

    info = malloc(sizeof(int)*2*n);

    /* fprintf(stderr, "[] INPUT:\n"); vect_debug(base_index); 
    syst_debug(ps);    */

    /*    set_static_bases_for_compare(ps->base, base_index);
    ps->inegalites = constraints_sort_info(ps->inegalites, 
					   base_index,
					   compare_the_constraints, 
					   info);
    reset_static_bases_for_compare();
    */

    ps->inegalites = 
	contrainte_sort_info_2levels(ps->inegalites, 
			     ps->base, base_index,
			     TRUE, TRUE, info);

    /*fprintf(stderr, "[] OUTPUT:\n"); syst_debug(ps); */

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
		info[abs(level)][level<0?0:1]--;
	    }
	}
    }
    sc_elim_empty_constraints(ps,0);
    ps = sc_kill_db_eg(ps);

 /*    fprintf(stderr, "[sc_triang_elim_redond] output\n");
       sc_fprint(stderr, ps,variable_default_name);  */
      

    free(info);
    return(ps);
}

/*   That is all
 */
