/* package vecteur - operations unaires */

/*LINTLIBRARY*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(p,t,f) free(p)

/* void vect_normalize(Pvecteur v): division de tous les coefficients de v
 * par leur pgcd; "normalisation" de vecteur directeur a coefficient entier
 *
 *    ->   ->      ->   ->
 * si v == 0 alors v := 0;
 * sinon
 *    pgcd = PGCD v[i];
 *	       i
 *    ->   ->
 *    v := v / pgcd
 * 
 * Le pgcd est toujours positif.
 *
 * Ancien nom: vect_norm()
 */
void vect_normalize(v)
Pvecteur v;
{
    int gcd = vect_pgcd_all(v);
    if ((gcd != 0) && (gcd != 1))
	(void) vect_div(v,gcd);
}

/* void vect_add_elem(Pvecteur * pvect, Variable var, Value val): 
 * addition d'un vecteur colineaire au vecteur de base var au vecteur vect
 * 
 * ----->    ----->       --->
 * *pvect := *pvect + val evar
 */
void vect_add_elem(pvect,var,val)
Pvecteur *pvect;
Variable var;
Value val;
{
    Pvecteur vect;

    if (val!=0) {
	for (vect=(*pvect);vect!=NULL;vect=vect->succ) {
	    if (var_of(vect)==var) {
		if ((val_of(vect) += val) == 0)
		    vect_erase_var(pvect, var_of(vect));
		return;
	    }
	}
	/* le coefficient valait 0 et n'etait pas represente */
	*pvect = vect_chain(*pvect,var,val);
    }
	/* sinon, le vecteur est inchange et on ne fait rien */
}

/* void vect_erase_var(Pvecteur * ppv, Variable v): projection du
 * vecteur *ppv selon la direction v (i.e. mise a zero de la
 * coordonnee v du vecteur pointe par ppv)
 *
 * Soit ev le vecteur de base correspondant a v:
 *
 *  --->    --->    --->   ->  ->
 *  *ppv := *ppv - <*ppv . ev> ev
 *
 * Note: cette routine ne fait pas l'hypothese que chaque coordonnee
 * n'apparait qu'une fois; on pourrait l'accelerer en forcant
 * pvcour a NULL des que la coordonnee est trouvee.
 */
void vect_erase_var(ppv, v)
Pvecteur *ppv;
Variable v;
{
    Pvecteur pvprec, pvcour;

    for (pvprec = NULL, pvcour = (*ppv); pvcour != NULL;) {
	/* A-t-on trouve la composante v? */
	if (pvcour->var == v) {
	    /* Si oui, est-il possible de la dechainer? */
	    if (pvprec != NULL) {
		/* elle n'est pas en tete de liste */
		Pvecteur pvprim = pvcour;
		pvcour = pvprec->succ = pvcour->succ;
		FREE((char *)pvprim, VECTEUR, "vect_erase_var");
	    }
	    else {
		/* Elle est en tete de liste; il faut modifier ppv */
		*ppv = pvcour->succ;
		FREE((char *)pvcour,VECTEUR,"vect_erase_var");
		pvcour = *ppv;
	    }
	}
	else {
	    /* Non, on passe a la composante suivante... */
	    pvprec = pvcour;
	    pvcour = pvcour->succ;
	}
    }
}

/* void vect_chg_coeff(Pvecteur *ppv, Variable var, Value val): mise
 * de la coordonnee var du vecteur *ppv a la valeur val
 *
 * --->   --->    --->   --->  --->       --->
 * *ppv = *ppv - <*ppv . evar> evar + val evar 
 */
void vect_chg_coeff(ppv,var,val)
Pvecteur   *ppv;
Variable var;
Value val;
{
    Pvecteur pvcour;

    if (val == 0) {
	vect_erase_var(ppv, var);
    }
    else {
	for (pvcour = (*ppv); pvcour != NULL; pvcour = pvcour->succ) {
	    if (pvcour->var == var) {
		pvcour->val = val;
		return;
	    }
	}
	/* on n'a pas trouve de composante var */
	*ppv = vect_chain(*ppv,var,val);
    }
}

/* void vect_chg_var(Pvecteur *ppv, Variable v_old, Variable v_new)
 * replace the variable v_old by v_new 
 */
void vect_chg_var(ppv,v_old,v_new)
Pvecteur *ppv;
Variable v_old,v_new;
{
    Pvecteur pvcour;

    for (pvcour = (*ppv); pvcour != NULL; pvcour = pvcour->succ) {
	if (pvcour->var == v_old){
	    pvcour->var = v_new;
	    return;
	}
    }
}



/* Pvecteur vect_del_var(Pvecteur v_in, Variable var): allocation d'un
 * nouveau vecteur egal a la projection de v_in selon la direction var
 * (i.e. le coefficient de la coordonnee var est mis a 0)
 *
 * Soit evar le vecteur de base correspondant a var:
 *
 *           ---->
 *  allocate v_out;
 *
 *  ---->    --->    ---->   --->  --->
 *  v_out := v_in - <v_out . evar> evar
 *
 *        ---->
 * return v_out;
 *
 */
Pvecteur vect_del_var(v_in,var)
Pvecteur v_in;
Variable var;
{
    if(v_in!=NULL){
	Pvecteur v_out = vect_dup(v_in);
	vect_erase_var(&v_out,var);
	return(v_out);
    }
    else
	return(NULL);
}

/* Variable vect_coeff(Variable var, Pvecteur vect): coefficient
 * de coordonnee var du vecteur vect
 *      --->
 * Soit evar le vecteur de base de nom var:
 * 
 *         --->   --->
 * return <vect . evar>; (i.e. return vect[var])
 *
 */
Value vect_coeff(var,vect)
Variable var;
Pvecteur vect;
{
    for ( ; vect != NULL ; vect = vect->succ)
	if (var_of(vect) == var) {
	    assert(val_of(vect)!=0);
	    return(val_of(vect));
	}
    return(0);
}

/* Value vect_coeff_sum(Pvecteur vect): coefficient sum
 * de tout les val de ce vecteur
 * 
 * return Value
 * Lei Zhou    Mar.25, 91
 */
Value vect_coeff_sum(vect)
Pvecteur vect;
{
    Value val = 0;

    if ( vect->var == TCST )
	return (val);
    for (vect = vect; vect != NULL ; vect = vect->succ) {
	val += vecteur_val(vect);
	assert(val_of(vect)!=0);
    }
    return (val);
}


/* Pvecteur vect_sign(Pvecteur v): application de l'operation signe au
 * vecteur v
 * 
 * ->         ->
 * v := signe(v );
 *        ->
 * return v ;
 */
Pvecteur vect_sign(v)
Pvecteur v;
{
    Pvecteur coord;

    for(coord = v; coord!=NULL; coord=coord->succ)
	val_of(coord) = SIGN(val_of(coord));

    return v;
}


/* void vect_sort_in_place(pv, compare)
 * Pvecteur *pv;
 * int (*compare)(Pvecteur *, Pvecteur *);
 *
 * Sorts the vector in place. It is an interface to qsort (stdlib).
 * see man qsort about the compare function, which tells < == or >.
 *
 * FC 29/12/94
 */
void vect_sort_in_place(pv, compare)
Pvecteur *pv;
int (*compare)();
{
    int 
	n = vect_size(*pv);
    Pvecteur 
	v, 
	*table,
	*point;

    if (n==0) return;

    /*  the temporary table is created and initialized
     */
    table = (Pvecteur*) malloc(sizeof(Pvecteur)*n);

    for (v=*pv, point=table; v!=(Pvecteur)NULL; v=v->succ, point++)
	*point=v;

    /*  sort!
     */
    qsort(table, n, sizeof(Pvecteur), compare);

    /*  the vector is regenerated in order
     */
    for (point=table; n>1; point++, n--)
	(*point)->succ=*(point+1);

    (*point)->succ=(Pvecteur) NULL;
    
    /*  clean and return
     */
    *pv=*table; free(table);
}

/* Pvecteur vect_sort(v, compare)
 * Pvecteur v;
 * int (*compare)();
 *
 *   --->           -->
 *   OUT  =  sorted IN
 */
Pvecteur vect_sort(v, compare)
Pvecteur v;
int (*compare)();
{
    Pvecteur
	new = vect_dup(v);

    vect_sort_in_place(&new, compare);
    return(new);
}

/*  for qsort, returns:
 *
 *     - if v1 < v2
 *     0 if v1 = v2
 *     + if v1 > v2
 */
int vect_compare(pv1, pv2)
Pvecteur *pv1, *pv2;
{
    return(strcmp((char *)&var_of(*pv1), (char *)&var_of(*pv2)));
}

/* void Pvecteur_separate_on_sign(v, pvpos, pvneg)
 * Pvecteur v, *pvpos, *pvneg;
 *
 *     IN: v
 *    OUT: pvpos, pvneg
 *
 * this function builds 2 vectors composed of the positive and negative
 * parts of the initial vector v which is not modified.
 * 
 * (c) FC 16/05/94
 */
void Pvecteur_separate_on_sign(v, pvpos, pvneg)
Pvecteur v, *pvpos, *pvneg;
{
    Pvecteur
	vc = VECTEUR_NUL;
    Value
	val = (Value) 0;
    Variable 
	var = (Variable) NULL;

    *pvneg = VECTEUR_NUL,
    *pvpos = VECTEUR_NUL;

    for(vc=v;
	vc!=VECTEUR_NUL;
	vc=vc->succ)
    {
	var = var_of(vc), 
	val = val_of(vc);
	if (val<(Value) 0)
	  vect_add_elem(pvneg, var, -val);
	else
	  vect_add_elem(pvpos, var, val);
    }
}


/* boolean vect_common_variables_p(Pvecteur v1, v2)    BA 19/05/94
 * input    : two vectors.
 * output   : TRUE if they have at least one common variable, 
 *            FALSE otherwise.
 * modifies : nothing.
 */
boolean vect_common_variables_p(v1, v2)
Pvecteur v1, v2;
{
    Pvecteur ev;

    for(ev = v1; !VECTEUR_NUL_P(ev); ev = ev->succ) {
	if(vect_contains_variable_p(v2, vecteur_var(ev)))
	    return TRUE;
    }
    return FALSE;
}


/* boolean vect_contains_variable_p(Pvecteur v, Variable var)    BA 19/05/94
 * input    : a vector and a variable
 * output   : TRUE if var appears as a component of v, FALSE otherwise.
 * modifies : nothing
 */
boolean vect_contains_variable_p(v, var)
Pvecteur v;
Variable var;
{
    boolean in_base;

    for(; !VECTEUR_NUL_P(v) && !variable_equal(vecteur_var(v), var); v = v->succ)
	;
    in_base = !VECTEUR_NUL_P(v);
    return(in_base);
}



/*
 * that is all
 */
