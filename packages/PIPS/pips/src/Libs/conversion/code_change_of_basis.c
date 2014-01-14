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

#include <stdio.h>
#include <stdlib.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "conversion.h"

/* void scanning_base_to_vect(matrice G,int n,Pbase base,Pvecteur pvg[n])
 * compute  G*base and put the result in a vector pvg[n]
 * ex:     1 2
       G =      , base = (I, J)   , alor pvg[2] = ((I+2J), (3I+4J))

           3 4
 */
void  scanning_base_to_vect(G, n, base, pvg)
matrice G;
int n;
Pbase base;
Pvecteur pvg[];
{
    int i,j;
    Pbase pb;
    Pvecteur pv;

    for (i=1; i<=n; i++) {
	for (j=1,pb=base,pv=NULL; j<=n; j++,pb=pb->succ)
	    vect_add_elem(&pv,pb->var,ACCESS(G,n,i,j));
	pvg[i] = pv;
    }
}

/* Pvecteur vect_change_base(Pvecteur pv_old, Pvecteur pvg[],
 * Pbase base_oldindex, Pbase base_newindex)
 * compute the new vector in the new basis pvg[]
 */
Pvecteur vect_change_base(pv_old, base_oldindex, pvg)
Pvecteur pv_old;
Pbase base_oldindex;
Pvecteur pvg[];
{
    Pvecteur pv,pv1,pv2 = NULL;
    Pbase to_erase = BASE_NULLE, b;
    int r;

    for (pv=pv_old; pv!=NULL; pv=pv->succ)
    {
	r = base_find_variable_rank(base_oldindex, pv->var, (get_variable_name_t) entity_name_or_TCST);

	if (r != -1)
	{ /* var is in base_oldindex */
	    pv1 = vect_multiply(vect_dup(pvg[r]),pv->val);
	    pv2 = vect_add(pv2,pv1);

	    /* on bousille tranquillement le vecteur sur lequel on itere;-) */
	    /* vect_erase_var(&pv_old,pv->var); */
	    to_erase = base_add_variable(to_erase, pv->var);
	}
    }

    /* clean vector
     */
    for(b=to_erase; b!=NULL; b=b->succ)
	vect_erase_var(&pv_old, b->var);
    base_rm(to_erase);

    pv2 = vect_add(pv2,pv_old);

    return pv2;
}


/* cons *listexpres_to_listexpres_newbase(cons *lex, Pvecteur pvg[],
 * Pbase base_oldindex)
 * compute the new list of expressions (cons* lex) in the new basis  pvg[]
 */
cons *listexpres_to_listexpres_newbase(lex, pvg, base_oldindex)
cons *lex;
Pvecteur pvg[];
Pbase base_oldindex;
{
    cons *l_new=NIL;
    expression ex, ex_new;
    
    for (; lex!=NULL; lex=CDR(lex)) {
	ex = EXPRESSION(CAR(lex));
	ex_new = expression_to_expression_newbase(ex,pvg,base_oldindex);
	/*if (ex_new != ex) free_expression(ex);*/
	l_new = CONS(EXPRESSION,ex_new,l_new);
    }
    return(gen_nreverse(l_new));
}

/* expression expression_to_expression_newbase(expression e_old,Pvecteur pvg[],
 * Pbase base_oldindex)
 * compute the new expression for e_old  in the new basis pvg[]
 */
expression expression_to_expression_newbase(e_old, pvg, base_oldindex)
expression e_old;
Pvecteur pvg[];
Pbase base_oldindex;
{
    normalized e_old_norm;   
    syntax syn; 
    expression e_new;
    reference ref;
    call cal;
    cons *l_ex;
    cons *l_ex_new;
    Pvecteur pve_old,pve_new;
    
    e_old_norm = NORMALIZE_EXPRESSION(e_old);

    pips_debug(8, "tag=%d\n", normalized_tag(e_old_norm));
    ifdebug(9) {
	pips_debug(9, "considering expression %p:\n", e_old);
	print_expression(e_old);
    }

    if (normalized_linear_p(e_old_norm))
    { /* linear */
	pve_old = (Pvecteur) normalized_linear(e_old_norm);
	pve_new = vect_change_base(pve_old,base_oldindex,pvg);
	e_new = make_vecteur_expression(pve_new); 
	return(e_new);
    }
    else 
    {				/* complex */
	syn = expression_syntax(e_old);
	if (syntax_reference_p(syn)) {	
	    ref = syntax_reference(syn);
	    l_ex = reference_indices(ref);
	    if (l_ex!=NULL){	
		l_ex_new = listexpres_to_listexpres_newbase(
		    l_ex,pvg,base_oldindex);
		reference_indices(ref) = l_ex_new;		
		gen_free_list(l_ex);
	    }
	}
	else if (syntax_call_p(syn)) {
	    cal = syntax_call(syn);
	    l_ex = call_arguments(cal);
	    if (l_ex!=NULL){
		l_ex_new = listexpres_to_listexpres_newbase(
		    l_ex,pvg,base_oldindex);
		call_arguments(cal) = l_ex_new;
		gen_free_list(l_ex);
	    }
	}
	return (e_old);
    }
}


        
/* statement_newbase(statement s, Pvecteur pvg[], Pbase base_oldindex)
 * compute the new statement by performing the change of basis 
 */
void statement_newbase(s, pvg, base_oldindex) 
statement s;
Pvecteur pvg[];
Pbase base_oldindex;
{
    instruction i;
    statement s1;
    test t;
    expression e, e1;
    loop l;
    range r;
    call c;
    cons *ls;
    cons *lex;

    i = statement_instruction(s);
    switch (instruction_tag(i)) {
    case is_instruction_block:
	for (ls=instruction_block(i); ls!=NULL; ls=CDR(ls)) {
	    s1 = STATEMENT(CAR(ls));
	    statement_newbase(s1,pvg,base_oldindex);
	}
	break;
    case is_instruction_test:
	t = instruction_test(i);
	e = test_condition(t);
	
	    e1 = expression_to_expression_newbase(e, pvg, base_oldindex);
	if (e != e1){
	    test_condition(t) = e1;
	    free_expression(e);
	}
	statement_newbase(test_true(t),pvg,base_oldindex);
	statement_newbase(test_false(t),pvg,base_oldindex);
	break;
    case is_instruction_loop:
	l = instruction_loop(i);
	r = loop_range(l);
	e = range_lower(r);
	e1 = expression_to_expression_newbase(e,pvg,base_oldindex);
	if (e != e1) {
	    range_lower(r) = e1;
	    free_expression(e);
	}
	e = range_upper(r);
	e1 = expression_to_expression_newbase(e,pvg,base_oldindex);
	if (e != e1) {
	    range_upper(r) = e1;
	    free_expression(e);
	}
	
	e = range_increment(r);
	e1 = expression_to_expression_newbase(e,pvg,base_oldindex);
	if (e != e1) {
	    range_increment(r) = e1;
	    free_expression(e);
	}
	statement_newbase(loop_body(l),pvg,base_oldindex);
	break;
    case is_instruction_forloop: {
	forloop fl = instruction_forloop(i);

	e = forloop_initialization(fl);
	e1 = expression_to_expression_newbase(e,pvg,base_oldindex);
	if (e != e1) {
	  forloop_initialization(fl) = e1;
	  free_expression(e);
	}
	e = forloop_condition(fl);
	e1 = expression_to_expression_newbase(e,pvg,base_oldindex);
	if (e != e1) {
	    forloop_condition(fl) = e1;
	    free_expression(e);
	}
	e = forloop_increment(fl);
	e1 = expression_to_expression_newbase(e,pvg,base_oldindex);
	if (e != e1) {
	    forloop_increment(fl) = e1;
	    free_expression(e);
	}
	statement_newbase(forloop_body(fl),pvg,base_oldindex);
	break;
    }
    case is_instruction_whileloop: {
	whileloop wl = instruction_whileloop(i);

	e = whileloop_condition(wl);
	e1 = expression_to_expression_newbase(e,pvg,base_oldindex);
	if (e != e1) {
	  whileloop_condition(wl) = e1;
	    free_expression(e);
	}
	statement_newbase(whileloop_body(wl),pvg,base_oldindex);
	break;
    }
    case is_instruction_goto:
	statement_newbase(instruction_goto(i),pvg,base_oldindex);
    case is_instruction_call:
	c = instruction_call(i);
	lex = call_arguments(c);
	call_arguments(c) = listexpres_to_listexpres_newbase(lex, pvg,
							     base_oldindex);
	gen_free_list(lex);
	break;
    case is_instruction_unstructured:
	break;
    default:
	pips_internal_error("unexpected tag %d",instruction_tag(i));
    }
}
