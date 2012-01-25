/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/* Name     : reindexing_utils.c
 * Package  : reindexing
 * Author   : Alexis Platonoff & Antoine Cloue
 * Date     : May 1994
 * Historic :
 * - 20 apr 95 : modification of create_new_entity(), AP
 *
 * Documents: SOON
 * Comments : contains some usefull functions.
 */
 
/* Ansi includes        */
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
 
/* Newgen includes      */
#include "genC.h"
 
/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "union.h"
#include "matrix.h"

/* Pips includes 	*/
#include "boolean.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "text.h"
#include "text-util.h"
#include "tiling.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "static_controlize.h"
#include "paf-util.h"
#include "pip.h"
#include "array_dfg.h"
#include "reindexing.h"

/* Macro functions  	*/

/* Internal variables 	*/

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;

#define SUCC(cp) (((cp) == NULL) ? NULL : (cp)->succ)


/*=======================================================================*/
/* Psyslist reverse_psyslist(l): reverse the psyslist l.
 *
 * AC 94/06/29
 */

Psyslist reverse_psyslist(l)

 Psyslist  l;
{
 Psyslist  next, next_next;

 if( l == NULL || l->succ == NULL ) return(l) ;

 next = l->succ;
 l->succ = NULL ;
 next_next = SUCC(next);
 
 for( ; next != NULL ; ) 
   {
    next->succ = l;
    l = next ;
    next = next_next ;
    next_next = SUCC(next_next);
   }
 
 return(l) ;
}

/*=======================================================================*/
/* expression psystem_to_expression(predicate pred): function that
 * transforms a predicate into an expression. Function that comes from
 * single_assign.c but I have modified it to include the case where
 * pred is NULL.
 *
 * AC 94/06/24
 */
expression psystem_to_expression(ps)
     Psysteme     ps;
{
  entity       and_ent, leq_ent, equ_ent;
  expression   exp1 = expression_undefined, exp2;
  Pcontrainte  pc;

  if (SC_UNDEFINED_P(ps))
    return(int_to_expression(0));
  else {
    and_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						      AND_OPERATOR_NAME),
				 entity_domain);
    leq_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						      LESS_OR_EQUAL_OPERATOR_NAME),
				 entity_domain);
    equ_ent = gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME,
						      EQUAL_OPERATOR_NAME),
				 entity_domain);

    if( (and_ent == entity_undefined) || (leq_ent == entity_undefined) ||
       (equ_ent == entity_undefined) ) {
      pips_internal_error("There is no entity for operators");
    }

    for(pc = ps->inegalites; pc!=NULL; pc = pc->succ) {
      Pvecteur pv = pc->vecteur;
      expression exp = make_vecteur_expression(pv);
      exp2 = MakeBinaryCall(leq_ent, exp, int_to_expression(0));

      if(exp1 == expression_undefined)  exp1 = exp2;
      else exp1 = MakeBinaryCall(and_ent, exp1, exp2);
    }

    for(pc = ps->egalites; pc!=NULL; pc = pc->succ) {
      Pvecteur pv = pc->vecteur;
      exp2 = MakeBinaryCall(equ_ent, make_vecteur_expression(pv),
			    int_to_expression(0));
      if(exp1 == expression_undefined)  exp1 = exp2;
      else  exp1 = MakeBinaryCall(and_ent, exp1, exp2);
    }

    if (get_debug_level() > 7) {
      fprintf(stderr, "\t[predicate_to_expression] Result: %s\n",
	      words_to_string(words_expression(exp1)));
    }
    
    return(exp1);
  }
}

/*=========================================================================*/
/* reference build_new_ref(int kind,int n,list subscripts,reference old_r)
 * 
 * builds a new array reference. Its entity name depends on "kind":
 *	kind == IS_TEMP => name is : SATn
 *	kind == IS_NEW_ARRAY => name is : SAIn
 * We first test if this entity does not exist yet. If not, we have create
 * it with a type_variable with the same basic as the one of the entity of
 * "old_ref" and a dimension depending again on kind:
 *	kind == IS_TEMP => dimension is: empty
 *	kind == IS_NEW_ARRAY => dimension is: dimension of the loop nest of
 *					      statement number n
 *
 * Its indices of the new reference are "subscripts".
 *
 * "subscripts" is a list of affine expressions.
 * If "subscripts" is empty, then this is a scalar.
 * AC : function that comes from single_assign.c but we have modified it
 * to include the new parameter BASE_NODE_NUMBER.
 */

reference my_build_new_ref(kind, n, subscripts, old_r)

 int        kind;
 int        n;
 list       subscripts;
 reference  old_r;
{
  list       sl;
  entity     ent;
  string     num, name = (string) NULL;
  entity mod_entity;
  code mod_code;

  /* we duplicate this list */
  sl = subscripts;

  num = atoi(n-BASE_NODE_NUMBER);
  if(kind == IS_TEMP)
    name = strdup(concatenate(SA_MODULE_NAME, MODULE_SEP_STRING,
			      SAT, num, (char *) NULL));
  else if(kind == IS_NEW_ARRAY)
    name = strdup(concatenate(SA_MODULE_NAME, MODULE_SEP_STRING,
			      SAI, num, (char *) NULL));
  else
    pips_internal_error("Bad value for kind");

  ent = gen_find_tabulated(name, entity_domain);
  if(ent == entity_undefined) {
    list dims = NIL;
    
    if(kind == IS_NEW_ARRAY)
      dims = dims_of_nest(n);
    else
      pips_internal_error("Bad value for kind");

    ent = create_entity(name, make_variable(basic_of_reference(old_r), dims));

    /* Declare the entity */
    mod_entity = get_current_module_entity();
    mod_code = entity_code(mod_entity);
    code_declarations(mod_code) = gen_nconc(code_declarations(mod_code),
					    CONS(ENTITY, ent, NIL));

    if (get_debug_level() > 6) {
      if(kind == IS_NEW_ARRAY)
	fprintf(stderr, "\n***\nCreate an ARRAY ENTITY : %s, %s\n",
		entity_local_name(ent), name);
    }
  }

  return(make_reference(ent, sl));
}

/*=========================================================================*/
/* void lhs_subs_in_ins(instruction ins, string SA, int n, list subscripts)
 * 
 * Substitutes to the lhs (left Hand Side) reference of "ins" the array
 * reference SAn[subscripts], cf. build_new_ref().
 *
 * "subscripts" is a list of entity, so we have transform it into a list of
 * expression.
 *
 * Note: "ins" must be an assign call
 */

void my_lhs_subs_in_ins(ins, SA, n, subscripts)

 instruction ins;
 string      SA;
 int         n;
 list        subscripts;
{
 switch(instruction_tag(ins))
   {
    case is_instruction_call : {
      call c = instruction_call(ins);
      if(ENTITY_ASSIGN_P(call_function(c))) {
	expression lhs_exp = EXPRESSION(CAR(call_arguments(c)));
	syntax sy = expression_syntax(lhs_exp);
	if(syntax_reference_p(sy)) {
	  reference lhs = syntax_reference(sy);
	  list exp_subs = entities_to_expressions(subscripts);
	  syntax_reference(sy) = my_build_new_ref(IS_NEW_ARRAY, n, 
						  exp_subs, lhs);

	}
	else pips_internal_error("Lhs is not a reference");
      }
      else pips_internal_error("Instruction is not an assign call");
      break;
    }
    case is_instruction_block :
    case is_instruction_test :
    case is_instruction_loop :
    case is_instruction_goto :
    case is_instruction_unstructured :
    default : pips_internal_error("Instruction is not an assign call");
  }
}

/*========================================================================*/
/* list ADD_LIST_TO_LIST(l1, l2): add the list l2 at the end of the list l1
 *
 * AC 94/06/07
 */

list ADD_LIST_TO_LIST(l1, l2)

 list  l1, l2;
{
 list  l3, l4;

 if (l1 == NIL) return(l2);
 if (l2 == NIL) return(l1);

 l3 = l1;

 while (l3 != NIL)
   {
    l4 = l3;
    l3 = l3->cdr;
   }
 l4->cdr = l2;

 return(l1);
}

/*========================================================================*/
/* void fprint_list_of_ins(fp, li): print a list of instruction.
 *
 * AC 94/06/07
 */

void fprint_list_of_ins(fp, li)

 FILE   *fp;
 list   li;
{
 int    i = 1;

 fprintf(fp,"\nListe of instruction:");
 fprintf(fp,"\n=====================");

 for (; li != NIL; li = CDR(li))
   {
    fprintf(fp, "\nInstruction n. %d:\n", i);
    sa_print_ins(fp, INSTRUCTION(CAR(li)));
    i++;
   }
}

/*========================================================================*/
/* void fprint_loop(fp, lp): print a loop.
 * 
 * AC 94/06/07
 */

void fprint_loop(fp, lp)

 FILE   *fp;
 loop   lp;
{
 fprintf(fp,"\nIndice de boucle :");
 fprint_entity_list(fp, CONS(ENTITY, loop_index(lp), NIL));
 fprintf(fp,"\nDomaine (lower, upper, inc):");
 fprint_list_of_exp(fp,
		    CONS(EXPRESSION, range_lower(loop_range(lp)), NIL));
 fprint_list_of_exp(fp, 
		    CONS(EXPRESSION, range_upper(loop_range(lp)), NIL));
 fprint_list_of_exp(fp, 
		    CONS(EXPRESSION, range_increment(loop_range(lp)), NIL));
 fprintf(fp,"\nCorps de boucle :");
 sa_print_ins(fp,statement_instruction(loop_body(lp)));
 fprintf(fp,"\nLabel:");
 fprint_entity_list(fp, CONS(ENTITY, loop_label(lp), NIL));
 fprintf(fp,"\nType d'execution");
 if (execution_sequential_p(loop_execution(lp)))
    fprintf(fp," ->sequentiel");
 else fprintf(fp, "->parallele");
 fprintf(fp, "\nLocals : ");
 fprint_entity_list(fp, loop_locals(lp));
}

/*========================================================================*/
/* void fprint_call(fp, ca): print a call.
 *
 * AC 94/06/07
 */

void fprint_call(fp, ca)

 FILE       *fp;
 call       ca;
{
 expression exp1, exp2;

 exp1 = EXPRESSION(CAR(call_arguments(ca)));
 exp2 = EXPRESSION(CAR((call_arguments(ca))->cdr));
 fprintf(fp, "\nCall : ");
 fprint_list_of_exp(fp, CONS(EXPRESSION, exp1, NIL));
 fprintf(fp, " %s ", entity_local_name(call_function(ca)));
 fprint_list_of_exp(fp, CONS(EXPRESSION, exp2, NIL));
}

/*=======================================================================*/
/* void print_detailed_ins(): the instruction is either a
 * test or an unstructured.
 *
 * AC 01/06/94
 */

void print_detailed_ins(ins)

 instruction ins;
{
 list        l;
 statement   stat;
 loop        lp;

 switch (instruction_tag(ins))
   {
    case is_instruction_block:
      {
       fprintf(stderr,"\nC'est un block de statement :");
       for (l = instruction_block(ins); l != NIL; l = CDR(l))
	 {
	  stat = STATEMENT(CAR(l));
          print_detailed_ins(statement_instruction(stat));
	 }
       fprintf(stderr,"\nFin du block");
       break;
      }

    case is_instruction_test:
      {
       fprintf(stderr,"\nC'est un test :");
       fprintf(stderr,"\nTrue statement :");
       print_detailed_ins(statement_instruction(
				    test_true(instruction_test(ins))));
       fprintf(stderr, "\nFin de la branche True");
       fprintf(stderr,"\nFalse statement :");
       print_detailed_ins(statement_instruction(
				    test_false(instruction_test(ins))));
       fprintf(stderr,"\nFin du test.");
       break;
      }

    case is_instruction_loop:
      {
       fprintf(stderr,"\nC'est une boucle :");
       lp = instruction_loop(ins);
       fprint_loop(stderr, lp);
       fprintf(stderr,"\nFin de boucle.");
       break;
      }

    case is_instruction_goto:
      {
       fprintf(stderr, "\nC'est une instruction goto ");
       sa_print_ins(stderr, statement_instruction(instruction_goto(ins)));
       fprintf(stderr,"\nFin du goto");
       break;
      }

    case is_instruction_call:
      {
       fprintf(stderr, "\nC'est un call ");
       fprint_call(stderr, instruction_call(ins));
       fprintf(stderr, "\nFin du call");
       break;
      }

    case is_instruction_unstructured:
      {
       fprintf(stderr,"\nC'est un unstructured ;");
/*       loop_normalize_of_unstructured(instruction_unstructured(ins));*/
       print_detailed_ins(control_statement(
			  unstructured_control(
                                      instruction_unstructured(ins))));
       fprintf(stderr, "\nFin du unstructured");
       break;
      }
   }
}

/*========================================================================*/
/* Psyslist add_sclist_to_sclist(l1, l2): add the list of systems l2 at
 * the end of the list of system l1.
 *
 * Ac 94/05/05
 */

Psyslist add_sclist_to_sclist(l1, l2)

 Psyslist  l1, l2;
{
 Psyslist  l = l1, ls;

 if (l2 == NULL) return(l1);

 if (l != NULL)
    {
     ls = l->succ;
     while (ls != NULL)
	{
	 l = l->succ;
	 ls = l->succ;
        }
     l->succ = l2;
     return(l1);
    }
 else return(l2);
}

/*========================================================================*/
/* Psysteme base_complete(Psysteme sys, list var_l, list par_l): "sys" gives
 * a family of free vectors {V1, ..., Vs} represented by a linear combinations
 * of indices from "var_l". This function wants to find the indices
 * (I1, ..., Id) of "var_l" for which we have that {V1, ..., Vs, I1, ..., Id}
 * is a family of free vectors.
 * "par_l" gives the symbolic constants that may appear in "sys".
 *
 * "s" is the number of equations of "sys" (its number of vectors).
 * "d" is the number of vectors we have to find in order to get as much
 * equations in "sys" as there are indices in "var_l".
 *
 * Example: with "sys" equal to {i+j = 0, i-k+2j = 0}, and "var_l" equal to
 * {i, j, k} we obtain the new system {i+j = 0, i-k+2j = 0, i = 0}
 *
 * function written by Alexis in "utils.c". The difference is that 
 * it returns the list of new free vectors added. (AC 94/03/18)
 */

Psysteme base_complete(sys, var_l, par_l, new_l)

 Psysteme  sys;
 list      var_l, par_l, *new_l;
{
 Psysteme  ps = sc_dup(sys), new_ps = sc_new();
 int       dim = gen_length(var_l) - sys->nb_eq;
 list      l;

 for (l = var_l; (!ENDP(l)) && (new_ps->nb_eq < dim); POP(l)) 
    {
     entity   var = ENTITY(CAR(l));
     Pvecteur pv = vect_new((Variable) var, 1);
     Psysteme aux_ps = sc_dup(ps);
     Psysteme aux_new_ps = sc_dup(new_ps);

     sc_add_egalite(aux_new_ps, contrainte_make(pv));
     aux_ps = append_eg(aux_ps, aux_new_ps);

     if (vecteurs_libres_p(aux_ps, list_to_base(var_l), list_to_base(par_l))) 
        {
         new_ps = aux_new_ps;
         ADD_ELEMENT_TO_LIST(*new_l, ENTITY, var);
        }
     else
         sc_rm(aux_new_ps);
    }

 ps = append_eg(ps, new_ps);
 ps->base = NULL;
 sc_creer_base(ps);

 return(ps);
}

/*========================================================================*/
/* Psysteme sc_add_egalite_at_end(ps, co): idem "sc_add_egalite()" except that
 * it puts the new constraint at the end of the list of constraints.
 *
 * The system basis is not updated. ps may be inconsistent on return.
 *
 * AC 94/03/17
 */
Psysteme sc_add_egalite_at_end(Psysteme ps,  Pcontrainte  co)
{
  Pcontrainte  co_aux;

  if (ps->egalites == NULL) ps->egalites = co;
  else
    {
      for (co_aux = ps->egalites; co_aux->succ != NULL; co_aux = co_aux->succ) ;
      co_aux->succ = co;
    }
  ps->nb_eq++;

  return ps;
}

/*=========================================================================*/
/* void matrix_coef_mult(A, nb):multiply all elements of matrix A by the
 * number nb.
 *
 * AC 94/03/21
 */

/* also elsewehre... */
void matrix_coef_mult(A, nb)

 Pmatrix   A;
 Value       nb;
{
 int       i, j;

 for (i = 1; i <= MATRIX_NB_LINES(A); i++)
    for (j = 1; j <= MATRIX_NB_COLUMNS(A); j++)
       value_product(MATRIX_ELEM(A,i,j), nb); 
}

/*======================================================================*/
/* void constraints_with_sym_cst_to_matrices(Pcontrainte pc,
 *      Pbase index_base const_base, matrice A B, int n m1 m2):
 *
 * constructs the matrices "A" and "B" corresponding to the linear
 * constraints "pc", so: A.ib + B1.cb + B2 = 0 <=> pc(ib, cb) = 0:
 *
 *      B = ( B1 | B2 ), B2 of dimension (n,1).
 *
 * The basis "ib" gives the variables of the linear system.
 * The basis "cb" gives the symbolic constants of the linear system.
 *
 * The matrices "A" and "B" are supposed to have been already allocated
 * in memory, respectively of dimension (n, m1) and (n, m2).
 *
 * "n" must be the exact number of constraints in "pc".
 * "m1" must be the exact number of variables in "ib".
 * "m2" must be equal to the number of symbolic constants (in "cb") PLUS
 * ONE (the actual constant).
 */
void my_constraints_with_sym_cst_to_matrices(pc,index_base,const_base,A,B)

 Pcontrainte  pc;
 Pbase        index_base,const_base;
 Pmatrix      A, B;
{
 int          i,j;
 Pcontrainte  eq;
 Pvecteur     pv;
 int          n = 0;
 int          m2 = vect_size(const_base) + 1; /* CHANGE HERE !!! */

 for (eq = pc; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,n++);
 matrix_nulle(B);
 matrix_nulle(A);

 for (eq = pc,i=1; !CONTRAINTE_UNDEFINED_P(eq); eq=eq->succ,i++)
   {
    for(pv = index_base, j=1; pv != NULL; pv = pv->succ, j++)
      { MATRIX_ELEM(A,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);}
   
    for(pv = const_base, j=1; pv != NULL; pv = pv->succ, j++)
      { MATRIX_ELEM(B,i,j) = vect_coeff(vecteur_var(pv),eq->vecteur);}

    MATRIX_ELEM(B,i,m2) = vect_coeff(TCST,eq->vecteur);
   }
}

/*======================================================================*/
/* void my_matrices_to_constraints_with_sym_cst(Pcontrainte *pc,
 *      Pbase index_base const_base, matrice A B,int n m1 m2):
 *
 * constructs the constraints "pc" corresponding to the matrices "A" and "B"
 * so: A.ib + B1.cb + B2 = nb <=> pc(nb, ib, cb) = 0, with:
 *
 *      B = ( B1 | B2 ), B2 of dimension (n,1).
 *
 * The basis "ib" gives the variables of the linear system.
 * The basis "cb" gives the symbolic constants of the linear system.
 * The basis "nb" gives the new variables of the linear system.
 *
 * The matrices "A" and "B" are respectively of dimension (n, m1) and (n,m2).
 *
 * "n" will be the exact number of constraints in "pc".
 * "m1" must be the exact number of variables in "ib".
 * "m2" must be equal to the number of symbolic constants (in "cb") PLUS
 * ONE (the actual constant).
 *
 * Note: the formal parameter pc is a "Pcontrainte *". Instead, the resulting
 * Pcontrainte could have been returned as the result of this function.
 * 
 * TAKE CARE : new_base should be well ordered !!
 *
 * AC 94/03/30
 */

void my_matrices_to_constraints_with_sym_cst(pc, new_base, index_base,
                                             const_base, A, B)

 Pcontrainte  *pc;
 Pbase        new_base, index_base, const_base;
 Pmatrix      A, B;
{
 Pvecteur     vect, pv = NULL;
 Pcontrainte  cp, newpc = NULL;
 int          i, j;
 Value cst, coeff, dena, denb, ppc;
 bool      found;
 list         lnew = gen_nreverse(base_to_list(new_base));
 entity       ent;

 int n= MATRIX_NB_LINES(A);
 int m1 = MATRIX_NB_COLUMNS(A);
 int m2 = MATRIX_NB_COLUMNS(B);
 dena = MATRIX_DENOMINATOR(A);
 denb = MATRIX_DENOMINATOR(B);

 ppc = ppcm(dena, denb);

 for (i = n; i >= 1; i--)
    {
     found = false;
     cp = contrainte_new();

     /* build the constant terme if it exists */
     if (value_notzero_p(cst = MATRIX_ELEM(B,i,m2)))
     {
	 Value x = value_mult(ppc,cst);
	 value_division(x, denb);
         pv = vect_new(TCST, x);
         found = true;
     }

     vect = index_base;
     for (j = 1; j <= m1; vect = vect->succ, j++) 
        {
         if (value_notzero_p(coeff = MATRIX_ELEM(A,i,j)))
	   {
	       Value x = value_div(ppc,dena);
	       value_product(x,coeff);
            if (found)
                 vect_chg_coeff(&pv, vecteur_var(vect), x);
            else
	      {
               /* build a new vecteur if there is not constant term */
               pv = vect_new(vecteur_var(vect), x);
               found = true;
              }
	   }
        }

     vect = const_base;
     for (j = 1; j <= m2-1; vect = vect->succ, j++) 
	{
         if (value_notzero_p(coeff = MATRIX_ELEM(B,i,j)))
	   {
	       Value x = value_div(ppc,denb);
	       value_product(x,coeff);
            if (found)
                 vect_chg_coeff(&pv, vecteur_var(vect), x);
            else 
	      {
               /* build a new vecteur if there is not constant term */
               pv = vect_new(vecteur_var(vect), x);
               found = true;
              }
	   }
        }

     ent = ENTITY(CAR(lnew));
     pv = vect_substract(pv, vect_new((Variable) ent, ppc));
     lnew = CDR(lnew);

     cp->vecteur = pv;
     cp->succ = newpc;
     newpc = cp;
    }

 *pc = newpc;
}

/*======================================================================*/
/* void my_matrices_to_constraints_with_sym_cst_2(Pcontrainte *pc,
 *      Pbase index_base const_base, matrice A B,int n m1 m2):
 *
 * constructs the constraints "pc" corresponding to the matrices "A" and "B"
 * so: A.ib + B1.cb + B2 = nb <=> pc(nb, ib, cb) = 0, with:
 *
 *      B = ( B1 | B2 ), B2 of dimension (n,1).
 *
 * The basis "ib" gives the variables of the linear system.
 * The basis "cb" gives the symbolic constants of the linear system.
 * The basis "nb" gives the new variables of the linear system.
 *
 * The matrices "A" and "B" are respectively of dimension (n, m1) and (n,m2).
 *
 * "n" will be the exact number of constraints in "pc".
 * "m1" must be the exact number of variables in "ib".
 * "m2" must be equal to the number of symbolic constants (in "cb") PLUS
 * ONE (the actual constant).
 *
 * Note: the formal parameter pc is a "Pcontrainte *". Instead, the resulting
 * Pcontrainte could have been returned as the result of this function.
 * 
 * TAKE CARE : new_base should be well ordered !!
 *
 * AC 94/03/30
 */

void my_matrices_to_constraints_with_sym_cst_2(pc, new_base, index_base,
                                               const_base, A, B)

 Pcontrainte  *pc;
 Pbase        new_base, index_base, const_base;
 Pmatrix      A, B;
{
 Pvecteur     vect, pv = NULL;
 Pcontrainte  cp, newpc = NULL;
 int          i, j;
 Value cst, coeff, dena, denb, ppc;
 bool      found;

 int n= MATRIX_NB_LINES(A);
 int m1 = MATRIX_NB_COLUMNS(A);
 int m2 = MATRIX_NB_COLUMNS(B);
 dena = MATRIX_DENOMINATOR(A);
 denb = MATRIX_DENOMINATOR(B);

 ppc = ppcm(dena, denb);

 for (i = n; i >= 1; i--)
    {
     found = false;
     cp = contrainte_new();

     /* build the constant terme if it exists */
     if (value_notzero_p(cst = MATRIX_ELEM(B,i,m2))) 
     {
	 Value x = value_div(ppc,denb);
	 value_product(x, cst);
	 pv = vect_new(TCST,  x);
	 found = true;
     }

     vect = index_base;
     for (j = 1; j <= m1; vect = vect->succ, j++) 
        {
         if (value_notzero_p(coeff = MATRIX_ELEM(A,i,j)))
	   {
	       Value x = value_div(ppc,dena);
	       value_product(x,coeff);
            if (found)
                 vect_chg_coeff(&pv, vecteur_var(vect), x);
            else
	      {
               /* build a new vecteur if there is not constant term */
               pv = vect_new(vecteur_var(vect), x);
               found = true;
              }
	   }
        }

     vect = const_base;
     for (j = 1; j <= m2-1; vect = vect->succ, j++) 
	{
         if (value_notzero_p(coeff = MATRIX_ELEM(B,i,j)))
	   {
	       Value x = value_div(ppc,denb);
	       value_product(x,coeff);
            if (found)
                 vect_chg_coeff(&pv, vecteur_var(vect), x);
            else 
	      {
               /* build a new vecteur if there is not constant term */
               pv = vect_new(vecteur_var(vect), x);
               found = true;
              }
	   }
        }

     cp->vecteur = pv;
     cp->succ = newpc;
     newpc = cp;
    }

 *pc = newpc;
}

/*===================================================================*/
/* Psysteme matrix_to_system(A, b): transform a Pmatrix in a system
 * following the variables in Pbase b.
 *
 * AC 94/05/15
 */

Psysteme matrix_to_system(A, b)

 Pmatrix  A;
 Pbase    b;
{
 Psysteme p = sc_new();
 list     l, lb = base_to_list(b);
 int      i, j;
 Pvecteur vect, nvect;

 for (i = 1; i <= MATRIX_NB_LINES(A); i++)
   {
    l = lb;
    vect = VECTEUR_NUL;

    for (j = 1; j <= MATRIX_NB_COLUMNS(A); j++)
      {
       nvect = vect_new((Variable) ENTITY(CAR(l)), MATRIX_ELEM(A,i,j));
       vect = vect_add(vect, nvect);
       l = CDR(l);
      }
    sc_add_egalite_at_end(p, contrainte_make(vect));
   }

 if (get_debug_level() > 5)
   {
    fprintf(stderr,"\nmatrcix_to_ps");
    fprint_psysteme(stderr,p);
   }

 p->base = vect_dup(b);

 return(p);
}

/*=========================================================================*/
/* Psysteme sc_reverse_constraints(ps): reverse the list of equalities and
 * inequalities in the system ps.
 *
 * AC 94/03/17
 */

Psysteme sc_reverse_constraints(ps)

 Psysteme     ps;
{
 Pcontrainte  pc1, pc2;

 pc2 = NULL;

 while (ps->egalites != NULL)
    {
     pc1 = ps->egalites;
     ps->egalites = (ps->egalites)->succ;
     pc1->succ = pc2;
     pc2 = pc1;
    }
 ps->egalites = pc2;
 pc2 = NULL;

 while (ps->inegalites != NULL)
    {
     pc1 = ps->inegalites;
     ps->inegalites = (ps->inegalites)->succ;
     pc1->succ = pc2;
     pc2 = pc1;
    }
 ps->inegalites = pc2;

 return(ps);
}


/*=======================================================================*/
/* bool vars_in_vect_p(Pvecteur pv, list vars):
 *
 * returns true if the vector "vec" contains at least one of the variables
 * of "vars". Else, returns FALSE.  */
bool vars_in_vect_p(pv, vars)
     Pvecteur pv;
     list vars;
{
  bool no_var = true;
  list lv;

  if(!VECTEUR_NUL_P(pv)) {
    for(lv = vars; !ENDP(lv) && no_var; POP(lv)) {
      entity cv = ENTITY(CAR(lv));

      if(value_notzero_p(vect_coeff((Variable) cv, pv)))
	no_var = false;
    }
  }
  return(! no_var);
}


/*====================================================================*/
/* expression merge_expressions(expression exp1, exp2, int max_or_min):
 *
 * Merges two expression into one using the MAX or MIN function
 *
 * For example, with expressions (N-1) and (M+2), and IS_MIN, we obtain
 * the following expression : MIN(N-1, M+2).
 *
 * Note : should be modified in order to avoid the MAX or MIN call when
 * the result can be known (AP) */
expression merge_expressions(exp1, exp2, max_or_min)
expression exp1, exp2;
int max_or_min;
{
  expression exp;
  entity op_ent = entity_undefined;

  if(max_or_min == IS_MAX)
    op_ent = entity_intrinsic("MAX");
  else if(max_or_min == IS_MIN)
    op_ent = entity_intrinsic("MIN");
  else
    user_error("merge_expressions", "Bad max or min tag\n");

  if(exp1 == expression_undefined)
    exp = exp2;
  else if(exp2 == expression_undefined)
    exp = exp1;
  else
    exp = make_expression(make_syntax(is_syntax_call,
				      make_call(op_ent,
						CONS(EXPRESSION, exp1,
						     CONS(EXPRESSION,
							  exp2, NIL)))),
			  normalized_undefined);
  return(exp);
}


/*====================================================================*/
/*
 * bool min_or_max_expression_p(expression exp)
 *
 * Returns true if this expression ("exp") is a call to the MAX or MIN
 * function. Else, returns FALSE.
 */
bool min_or_max_expression_p(exp)
expression exp;
{
  syntax sy;
  call ca;
  entity func;

  sy = expression_syntax(exp);
  if(syntax_tag(sy) != is_syntax_call)
    return(false);

  ca = syntax_call(sy);
  func = call_function(ca);
  
  return(ENTITY_MIN_OR_MAX_P(func));
}


/*========================================================================*/
/* list extract_bdt(b, cn):extract in the global bdt b the bdt of the
 * instruction of statement cn.
 *
 * AC 94/03/15
 */

list extract_bdt(b, cn)
bdt       b;
int       cn;
{
  list      lc = NIL, ba = bdt_schedules(b);
  schedule  sched;

  for (; !ENDP(ba); POP(ba)) {
    sched = SCHEDULE(CAR(ba));
    if (cn == schedule_statement(sched))
      ADD_ELEMENT_TO_LIST(lc, SCHEDULE, sched); 
  }
  
  return(lc);
}

/*=========================================================================*/
/* placement extract_plc(p, cn): extract in the global distribution p, the
 * placement function of statement "cn".
 *
 * AC 94/03/15
 */

placement extract_plc(p, cn)
plc       p;
int       cn;
{
  list      pa = plc_placements(p);
  placement pla = placement_undefined;
  
  for (; !ENDP(pa); POP(pa)) {
    pla = PLACEMENT(CAR(pa));
    if (cn == placement_statement(pla)) break;
  }

  if(pla == placement_undefined)
    user_error("extract_plc", "A plc is undefined\n");
  
  return(pla);
}

/*=========================================================================*/
/* entity create_new_entity(st, typ, nb): create a new entity with the
 * following form : "STAT_SYMstTYPnb" where st is the statement number.
 *
 * This new entity is put in the list of entities that have to be
 * declared.
 *
 * AC 94/03/15
 *
 * AP 95/04/20: entities must have in their full name the current module
 * name instead of RE_MODULE_NAME */

entity create_new_entity(st, typ, nb)
int     st, nb;
char    *typ;
{
  entity  ent, mod_entity;
  code mod_code;
  char    *name;
  string  f_name;

  /* Create the entity */
  name = (char*) malloc(32);
  sprintf(name,"%s%d%s%d", STAT_SYM, st, typ, nb);

  /* f_name = concatenate(RE_MODULE_NAME, MODULE_SEP_STRING, name, NULL); */
  f_name = concatenate(strdup(db_get_current_module_name()),
		       MODULE_SEP_STRING, name, NULL);
  free(name);

  ent = make_entity(strdup(f_name),
		    make_type(is_type_variable,
			      make_variable(make_basic(is_basic_int, 4),
					    NIL)),
		    make_storage(is_storage_rom, UU),
		    make_value(is_value_unknown, UU));
  
  /* Declare the entity */
  mod_entity = get_current_module_entity();
  mod_code = entity_code(mod_entity);
  code_declarations(mod_code) = gen_nconc(code_declarations(mod_code),
					  CONS(ENTITY, ent, NIL));

  return(ent);
}


/*=========================================================================*/
/* Psysteme change_base_in_sc(ps, lvar, lequ)
 *
 * Change in the system ps all variables given by lvar by the
 * corresponding equation of lequ. In fact, to each variable of lvar
 * corresponds an equality of lequ which gives its value. We then
 * substitute in ps each variable of lvar by its value given by lequ.
 *
 * The system ps is modified and its new value is returned.
 *
 * AC 94/03/22 */

Psysteme change_base_in_sc(ps, lvar, lequ)
Psysteme    ps, lequ;
list        lvar;
{
  entity      cvar;
  Pcontrainte cequ;
  list        le;

  /* There should be as many equalities in lequ as there are variables in
   * lvar.*/
  if (lequ->nb_eq != gen_length(lvar)) {
    if (get_debug_level() > 5) {
      fprintf(stderr,
	      "\nErreur dans le nombre d'equations ds chg_base()!\n");
      fprintf(stderr,"\nnb eq = %d",lequ->nb_eq);
      fprintf(stderr,"\nlvar = %d", gen_length(lvar));
    }
    user_error("change_base_in_sc",
	       "Erreur dans le nombre d'equations !\n");
  }
  else {
    cequ = lequ->egalites;
    for (le = lvar; le != NIL; POP(le), cequ = cequ->succ) {
	Value val;
	Pvecteur vec = vect_dup(cequ->vecteur);

      cvar = ENTITY(CAR(le));

      /* We have to substitute the occurrences of "cvar" in "ps" by its
       * counterpart in "cequ->vecteur", which represents the following
       * equality : cequ->vecteur = 0. We note it "vec" : val*cvar-vec=0,
       * so we have : vec = -(cequ->vecteur-val*cvar) */
      val = vect_coeff((Variable) cvar, vec);
      vect_erase_var(&vec, (Variable) cvar);
      vect_chg_sgn(vec);
      substitute_var_with_vec(ps, cvar, val, vec);
    }
  }
  
  ps->base = NULL;
  sc_creer_base(ps);
  
  return(ps);
}

/*=========================================================================*/
/* list find_new_variables(l1, l2): replace the n first variables of list
 * l1 by those of list l2 (gen_length(l2) = n).
 *
 * AC 93/03/31
 */

list find_new_variables(l1, l2)
list    l1, l2;
{
  int     n;
  entity  e;

  n = gen_length(l2);

  while (n != 0) {
    l1 = CDR(l1);
    n--;
  }

  while (l1 != NIL) {
    e = ENTITY(CAR(l1));
    ADD_ELEMENT_TO_LIST(l2, ENTITY, e);
    l1 = CDR(l1);
  }

  return(l2);
}


/*========================================================================*/
/* Psysteme my_clean_ps(ps):
 *
 * put nb_eq and nb_ineq at the rigth value.
 * 
 * AC 94/05/09
 */

Psysteme my_clean_ps(ps)
Psysteme     ps;
{
  int          count;
  Pcontrainte  cont;

  count = 0;
  cont = ps->egalites;
  while (cont != NULL) {
    count++;
    cont = cont->succ;
  }
  ps->nb_eq = count;

  count = 0;
  cont = ps->inegalites;
  while (cont != NULL) {
    count++;
    cont = cont->succ;
  }
  ps->nb_ineq = count;
  
  return(ps);
}


/*=========================================================================*/
/* bool is_vect_constant_p(v): test if a vecteur is constant.
 *
 * AC 94/07/01
 */

bool is_vect_constant_p(v)
Pvecteur v;
{
  if (VECTEUR_NUL_P(v))
    return(true);
  else {
    if ((v->var == TCST)&&(v->succ == NULL))
      return(true);
    else
      return(false);
  }
}

/*=========================================================================*/
/* bool cst_vector_p(Pvecteur v, Pbase b)
 *
 * Tests if a vector is constant, i.e. does not depend on one of the
 * variables given in b.
 *
 * AC 94/07/01 */

bool cst_vector_p(v, b)
Pvecteur v;
Pbase b;
{
  bool not_found = true;
  Pvecteur av;

  if (!VECTEUR_NUL_P(v)) {
    for(av = b; (av != NULL) && not_found; av = av->succ) {
      Variable cv = av->var;
      not_found = value_zero_p(vect_coeff(cv, v));
    }
  }
  return(not_found);
}


/*======================================================================*/
/* list remove_minmax(list le)
 *
 * Parameters:
 *
 * Result:
 *
 * AP 95/02/1 
 */
list remove_minmax(le)
list le;
{
  list res_l = NIL, lle;

  for(lle = le; !ENDP(lle); POP(lle)) {
    expression exp = EXPRESSION(CAR(lle));
    normalized nor;

    ifdebug(9) {
	pips_debug(9, "considering expression:\n");
	print_expression(exp);
    }

    nor = NORMALIZE_EXPRESSION(exp);

    if(normalized_tag(nor) == is_normalized_complex) {
      syntax sy = expression_syntax(exp);

      if(syntax_tag(sy) == is_syntax_call) {

	call ca = syntax_call(sy);
	if(ENTITY_MIN_OR_MAX_P(call_function(ca)))
	  res_l = remove_minmax(call_arguments(ca));
	else
	  pips_internal_error("A complex exp is not a call to MIN or MAX: %s",
		     words_to_string(words_expression(exp)));
      }
      else
	pips_internal_error("A complex exp is not a call : %s",
		   words_to_string(words_expression(exp)));
    }
    else
      ADD_ELEMENT_TO_LIST(res_l, EXPRESSION, exp);
  }

  return(res_l);
}


/*=======================================================================*/
/* bool array_ref_exp_p(e)
 *
 * Tests if an expression is an array, that is we test the field
 * "reference" of the expression which should not be empty.
 * 
 * AC 94/07/28 */

bool array_ref_exp_p(e)
expression   e;
{
  bool      b = false;

  if(syntax_reference_p(expression_syntax(e)))  {
    reference r = syntax_reference(expression_syntax(e));
    return(reference_indices(r) != NIL);
  }
  else
    pips_internal_error("expression syntax is not no reference !");

  /* Never reached. */
  return(b);
}
