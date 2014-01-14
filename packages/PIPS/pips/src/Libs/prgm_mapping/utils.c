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

#ifndef lint
char vcid_prgm_mapping_utils[] = "$Id$";
#endif /* lint */

/* Name     : utils.c
 * Package  : prgm_mapping
 * Author   : Alexis Platonoff
 * Date     : 23 september 1993
 *
 * Historic :
 * -  6 dec 93, some new functions, AP
 * - 14 nov 94, remove find_implicit_equation(), put in paf-util, AP
 *
 * Documents:
 *
 * Comments : This file contains useful functions used for the computation of
 * prgm_mapping.
 */

/* Ansi includes 	*/
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Newgen includes 	*/
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
#include "matrice.h"
#include "matrix.h"

/* Pips includes 	*/
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "graph.h"
#include "paf_ri.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"
#include "static_controlize.h"
#include "paf-util.h"
#include "prgm_mapping.h"

/* Useful constants */

/* Macro functions  	*/

/* Global variables 	*/
extern list prgm_parameter_l;

/* Internal variables 	*/

/* Local defines */
typedef dfg_vertex_label vertex_label;
typedef dfg_arc_label arc_label;


/* ======================================================================== */
/* list insert_sort(list l, bool (*compare_obj)()): returns the result of
 * sorting the list "l" using the comparison function "compare_obj". This
 * bool function should retuns true if its first argument has to be placed
 * before its second argument in the sorted list, else FALSE.
 *
 * This is a generic function that accepts any homogene list of newgen
 * objects. The comparison function must be coded by the user, its prototype
 * should be: bool my_compare_obj(chunk * obj1, chunk * obj2);
 *
 * This function uses the insert sort algorithm which has a mean and worst case
 * complexity of n^2.
 */
list insert_sort(l, compare_obj)
list l;
bool (*compare_obj)();
{
  list al, aal, nl = NIL, nnl, nnl_q;
  chunk * ngo, * aux_ngo;
  bool not_inserted;

  for(al = l; al != NIL; al = CDR(al)) {
    ngo = CHUNK(CAR(al));
    not_inserted = true;
    nnl = NIL;
    nnl_q = nnl;

    for(aal = nl; (aal != NIL) && not_inserted ; aal = CDR(aal)) {
       aux_ngo = CHUNK(CAR(aal));
       if(compare_obj(ngo, aux_ngo)) {
          nnl = gen_nconc(gen_nconc(nnl, CONS(CHUNK, ngo, NIL)), aal);
          not_inserted = false;
       }
       else
          nnl = gen_nconc(nnl, CONS(CHUNK, aux_ngo, NIL));
    }
    if(not_inserted)
       nnl = gen_nconc(nnl, CONS(CHUNK, ngo, NIL));

    nl = nnl;
  }

  return(nl);
}


/* ======================================================================== */
bool is_index_coeff_p(e)
entity e;
{
  if( (strncmp(entity_local_name(e), INDEX_COEFF, 1) == 0) ||
      (strncmp(entity_local_name(e), MU_COEFF, 1) == 0) )
    return(true);
  else
    return(false);
}


/* ======================================================================== */
/* bool compare_coeff(c1, c2):
 *
 * returns a bool saying true if "c1" is before "c2" in the lexicographic
 * order, else FALSE. "c1" and "c2" are entities, we compare their
 * name. */
bool compare_coeff(c1, c2)
chunk * c1, * c2;
{
  return strcmp(entity_local_name((entity) c1),
	 entity_local_name((entity) c2))<0;
}


/* ======================================================================== */
bool compare_nodes_dim(n1, n2)
chunk * n1, * n2;
{
  extern hash_table StmtToDim;

  return((int) hash_get(StmtToDim, (char *) vertex_int_stmt((vertex) n1)) >
	 (int) hash_get(StmtToDim, (char *) vertex_int_stmt((vertex) n2)) );
}


/* ======================================================================== */
bool compare_dfs_weight(d1, d2)
chunk * d1, * d2;
{
  extern hash_table DtfToWgh;

  return( (int) hash_get(DtfToWgh, (char *) d1) >
	  (int) hash_get(DtfToWgh, (char *) d2) );
}


/* ======================================================================== */
bool compare_unks_frenq(e1, e2)
chunk * e1, * e2;
{
  extern hash_table UnkToFrenq;

  return( (int) hash_get(UnkToFrenq, (char *) e1) >
	  (int) hash_get(UnkToFrenq, (char *) e2) );
}


/* ======================================================================== */
/* entity make_coeff(string prefix, int n): returns a new entity which
 * will be used as a coefficient in a prototype of a placement function. All
 * coefficients differ in their name which is something like:
 *	"MAPPING:prefix#"
 * where # is the integer value of "n".
 */
entity make_coeff(prefix, n)
string prefix;
int n;
{
 string num, name;
 entity new_coeff;

 num=i2a(n);

 name = strdup(concatenate(MAPPING_MODULE_NAME, MODULE_SEP_STRING,
                           prefix, num, (char *) NULL));

 new_coeff = make_entity(name,
                         make_type(is_type_variable,
                                   make_variable(make_basic(is_basic_int, 4),
                                                 NIL)),
                         make_storage(is_storage_rom, UU),
                         make_value(is_value_unknown, UU));

 free(num);
 return(new_coeff);
}


/* ======================================================================== */
/* entity find_or_create_coeff(string prefix, int n): returns the entity
 * that represent the coefficient numbered "n" and prefixed by "prefix". If it
 * does not exist yet, we create it.
 */
entity find_or_create_coeff(prefix, n)
string prefix;
int n;
{
 string num, name;
 entity new_coeff;

 num=i2a(n);
 name = strdup(concatenate(MAPPING_MODULE_NAME, MODULE_SEP_STRING,
                           prefix, num, (char *) NULL));
 new_coeff = gen_find_tabulated(name, entity_domain);

 /* We create it, if it does not exist yet */
 if(new_coeff == entity_undefined)
    new_coeff = make_coeff(prefix, n);

 free(num);
 return(new_coeff);
}


/* ======================================================================== */
list rm_non_x_var(l)
list l;
{
  list prec, aux_l;

if(get_debug_level() > 6) {
fprintf(stderr, "\n[rm_non_x_var] Bases de depart :\n");
fprint_entity_list(stderr, l);
fprintf(stderr, "\n");
}

  prec = NIL;
  for(aux_l = l; !ENDP(aux_l); POP(aux_l)) {
    entity crt_var = ENTITY(CAR(aux_l));
    if(!is_index_coeff_p(crt_var)) {
      if(prec == NIL)
	l = CDR(l);
      else
	CDR(prec) = CDR(aux_l);
    }
    else
      prec = aux_l;
  }

if(get_debug_level() > 6) {
fprintf(stderr, "\n[rm_non_x_var] Base d'arrivee :\n");
fprint_entity_list(stderr, l);
fprintf(stderr, "\n");
}

  return(l);
}


/* ======================================================================== */
list unify_lists(l1, l2)
list l1, l2;
{
  list l;
  
  if(l1 == NULL)
    l = gen_append(l2, NIL);
  else {
    l = gen_append(l1, NIL);

    for(; !ENDP(l2); POP(l2)) {
      chunk * c2 = CHUNK(CAR(l2));
      bool found = false;
      for(; (!ENDP(l1)) && (!found); POP(l1)) {
        chunk *c1 = CHUNK(CAR(l1));
        if(c1 == c2)
	  found = true;
      }
      if(! found)
        gen_nconc(l, CONS(CHUNK, c2, NIL));
    }
  }

  return(l);
}


/* ======================================================================== */
/* Psysteme new_elim_var_with_eg(Psysteme ps, list *init_l, list *elim_l):
 * Computes the gaussian elimination of variables in the system "ps". The
 * modifications are directly done on "ps".
 *
 * However, we keep all the eliminated equations in "sc_elim", and return that
 * system.
 *
 * Initially, "init_l" gives the list of variables that can be eliminated, at
 * the end, it only contains the variables that were not eliminated. We take
 * the order of this list in our process of eliminating.
 *
 * Initially, "elim_l" is empty, at the end it contains the variables that were
 * eliminated.
 *
 */
Psysteme new_elim_var_with_eg(ps, init_l, elim_l)
Psysteme ps;
list *init_l, *elim_l;
{
  Psysteme sc_elim = sc_new();

  /* During the computation, we modify *init_l, so we duplicate it.
   * We use "el" not *elim_l, which should be  empty at the beginning.
   */
  list vl = gen_append(*init_l, NIL),
       el = NIL,         
       l;
  Pcontrainte eq, eg;

  for(l = vl; !ENDP(l); POP(l)) {
    Variable v = (Variable) ENTITY(CAR(l));
    Value coeff;

    if ((eq = contrainte_var_min_coeff(ps->egalites,v, &coeff, true))
	!= NULL) {

 if(get_debug_level() > 7) {
fprintf(stderr, "System is :");
fprint_psysteme(stderr, ps);
fprintf(stderr, "\t\tElim var %s in equation:", entity_local_name((entity) v));
pu_vect_fprint(stderr, eq->vecteur);
fprintf(stderr, "\n");
 }

      if(!egalite_normalize(eq))
        pips_internal_error("Egalite bizarre");

      sc_nbre_egalites(ps)--;
      if (eq == (ps->egalites)) ps->egalites = eq->succ;
      /* si eq etait en tete il faut l'enlever de la liste, sinon, eq a
         ete enleve dans la fonction contrainte_var_min_coeff(). */

      for(eg = ps->egalites; eg != NULL; eg = eg->succ)
        (void) contrainte_subst_ofl_ctrl(v, eq, eg, true, NO_OFL_CTRL);
      for(eg = ps->inegalites; eg != NULL; eg = eg->succ)
        (void) contrainte_subst_ofl_ctrl(v, eq, eg, false, NO_OFL_CTRL);

      for(eg = sc_elim->egalites; eg != NULL; eg = eg->succ)
        (void) contrainte_subst_ofl_ctrl(v, eq, eg, true, NO_OFL_CTRL);
      for(eg = sc_elim->inegalites; eg != NULL; eg = eg->succ)
        (void) contrainte_subst_ofl_ctrl(v, eq, eg, false, NO_OFL_CTRL);

      sc_add_egalite(sc_elim, eq);
      gen_remove(init_l, (chunk *) v);
      el = CONS(ENTITY, (entity) v, el);
    }
  }

  *elim_l = el;
  sc_elim->base = NULL;
  sc_creer_base(sc_elim);

 if(get_debug_level() > 6) {
fprintf(stderr, "[new_elim_var_with_eg] Results:\n");
fprintf(stderr, "Elim sys:\n");
fprint_entity_list(stderr, el);
fprint_psysteme(stderr, sc_elim);
fprintf(stderr, "Remnants sys:\n");
fprint_entity_list(stderr, *init_l);
fprint_psysteme(stderr, ps);
fprintf(stderr, "\n");
 }

  return(sc_elim);
}


/* ======================================================================== */
/* Psysteme plc_elim_var_with_eg(Psysteme ps, list *init_l, list *elim_l):
 * Computes the gaussian elimination of variables in the system "ps". The
 * modifications are directly done on "ps".
 *
 * However, we keep all the eliminated equations in "sc_elim", and return that
 * system.
 *
 * Initially, "init_l" gives the list of variables that can be eliminated, at
 * the end, it only contains the variables that were not eliminated. We take
 * the order of this list in our process of eliminating.
 *
 * Initially, "elim_l" is empty, at the end it contains the variables that were
 * eliminated.
 *
 */
Psysteme plc_elim_var_with_eg(ps, init_l, elim_l)
Psysteme ps;
list *init_l, *elim_l;
{
 bool var_not_found;
 Psysteme sc_elim = sc_new();
 Pcontrainte eqs;
 Pvecteur ve, pv_elim;
 entity crt_var = entity_undefined;
 int crt_val = 0;
 list vl = *init_l,     /* We use "vl" during the computation, not *init_l */
      el = NIL,         /* We use "el" during the computation, not *elim_l */
      l;

 /* This elimination works only on equalities. While there remains equalities,
  * we can eliminate variables.
  */
 eqs = ps->egalites;
 while(eqs != NULL)
   {
    ve = eqs->vecteur;

 if(get_debug_level() > 8) {
fprintf(stderr, "System is :");
fprint_psysteme(stderr, ps);
fprintf(stderr, "\t\tConsidered Vect :");
pu_vect_fprint(stderr, ve);
fprintf(stderr, "\n");
 }

    /* We look, in vl (i.e. init_l), for a variable that we can eliminate in
     * ve, i.e. with a coefficient equal to 1 or -1.
     */
    var_not_found = true;
    for(l = vl ; (l != NIL) && var_not_found; l = CDR(l)) {
      crt_var = ENTITY(CAR(l));
      crt_val = (int) vect_coeff((Variable) crt_var, ve);
      if((crt_val == 1) || (crt_val == -1))
	var_not_found = false;
    }

    /* If we get such a variable, we eliminate it. */
    if(! var_not_found)
      {
       Pvecteur init_vec;

       /* First, we remove it from "vl". */
       gen_remove(&vl, (chunk *) crt_var);

       /* Then, we add it to "el". */
       el = CONS(ENTITY, crt_var, el);

       /* We keep a copy of the initial vector. */
       init_vec = vect_dup(eqs->vecteur);

       /* We compute the expression (pv_elim) by which we are going to
        * substitute our variable (var):
        *
        * We have: var = pv_elim
        *
        * The equality is: V = 0, with: V = val*var + Vaux, and: val in {-1,1}
        *
        * So, we have: pv_elim = -val*Vaux, with: Vaux = V - val*var
        *
        * So: pv_elim = -val(V - val*var)
        * i.e.: pv_elim = -val+V + (val^2)*var
        * but: val in {-1,1}, so: val^2 = 1
        *
        * so: pv_elim = -val*V + var
        *
        */
       pv_elim = vect_cl2_ofl_ctrl((crt_val)*(-1), eqs->vecteur,
				   1, vect_new((Variable) crt_var, 1),
				   NO_OFL_CTRL);

 if(get_debug_level() > 7) {
fprintf(stderr, "\t\tElim with %s = ", entity_local_name(crt_var));
pu_vect_fprint(stderr, pv_elim);
fprintf(stderr, "\n");
 }

       /* We substitute var by its value (pv_elim) in "ps". */
       my_substitute_var_with_vec(ps, crt_var, 1, vect_dup(pv_elim));

       /* We substitute var by its value (pv_elim) in "sc_elim". */
       my_substitute_var_with_vec(sc_elim, crt_var, 1, vect_dup(pv_elim));

       ps = sc_normalize(ps);
       vect_rm(pv_elim);

 if(get_debug_level() > 7) {
fprintf(stderr, "New System is :");
fprint_psysteme(stderr, ps);
 }

       /* The initial equality is added to "sc_elim". */
       sc_add_egalite(sc_elim, contrainte_make(init_vec));


       /* We reinitialize the list of equalities. */
       eqs = ps->egalites;
      }
    /* Else, we try on the next equality. */
    else
       eqs = eqs->succ;
   }
 *init_l = vl;
 *elim_l = el;
 sc_elim->base = NULL;
 sc_creer_base(sc_elim);

 if(get_debug_level() > 6) {
fprintf(stderr, "[plc_elim_var_with_eg] Results:\n");
fprintf(stderr, "Elim sys:\n");
fprint_entity_list(stderr, el);
fprint_psysteme(stderr, sc_elim);
fprintf(stderr, "Remnants sys:\n");
fprint_entity_list(stderr, vl);
fprint_psysteme(stderr, ps);
fprintf(stderr, "\n");
 }

 return(sc_elim);
}


/* ======================================================================== */
/* int new_count_implicit_equation(Psysteme ps): returns the number of implicit
 * equations there are in the system "ps".
 *
 * Practically, we construct a system containing all the implicit equations,
 * then we count the number of equations (there should be no redondant
 * equation since find_implicit_equation() normalizes its result).
 *
 * See the description of find_implicit_equation().
 */
int count_implicit_equation(ps)
Psysteme ps;
{
  Psysteme impl_ps;

  if(ps == NULL)
    return(0);

  impl_ps = find_implicit_equation(ps);

  /* If impl_ps is empty, then no data are communicated */
  if(impl_ps == NULL)
    return(ps->nb_ineq + ps->nb_eq);

  if(get_debug_level() > 6) {
    fprintf(stderr, "Number of equations in Implicit system : %d\n", impl_ps->nb_eq);
  }
  return(impl_ps->nb_eq);
}


/* ======================================================================== */
/* list put_source_ind(list le): returns a list of expressions computed from
 * the list given in argument ("le").
 *
 * We want to associate a variable (var) to each expression (exp) in order to
 * have some kind of equality : 0 = exp - var
 * which means in fact : var = exp
 *
 * The name of the variables are not important, however, we need a different
 * variable for each expression.
 *
 * Note: the variables are entities.
 */
list put_source_ind(le)
list le;
{
 int count = 1; /* This counter makes sure that each new variable is different
                 * for the preceding one.
		 */
 expression texp, new_texp;
 entity en;
 list l,
      new_le = NIL;	/* The resulting list is initialized to NIL. */

 /* For each expression we compute the new expression. */
 for(l = le; l != NIL; l = CDR(l), count++)
   {
    normalized nor;

    /* We get a new variable (an entity). */
    en = find_or_create_coeff(INDEX_VARIA, count);

    texp = EXPRESSION(CAR(l));
    NORMALIZE_EXPRESSION(texp);
    nor = expression_normalized(texp);

    if(normalized_tag(nor) == is_normalized_linear) {
      Pvecteur pv = normalized_linear(nor);
      pv = vect_cl_ofl_ctrl(pv, -1, vect_new((Variable) en, 1), NO_OFL_CTRL);
      new_texp = make_vecteur_expression(pv);
    }
    else {
      /* We make the new expression : exp - var. */
      new_texp = make_op_exp(MINUS_OPERATOR_NAME,
			     texp, make_entity_expression(en, NIL));
    }

    /* This expression is added to the new list. The order is kept. */
    new_le = gen_nconc(new_le, CONS(EXPRESSION, new_texp, NIL));
   }
 return(new_le);
}


/* ======================================================================== */
/* Ppolynome apply_farkas(Ppolynome F, Psysteme D,
 *			  list *L, int *count_lambdas)
 * returns the affine from of farkas lemma computed from the polynome "F"
 * (the affine form) and the domain of its variables "ps_dom".
 *
 * Farkas Lemma:
 * -------------
 * Let D be a non empty polyhedron defined by n affine inequalities:
 * 	Ik(x) >= 0, k in 1,..,n
 *
 * Then an affine form F is a non negative everywhere in D iff it is a
 * positive affine combination of the above inequalities:
 *	F(x) >= 0 in D <=> F(x) == L0 + L1.I1(x) + .. + Ln.In(x)
 *      with Lk >= 0, k in 1,..,n
 *
 * Here, *L" is the list (L0,..,Ln).
 * If we note P(x) = L0 + L1.I1(x) + .. + Ln.In(x), the polynome returned
 * is the Farkas polynome: FP = F(x) - P(x)
 *
 * We also have to add the structure parameters that do not appear in D and
 * that are in the global var "prgm_parameter_l". For instance, if "p"
 * (in "prgm_parameter_l") does not appear in D, we'll have:
 *	P(x) = L0 + L1.I1(x) + .. + Ln.In(x) + Ln+1.p
 * and "*L" will be (L0,..,Ln,Ln+1)
 *
 * In the following code, variable "P" is the opposite of our P(x). In doing so,
 * we'll be able to simplified the final calculus (F(x) - P(x)) by an addition.
 */

typedef bool (*argh)(Pvecteur*, Pvecteur*);

Ppolynome apply_farkas(F, D, L, count_lambdas)
Ppolynome F;
Psysteme D;
list * L;
int *count_lambdas;
{
  Ppolynome FP, pp_v, P, last_P;
  int cl = *count_lambdas;
  entity nl;
  list local_l = NIL, params_in_D, l, ll;
  Pcontrainte pc;

  nl = find_or_create_coeff(LAMBD_COEFF, cl++);
  local_l = CONS(ENTITY, nl, local_l);

  /* Constant term: L0 */
  P = make_polynome(-1.0, (Variable) nl, 1);
  last_P = P;

  /* We associate to each equation or inequation a different coefficient, so
   * the addition is never simplified, it just put a few monomes to the end
   * of "P", i.e. "last_P". We do not used "polynome_add()" because this
   * function looks for similar monomes.
   */
  if(D != NULL) {
    for(pc = D->inegalites; pc != NULL; pc = pc->succ) {
      Pvecteur pv = pc->vecteur;

      nl = find_or_create_coeff(LAMBD_COEFF, cl++);
      local_l = CONS(ENTITY, nl, local_l);

      pp_v = vecteur_mult(pv, vect_new((Variable) nl, 1));

      for(last_P->succ = pp_v; last_P->succ != NULL; last_P = last_P->succ) {}
    }

    /* An equality (v == 0) is also two inequalities (v >= 0, v <= 0) */
    for(pc = D->egalites; pc != NULL; pc = pc->succ) {
      Pvecteur pv = pc->vecteur;

      nl = find_or_create_coeff(LAMBD_COEFF, cl++);
      local_l = CONS(ENTITY, nl, local_l);

      pp_v = vecteur_mult(pv, vect_new((Variable) nl, 1));

      for(last_P->succ = pp_v; last_P->succ != NULL; last_P = last_P->succ) {}

      nl = find_or_create_coeff(LAMBD_COEFF, cl++);
      local_l = CONS(ENTITY, nl, local_l);

      pp_v = vecteur_mult(pv, vect_new((Variable) nl, -1));

      for(last_P->succ = pp_v; last_P->succ != NULL; last_P = last_P->succ) {}
    }
  }

  /* We add to (-P) the parameters that do not appear in D */
  /* pu_is_inferior_var is not implemented and does not match prototype... */
  params_in_D = base_to_list(polynome_used_var(P, (argh) pu_is_inferior_var));
  for(l = gen_append(prgm_parameter_l, NIL); !ENDP(l); POP(l)) {
    bool not_found = true;
    entity p = ENTITY(CAR(l));
    for(ll = params_in_D; !ENDP(ll) && (not_found); POP(ll)) {
      if( same_entity_p(p, ENTITY(CAR(ll))) )
	not_found = false;
    }
    if(not_found) {
      nl = find_or_create_coeff(LAMBD_COEFF, cl++);
      local_l = CONS(ENTITY, nl, local_l);

      pp_v = vecteur_mult(vect_new((Variable) nl, -1),
			  vect_new((Variable) p, 1));

      for(last_P->succ = pp_v; last_P->succ != NULL; last_P = last_P->succ) {}
    }
  }

  /* FP = F + (-P)
   * We are sure that F-P can not be simplified, i.e. that there are similar
   * monomes; this is because each monome of P has a variable that never
   * appear outside of P.
   */
  FP = P;
  last_P->succ = F;

  *L = local_l;
  *count_lambdas = cl;
  return(FP);
}


/* ======================================================================== */
/* list get_graph_dataflows(graph g): returns the list of dataflows of the DFG
 * graph "g". Each edge of "g" contains a list of dataflows, so we concatenate
 * these lists into one.
 */
list get_graph_dataflows(g)
graph g;
{
  list l, su_l, g_dfs;

  g_dfs = NIL;
  for(l = graph_vertices(g); !ENDP(l); POP(l)) {
    for(su_l = vertex_successors(VERTEX(CAR(l))); !ENDP(su_l); POP(su_l)) {
      list dfs = gen_append(SUCC_DATAFLOWS(SUCCESSOR(CAR(su_l))), NIL);
      g_dfs = gen_nconc(g_dfs, dfs);
    }
  }

  return(g_dfs);
}


/* ======================================================================== */
/*
 * list get_stmt_index_coeff(int stmt, int n, hash_table StoL): returns the
 * index coeff from the coeff used in the prototype of statement "stmt".
 *
 * The hash table "StoL" maps a statement number to the lists of coeff used
 * for the prototype. When this list is gotten, we have to remove the non index
 * coeff.
 * An index coeff is recognized with is_index_coeff_p().
 */
list get_stmt_index_coeff(stmt, StoL)
int stmt;
hash_table StoL;
{
  list proto_lambda, prec, pl;

  proto_lambda = gen_append((list) hash_get(StoL, (char *) stmt), NIL);

  prec = NIL;
  for(pl = proto_lambda; !ENDP(pl); POP(pl)) {
    entity e = ENTITY(CAR(pl));
    if(is_index_coeff_p(e))
      prec = pl;
    else
      if(prec == NIL)
        proto_lambda = CDR(proto_lambda);
      else
        CDR(prec) = CDR(pl);
  }

  return(proto_lambda);
}


/* ======================================================================== */
/*
 * Psysteme completer_base(Psysteme sys, list var_l, list par_l): "sys" gives
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
 */
Psysteme completer_base(sys, var_l, par_l)
Psysteme sys;
list var_l, par_l;
{
  Psysteme ps = sc_dup(sys), new_ps = sc_new();
  int dim = gen_length(var_l) - sys->nb_eq;
  list l;

  for(l = var_l; (!ENDP(l)) && (new_ps->nb_eq < dim); POP(l)) {
    entity var = ENTITY(CAR(l));
    Pvecteur pv = vect_new((Variable) var, 1);
    Psysteme aux_ps = sc_dup(ps);
    Psysteme aux_new_ps = sc_dup(new_ps);

    sc_add_egalite(aux_new_ps, contrainte_make(pv));
    aux_ps = append_eg(aux_ps, aux_new_ps);

    if(vecteurs_libres_p(aux_ps, list_to_base(var_l), list_to_base(par_l))) {
      new_ps = aux_new_ps;
    }
    else
      sc_rm(aux_new_ps);
  }
  ps = append_eg(ps, new_ps);
  ps->base = NULL;
  sc_creer_base(ps);
  return(ps);
}


/* ======================================================================== */
/*
 * list diff_lists(list l1, list l2): return the difference between two lists.
 *
 * Example: the diff between (e1, e2, e3) and (e2, e4) is (e1, e3).
 */
list diff_lists(l1, l2)
list l1, l2;
{
  list l = NIL;

  if(l2 == NULL)
    return(l1);

  for( ; !ENDP(l1); POP(l1)) {
    chunk *c1 = CHUNK(CAR(l1));
    bool found = false;
    for(; (!ENDP(l2)) && (!found); POP(l2)) {
      chunk *c2 = CHUNK(CAR(l2));
      if(c1 == c2)
        found = true;
    }
    if(!found)
      l = gen_nconc(l, CONS(CHUNK, c1, NIL));
  }
  return(l);
}


/* ======================================================================== */
/*
 * bool vecteurs_libres_p(Psysteme sys, Pbase v_base, Pbase c_base):
 * returns true if "sys" contains a list of equalities that can represent a
 * family of free vectors.
 * "v_base" is the list unit vectors, and "c_base" is the list of symbolic
 * constants.
 *
 * Example: (i,j) are unit vectors, n is a symbolic constant
 *		{i + n == 0, i - n == 0} is not a free family
 *              {i + j == 0, i - j == 0} is a free family
 */
bool vecteurs_libres_p(sys, v_base, c_base)
Psysteme sys;
Pbase v_base, c_base;
{
  int n, m1, m2, r;
  Value det_p, det_q;
  matrice A, B, P, H, Q;

  n = sys->nb_eq;
  m1 = base_dimension(v_base);

  if(m1 < n)
    return(false);

  m2 = base_dimension(c_base) + 1; /* add 1, because of the TCST */
  A = matrice_new(n,m1);
  B = matrice_new(n,m2);

  contraintes_with_sym_cst_to_matrices(sys->egalites,v_base,c_base,
				       A,B,n,m1,m2);

  P = matrice_new(n,n);
  Q = matrice_new(m1,m1);
  H = matrice_new(n,m1);
  matrice_hermite(A, n, m1, P, H, Q, &det_p, &det_q);

  r = matrice_hermite_rank(H, n, m1);

  matrice_free(A);
  matrice_free(B);
  matrice_free(P);
  matrice_free(Q);
  matrice_free(H);

  return(r == n);
}


/* ======================================================================== */
/*
 * Psysteme append_eg(M1, M2): returns a system containing all the equations of
 * "M1" and all the equations of "M2". The order is kept, i.e. "M1" and "M2".
 */
Psysteme append_eg(M1, M2)
Psysteme M1, M2;
{
  Pcontrainte pc1, pc2, prec;
  Pvecteur pv;

  if(M1 == NULL)
    return(M2);
  if(M2 == NULL)
    return(M1);

  pc1 = M1->egalites;
  pc2 = M2->egalites;
  for(prec = NULL; pc1 != NULL; pc1 = pc1->succ)
    prec = pc1;

  if(prec == NULL)
    return(M2);

  prec->succ = pc2;
  M1->nb_eq += M2->nb_eq;

  pv = M1->base;
  M1->base = NULL;
  vect_rm(pv);
  sc_creer_base(M1);

  return(M1);
}

/* ======================================================================== */
/*
 * void di_polynome_var_subst_null(Ppolynome *pp, entity var)
 */
void di_polynome_var_subst_null(pp, var)
Ppolynome *pp;
entity var;
{
  Ppolynome ppp, p = POLYNOME_UNDEFINED;
  Pvecteur pv;

  for(ppp = *pp; ppp != NULL; ppp = ppp->succ) {
    entity first = entity_undefined,
           second = entity_undefined;
    pv = (ppp->monome)->term;
    for(; (pv != NULL) && (second == entity_undefined); pv = pv->succ) {
      second = first;
      first = (entity) pv->var;
    }
    if(pv != NULL)
      pips_internal_error("Vecteur should contains 2 var");
    else if(same_entity_p(first, var) || same_entity_p(second, var)) {
      if(POLYNOME_UNDEFINED_P(p)) {
	*pp = ppp->succ;
      }
      else
        p->succ = ppp->succ;
    }
    else
      p = ppp;
  }
}


/* ======================================================================== */
/*
 * Psysteme nullify_factors(Ppolynome *pp, list var_l, bool with_remnants):
 * returns a system of equalities ("new_ps") computed from a polynome "pp"
 * and a list of variables "var_l".
 *
 * This list gives the variables of the polynome for which we need to nullify
 * the factor. Thus, the resulting system contains the equations that nullify
 * these factors (the degree of the polynome must be less or equal to two).
 *
 * If "with_remnants" is true, then the remaining polynome, from which we
 * have removed all the occurences of these variables, is also nullify and the
 * equation added to the system (then, these remnants must be of degree 1).
 *
 * Note: "pp" is modified, it contains at the end these remnants.
 */
Psysteme nullify_factors(pp, var_l, with_remnants)
Ppolynome *pp;
list var_l;
bool with_remnants;
{
  Ppolynome aux_pp = *pp;
  Psysteme new_ps = sc_new();
  list l;

  if(get_debug_level() > 4) {
    fprintf(stderr, "[nullify_factors] polynome is :");
    polynome_fprint(stderr, aux_pp, pu_variable_name, pu_is_inferior_var);
    fprintf(stderr, "\n");
  }

  if(!POLYNOME_NUL_P(aux_pp)) {

    /* For each variable, we nullify its factor in the polynome. */
    for(l = var_l; l != NIL; l = CDR(l)) {
      /* We get the current variable. */
      entity var = ENTITY(CAR(l));

      /* We get its factor in the polynome. */
      Pvecteur pv_fac = prototype_factorize(aux_pp, (Variable) var);

      if(get_debug_level() > 5) {
	fprintf(stderr, "[nullify_factors] factor is :");
	pu_vect_fprint(stderr, pv_fac);
	fprintf(stderr, "\n");
      }

      if(!VECTEUR_NUL_P(pv_fac)) {
	/* We add a new equality in the system. */
	sc_add_egalite(new_ps, contrainte_make(pv_fac));

	/* We delete the occurences of this variable in the polynome. */
	aux_pp = prototype_var_subst(aux_pp, (Variable) var, POLYNOME_NUL);
      }
    }

    if( (with_remnants) && (!POLYNOME_NUL_P(aux_pp)) )
      /* The remnants are zero out and are added as one equation. */
      sc_add_egalite(new_ps, polynome_to_contrainte(aux_pp));

    if(get_debug_level() > 4) {
      fprintf(stderr, "[nullify_factors] final new sys :");
      fprint_psysteme(stderr, new_ps);
      fprintf(stderr, "\n");
    }

    sc_creer_base(new_ps);
    sc_normalize(new_ps);
    *pp = aux_pp;
  }
  return(new_ps);
}


/* ======================================================================== */
/* bool is_broadcast_p(dataflow df): returns true if the dataflow "df"
 * has a defined broadcast communication. Otherwise it returns FALSE.
 */
bool is_broadcast_p(df)
dataflow df;
{
  communication com;
  predicate bp;
  Psysteme ps;

  com = dataflow_communication(df);
  if(com == communication_undefined)
    return(false);

  bp = communication_broadcast(com);
  if(bp == predicate_undefined)
    return(false);

  ps = (Psysteme) predicate_system(bp);
  if(ps == NULL)
    return(false);
  else
    return(true);
}


/* ======================================================================== */
/* int communication_dim(dataflow df): returns the number of directions
 * vectors of the dataflow if it a broadcast or a reduction. Oterwise,
 * returns 0.
 */
int communication_dim(df)
dataflow df;
{
 communication com;
 predicate bp, rp;
 Psysteme ps;
 Pcontrainte pc;
 int dim = 0;

 com = dataflow_communication(df);
 if(com == communication_undefined)
    return(0);

 bp = communication_broadcast(com);
 rp = communication_reduction(com);
 if((bp == predicate_undefined) && (rp == predicate_undefined))
    return(0);

 if(bp == predicate_undefined)
   ps = (Psysteme) predicate_system(rp);
 else
   ps = (Psysteme) predicate_system(bp);

 for(pc = ps->egalites; pc != NULL; pc = pc->succ)
   dim++;

 return(dim);
}


/* ======================================================================== */
/* bool is_reduction_p(dataflow df): returns true if the dataflow "df"
 * has a defined reduction communication. Otherwise it returns FALSE.
 */
bool is_reduction_p(df)
dataflow df;
{
 communication com = dataflow_communication(df);
 predicate bp;
 Psysteme ps;

 if(com == communication_undefined)
    return(false);

 bp = communication_reduction(com);
 if(bp == predicate_undefined)
   return(false);

 ps = (Psysteme) predicate_system(bp);
 if(ps == NULL)
   return(false);
 else
   return(true);
}


/* ======================================================================== */
/* bool is_shift_p(dataflow df): returns true if the dataflow "df" has a
 * defined shift communication. Otherwise it returns FALSE.
 */
bool is_shift_p(df)
dataflow df;
{
 communication com = dataflow_communication(df);
 predicate bp;
 Psysteme ps;

 if(com == communication_undefined)
    return(false);

 bp = communication_shift(com);
 if(bp == predicate_undefined)
   return(false);

 ps = (Psysteme) predicate_system(bp);
 if(ps == NULL)
   return(false);
 else
   return(true);
}


/*============================================================================*/
/* void my_substitute_var_with_vec(Psysteme ps, entity var, int val, Pvecteur vec):
 * Substitutes in a system ("ps") a variable ("var"), factor of a positive
 * value ("val"), by an expression ("vec").
 *
 * This substitution is done on all assertions of the system (equalities and
 * inequalities). For each assertion (represented by a vector Vold) we have:
 *
 *      Vold = c*var + Vaux
 *      val*var = vec
 *
 * Vnew represents the new assertion.  With: p = gcd(c, val) >= 1, we have:
 *
 *	Vnew = (c/p)*vec + (val/p)*Vaux = (c/p)*vec + (val/p)*(Vold - c*var)
 *
 * Note: we have: Vold == 0 <=> (val/p)*Vold == 0
 *                Vold > 0 <=> (val/p)*Vold > 0
 *                ...
 *
 * because "val" is positive.
 */
void my_substitute_var_with_vec(ps, var, val, vec)
Psysteme ps;
entity var;
int val;
Pvecteur vec;
{
 Variable Var = (Variable) var;
 Pcontrainte assert;

 if(get_debug_level() > 8) {
fprintf(stderr, "\t\t\tAvant Sub: \n");
fprint_psysteme(stderr, ps);
fprintf(stderr, "\n");
 }

  /* If we want to substitute a NULL vector, we just erase "Var" */
  if(VECTEUR_NUL_P(vec)) {
    for(assert = ps->egalites; assert != NULL; assert = assert->succ) {
      Pvecteur v_old = assert->vecteur;
      vect_erase_var(&v_old, Var);
    }
    for(assert = ps->inegalites; assert != NULL; assert = assert->succ) {
      Pvecteur v_old = assert->vecteur;
      vect_erase_var(&v_old, Var);
    }
  }

  /* "val" must be positive. */
  if(val < 0) {
    val = 0-val;
    vect_chg_sgn(vec);
  }

  /* Vnew = (c/p)*vec + (val/p)*Vaux = (c/p)*vec + (val/p)*(Vold - c*var) */
  for(assert = ps->egalites; assert != NULL; assert = assert->succ) {
    Pvecteur v_old = assert->vecteur;
    int coeff = vect_coeff(Var, v_old);
    if(coeff != 0) {
      int p = pgcd(coeff, val);

      assert->vecteur = vect_cl2_ofl_ctrl(coeff/p, vec,
					  val/p,
					  vect_cl2_ofl_ctrl(1, v_old, -1,
							    vect_new(Var,
								     coeff),
							    NO_OFL_CTRL),
					  NO_OFL_CTRL);
    }
  }
  for(assert = ps->inegalites; assert != NULL; assert = assert->succ) {
    Pvecteur v_old = assert->vecteur;
    int coeff = vect_coeff(Var, v_old);
    if(coeff != 0) {
      int p = pgcd(coeff, val);

      assert->vecteur = vect_cl2_ofl_ctrl(coeff/p, vec,
					  val/p,
					  vect_cl2_ofl_ctrl(1, v_old, -1,
							    vect_new(Var,
								     coeff),
							    NO_OFL_CTRL),
					  NO_OFL_CTRL);
    }
  }
  vect_rm((Pvecteur) ps->base);
  ps->base = (Pbase) NULL;
  sc_creer_base(ps);

 if(get_debug_level() > 8) {
fprintf(stderr, "\t\t\tApres Sub: \n");
fprint_psysteme(stderr, ps);
fprintf(stderr, "\n");
 }

}


/* ======================================================================== */
/*
 * Pvecteur old_prototype_factorize(Ppolynome pp, Variable var)
 */
Pvecteur old_prototype_factorize(pp, var)
Ppolynome pp;
Variable var;
{
  Pvecteur pv = NULL;

  if(POLYNOME_NUL_P(pp))
    pv = VECTEUR_NUL;
  else if(var == TCST)
    pv = vect_new(TCST, (int) polynome_TCST(pp));
  else {
    Ppolynome ppp;

    for(ppp = pp; ppp != NULL; ppp = ppp->succ) {
      Variable newvar = VARIABLE_UNDEFINED;
      int newval;
      Pvecteur vec, newpv;
      entity first = entity_undefined, second = entity_undefined;
      bool factor_found = true;

      vec = (ppp->monome)->term;
      for(; (vec != NULL) && (second == entity_undefined); vec = vec->succ) {
        second = first;
        first = (entity) vec->var;
      }
      if(vec != NULL)
        pips_internal_error("Vecteur should contains 2 var");
      else if(same_entity_p(first,  (entity) var))
	if(second == entity_undefined)
	  newvar = TCST;
        else
	  newvar = (Variable) second;
      else if(same_entity_p(second, (entity) var))
	newvar = (Variable) first;
      else
	factor_found = false;

      if(factor_found) {
        newval = (int) (ppp->monome)->coeff;
        newpv = vect_new(newvar, newval);
        newpv->succ = pv;
        pv = newpv;
      }
    }
  }

  return(pv);
}

