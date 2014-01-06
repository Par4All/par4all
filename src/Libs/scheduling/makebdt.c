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
#include <setjmp.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"

/* C3 includes          */
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
#include "sparse_sc.h"

/* Pips includes        */
#include "complexity_ri.h"
#include "database.h"
#include "dg.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "tiling.h"
/* GO:
#include "loop_normalize.h"
*/


#include "text.h"
#include "text-util.h"
#include "graph.h"
#include "paf_ri.h"
#include "paf-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "array_dfg.h"
#include "pip.h"
#include "static_controlize.h"
#include "scheduling.h"

/* Local defines */
typedef dfg_arc_label arc_label;
typedef dfg_vertex_label vertex_label;

#define my_polynome_fprint(fp,p) \
   polynome_fprint(fp,p,entity_local_name,default_is_inferior_var);

hash_table h_node;

typedef struct n_coef {
	  entity           n_ent;
	  Pvecteur         n_vect;
	  struct n_coef    *next;
		      } n_coef, *Pn_coef;

typedef struct bdt_node {
          Ppolynome        n_poly;
          bdt              n_bdt;
          Pn_coef          n_var;
                        } bdt_node,*Pbdt_node;

typedef struct sys_list {
          int              nb;
          Psysteme         sys;
          struct sys_list  *next;
                        } sys_list, *Psys_list;

/*======================================================================*/
/* entity create_var_name(type,source,dest,count) : create the variable
 * to iut in the vector that we construct in "create_farkas_poly".
 * Returns an entity for printing commmodities.
 *
 * AC 93/11/28
 */

entity create_var_name(Type, source, dest, count)

 char    *Type;
 int     source;
 int     dest;
 int     count;
{
 char    *name;

 asprintf(&name, "%s_%d_%d_%d", Type, source, dest, count);

 entity e = (create_named_entity(name));
 free(name);
 return e;
}

/*======================================================================*/
/* Psys_list add_elt_to_sys_list(ls, s, ps): add a system in the list
 * of Psysteme.
 *
 * AC 94/01/27
 */

static Psys_list add_elt_to_sys_list(ls, s, ps)

 Psys_list    ls;
 int          s;
 Psysteme     ps;
{
 Psys_list    aux, la, lb;

 aux = (Psys_list)malloc(sizeof(sys_list));
 aux->nb = s;
 aux->sys = sc_dup(ps);
 aux->next = NULL;

 if (ls == NULL) ls = aux;
 else
    {
     lb = ls;
     while (lb != NULL) 
        {
         la = lb;
         lb = lb->next;
        }
     la->next = aux;
    }

 return(ls);
}

/*======================================================================*/
/* void fprint_sys_list(fp,ls): print in the file fp the list of systems
 * ls.
 *
 * AC 94/01/27
 */

static void fprint_sys_list(fp,ls)

 FILE       *fp;
 Psys_list  ls;
{
 for (; ls != NULL; ls = ls->next)
   {
    fprintf(fp,"\n");
    fprint_psysteme(fp, ls->sys);
    fprintf(fp,"\n");
   }
}

/*======================================================================*/
/* Pn_coef make_n_coef(e,v): allocating the space for a Pn_coef, that is
 * a cell containing an entity and a Pvecteur, and returns a Pn_coef.
 *
 * AC 93/11/15
 */

static Pn_coef make_n_coef(e, v)

 entity      e;
 Pvecteur    v;
{
 Pn_coef     aux;

 aux = (Pn_coef)malloc(sizeof(n_coef));
 aux->n_ent = e;
 aux->n_vect = v;
 aux->next = NULL;

 return (aux);
}

/*======================================================================*/
/* Pn_coef add_n_coef_to_list(l1,l2): add the n_coef l2 at the end of
 * the list l1.
 *
 * AC 93/11/15
 */

static Pn_coef add_n_coef_to_list(l1, l2)

 Pn_coef     l1, l2;
{
 Pn_coef     p = l1;

 if (p == NULL) l1 = l2;
 else
    {
     while (p->next != NULL) p = p->next;
     p->next = l2;
    }

 return(l1);
}

/*======================================================================*/
/* void fprint_coef_list(fp,l): print in the file fp the list of Pn_coef
 * l.
 *
 * AC 93/11/15
 */

static void fprint_coef_list(fp,l)
 
 FILE      *fp;
 Pn_coef   l;
{
 while (l != NULL)
    {
     fprintf(fp,"Entite : ");
     fprintf(fp,"%s",entity_local_name(l->n_ent));
     fprintf(fp,", Vecteur : ");
     pu_vect_fprint(fp,l->n_vect);
     l = l->next;
    }
}

/*==================================================================*/
/* Pn_coef add_lcoef_to_lcoef(l1, l2): appends the list of Pn_coef l2
 * to l1.
 *
 * AC 93/11/16
 */

static Pn_coef add_lcoef_to_lcoef(l1, l2)

 Pn_coef    l1, l2;
{
 Pn_coef    l;

 while (l2 != NULL)
    {
     l = l2;
     l2 = l2->next;
     l->next = NULL;
     l1 = add_n_coef_to_list(l1, l);
    }
 return (l1);
}

/*==================================================================*/
/* list extract_lambda_list(l): from the list l, extract the sublist
 * of entities whose name is beginning by "LAMBDA".
 *
 * AC 93/11/15
 */

list extract_lambda_list(l)

 list    l;
{
 entity  ent;
 char    *name;
 list    laux = NIL;

 while (l != NIL)
    {
     ent = ENTITY(CAR(l));
     name = entity_local_name(ent);
     if (!strncmp(name, "LAMBDA", 6))
         ADD_ELEMENT_TO_LIST(laux, ENTITY, ent);
     l = CDR(l);
    }

 return (laux);
}

/*==================================================================*/
/* list reorder_base(l,lu) : reorder the base list l with the
 * unknows lu contains in first positions in the list.
 *
 * AC 93/11/17
 */

list reorder_base(l, lu)

 list    l, lu;
{
 list    laux = NIL;
 entity  ent;

 for (; lu != NIL; lu = CDR(lu))
    {
     ent = ENTITY(CAR(lu));
     ADD_ELEMENT_TO_LIST(laux, ENTITY, ent);
    }

 for (; l != NIL; l = CDR(l))
    {
     ent = ENTITY(CAR(l));
     if (!(is_entity_in_list_p(ent, laux)))
        ADD_ELEMENT_TO_LIST(laux, ENTITY, ent);
    }
 return(laux);
}

/*==================================================================*/
/* Pn_coef add_lambda_list(lunk, lvar): lvar contains the lambda list
 * that we transform in a list of Pn_coef that we append to lunk.
 *
 * AC 93/11/15
 */

static Pn_coef add_lambda_list(lunk, lvar)

 Pn_coef    lunk;
 list       lvar;
{
 Pn_coef    aux;

 while (lvar != NIL)
    {
     aux = make_n_coef(ENTITY(CAR(lvar)), VECTEUR_NUL);
     lunk = add_n_coef_to_list(lunk, aux);
     lvar = CDR(lvar);
    }

 return (lunk);
}

/*==================================================================*/
/* list make_x_list(c): build a list of entity of type X_nb where
 * nb is a number between 0 and c.
 *
 * AC 94/01/27
 */

list make_x_list(c)
 
 int      c;
{
 entity   ent;
 char     *name;
 list     laux = NIL;
 int      i;

 name = (char*) malloc(10);

 for (i = 0; i <= c; i++)
    {
     sprintf(name, "X_%d", i);
     ent = create_named_entity(name);
     ADD_ELEMENT_TO_LIST(laux, ENTITY, ent);
    }
 free(name);

 return(laux);
}

/*==================================================================*/
/* Pn_coef add_x_list(lunk, lvar, me): create the list of Pn_coef
 * for the X, the second member is the big parameter we will use
 * for PIP.
 *
 * AC 94/01/27
 */

static Pn_coef add_x_list(lunk, lvar, me)

 Pn_coef    lunk;
 list       lvar;
 entity     *me;
{
 Pn_coef    aux, laux = NULL;
 Pvecteur   vect_m;

 *me = create_named_entity("My_Own_Private_M");
 vect_m = vect_new((Variable)(*me), -1);

 while (lvar != NIL)
    {
     aux = make_n_coef(ENTITY(CAR(lvar)), vect_m);
     laux = add_n_coef_to_list(laux, aux);
     lvar = CDR(lvar);
    }
 lunk = add_lcoef_to_lcoef(laux, lunk);

 return(lunk);
}

/*==================================================================*/
/* list extract_var_list(lc): extract from the list of Pn_coef lc
 * the variables and put them in a list it returns.
 *
 * AC 93/11/15
 */

static list extract_var_list(lc)

 Pn_coef    lc;
{
 list       l = NIL;

 for ( ; lc != NULL; lc = lc->next)
     ADD_ELEMENT_TO_LIST(l, ENTITY, lc->n_ent);

 return(l);
}

/*==================================================================*/
/* void clean_list_of_unk(l, sc): update the list of unknown for the
 * problem by erasing the unknown that have been already calculated.
 *
 * AC 94/01/27
 */

static void clean_list_of_unk(l, sc)

 Pn_coef     *l;
 Psysteme    *sc;
{
 Pn_coef     lout, laux, lin = *l;
 list        base;

 lout = NULL;
 base = base_to_list((*sc)->base);

 while (lin != NULL)
    {
     laux = lin;
     lin = lin->next;
     laux->next = NULL;
     if (is_entity_in_list_p(laux->n_ent, base))
        lout = add_n_coef_to_list(lout, laux);
    }

 base = extract_var_list(lout);
 (*sc)->base = list_to_base(base);

 *l = lout;
}

/*==================================================================*/
/* list make_list_of_n(n,c) : function that creates a list of new
 * entities as "u1" or "u2" it returns.
 *
 * AC 93/11/16
 */

list make_list_of_n(n, c)

 char    *n;
 int     c;
{
 entity  ent;
 list    l = NIL;
 char    *name;
 int     i;

 name = (char*) malloc(10);

 for (i = 1; i <= c; i++)
    {
     sprintf(name, "%s%d", n, i);
     ent = create_named_entity(name);
     ADD_ELEMENT_TO_LIST(l, ENTITY, ent);
    }
 free(name);
 return(l);
}

/*======================================================================*/
/* bool is_stat_in_pred_list(stat, pred_list): tests if the stat is
 * in the list pred_list.
 *
 * AC 93/12/06
 */

bool is_stat_in_pred_list(stat, pred_list)

 int      stat;
 list     pred_list;
{
    list     l;
    bool  bool = false;
    int      s;

    l = pred_list;
    
    if (l == NIL)
	bool = false;
    else {
	for ( ; l != NIL; l = CDR(l)) {
	    s = INT(CAR(l));
	    if (s == stat)
		bool = true;
        }
    }
    return(bool);
}

/*======================================================================*/
/* Psysteme include_parameters_in_sc(sys,source): include in the systeme
 * "sys" the parameters that it does not contain at that moment, and 
 * put them under an inequality (for example: n>=0).
 *
 * AC 94/01/05
 */

Psysteme include_parameters_in_sc(sys, source)

 Psysteme        sys;
 int             source;
{
 list            lent;
 Variable        var;
 Pvecteur        vect;
 Pcontrainte     cont;
 static_control  stct;

 stct = get_stco_from_current_map(adg_number_to_statement(source));
 lent = static_control_params(stct);

 if (sys == NULL) sys = sc_new();

 for (; lent != NIL; lent = CDR(lent))
    {
     var = (Variable)ENTITY(CAR(lent));

     if (!system_contains_var(sys, var))
        {
         vect = vect_new(var, (Value)-1);
         cont = contrainte_make(vect);
         sc_add_inegalite(sys, cont); 
	 sys->base = base_add_variable(sys->base, var);
	 sys->dimension++;
        }
    }

 return(sys);
}

/*======================================================================*/
/* create_farkas_poly() : that function creates the farkas polynom of a
 * node, from the execution domain "psys". The coefficients created have
 * the form NAME_i_j_k with i the statement of the source node, j the
 * statement of the destination node (=0 in the case of a MU polynom)
 * and k the number of the coefficient. The name NAME is given by Type.
 * The polynome created is put in "p", each coefficient created is
 * put in the list of Pn_coef "l" with the vector he is the coef. of,
 * (0 in case of a Lambda coef), and "c" is a counter of tthe created
 * coefficient.
 *
 * AC 93/10/28
 */
 
static void create_farkas_poly(Psys, Type, source, dest, p, l, c)

 Psysteme       Psys;
 char           *Type;
 int            source, dest;
 Ppolynome      *p;
 Pn_coef        *l;
 int            *c;
{
 int            i, count = *c;
 Pcontrainte    cont;
 Pvecteur       vect, vect_aux;
 Ppolynome      poly, poly_aux;
 entity         ent;
 Pn_coef        coef;

 poly = *p;

 if (get_debug_level() > 5)
    fprintf(stderr,"\nXXX creer equation farkas XXX\n");

 if (POLYNOME_UNDEFINED_P(poly))
    {
     /* creation of the constant term of the polynom */
     ent = create_var_name(Type, source, dest, count);
     vect = vect_new((Variable)TCST, (Value)1);
     coef = make_n_coef(ent, vect);
     *l = add_n_coef_to_list(*l, coef);
     poly = make_polynome((float)1, (Variable)ent, (Value)1);
     count++;

     if (!SC_UNDEFINED_P(Psys))
        {
         sc_transform_eg_in_ineg(Psys);
         cont = Psys->inegalites;

         /* exploring the Psystem, i.e. each inegality of the execution */
         /* domain, and making each component                           */
         for (i = 0 ; i < Psys->nb_ineq ; i++)
            {
             vect = vect_dup(cont->vecteur);
             ent = create_var_name(Type, source, dest, count);

             vect_aux = vect_new((Variable)ent, (Value)1);
             vect_chg_sgn(vect);

             coef = make_n_coef(ent, vect);
             *l = add_n_coef_to_list(*l, coef);

             poly_aux = vecteur_mult(vect, vect_aux);
             polynome_add(&poly, poly_aux);

             count++;
             cont = cont->succ;
            }
        }
    }
 else
    {
     if (!SC_UNDEFINED_P(Psys))
        {
         sc_transform_eg_in_ineg(Psys);
         cont = Psys->inegalites;

         /* exploring the Psystem, i.e. each inegality of the execution */
         /* domain, and making each component                           */
         for (i = 0 ; i< Psys->nb_ineq ; i++)
            {
             vect = vect_dup(cont->vecteur);
             ent = create_var_name(Type, source, dest, count);

             vect_aux = vect_new((Variable)ent, (Value)1);
	     vect_chg_sgn(vect);

             coef = make_n_coef(ent, vect);
             *l = add_n_coef_to_list(*l, coef);

             poly_aux = vecteur_mult(vect, vect_aux);  
	     polynome_add(&poly, poly_aux);

             count++;
             cont = cont->succ;
            }
        }
    }

 *p = poly;

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nP[%d] = ",source);
     my_polynome_fprint(stderr, poly);
     fprintf(stderr,"\n");
    }

 *c = count;
}

/*==================================================================*/
/* void make_proto((hash_table)h, (sccs)rg): function that goes through
 * the reverse graph "rg", and for each node of each scc, initialize
 * the node structure associated with each node, that is, creates the
 * Farkas polynom of each node.
 *
 * AC 93/10/28
 */

static void make_proto(h, rg)

 hash_table   h;
 sccs         rg;
{
 list         lscc, lver;
 Pbdt_node    this_node;
 Ppolynome    this_poly;
 Psysteme     this_sys;
 int          this_stat, count;
 vertex       this_ver;
 Pn_coef      lvar;

 for (lscc = sccs_sccs(rg); lscc != NIL; lscc = CDR(lscc))
    {
     scc scc_an = SCC(CAR(lscc));
     for (lver = scc_vertices(scc_an); lver != NIL; lver = CDR(lver))
        {
         lvar = NULL;
	 this_poly = POLYNOME_UNDEFINED;
         this_ver  = VERTEX(CAR(lver));
         this_stat = dfg_vertex_label_statement((dfg_vertex_label)\
                                     vertex_vertex_label(this_ver));
         this_sys  = predicate_system(dfg_vertex_label_exec_domain(\
                     (dfg_vertex_label)vertex_vertex_label(this_ver)));
         this_sys  = include_parameters_in_sc(this_sys, this_stat);
         count = 0;
         create_farkas_poly(this_sys, "MU", this_stat, 0, &this_poly,\
			    &lvar, &count);
         this_node = (Pbdt_node)malloc(sizeof(bdt_node));
         this_node->n_poly = this_poly;
         this_node->n_bdt  = (bdt)NIL;
         this_node->n_var = lvar;

         hash_put(h, (char *) this_stat, (char *) this_node); 
        }
    }
}

/*==================================================================*/
/* build_bdt_null(v): update the hash table by making the
 * schedule null
 *
 * AC 93/10/29
 */

bdt build_bdt_null(v)

 vertex     v;
{
 Pbdt_node  node;
 int        stat;
 schedule   sche;
 expression exp;
 list       lexp, lsche;
 bdt        b;

 stat = dfg_vertex_label_statement((dfg_vertex_label)vertex_vertex_label(v));
 node = (Pbdt_node)hash_get(h_node, (char *) stat);
 exp  = int_to_expression(0);

 lexp = CONS(EXPRESSION, exp, NIL);
 sche = make_schedule(stat, predicate_undefined, lexp); 
 lsche = CONS(SCHEDULE, sche, NIL);
 b = make_bdt(lsche);
 node->n_bdt = true_copy_bdt(b);

 if (get_debug_level() > 5) fprint_bdt(stderr,node->n_bdt);

 return(b);
}

/*==================================================================*/
/* if_no_pred(s, b): that function tests if the scc studied has
 * only 1 vertex and no predecessor. Returns true in this case. At
 * the same time, this function updates the field schedule of the
 * hash table with the proper schedule, that is time=0
 *
 * AC 93/10/29
 */

bool if_no_pred(s, b) 

 scc      s;
 bdt      *b;
{
 list     lver_a, lver_b, lsucc;
 vertex   v;

 lver_a = scc_vertices(s);
 lver_b = CDR(lver_a);
 v = VERTEX(CAR(lver_a));
 lsucc = vertex_successors(v);

 if ((lver_b == NIL) && (lsucc == NIL)) 
   {
    *b = build_bdt_null(v);
    return(true);
   }
   else  return(false) ;
}

/*==================================================================*/
/* Psysteme erase_trivial_ineg(p): erase from the psysteme p all
 * inequalities like -MU<=0.
 *
 * AC 93/11/22
 */

Psysteme erase_trivial_ineg(p)

 Psysteme        p;
{
 Pcontrainte     cont;
 Pvecteur        vect;

 if (!SC_UNDEFINED_P(p))
    {
     for (cont = p->inegalites ; cont != NULL ; cont = cont->succ)
        {
         vect = cont->vecteur;
         if ((vect != NULL)&&value_negz_p(vect->val)&&(vect->succ == NULL))
	    {
             vect_rm(cont->vecteur);
	     cont->vecteur = NULL;
            }
	}
    }

 return(p);
}

/*==================================================================*/
/* Ppolynome include_trans_in_poly((int)s,(Ppolynome)p,(list)l):
 * list contains the list of expression defining the edge trans-
 * formation we want to apply to the node of statement s.
 * This function replaces the old variable values by their new
 * values in the polynome p. This is done in 2 steps; first we replace
 * each variable by a local one,and put this one in a list; then we
 * replace the local variable by the expression of the transformation.
 *
 * AC 93/11/03
 */


Ppolynome include_trans_in_poly(s, p, l, d)

 int             s, *d;
 Ppolynome       p;
 list            l;
{
  list            lindice, lexp, lind, lvar, lv;
  static_control  stct;
  char            *name;
  expression      exp;
  Ppolynome       poly_trans;
  int             count = 0, den = 1;
  Variable        var, v;
  entity          ent;

  if ((p != (Ppolynome)NIL)||(l != NIL))
  {
    /* we get the list of the englobing loop of the studied node */
    stct = get_stco_from_current_map(adg_number_to_statement(s));
    lindice = static_control_to_indices(stct);

    lvar = NIL;
    name = (char*) malloc(100);

    /* replace all variables by a local one */
    for (lind = lindice; lind != NIL; lind = CDR(lind))
    {
      var = (Variable)ENTITY(CAR(lind));
      sprintf(name, "myownprivatevariable_%d", count);
      ent = create_named_entity(name);
      
      if (polynome_contains_var(p, var))
	poly_chg_var(p, var, (Variable)ent);
      ADD_ELEMENT_TO_LIST(lvar, ENTITY, ent);
      count++;
    }
    free(name);

    /* replace all local variables by the transformation */
    lexp = l;

    for (lv = lvar; lv != NIL; lv = CDR(lv))
    {
      exp = EXPRESSION(CAR(lexp));
      analyze_expression(&exp, &den);
      poly_trans = expression_to_polynome(exp);
      v = (Variable)ENTITY(CAR(lv));
      if (polynome_contains_var(p, v))
	p = prototype_var_subst(p, v, poly_trans); 
      lexp = CDR(lexp);
    }

    gen_free_list(lindice);
    gen_free_list(lvar);
  }

  *d = den;

  return(p);
}

/*==================================================================*/
/* Psysteme transform_in_ineq((Psysteme)sc,(list)l):change a system of 
 * equalities in a system of inequalities, and at the same time, try
 * to eliminate some lambda variables that are useless.
 * 
 * AC 93/11/08
 */


Psysteme transform_in_ineq(sc, l)

 Psysteme         sc;
 list             l;
{
 Psysteme         Psyst = sc_dup(sc);
 Pvecteur         v;
 Pcontrainte      c;
 Variable         var;
 Value            val;

 /* in each vector, isolate a lambda variable if it exists */
 c = Psyst->egalites;

 while ((c != NULL) && (l != NIL))
    {
     var = (Variable)ENTITY(CAR(l));
     v = c->vecteur;
       
     if (base_contains_variable_p((Pbase)v, var))
	{
         val = vect_coeff(var, v);
         if (value_negz_p(val))  vect_chg_sgn(c->vecteur);
         vect_erase_var(&c->vecteur, var);
        }
     c = c->succ;
     l = CDR(l);
    }

 /* transform each equality in an inequality */
 sc_elim_empty_constraints(Psyst, true);
 sc->inegalites = Psyst->egalites;
 Psyst->egalites = sc->egalites;
 Psyst->nb_eq = sc->nb_eq;
 sc->nb_ineq = sc->nb_eq;
 sc->nb_eq = 0;
 sc->egalites = NULL;
 
 sc_rm(Psyst);
 sc->base = NULL;
 sc_creer_base(sc);

 return(sc);
}

/*==================================================================*/
/* int get_m_coef(e, de): get the coefficient of the variable "m"
 * in the expression "e".
 *
 * AC 94/01/20
 */

int get_m_coef(e, de)

 expression    *e;
 int           *de;
{
    int d, mc = 0;
 Pvecteur      vect;
 expression    exp, e_aux = *e;
 entity        ent;

 analyze_expression(&e_aux, &d);

 if (d == 1)
    {
     NORMALIZE_EXPRESSION(e_aux);
     vect = normalized_linear(expression_normalized(e_aux));

     while (!VECTEUR_NUL_P(vect))
        {
         if (vect->var != TCST)
            {
             ent = (entity)vect->var;
             if (!strncmp(entity_local_name(ent), "My_Own_Private_M", 16))
                 mc = VALUE_TO_INT(vect->val);
            }
         vect = vect->succ;
        }
    }
 else
    {
     exp = EXPRESSION(CAR(call_arguments(syntax_call\
                                     (expression_syntax(e_aux)))));
     NORMALIZE_EXPRESSION(exp);
     vect = normalized_linear(expression_normalized(exp));

     while (!VECTEUR_NUL_P(vect))
        {
         if (vect->var != TCST)
            {
             ent = (entity)vect->var;
             if (!strncmp(entity_local_name(ent), "My_Own_Private_M", 16))
                 mc = VALUE_TO_INT(vect->val);
            }
         vect = vect->succ;
        }
     mc/=d;
    }

 unnormalize_expression(e_aux);

 *de = d;
 *e = e_aux;

 return (mc);
}

/*=======================================================================*/
/* bool coef_of_M_equal(exp1, exp2): tests if two expressions have the
 * same coefficient of "M".
 *
 * AC 94/02/08
 */

bool coef_of_M_equal(exp1, exp2)

 expression  *exp1, *exp2;
{
 expression  e1 = *exp1, e2 = *exp2;
 int         d1, d2, m1, m2;

 m1 = get_m_coef(&e1, &d1);
 m2 = get_m_coef(&e2, &d2);

 return( !(m1-m2) );
}

/*=======================================================================*/
/* bool list_of_exp_equals_1n_p(l1,l2,n): tests the equality of 2 lists 
 * of expressions on the first n terms.
 *
 * AC 94/01/25
 */

bool list_of_exp_equals_1n_p(l1, l2, n)

 list        l1, l2;
 int         n;
{
 bool     is_equal = true;
 expression  exp1, exp2;
 int         i;

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nPremiere liste: ");
     fprint_list_of_exp(stderr, l1);
     fprintf(stderr,"\nDeuxieme liste: ");
     fprint_list_of_exp(stderr, l2);
    }

 exp1 = EXPRESSION(CAR(l1));
 exp2 = EXPRESSION(CAR(l2));
 is_equal = exp_equals_p(exp1, exp2);
 l1 = CDR(l1);
 l2 = CDR(l2);

 for (i = 1; (i <= (n-1))&&(is_equal) ; i++)
    {
     exp1 = EXPRESSION(CAR(l1));
     exp2 = EXPRESSION(CAR(l2));
     is_equal = coef_of_M_equal(&exp1, &exp2);
     l1 = CDR(l1);
     l2 = CDR(l2);
    }

 return(is_equal);
}

/*=======================================================================*/
/* quast compact_quast(q, n): try to reduce a quast when it is a condi-
 * tional quast by testing the possible equality between the false edge
 * and the true edge, and if it exists, suppress one.
 *
 * AC 94/01/03
 */

quast compact_quast(q, n)

 quast           q;
 int             n;
{
 quast           tq, fq;
 quast_value     qv;
 conditional     cond;
 list            lf, lt;

 if (q == quast_undefined)         return (q);

 qv = quast_quast_value(q);
 if (qv == quast_value_undefined)  return(q);

 if (quast_value_quast_leaf_p(qv)) return(q);
 else
    {
     cond = quast_value_conditional(qv);
     tq = conditional_true_quast(cond);
     fq = conditional_false_quast(cond);
  
     /* may be possible source of bug here */
     if (quast_undefined_p(tq) || \
         quast_value_undefined_p(quast_quast_value(tq)))
	return(fq);

     else if (quast_undefined_p(fq) ||\
	 quast_value_undefined_p(quast_quast_value(fq)))
	return(tq);

     else if (quast_value_quast_leaf_p(quast_quast_value(tq)) &&\
         quast_value_quast_leaf_p(quast_quast_value(fq)))
        {
         lt = quast_leaf_solution(quast_value_quast_leaf\
                                           (quast_quast_value(tq)));
         lf = quast_leaf_solution(quast_value_quast_leaf\
                                           (quast_quast_value(fq)));

         if (list_of_exp_equals_1n_p(lt, lf, n))
            {
             free_quast(fq);
             return(tq);
            }
         else return(q);
        }
     else
        {
         tq = compact_quast(tq, n);
         fq = compact_quast(fq, n);

         if (quast_value_quast_leaf_p(quast_quast_value(tq)) &&\
             quast_value_quast_leaf_p(quast_quast_value(fq)))
            {
             lt = quast_leaf_solution(quast_value_quast_leaf\
                                               (quast_quast_value(tq)));
             lf = quast_leaf_solution(quast_value_quast_leaf\
                                               (quast_quast_value(fq)));
             if (list_of_exp_equals_1n_p(lt, lf, n))
                {
                 q = tq;
                 free_quast(fq);
                 return(q);
                }
             else
                {
                 conditional_true_quast(cond) = tq;
                 conditional_false_quast(cond) = fq;
                 return(q);
                }
            }
         else
            {
             conditional_true_quast(cond) = tq;
             conditional_false_quast(cond) = fq;
             return(q);
            }
        }
    }
}

/*==================================================================*/
/* list get_list_of_all_param(s, t): get the entire list of the parame-
 * ters for a deisganted node.
 *
 * AC 94/02/08
 */

list get_list_of_all_param(s, t)

 int             s, t;
{
 static_control  stct;
 list            l, m;
 entity          ent;

 stct = get_stco_from_current_map(adg_number_to_statement(s));
 l = static_control_params(stct);
 l = gen_append(l, static_control_to_indices(stct));

 stct = get_stco_from_current_map(adg_number_to_statement(t));
 m = static_control_params(stct);
 m = gen_append(m, static_control_to_indices(stct));

 for (; m != NIL; m = CDR(m))
    {
     ent = ENTITY(CAR(m));
     if (!is_entity_in_list_p(ent, l))
        l = gen_append(l, CONS(ENTITY, ent, NIL));
    }

 return(l);
}

/*==================================================================*/
/* bool is_uniform_rec(l, p): test if a causality condition is
 * linear, that is independent of the structure parameters and all
 * the loop counters.
 *
 * AC 93/11/22
 */

bool is_uniform_rec(l, p)

 list            l;
 Ppolynome       p;
{
 list            lindice = l;
 Variable        var;
 bool         bool = false;

 for ( ; lindice != NIL; lindice = CDR(lindice))
    {
     var = (Variable)ENTITY(CAR(lindice));
     bool = (polynome_contains_var(p, var)) || (bool);
    }

 return(!bool);
}

/*==================================================================*/
/* Psysteme include_trans_in_sc(s,sys,l): include in a system the
 * indice transformation introduced by an edge.
 *
 * AC 93/12/20
 */

Psysteme include_trans_in_sc(s, sys, l)

 int          s;
 Psysteme     sys;
 list         l;
{
 Pcontrainte  cont;
 Pvecteur     vect;
 Ppolynome    poly;
 int          d;

 for (cont = sys->inegalites; cont != NULL; cont = cont->succ)
    {
     vect = cont->vecteur;
     poly = vecteur_to_polynome(vect);
     poly = include_trans_in_poly(s, poly, l, &d);
     if (d != 1) cont->vecteur = vect_multiply(polynome_to_vecteur(poly), d);
     else cont->vecteur = polynome_to_vecteur(poly);
    }

 sys->base = BASE_UNDEFINED;
 sc_creer_base(sys);

 return(sys);
}

/*==================================================================*/
/* Ppolynome make_polynome_Xe(xc, xe): make the polynom of unknown
 * xe = "X_xc".
 *
 * AC 94/01/27
 */

Ppolynome make_polynome_Xe(xc, xe)

 int        xc;
 entity     *xe;
{
 Ppolynome  p;
 entity     e;
 char       *n;

 asprintf(&n, "X%d", xc);
 e = create_named_entity(n);
 free(n);

 p = make_polynome((float)1, (Variable)e, 1);

 *xe = e;
 return(p);
}

/*==================================================================*/
/* list add_others_variables(lvar, n_lvar): check if the elements of
 * n_lvar are in lvar, if not add them in lvar.
 *
 * AC 94/02/10
 */

list add_others_variables(lvar, n_lvar)

 list    lvar, n_lvar;
{
 entity  ent;

 for (; n_lvar != NULL; n_lvar = CDR(n_lvar))
    {
     ent = ENTITY(CAR(n_lvar));
     if (!is_entity_in_list_p(ent, lvar))
	lvar = gen_append(lvar, CONS(ENTITY, ent, NIL));
    }

 return(lvar);
}

/*==================================================================*/
/* Psysteme make_causal_internal(): build the causality condition.
 * If the causality condition is already linear, we do not need to
 * apply Farkas lemma to create a Lambda polynom. This function 
 * identifies the different variables and writes them in a Psystem
 * it returns. The integer c is usefull for the creation of the
 * parameters LAMBDA in the case of a multi-edge. This function
 * is used in the case of an internal edge of the scc where we
 * introduce the variable "switch" Xe. The causality condition
 * is written as:
 *     Pdest - Pinit >= Xe
 *
 * AC 94/01/27
 */

Psysteme make_causal_internal(stat_dest, sys_dest, poly_dest, ldata,\
                              stat_pred, sys_pred, poly_source,\
			      c, xc, xe, den)

 int           stat_dest, stat_pred;
 Psysteme      sys_dest, sys_pred;
 Ppolynome     poly_dest, poly_source;
 int           *c, xc, den;
 list          ldata;
 entity        *xe;
{
 list          ltrans, lvar, llambda, llambda_el;
 predicate     pred_edge;
 Psysteme      psys_aux, sys_edge, pred_sys = SC_RN;
 Ppolynome     poly_lambda, poly_res, p_source, poly_x;
 dataflow      pred_data;
 Pn_coef       llam1, llam2;
 entity        xe_a;
 int           co = *c, d, ppc;

 lvar = get_list_of_all_param(stat_dest, stat_pred);
 llam1 = NULL;
 llam2 = NULL;
 xe_a = *xe;
 sys_edge = SC_RN;
 psys_aux = sc_new();
 poly_lambda = POLYNOME_UNDEFINED;
 poly_res = POLYNOME_UNDEFINED;
 p_source = polynome_dup(poly_source);
 pred_sys = sc_dup(sys_pred);

 pred_data = DATAFLOW(CAR(ldata));
 ltrans = dataflow_transformation(pred_data);

 p_source = include_trans_in_poly(stat_pred, p_source, ltrans, &d);

 ppc = sol_ppcm(den, d);

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nTRANSFORMATION :\n");
     fprint_list_of_exp(stderr,ltrans);
     fprintf(stderr,"\n\nPoly_source : ");
     my_polynome_fprint(stderr,p_source);
    }

 /* write the causality condition under the following form: */
 /* P_dest - P_source -Xe = 0                               */
 poly_res = polynome_dup(poly_dest);
 polynome_negate(&p_source);
 polynome_add(&poly_res, p_source);
 poly_x = make_polynome_Xe(xc, &xe_a);
 polynome_negate(&poly_x);
 polynome_add(&poly_res, poly_x);

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nInequation de causalite :\n");
     my_polynome_fprint(stderr,poly_res);
    }

 if (is_uniform_rec(lvar, poly_res))
    {
     /* case of a uniform causality condition */
     if (get_debug_level() > 5) fprintf(stderr,"\nRecurrence uniforme :\n");
     polynome_negate(&poly_res);
     psys_aux = polynome_to_sc(poly_res, lvar);
     psys_aux = transform_in_ineq(psys_aux, NIL);
    }
 else
    {
     /* make the polynom of unknown lambda */
     sys_edge = sc_dup(sys_dest);
     sys_edge = sc_append(sys_edge, pred_sys);
     sys_edge = include_parameters_in_sc(sys_edge, stat_dest);
     my_sc_normalize(sys_edge);

     /* list of the variables to identify */
     if (!SC_UNDEFINED_P(sys_edge))
	lvar = add_others_variables(lvar, base_to_list(sys_edge->base));

     create_farkas_poly(sys_edge, "LAMBDA", stat_pred, stat_dest,\
                                          &poly_lambda, &llam1, &co);

     /* Now we add the conditions on the edge, caus' we livin' on */
     /* the edge                                                  */
     pred_edge = dataflow_governing_pred(pred_data);
     sys_edge = sc_dup(predicate_to_system(pred_edge));

     /* list of the variables to identify */
     if (!SC_UNDEFINED_P(sys_edge))
	lvar = add_others_variables(lvar, base_to_list(sys_edge->base));

     create_farkas_poly(sys_edge, "LAMBDA", stat_pred, stat_dest,\
                                          &poly_lambda, &llam2, &co);

     sc_rm(sys_edge);

     llam1 = add_lcoef_to_lcoef(llam1, llam2);
     llambda = extract_var_list(llam1);

     /* write the causality condition under the following form: */
     /* P_dest - P_source - P_lambda -1 = 0                     */
     polynome_negate(&poly_lambda);
     polynome_add(&poly_res, poly_lambda);

     if (ppc != 1) polynome_scalar_mult(&poly_res, (float)ppc); 

     psys_aux = polynome_to_sc(poly_res, lvar);
     psys_aux = elim_var_with_eg(psys_aux, &llambda, &llambda_el);

     /* simplify the system and transform it in inequalities */
     psys_aux = my_sc_normalize(psys_aux);
     llambda_el = gen_nreverse(llambda_el);
     psys_aux = transform_in_ineq(psys_aux, llambda_el);
    }

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nPoly_resultat : ");
     my_polynome_fprint(stderr,poly_res);
     fprintf(stderr,"\n\n");
    }

 polynome_rm(&poly_res);
 polynome_rm(&p_source);

 *xe = xe_a;
 *c = co;

 return(psys_aux);
}

/*==================================================================*/
/* Psysteme make_causal_external(): same function as make_causal
 * internal, but in this case we do not introduce the new variables
 * Xe, and we write the causality condition under the following
 * form:
 *      Pdest - Pinit >= 1
 *
 * AC 94/01/27
 */

Psysteme make_causal_external(stat_dest, sys_dest, poly_dest, ldata,\
		              stat_pred, sys_pred, poly_source, c, den)

 int           stat_dest, stat_pred;
 Psysteme      sys_dest, sys_pred;
 Ppolynome     poly_dest, poly_source;
 int           *c, den;
 list          ldata;
{
 list          ltrans, lvar, llambda, llambda_el;
 predicate     pred_edge;
 Psysteme      psys_aux, sys_edge, pred_sys = SC_RN;
 Ppolynome     poly_lambda, poly_res, p_source;
 dataflow      pred_data;
 Pn_coef       llam1, llam2;
 int           co = *c, d, ppc;

 lvar = get_list_of_all_param(stat_dest, stat_pred);
 llam1 = NULL;
 llam2 = NULL;
 sys_edge = SC_RN;
 psys_aux = sc_new();
 poly_lambda = POLYNOME_UNDEFINED;
 p_source = polynome_dup(poly_source);

 pred_data = DATAFLOW(CAR(ldata));
 ltrans = dataflow_transformation(pred_data);
 pred_sys = sc_dup(sys_pred);

 p_source = include_trans_in_poly(stat_pred, p_source, ltrans, &d);

 ppc = sol_ppcm(den, d);

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nTRANSFORMATION :\n");
     fprint_list_of_exp(stderr,ltrans);
     fprintf(stderr,"\nPred_sys :");
     fprint_psysteme(stderr,pred_sys);
     fprintf(stderr,"\n\nPoly_source : ");
     my_polynome_fprint(stderr,p_source);
    }

 /* write the causality condition under the following form: */
 /* P_dest - P_source -1 = 0                                */
 poly_res = polynome_dup(poly_dest);
 polynome_negate(&p_source);
 polynome_add(&poly_res, p_source);
 polynome_decr(poly_res);

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nInequation de causalite :\n");
     my_polynome_fprint(stderr,poly_res);
    }

 if (is_uniform_rec(lvar, poly_res))
    {
     /* case of a uniform causality condition */
     if (get_debug_level() > 5) fprintf(stderr,"\nRecurrence uniforme :\n");
     polynome_negate(&poly_res);
     psys_aux = polynome_to_sc(poly_res, lvar);
     psys_aux = transform_in_ineq(psys_aux, NIL);
    }
 else
    {
     /* make the polynom of unknown lambda */
     sys_edge = sc_dup(sys_dest);
     sys_edge = sc_append(sys_edge, pred_sys);
     sys_edge = include_parameters_in_sc(sys_edge, stat_dest);
     my_sc_normalize(sys_edge);

     /* list of the variables to identify */
     if (!SC_UNDEFINED_P(sys_edge))
	lvar = add_others_variables(lvar, base_to_list(sys_edge->base));

     create_farkas_poly(sys_edge, "LAMBDA", stat_pred, stat_dest,\
					  &poly_lambda, &llam1, &co);
     sc_rm(sys_edge);
     sys_edge = NULL;

     /* Now we add the conditions on the edge, caus' we livin' on */
     /* the edge                                                  */
     pred_edge = dataflow_governing_pred(pred_data);
     sys_edge = sc_dup(predicate_to_system(pred_edge));

     if (!SC_UNDEFINED_P(sys_edge))
	lvar = add_others_variables(lvar, base_to_list(sys_edge->base));

     create_farkas_poly(sys_edge, "LAMBDA", stat_pred, stat_dest,\
					  &poly_lambda, &llam2, &co);

     llam1 = add_lcoef_to_lcoef(llam1, llam2);
     llambda = extract_var_list(llam1);

     /* write the causality condition under the following form: */
     /* P_dest - P_source - P_lambda -1 = 0                     */
     polynome_negate(&poly_lambda);
     polynome_add(&poly_res, poly_lambda);

     if (ppc != 1) polynome_scalar_mult(&poly_res, (float)ppc);

     psys_aux = polynome_to_sc(poly_res, lvar);

     psys_aux = elim_var_with_eg(psys_aux, &llambda, &llambda_el);

     /* simplify the system and transform it in inequalities */
     llambda_el = gen_nreverse(llambda_el);
     psys_aux = transform_in_ineq(psys_aux, llambda_el);
    }

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nPoly_resultat : ");
     my_polynome_fprint(stderr,poly_res);
     fprintf(stderr,"\n\n");
    }

 polynome_rm(&poly_res);
 polynome_rm(&p_source);
 *c = co;

 return(psys_aux);
}

/*==================================================================*/
/* list make_sched_proto(h,lin,,col,lu): make the vector-list 
 * representing the prototype of the schedule we search.
 *
 * AC 93/12/06
 */

list make_sched_proto(h, lin, col, lu)

 matrice h;
 int lin, col;
 list       lu;
{
 list       p, lvp = NIL;
 int        i;

 p = lu;
 i = 1;

 while (p != NIL)
    {
	Value v = ACCESS(h,lin,i,1);
	ADD_ELEMENT_TO_LIST(lvp, INT, VALUE_TO_INT(v));
     i++;
     p = CDR(p);
    }

 /* complete the vector to the dimension of the dual problem */
 if (gen_length(lvp) != col)
    {
     for (i = 1; i <= (col-lin); i++)
        ADD_ELEMENT_TO_LIST(lvp, INT, 0);
    }

 return(lvp);
}

/*==================================================================*/
/* Psysteme system_new_var_subst(sys, l): replace in a psyteme the
 * new variables introduced by PIP by their value.
 *
 * AC 93/12/20
 */

Psysteme system_new_var_subst(sys, l)

 Psysteme    sys;
 list        l;
{
 Pcontrainte cont;
 Pvecteur    new_vect;
 list        le, lexp, lvect;
 entity      ent;
 Value       val, va;
 expression  exp, exp1, exp2;
 var_val     vv;
 call        ca;
 int         d;

 for (cont = sys->inegalites; cont != NULL; cont = cont->succ)
    {
     for (le = l; le != NIL; le = CDR(le))
        {
         /* extract the new parameter */
         vv = VAR_VAL(CAR(le));
         ent = var_val_variable(vv);

         lvect = vecteur_to_list(cont->vecteur);
         if (is_entity_in_list_p(ent, lvect))
            {
	     exp = var_val_value(vv);
	     analyze_expression(&exp, &d);
	     ca = syntax_call(expression_syntax(exp));
	     lexp = call_arguments(ca);

             /* extract the numerator */
	     exp1 = EXPRESSION(CAR(lexp));
             NORMALIZE_EXPRESSION(exp1);
             new_vect = normalized_linear(expression_normalized(exp1));

             /* extract the denominator */
	     lexp = CDR(lexp);
	     exp2 = EXPRESSION(CAR(lexp));
             val = (Value)expression_to_int(exp2);
             va = vect_coeff((Variable)ent, cont->vecteur);
             vect_erase_var(&(cont->vecteur), (Variable)ent);
             cont->vecteur = vect_multiply(cont->vecteur, val);
             new_vect = vect_multiply(new_vect, va);
	     cont->vecteur = vect_add(cont->vecteur, new_vect);
             vect_normalize(cont->vecteur);
            }
        }
    }

 sys->base = BASE_NULLE;
 sc_creer_base(sys);
 return(sys);
}

/*==================================================================*/
/* Psysteme add_constraint_on_x(psys, lx): in the Psysteme psys, we
 * add the constraints on the introduced variables Xe: Xe <= 1.
 *
 * AC 94/01/27
 */

Psysteme add_constraint_on_x(psys, lx)

 Psysteme     psys;
 list         lx;
{
 entity       e;
 Ppolynome    p;
 Pcontrainte  c;
 list         l, lx_r;

 lx_r = gen_nreverse(lx);

 for (l = lx_r; l != NIL; l = CDR(l))
    {
     e = ENTITY(CAR(l));
     if (!strncmp(entity_local_name(e), "X", 1))
        {
         p = make_polynome((float)1, (Variable) e, 1);
         p = polynome_decr(p);
         c = polynome_to_contrainte(p);
         sc_add_inegalite(psys, c);
	 psys->base = base_add_variable(psys->base, (Variable) e);
	 psys->dimension++;
        }
    }

 lx = gen_nreverse(lx_r);

 return(psys);
}

/*==================================================================*/
/* Psysteme make_primal(psys,lvar_u,lvp,lxe): build the primal problem.
 * In fact, it does more then that, because it prepares the construc-
 * tion of the dual problem by transposing the matrice coming from
 * the input Psysteme.
 *
 * AC 93/12/20
 */

static Psysteme make_primal(psys, lvar_u, lvp, lxe)

 Psysteme     psys;
 list         *lvar_u, *lvp, lxe;
{
 int          lin, col;
 matrice G, tG, h, f;
 Pbase        base;
 Pcontrainte  cont;

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nPsysteme primal:\n");
     fprint_psysteme(stderr, psys);
    }

 col = gen_length(base_to_list(psys->base));
 lin = psys->nb_ineq;

 G = matrice_new(lin, col);
 tG = matrice_new(col, lin);
 h = matrice_new(lin, 1);
 f = matrice_new(col, 1);

 /* transform the system into a matrix */
 sc_to_matrices(psys, psys->base, G, h, lin, col);

 /* Now, we begin to make the dual problem */
 matrice_transpose(G, tG, lin, col);
 matrice_nulle(f, col, 1);

 *lvar_u = make_list_of_n("u", lin);
 base = list_to_base(*lvar_u);

 *lvp = make_sched_proto(h, lin, col, *lvar_u);

 pu_matrices_to_contraintes(&cont, base, tG, f, col, lin);

 psys = sc_make(NULL, cont);

 matrice_free(G);
 matrice_free(tG);
 matrice_free(h);
 matrice_free(f);
 
 return(psys);
}

/*==================================================================*/
/* list get_exp_schedule(e,me,d): extract form the expression exp
 * given by the resulting quast of PIP the expression of the schedule
 * we search: it takes the expression, put the coef of "M" to 0, and
 * change the sign of the vector.
 *
 * AC 94/01/27
 */

list get_exp_schedule(e, me, d)

 expression   e;
 entity       me;
 int          d;
{
 list         lexp;
 Pvecteur     v;
 expression   exp;
 call         ca;
 syntax       syn;
 entity       func;

 if (!expression_undefined_p(e))
 {
 if (d == 1)
    {
     NORMALIZE_EXPRESSION(e);
     v = normalized_linear(expression_normalized(e));
     vect_chg_coeff(&v, (Variable) me, 0);
     vect_chg_sgn(v);
     e = make_vecteur_expression(v);
     unnormalize_expression(e);
    }
 else
    {
     ca = syntax_call(expression_syntax(e));
     lexp = call_arguments(ca);
     func = call_function(ca);
     exp = EXPRESSION(CAR(lexp));
     NORMALIZE_EXPRESSION(exp);
     lexp = CDR(lexp);
     v = normalized_linear(expression_normalized(exp));
     vect_chg_coeff(&v, (Variable) me, 0);
     vect_chg_sgn(v);
     exp = make_vecteur_expression(v);
     unnormalize_expression(exp);
     lexp = CONS(EXPRESSION, exp, lexp);
     ca = make_call(func, lexp);
     syn = make_syntax(is_syntax_call, ca);
     e = make_expression(syn, normalized_undefined);
    }
 }

 return(CONS(EXPRESSION, e, NIL));
}

/*==================================================================*/
/* Psysteme get_unsatisfied_system(): we check the
 * coefficient of "m" in the solution of the dual variable of "xe",
 * if it is nil we update the list of system that could be unsatis-
 * fied,  the list of Xe, and we build the new system for the
 * primal problem.
 * lsys is updated, lunk too. "me" contains the entity "M".
 *
 * AC 94/01/20
 */

static Psysteme get_unsatisfied_system(lexp, lsys, lxe, lunk, me, de)

 list        lexp, *lxe;
 Psys_list   *lsys;
 Pn_coef     *lunk;
 entity      me;
 int         de;
{
 Psys_list   ls, ls_new;
 list        lx, lx_new;
 Psysteme    sys;
 expression  exp;

 lexp = CDR(lexp);
 lx = *lxe;
 sys = sc_new();
 lx_new = NIL;
 ls_new = NULL;

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nList :");
     fprint_list_of_exp(stderr,lexp);
    }

 for (ls = *lsys; ls != NULL; ls = ls->next)
    {
     exp = EXPRESSION(CAR(lexp));
     if (get_m_coef(&exp, &de) == 0)
	{
         /* this edge is unsatisfied */
	 ls_new = add_elt_to_sys_list(ls_new, 0, ls->sys);
	 ADD_ELEMENT_TO_LIST(lx_new, ENTITY, ENTITY(CAR(lx)));
	 sys = sc_append(sys, ls->sys);
	}
     lexp = CDR(lexp);
     lx = CDR(lx);
    }

 /* we have to add the constraints X<=1 */
 sys = add_constraint_on_x(sys, lx_new);

 *lsys = ls_new;
 *lxe = lx_new;

 return(sys);
}

/*==================================================================*/
/* void make_list_of_unk(l, sc, me, lx): build the complete list of
 * Pn_coef for the primal problem. At the beginning, l only contains
 * the "MU" introduced by the inequation of causality. We check if
 * SOME "MU" have been eliminated, we build the list of Xe that we
 * put in first position, then we append the list of LAMBDA.
 *
 * AC 94/01/27
 */

static void make_list_of_unk(l, sc, me, lx)

 Pn_coef     *l;
 Psysteme    *sc;
 entity      *me;
 list        lx;
{
 Pn_coef     laux, lout, lin = *l;
 list        llambda, base;
 entity      m;

 lout = NULL;

 base = base_to_list((*sc)->base);

 /* First we check if some MU have been eliminated */
 while (lin != NULL)
    {
     laux = lin;
     lin = lin->next;
     laux->next = NULL;
     if (is_entity_in_list_p(laux->n_ent, base))
        lout = add_n_coef_to_list(lout, laux);
    }

 /* we add at the beginning of lout the list about the Xs */
 lout = add_x_list(lout, lx, &m);

 /* we add at the end of lout the list about the lambda */
 llambda = extract_lambda_list(base);
 lout = add_lambda_list(lout, llambda);

 base = extract_var_list(lout);
 (*sc)->base = list_to_base(base);

 *me = m;
 *l = lout;
}

/*==================================================================*/
/* Psysteme get_predicate_system_of_node(st,s): get the system of
 * constraints of a node of statement "st" in a given scc.
 *
 * AC 93/12/20
 */

Psysteme get_predicate_system_of_node(st, s)

 int       st;
 scc       s;
{
 list      lver;
 vertex    ver;
 Psysteme  sys = SC_UNDEFINED;
 bool   not_found = true;

 lver = scc_vertices(s);

 while ((lver != NIL) && (not_found))
    {
     ver = VERTEX(CAR(lver));

     if (st == vertex_int_stmt(ver))
        {
         sys = predicate_to_system(dfg_vertex_label_exec_domain(\
                      (dfg_vertex_label)vertex_vertex_label(ver)));
         not_found = false;
        }
     lver = CDR(lver);
    }

 return(sys);
}

/*==================================================================*/
/* Psyslist add_sc_to_sclist(sc,lsys): add a system in the list of
 * systems defined by Psyslist.
 *
 * AC 93/12/23
 */

Psyslist add_sc_to_sclist(sc, lsys)

 Psysteme    sc;
 Psyslist    lsys;
{
 Psyslist    lsys_aux;

 lsys_aux = (Psyslist)malloc(sizeof(Ssyslist));

 lsys_aux->psys = sc;
 lsys_aux->succ = lsys;

 return(lsys_aux);
}

/*==================================================================*/
/* Psysteme simplify_predicate(ps, ps_eq, l): simplify the psysteme
 * ps by replacing the new variables in l by their expression in
 * ps_eq (in the same order).
 *
 * AC 94/03/28
 */

Psysteme simplify_predicate(ps, ps_eq, l)

 Psysteme     ps, ps_eq;
 list         l;
{
 Pcontrainte  cti, cte = ps_eq->egalites;
 list         lv;
 Value          coi, coe, ppc;
 Pvecteur     vi, ve;
 entity       ent;

 for (lv = l; lv != NIL; lv = CDR(lv))
    {
     ent = ENTITY(CAR(lv));

     for (cti = ps->inegalites; cti != NULL; cti = cti->succ)
	{
         vi = cti->vecteur;
	 if (base_contains_variable_p(vi, (Variable) ent))
	    {
	     coe = vect_coeff((Variable) ent, cte->vecteur);
             coi = vect_coeff((Variable) ent, cti->vecteur);
	     ve = vect_dup(cte->vecteur);
             ppc = value_abs(ppcm(coe, coi));
             vi = vect_multiply(vi, value_abs(value_div(ppc,coi)));
             ve = vect_multiply(ve, value_abs(value_div(ppc,coe)));
             if (value_posz_p(value_mult(coe,coi)))
		 vi = vect_substract(vi,ve); 
             else vi = vect_add(vi, ve);
	     cti->vecteur = vi;
	    }
	}
     cte = cte->succ;
    }

 ps->egalites = ps_eq->egalites;

 return(ps);
}

/*==================================================================*/
/* list simplify_dimension(ld, ps_eq, l): implify the different 
 * dimensions f the schedule in the list ld with the variables in l
 * with the equalities that ps_eq contains.
 *
 * AC 94/03/28
 */

list simplify_dimension(ld, ps_eq, l)

 Psysteme     ps_eq;
 list         ld, l;
{
 Pcontrainte  cte = ps_eq->egalites;
 list         lv, ln = NIL, ldi;
 Value        coi, coe, ppc;
 int d;
 Pvecteur     vi, ve;
 expression   exp, ex;
 entity       ent;
 bool      change = false;

 /* loop on the dimension of the schedule to process */
 for (ldi = ld; ldi != NIL; ldi = CDR(ldi))
    {
     exp = EXPRESSION(CAR(ldi));
     analyze_expression(&exp, &d);
     cte = ps_eq->egalites;

     if (get_debug_level() > 5)
	{
         fprintf(stderr,"\nExpression en entree :");
         fprint_list_of_exp(stderr, CONS(EXPRESSION,exp,NIL));
	 fprintf(stderr,"\nd = %d",d);
        }

     if (d == 1)
        {
         NORMALIZE_EXPRESSION(exp);
         vi = normalized_linear(expression_normalized(exp));
        }
     else
        {
         ex = EXPRESSION(CAR(call_arguments(syntax_call\
                                     (expression_syntax(exp)))));
         NORMALIZE_EXPRESSION(ex);
         vi = normalized_linear(expression_normalized(ex));
        }

     /* loop on the variables to eliminate */
     for (lv = l; lv != NIL; lv = CDR(lv))
        {
         ent = ENTITY(CAR(lv));
	 if (get_debug_level() > 5)
	    {
             fprintf(stderr,"\nEntite a eliminer :");
             fprint_entity_list(stderr,CONS(ENTITY,ent,NIL));
            }

         if (base_contains_variable_p(vi, (Variable) ent))
            {
	     change = true;
             coe = vect_coeff((Variable) ent, cte->vecteur);
             coi = vect_coeff((Variable) ent, vi);
             ve = vect_dup(cte->vecteur);
             ppc = ppcm(coe, coi);
             vi = vect_multiply(vi, value_abs(value_div(ppc,coi)));
             ve = vect_multiply(ve, value_abs(value_div(ppc,coe)));
             if (value_posz_p(value_mult(coe,coi)))
		 vi = vect_substract(vi,ve);
             else vi = vect_add(vi, ve);
            }
         /* try next variable */
         cte = cte->succ;
        }

     if (change)
	{
         if (VECTEUR_UNDEFINED_P(vi)) 
       	      exp = int_to_expression(0);
         else 
	    {
	     if (d == 1) exp = Pvecteur_to_expression(vi);
	     else exp = make_rational_exp(vi, d);
            }
        }

     if (get_debug_level() > 5)
	{
	 fprintf(stderr,"\nvi = ");
	 pu_vect_fprint(stderr, vi);
	 fprintf(stderr,"\nd = %d", d);
         fprintf(stderr,"\nExpression en sortie :");
         fprint_list_of_exp(stderr, CONS(EXPRESSION,exp,NIL));
        }

     ADD_ELEMENT_TO_LIST(ln, EXPRESSION, exp);
    }

 return(ln);
}

/*==================================================================*/
/* bdt simplify_bdt(b,s): simplify the list of schedule of a node
 * by comparing the predicate of the schedule with the domain on
 * which the node is defined. Moreover, replace variables by their
 * value when it is possible in the predicate and the expression.
 * (i.e. if I == N and dims==N+I+1 => dims==2*N+1 )
 *
 * AC 93/12/20
 */

bdt simplify_bdt(b, s)

 bdt             b;
 scc             s;
{
 schedule        sc;
 list            ls, lvar, lvar_e, ldims;
 int             st;
 Psysteme        s_bdt, s_node, s_eq, s_aux;
 static_control  stct;

 if (get_debug_level() > 5) fprintf(stderr,"\nSIMPLIFY BEGIN\n");

 for (ls = bdt_schedules(b); ls != NIL; ls = CDR(ls))
    {
     if (get_debug_level() > 5) fprintf(stderr,"\nSIMPLIFY new branche\n");
     sc = SCHEDULE(CAR(ls));
     st = schedule_statement(sc);
     ldims = schedule_dims(sc);
     stct = get_stco_from_current_map(adg_number_to_statement(st));
     lvar = static_control_to_indices(stct);
     lvar = gen_append(lvar, static_control_params(stct));
     lvar_e = NIL;

     if (get_debug_level() > 5)
	{
         fprintf(stderr, "\nListe des variables;");
         fprint_entity_list(stderr, lvar);
        }

     s_bdt = predicate_to_system(schedule_predicate(sc));
     s_node = get_predicate_system_of_node(st, s);

     if (!SC_RN_P(s_bdt))
        {
         if (get_debug_level() > 5)
	    {
             fprintf(stderr,"\nSYSTEME BDT en entree :");
             fprint_psysteme(stderr,s_bdt);
            }

         s_eq = find_implicit_equation(s_bdt);
         s_aux = elim_var_with_eg(s_eq, &lvar, &lvar_e);

         if (get_debug_level() > 5)
	    {
	     fprintf(stderr,"\nSYSTEME D EGALITES apres :");
             fprint_psysteme(stderr,s_aux);
             fprintf(stderr,"\nListe des eliminees:");
             fprint_entity_list(stderr, lvar_e);
            }

         if ((lvar_e != NIL)&&(!SC_RN_P(s_bdt)))
            {
             ldims = simplify_dimension(ldims, s_aux, lvar_e);
             s_bdt = simplify_predicate(s_bdt, s_aux, lvar_e);
            }

         if (get_debug_level() > 5)
	    {
	     fprintf(stderr,"\nSYSTEME BDT en sortie :");
             fprint_psysteme(stderr,s_bdt);
            }

         s_bdt = suppress_sc_in_sc(s_bdt, s_node);
         my_sc_normalize(s_bdt);
         schedule_predicate(sc) = make_predicate(s_bdt);
         schedule_dims(sc) = ldims;
        }
    }

 if (get_debug_level() > 5) fprintf(stderr,"\nSIMPLIFY END\n");

 return(b);
}

/*==================================================================*/
/* Psysteme make_dual(psys,pcont,l,lvar_u,lvp): makes the 
 * dual problem in the case of a scc containing a single node. In 
 * this case, we do not introduce a set of "v" in the second member
 * we put directly the proper value function of the program 
 * parameters.
 *
 * AC 93/12/20
 */

static Psysteme make_dual(psys, pcont, l, lvar_u, lvp)

 Psysteme     psys, *pcont;
 Pn_coef      l;
 list         *lvar_u, lvp;
{
 Pcontrainte  cont;
 Psysteme     sys_cont;
 Pvecteur     vect, vect_aux, vect_pro, vp;
 list         lbase, lp, laux;
 entity       ent;

 sys_cont = sc_new();
 lp = lvp;
 vp = VECTEUR_NUL;

 /* for each inequalities, build the second member of the system */
 /* and at the same time the system of constraints.              */

 for (cont = psys->inegalites ; cont != NULL ; cont = cont->succ)
    {
     vect = vect_dup(cont->vecteur);
     vect_chg_sgn(vect);
     vect_aux = vect_dup(l->n_vect);
     vect_chg_sgn(vect_aux);
     cont->vecteur = vect_add(vect, vect_aux);

     sc_add_inegalite(sys_cont, contrainte_make(vect_aux));

     vect_pro = vect_new((Variable) TCST, INT(CAR(lp)));
     vp = vect_add(vp, vect_pro);

     l = l->next;
     lp = CDR(lp);
    }

 /* build and put the economic function in 1st position in psys */

 ent = create_named_entity("u0");
 vect = vect_new((Variable)ent,1);
 lp = lvp;

 for (laux = *lvar_u; laux != NIL; laux = CDR(laux))
    {
     vect_aux = vect_new((Variable) ENTITY(CAR(laux)), INT(CAR(lp)));
     vect = vect_add(vect, vect_aux);
     lp = CDR(lp);
    }

 vect_chg_sgn(vect);
 sc_add_inegalite(psys, contrainte_make(vect));
 *lvar_u = CONS(ENTITY, ent, *lvar_u);

 psys->base = NULL;
 sys_cont->base = NULL;
 sc_creer_base(psys);
 sc_creer_base(sys_cont);

 lbase = base_to_list(psys->base);
 lbase = reorder_base(lbase, *lvar_u);
 psys->base = list_to_base(lbase);

 *pcont = my_sc_normalize(sys_cont);

 return(psys);
}

/*==================================================================*/
/* static Pn_coef extract_stat_lunk(stat, lunk): creates the good
 * second member for the dual problem based on the general second
 * member lunk. This function is used when we search the schedule
 * of each node in a scc one after the other.
 *
 * AC 94/01/10
 */

static Pn_coef extract_stat_lunk(stat, lunk)

 int      stat;
 Pn_coef  lunk;
{
 Pn_coef  lv_coef, laux, pn;
 char     *name;
 int      len;
 entity   ent;

 asprintf(&name, "%s_%d", "MU", stat);
 len = strlen(name);
 lv_coef = NULL;

 for (laux = lunk; laux != NULL; laux = laux->next)
    {
     ent = laux->n_ent;
     if (!strncmp(name, entity_local_name(ent), len))
            pn = make_n_coef(ent, laux->n_vect);
     else if (!strncmp("X", entity_local_name(ent), 1))
	    pn = make_n_coef(ent, laux->n_vect);
     else   pn = make_n_coef(ent, VECTEUR_NUL);
     lv_coef = add_n_coef_to_list(lv_coef, pn);
    }
 free(name);

 return(lv_coef);
}

/*==================================================================*/
/* bdt include_results_in_bdt(b, baux, lexp): in case of a multi
 * dimensionnal bdt, this function is used to append the dimension
 * calculated at this step (lexp) with the dimensions calculated
 * before (b) and the dimensions calculated after (baux);
 *
 * AC 94/10/27
 */

bdt include_results_in_bdt(b, baux, lexp)

 bdt        b, baux;
 list       lexp;
{
 list       ls, le, lexp_aux;
 schedule   sched;
 Psysteme   sys, sc;
 predicate  pred;

 if (!bdt_undefined_p(baux))
    {
     ls = bdt_schedules(baux);
     pred = schedule_predicate(SCHEDULE(CAR(bdt_schedules(b))));
     sys = predicate_to_system(pred);

     for (; ls != NIL; ls = CDR(ls))
        {
	 lexp_aux = CONS(EXPRESSION,(EXPRESSION(CAR(lexp))),NIL);

         /* update the predicate */
         sched = SCHEDULE(CAR(ls)); 
         sc = predicate_to_system(schedule_predicate(sched));
         sc = sc_append(sc,sys);
         schedule_predicate(sched) = make_predicate(sc);

         /* update the list of expressions */
         le = schedule_dims(sched);
         le = gen_nconc(lexp_aux, le);
         schedule_dims(sched) = le;
        }
     b = baux;
    }
 else
    {
     ls = bdt_schedules(b);
     sched = SCHEDULE(CAR(ls));
     schedule_dims(sched) = lexp;
     bdt_schedules(b) = CONS(SCHEDULE, sched, NIL);
    }

 return(b);
}

/*==================================================================*/
/* bool is_mu_stat_in_sc(stat, sc): check if the system "sc" 
 * contains some variable of type "MU_stat".
 *
 * AC 94/01/27
 */

bool is_mu_stat_in_sc(stat, sc)

 int       stat;
 Psysteme  sc;
{
 entity    ent;
 char      *name;
 int       len;
 list      lbase;
 bool   is_here = false;

 asprintf(&name, "%s_%d", "MU", stat);
 len = strlen(name);

 lbase = base_to_list(sc->base);

 for (; lbase != NIL; lbase = CDR(lbase))
    {
     ent = ENTITY(CAR(lbase));
     if (!strncmp(name, entity_local_name(ent), len))
	is_here = true;
    }
 free(name);
 return(is_here);
}

/*==================================================================*/
/* bdt analyze_quast(q,lu,nb): this function substitute the new
 * parameters that can appear, go down to each branch of the quast
 * and analyze if all edge have been satisfied, if not find recur-
 * sively the dimension of the schedule.
 *
 * AC 94/01/27
 */

static bdt analyze_quast(q, stat, lunk, lsys, b, lxe, me)

 quast        q;
 int          stat;
 Pn_coef      lunk;
 Psys_list    lsys;
 bdt          b;
 list         lxe;
 entity       me;
{
 list         lsol, new_lsol, lnew, lst, lsf, ls, lu, lvp;
 predicate    pred;
 conditional  cond, cond_aux;
 quast_value  quv;
 expression   exp;
 quast_leaf   qul;
 entity       ent;
 var_val      vv;
 Psysteme     sys, st_sys, sf_sys;
 int          m_coef, nb_arc, d;
 schedule     st, sf;
 Pdisjunct    dj;
 bdt          bt, bf;
 quast        qua;
 Pvecteur     vu;

 new_lsol = NIL;
 lnew = NIL;
 quv = quast_quast_value(q);
 lnew = quast_newparms(q);

 ls = NIL;
 lst = NIL;
 lsf = NIL;

 if (quast_undefined_p(q))
    pips_internal_error("Quast should not be undefined!");

 if (bdt_undefined_p(b))
     {
      st = make_schedule(stat, predicate_undefined, NIL);
      b = make_bdt(CONS(SCHEDULE, st, NIL));
     }
 else st = SCHEDULE(CAR(bdt_schedules(b)));

 /* see if there are some new parameters to replace */
 if ((lnew != NIL) && (get_debug_level() > 5))
    {
     vv = VAR_VAL(CAR(lnew));
     ent = var_val_variable(vv);
     exp = var_val_value(vv);
     fprintf(stderr,"\nNouveau parametre :\n");
     fprint_entity_list(stderr, CONS(ENTITY,ent,NIL));
     fprintf(stderr," => ");
     fprint_list_of_exp(stderr, CONS(EXPRESSION,exp,NIL));
    }
 
 switch(quast_value_tag(quv))
    {
     case is_quast_value_conditional:
        cond = quast_value_conditional(quv);
        pred = conditional_predicate(cond);
        sys = sc_dup(predicate_to_system(pred));

        /* replace the new parameters in the predicate system */
        sys = system_new_var_subst(sys, lnew);
        sf = true_copy_schedule(st);
        st_sys = predicate_to_system(schedule_predicate(st));
        sf_sys = predicate_to_system(schedule_predicate(sf));

        /* we process the true edge */
        st_sys = sc_append(st_sys, sys);

        if (sc_rational_feasibility_ofl_ctrl(st_sys, NO_OFL_CTRL, true)) {
	  st = make_schedule(stat, make_predicate(st_sys), NIL);
	  bt = make_bdt(CONS(SCHEDULE, st, NIL));
	  cond_aux = conditional_true_quast(cond);
	  bt = analyze_quast(cond_aux, stat, lunk, lsys, bt, lxe, me);
	  lst = bdt_schedules(bt);
	}
        else
	{
	  lst = NIL;
	  sc_rm(st_sys);
	}

        /* we process the false edge */
        dj = dj_system_complement(sys);
        sys = dj->psys;
        sf_sys = sc_append(sf_sys, sys);

        if (sc_rational_feasibility_ofl_ctrl(sf_sys, NO_OFL_CTRL, true))
	{
	  sf = make_schedule(stat, make_predicate(sf_sys), NIL);
	  bf = make_bdt(CONS(SCHEDULE, sf, NIL));
	  cond_aux = conditional_false_quast(cond);
	  bf = analyze_quast(cond_aux, stat, lunk, lsys, bf, lxe, me);
	  lsf = bdt_schedules(bf);
	}
        else
	{
	  lsf = NIL;
	  sc_rm(sf_sys);
	}

        bdt_schedules(b) = gen_nconc(lst, lsf);
        sc_rm(sys);
     break;

     case is_quast_value_quast_leaf:
        qul = quast_value_quast_leaf( quv );
        lsol = quast_leaf_solution( qul );
        exp = EXPRESSION(CAR(lsol));
        m_coef = get_m_coef(&exp, &d);
        nb_arc = gen_length(lxe);
        if (m_coef == nb_arc)
           {
            /* all the edges are satisfied, we get the schedule */
	    if (get_debug_level() > 5)
	       {
	        fprintf(stderr,"\nTous les arcs sont satisfaits car\
	        le coef de M est %d\n",m_coef);
	        fprint_entity_list(stderr, lxe);
	        fprint_entity_list(stderr,CONS(ENTITY,me,NIL));
               }

            schedule_dims(st) = get_exp_schedule(exp, me, d);
           }
        else
           {
            /* some edges have not been satisfied: we get the */
            /* dimension of the schedule already calculated   */
            /* and search the others                          */
	    if (get_debug_level() > 5)
	       {
	        fprintf(stderr,"\nTous les arcs ne sont pas satisfaits car\
       	        le coef de M est %d\n",m_coef);
               }

            new_lsol = get_exp_schedule(exp, me, d);

            sys = get_unsatisfied_system(lsol, &lsys, &lxe, me, d);
            clean_list_of_unk(&lunk, &sys); 

	    if (get_debug_level() > 5)
	       {
	        fprintf(stderr,"\nDimension trouvee :");
	        fprint_list_of_exp(stderr, new_lsol);
	        fprintf(stderr,"\n\n\nNb d'arc = %d", nb_arc);
	        fprintf(stderr,"\nSYS LIST:\n");
	        fprint_sys_list(stderr, lsys);
	        fprint_entity_list(stderr, lxe);
               }

	    bt = bdt_undefined;

	    if (is_mu_stat_in_sc(stat, sys))
	       {
                sys = make_primal(sys, &lu, &lvp, lxe);

                sys = make_dual(sys, &st_sys, lunk, &lu, lvp);

                vu = list_to_base(lu);

		if (get_debug_level() > 5)
		   {
                    fprint_psysteme(stderr,sys);
                    fprintf(stderr,"\nSysteme de contraintes :\n");
                    fprint_psysteme(stderr,st_sys);
                    fprintf(stderr,"\nVct d'inconnues = \n");
		    pu_vect_fprint(stderr, vu);
                    fprintf(stderr,"\nBase de psys: ");
                    pu_vect_fprint(stderr,sys->base);
                    fprintf(stderr,"\nBase de sys_cont:");
                    pu_vect_fprint(stderr,st_sys->base);
                   }

                qua = pip_solve_min_with_big(sys, st_sys, vu, "My_Own_Private_M");

                qua = compact_quast(qua, gen_length(lxe));

		if (get_debug_level() > 5)
		   {
                    fprintf(stderr,"\n\nQuast de PIP (n_dim):");
		    imprime_quast(stderr,qua);
		   }

                bt = analyze_quast(qua, stat, lunk, lsys, bt, lxe, me);

		if (get_debug_level() > 5)
		   {
		    fprintf(stderr,"\nBdt apres analyze :");
		    fprint_bdt(stderr,bt);
                   }
	       }
            b = include_results_in_bdt(b, bt, new_lsol);
           }
     break;
    }

 return(b);
}

/*==================================================================*/
/* bdt make_bdt_initial(stat, s): write the initial bdt with as a 
 * predicate, the definition domaine of the node.
 *
 * 94/01/27
 */

bdt make_bdt_initial(stat, s)

 int        stat;
 scc        s;
{
 bdt        b;
 schedule   st;
 predicate  pred;
 Psysteme   sc;

 sc = sc_dup(get_predicate_system_of_node(stat, s));
 pred = make_predicate(sc);
 st = make_schedule(stat, pred, NIL);
 b = make_bdt(CONS(SCHEDULE, st, NIL));

 return(b);
}

/*==================================================================*/
/* bdt write_resulting_bdt(s, stat, bs, bg): simplify the bdt found
 * and update the hash table.
 *
 * AC 94/01/27
 */

bdt write_resulting_bdt(s, stat, bs, bg)

 scc        s;
 int        stat;
 bdt        bs, bg;
{
 Pbdt_node  node;
 list       lsc;

 bs = simplify_bdt(bs, s);
 node = (Pbdt_node) hash_get(h_node, (char *) stat);
 node->n_bdt = true_copy_bdt(bs);
 lsc = bdt_schedules(true_copy_bdt(bs));
 if (bdt_undefined_p(bg)) bg = bs;
 else bdt_schedules(bg) = gen_nconc(bdt_schedules(bg), lsc);

 return(bg);
}

/*==================================================================*/
/* bdt search_scc_bdt( (scc) s) : from a given scc, make for each
 * vertex the causality condition under its Farkas form and place it
 * in a system. Then build the primal problem, then the dual problem,
 * and finally solve it by PIP. 
 *
 * AC 93/10/29
 */

static bdt search_scc_bdt(s)

 scc        s;
{
 Psysteme   psys, psys_aux, sys_dest, sys_pred, sys;
 list       lver, lpred, lsched, lexp, ldata, lu, lstat, ltrans;
 list       lxe, lvp;
 Ppolynome  poly_dest, poly_source;
 int        stat_dest, stat_pred, count, stat, xcount, den;
 Pbdt_node  node_dest, node_pred;
 schedule   sched;
 predicate  pred_dest, pred_pred;
 bdt        bdt_pred, scc_bdt, stat_bdt;
 expression exp;
 vertex     vert_pred;
 Pn_coef    lunk, lunk2, lunkx;
 quast      qua;
 Psyslist   lbdt_pred = NULL;
 Psys_list  lsys = NULL;
 entity     xe, me;
 bool    all_external = false;
 Pvecteur   vu;

 psys = sc_new();
 lunk = NULL;
 lunk2 = NULL;
 lunkx = NULL;
 lstat = NIL;
 scc_bdt = bdt_undefined;
 xcount = 0;
 lxe = NIL;
 lvp = NIL;

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nDEBUT DE CALCUL SUR UNE SCC :\n");
     fprintf(stderr,"\n=============================\n");
    }

 lver = scc_vertices(s);
 if (lver->cdr == NIL) all_external = true;

 /* for each node of the studied scc, get the characteristics of */
 /* the node we want the schedule                                */
 for (lver = scc_vertices(s); lver != NIL; lver = CDR(lver))
    {
     vertex ver = VERTEX(CAR(lver));
     stat_dest = vertex_int_stmt(ver);
     pred_dest = dfg_vertex_label_exec_domain((dfg_vertex_label)\
			                      vertex_vertex_label(ver));
     sys_dest  = sc_dup(predicate_to_system(pred_dest));
     node_dest = (Pbdt_node) hash_get(h_node, (char *) stat_dest);
     poly_dest = node_dest->n_poly;
     lunk = add_lcoef_to_lcoef(lunk, node_dest->n_var);
     ADD_ELEMENT_TO_LIST(lstat, INT, stat_dest);

     if (get_debug_level() > 5)
	{
         fprintf(stderr,"\nOn s'interesse au noeud n.%d",stat_dest);
         fprintf(stderr,"\n===========================\n");
         fprintf(stderr,"de predicat : ");
         fprint_psysteme(stderr,sys_dest);
         fprintf(stderr,"\nde polynome : ");
         my_polynome_fprint(stderr,poly_dest);
        }

     /* for each predecessor of the studied node, make the causality */
     /* condition under its Farkas form and write it in a Psystem    */
     for (lpred=vertex_successors(ver); lpred!=NIL; lpred=CDR(lpred))
	{
	 successor suc = SUCCESSOR(CAR(lpred));
	 vert_pred = successor_vertex(suc);
	 stat_pred = vertex_int_stmt(vert_pred);
         node_pred = (Pbdt_node)hash_get(h_node, (char *) stat_pred);
         count = 0;
	 den = 1;
         ldata = dfg_arc_label_dataflows((dfg_arc_label)\
                                          successor_arc_label(suc));

         while (ldata != NIL)
            {
             /* test if the vertex has already its schedule, and if yes */
	     /* convert it to use it in the causality condition         */
             if (node_pred->n_bdt == (schedule)NIL)
	       {
		/* this edge is an internal edge, so the predececssor */
		/* has not already a schedule                         */
		xcount++;

		if (get_debug_level() > 5)
		   {
                    fprintf(stderr,"\n\nPredecesseur n.%d:pas de\
		    schedule\n", stat_pred);
                    fprintf(stderr,"-------------------------------\n\n");
                   }

	        /* get the polynom of the source */
                poly_source = polynome_dup(node_pred->n_poly);

		psys_aux = make_causal_internal(stat_dest, sys_dest,\
			         poly_dest, ldata, stat_pred, SC_RN,\
			         poly_source, &count, xcount, &xe, den);
	   
                all_external = false; 
		ADD_ELEMENT_TO_LIST(lxe, ENTITY, xe);
                lsys = add_elt_to_sys_list(lsys, xcount, psys_aux); 
               }
            else 
	       {
		/* the edge is an external edge, so the predecessor    */
		/* has alredy a schedule. We only consider the first   */
		/* dimension of this schedule in case of a multidimen- */
		/* sionnal one.                                        */
		if (get_debug_level() > 5)
		   {
                    fprintf(stderr,"\n\nPredecesseur n.%d : schedule!!\n",\
	                                                   stat_pred);
                    fprintf(stderr,"------------------------------\n");
                   }

		psys_aux = SC_RN;
                /* get the list of schedules of the source */
                bdt_pred = true_copy_bdt(node_pred->n_bdt);

		if (get_debug_level() > 5)
		   {
		    fprintf(stderr,"\nBdt du predecesseur :\n");
		    fprint_bdt(stderr,bdt_pred);
                   }

		/* for each schedule, write the causality condition */
                for (lsched=bdt_schedules(bdt_pred); lsched!=NIL;lsched=CDR(lsched))
                   {
                    sched = SCHEDULE(CAR(lsched));
                    pred_pred = schedule_predicate(sched);
                    sys_pred = sc_dup(predicate_to_system(pred_pred));
		    ltrans = dataflow_transformation(DATAFLOW(CAR(ldata)));

		    if (!SC_UNDEFINED_P(sys_pred))
		       {
			sys_pred = include_trans_in_sc(stat_pred, sys_pred,\
						       ltrans);
		        lbdt_pred = add_sc_to_sclist(sys_pred, lbdt_pred);
                       }

                    lexp = schedule_dims(sched);
		    exp = EXPRESSION(CAR(lexp));
                    analyze_expression(&exp,&den);

		    poly_source = expression_to_polynome(exp);

		    if (get_debug_level() > 5)
		       {
		        fprint_list_of_exp(stderr,lexp);
                        fprintf(stderr,"\nPsystem du predecesseur: \n");
		        fprint_psysteme(stderr,sys_pred);
                       }

                    if (all_external)
                       {
			/* this is a particular case where we introduce */
			/* an Xe if we want the primal problem to be    */
			/* consistent with the theory.                  */
			xcount++;
                        sys = make_causal_internal(stat_dest, sys_dest,\
				      poly_dest, ldata, stat_pred, sys_pred,\
				      poly_source, &count, xcount, &xe, den);

                        ADD_ELEMENT_TO_LIST(lxe, ENTITY, xe);
			lsys = add_elt_to_sys_list(lsys, xcount, sys);
			all_external = false;
		       }
                    else
                        sys = make_causal_external(stat_dest, sys_dest,\
				      poly_dest, ldata, stat_pred, sys_pred,\
				      poly_source, &count, den);
                if (get_debug_level() > 5)
		   {
		   fprintf(stderr,"\nSysteme pour arc:\n");
		   fprint_psysteme(stderr,sys);
		   }

                    psys_aux = sc_append(psys_aux, sys);
                   }
               }

	    if (get_debug_level() > 5)
	       {
                fprintf(stderr,"\nSysteme:\n");
		fprint_psysteme(stderr,psys_aux);
	       }

            psys = sc_append(psys, psys_aux);
	    sc_rm(psys_aux);
	    ldata = CDR(ldata);
	   }
	}
     psys = erase_trivial_ineg(psys);
    }

 my_sc_normalize(psys);
 psys = sc_dup(psys);

 /* we have to add the constraints on the Xs: X<=1 */
 psys = add_constraint_on_x(psys, lxe);

 /* Now we have in "psys" the complete Psystem in mu-lambda to solve */
 /* We now transform it in the primal problem                        */

 make_list_of_unk(&lunk, &psys, &me, lxe);
 psys = make_primal(psys, &lu, &lvp, lxe);

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nSysteme primal :\n");
     fprint_psysteme(stderr,psys);
     fprintf(stderr,"\nVariables :\n");
     fprint_coef_list(stderr,lunk);
    }

 for (; lstat != NIL; lstat = CDR(lstat))
    {
     psys_aux = sc_dup(sc_dup(psys));
     stat = INT(CAR(lstat));
     lunk2 = extract_stat_lunk(stat, lunk);
     psys_aux = make_dual(psys_aux, &sys_dest, lunk2, &lu, lvp);
     sys_dest = sc_dup(sys_dest);
     vu = list_to_base(lu);

     if (get_debug_level() > 5)
	{
         fprintf(stderr,"\nSysteme dual :\n");
         fprint_psysteme(stderr,psys_aux);
         fprintf(stderr,"\nSysteme de contraintes :\n");
         fprint_psysteme(stderr,sys_dest);
         fprintf(stderr,"\nVct d'inconnues = \n");
	 pu_vect_fprint(stderr, vu);
         fprintf(stderr,"\nBase de psys: ");
         pu_vect_fprint(stderr,psys_aux->base);
         fprintf(stderr,"\nBase de sys_cont:");
         pu_vect_fprint(stderr,sys_dest->base);
        }

     qua = pip_solve_min_with_big(psys_aux, sys_dest, vu, "My_Own_Private_M");

     qua = compact_quast(qua, gen_length(lxe));

     if (get_debug_level() > 5)
	{
         fprintf(stderr,"\n\nQuast de PIP pour %d:", stat);
         imprime_quast(stderr,qua);
        }

     stat_bdt = make_bdt_initial(stat, s);
     stat_bdt = analyze_quast(qua, stat, lunk2, lsys, stat_bdt, lxe, me);

     if (get_debug_level() > 5)
	{
         fprintf(stderr,"\nBDT AVANT:\n");
         fprint_bdt(stderr, stat_bdt);
        }

     scc_bdt = write_resulting_bdt(s, stat, stat_bdt, scc_bdt);

     if (get_debug_level() > 5)
        {
         fprintf(stderr,"\nBDT APRES:\n");
         fprint_bdt(stderr, stat_bdt);
        }

     lu = CDR(lu);
    }

 if (get_debug_level() > 5)
    {
     fprintf(stderr,"\nBDT :\n");
     fprint_bdt(stderr,scc_bdt);
    }

 sc_rm(psys);

 return(scc_bdt);
}

/*==================================================================*/
/* bdt search_graph_bdt( (sccs) rgraph ) : function that goes through 
 * the reverse graph "rgraph" and search the set of bdt for each
 * reduced node.
 *
 * AC 93/10/27
 */

bdt search_graph_bdt(rgraph)

 sccs   rgraph;
{
 list   lscc, lsched, lschedg;
 bdt    bdt_graph, bdt_scc;
 bool   no_pred;

 bdt_graph = bdt_undefined;
 h_node = hash_table_make(hash_int, 0);
 make_proto(h_node, rgraph);

 for (lscc = sccs_sccs(rgraph); lscc != NIL; lscc = CDR(lscc))
    {
     scc scc_an = SCC(CAR(lscc));
     bdt_scc = bdt_undefined;

     /* test of the existence of a predecessor */
     no_pred = if_no_pred(scc_an, &bdt_scc);

     /* search the set of bdt for this scc */
     if (!no_pred)  bdt_scc = search_scc_bdt(scc_an);

     /* add the found bdt to the already calculated ones */
     if (!bdt_undefined_p(bdt_graph))
	{
         lsched = bdt_schedules(bdt_scc);
	 lschedg = bdt_schedules(bdt_graph);
	 bdt_schedules(bdt_graph) = gen_nconc(lschedg, lsched);
	}
     else  bdt_graph = bdt_scc;
    }

 hash_table_free(h_node);

 return (bdt_graph);
}
