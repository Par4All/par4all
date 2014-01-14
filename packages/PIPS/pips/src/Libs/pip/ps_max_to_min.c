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
/* Name     : ps_max_to_min.c
 * Package  : pip
 * Author   : F. Dumontet
 * Date     : july 93
 * Historic :
 * Documents:
 *
 * Comments :
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <setjmp.h>
#include <string.h>
#include <errno.h>

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
#include "matrix.h"

#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "paf-util.h"
#include "pip.h"



#define VARSUPP "Variable_pour_max_en_min_q"


/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name : add_coeff_vect                                                      */
/*                                                                            */
/* Parameters: p_vect: le vecteur a sommer.                                   */
/*             p_base: la base associee au vecteur.                           */
/*             nb_var: le nombre de variables dans la base.                   */
/*                                                                            */
/* Result: un nombre sous la forme d'une value.                               */
/*                                                                            */
/* Aims: sommer des coefficients des nb_var premieres variables du vecteur    */
/*        p_vect.                                                             */
/*                                                                            */
/* Author: F Dumontet                                                         */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

Value
add_coeff_vect (p_vect,p_base,nb_var)


        Pvecteur        p_vect;
        Pbase           p_base;
        int             nb_var;

{

        Pvecteur        p_vect_baux = (Pvecteur) p_base;
        Value           aux = VALUE_ZERO;
        int             aux1;

        for(aux1=0 ;((aux1 < nb_var) && (p_vect != NULL));\
            p_vect_baux = p_vect_baux->succ )
            {
            
            if (p_vect->var != NULL) 
                  if (p_vect_baux->var == p_vect->var)
                        {
			    value_addto(aux,p_vect->val);
			    p_vect = p_vect->succ;
                        }
            aux1++;
            }
        return aux;
}

/*----------------------------------------------------------------------------*/
/*            BIDOUILLE                                                       */
/* Name:                                                                      */
/*                                                                            */
/* Parameters: name: nom local attribue a l'"entite" cree.                    */
/*             module_name: TOP_LEVEL_MODULE ????                             */
/*                                                                            */
/* Side effect:                                                               */
/*                                                                            */
/* Result:                                                                    */
/*                                                                            */
/* Aims:                                                                      */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
/******************************************************************************/
entity
creer_nom_var(name, module_name)
char *name;
char *module_name;
{
    string full_name;
    entity e ;
    basic b ;

    debug(8,"make_scalar_integer_entity", "begin name=%s, module_name=%s\n",
          name, module_name);

    full_name = strdup(concatenate(module_name, MODULE_SEP_STRING, name, (char *) NULL));

    e = gen_find_tabulated(full_name, entity_domain);

    if(e == entity_undefined) {
      e = make_entity(strdup(full_name),
                      type_undefined,
                      storage_undefined,
                      value_undefined);
  
      b = make_basic(is_basic_int, 4);

      entity_type(e) = (type) MakeTypeVariable(b, NIL);
   }


        return (e);
}

/******************************************************************************/
/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: add_fin_mat                                                          */
/*                                                                            */
/* Parameters: p_cont: pointeur sur la premiere contrainte de la liste        */
/*                     d'egalites ou d'inegalites.                            */
/*             p_base: base du Psysteme.                                      */
/*             nb_var: le nombre de variables dans la base.                   */
/*                                                                            */
/* Result:                                                                    */
/*                                                                            */
/* Aims: ajouter A.1 dans la colonne correspondant au nouveau parametre var_q.*/
/*                                                                            */
/* Author: F Dumontet                                                         */
/*                                                                            */
/* Date of last change: */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
add_fin_mat(p_cont , p_base , nb_var, var_q)

        Pcontrainte      p_cont;
        Pbase            p_base;
        int              nb_var;
	Variable	 var_q;

{
      
        Pvecteur         p_vect_aux;
	Pvecteur	 v2;

        for( ; p_cont != NULL;p_cont = p_cont->succ)
            {



            for(p_vect_aux=p_cont->vecteur; p_vect_aux->succ != NULL;\
                  p_vect_aux = p_vect_aux->succ)
                  {}
/* Old version of FD
                  vect_add_elem(&p_vect_aux->succ, var_q,\
                              add_coeff_vect(p_cont->vecteur,\
                                          p_base,nb_var));
*/
		v2 = (Pvecteur) malloc(sizeof(Svecteur));
                vect_add_elem(&v2, var_q,\
                              add_coeff_vect(p_cont->vecteur,\
                                          p_base,nb_var));
		p_vect_aux->succ = v2;
            }
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: opp_var_in_mat.                                                      */
/*                                                                            */
/* Parameters: p_cont: pointeur sur la premiere contrainte de la liste        */
/*                     d'egalites ou d'inegalites.                            */
/*             p_base: base du Psysteme.                                      */
/*             nb_var: le nombre de variables dans la base.                   */
/*                                                                            */
/* Result:                                                                    */
/*                                                                            */
/* Aims: transformer les coefficients des vecteurs qui portent sur des        */
/*       en leur oppose.                                                      */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
opp_var_in_mat (p_cont , p_base , nb_var)

        Pcontrainte      p_cont;
        Pbase            p_base;
        int              nb_var;

{

        Pvecteur         p_vect_aux;      /* vecteur d'une contrainte.       */
        Pvecteur         p_vect_baux;     /* vecteur d'une base.             */
        int              aux1;            /* compteur de variables deja      */
                                          /* traitee.                        */ 
   

        for( ; p_cont != NULL; p_cont = p_cont->succ)
            {
            p_vect_baux = p_base;
            p_vect_aux = p_cont->vecteur;
                  /* on traite les nb_var 1eres variables de la base    */
            for(aux1 = 0 ; ((p_vect_aux != NULL) && (aux1<nb_var));\
                  p_vect_baux = p_vect_baux->succ)
                  {
                  if ((p_vect_baux->var != NULL) &&\
                              (p_vect_aux->var !=NULL))
                        {
                        if (p_vect_baux->var == p_vect_aux->var)
                              {
				  value_oppose(p_vect_aux->val);
				  p_vect_aux = p_vect_aux->succ;
                              }

                        }
                  if (p_vect_baux->var != NULL)
                        aux1++;
                  }
            }
}
/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: converti_psysmin_psysmax                                             */
/*                                                                            */
/* Parameters: p_systmin: le Psysteme (ordonne) a transformer.                */
/*             nb_var: le nombre de variables dans le Psysteme (ce sont les   */
/*                     premieres de la base).                                 */
/*                                                                            */
/* Result: Psysteme ordonne.                                                  */
/*                                                                            */
/* Aims: convertir un probleme de maximum en son dual de minimum, le probleme */
/*       et le resultat sont  sous formes de Psystemes.                       */
/*                                                                            */
/* Author:  F Dumontet.                                                       */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

Psysteme
converti_psysmin_psysmax (p_systmin, nb_var)
Psysteme        p_systmin;
int             nb_var;
{
    Psysteme        p_syst;       /* le Psysteme produi                 */
    Psysteme        p_syst1;      /* permet la double copie sinon rien  */
    Pcontrainte     p_cont_aux, pc;
    Pvecteur        p_vect_aux, p_vect_baux, p_base;
    int             aux1;
    Variable        var_q = (Variable) creer_nom_var(VARSUPP,
						     TOP_LEVEL_MODULE_NAME);

    /* double dup pour obtenir une copie qui respecte l'odre des variables
    * sc_dup rend une image inversee */
    p_syst1=sc_dup(p_systmin);      
    p_syst = sc_dup(p_syst1);
    
    /* ajout d'une variable dans la base essayer de trouver une fonction
     * C3 */
    p_syst->dimension++;

    for(p_vect_aux=(Pvecteur) p_syst->base; p_vect_aux->succ != NULL;
	p_vect_aux = p_vect_aux->succ) {}
    vect_add_elem(&p_vect_aux->succ, var_q, (Value) 1);

    /* ajout de A.1 en fin de matrice , le parametre var_q puis -A */

    /* These four function calls are replaced by the following loops */
    /*
       add_fin_mat((Pcontrainte) p_syst->egalites, p_syst->base, nb_var, var_q);

       opp_var_in_mat((Pcontrainte) p_syst->egalites,
       p_syst->base , nb_var);
       add_fin_mat((Pcontrainte) p_syst->inegalites,
       p_syst->base, nb_var, var_q);
       
       opp_var_in_mat((Pcontrainte) p_syst->inegalites,
       p_syst->base , nb_var);
       */

    for(aux1=0, p_base = p_syst->base ;((aux1 < nb_var) && (p_base != NULL));
	p_base = p_base->succ, aux1++) {
      for(pc = p_syst->egalites; pc != NULL; pc = pc->succ) {
	pc->vecteur = vect_var_subst(pc->vecteur, p_base->var,
				     vect_cl_ofl_ctrl
				     (vect_new(var_q, VALUE_ONE), 
				      VALUE_MONE,
				      vect_new(p_base->var, VALUE_ONE),
				      NO_OFL_CTRL));
      }
      for(pc = p_syst->inegalites; pc != NULL; pc = pc->succ) {
	pc->vecteur = vect_var_subst(pc->vecteur, p_base->var,
				     vect_cl_ofl_ctrl
				     (vect_new(var_q, VALUE_ONE), VALUE_MONE,
				      vect_new(p_base->var, VALUE_ONE),
				      NO_OFL_CTRL));
      }
    }

    /* ajout de -I , 1 en bas de la matrice           */
    p_vect_baux = (Pvecteur) p_syst->base;
    for(aux1 = 0; aux1 < nb_var ; aux1++) {
	p_vect_aux = vect_new(p_vect_baux->var, VALUE_ONE);
	vect_add_elem(&p_vect_aux, var_q, (Value) VALUE_MONE);
	p_cont_aux = contrainte_make(p_vect_aux);
	insert_ineq_end_sc(p_syst,p_cont_aux);
	p_vect_baux = p_vect_baux->succ;      
    }

    return p_syst;
}
