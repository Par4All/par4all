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
/* Name     : ps_to_fic_pip.c
 * Package  : pip
 * Author   : F. Dumontet
 * Date     : july 93
 * Historic :
 * - 18 oct 93. L'ordre dans le vecteur n'est plus important. AP
 *
 * Documents:
 *
 * Comments :
 */

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "genC.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"
#include "paf_ri.h"
#include "pip.h"

#define PIP_IN_FILE "pip_in"

#define OK 0
#define EGALITE 1
#define INEGALITE 2
#define ERREUR_TAILLE 3

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: ecrit_commentaire2                                                   */
/*                                                                            */
/* Parameters: commentaire: le texte du commentaire.                          */
/*             p_sys_tab: le Psysteme traite. Contient le nom des variables   */
/*                        et des parametres.                                  */
/*             nb_var: discrimine les variables des parametres                */
/*                                                                            */
/* Result: void                                                               */
/*                                                                            */
/* Aims: ecrire le commentaire et le  nom des variables et des parametres.    */
/*                                                                            */
/* Author: F Dumontet                                                         */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
ecrit_commentaire2(commentaire, p_sys_tab, nb_var, nom_fichier)
        char            *commentaire;
        Psysteme        p_sys_tab;
        int             nb_var;
        FILE            *nom_fichier;
	

{
        int             aux;
        Pvecteur        p_vecteur_aux;

        fprintf(nom_fichier, "(%s (variables", commentaire);

        p_vecteur_aux = (Pvecteur) p_sys_tab->base;
        for (aux = 1; aux <= nb_var; aux++) {
            fprintf(nom_fichier, " %s",
                    (char *) entity_local_name(
                                    (entity) p_vecteur_aux->var));
            p_vecteur_aux = p_vecteur_aux->succ;
        }
/*
        for (aux = 1; aux <= nb_var; aux++) {
            fprintf(nom_fichier, " %s", (char *) p_vecteur_aux->var);
            p_vecteur_aux = p_vecteur_aux->succ;
        }
*/
        fprintf(nom_fichier, ") (parametres");
        while (p_vecteur_aux != NULL) {
               fprintf(nom_fichier, " %s",
                       (char *) entity_local_name(
                                       (entity) p_vecteur_aux->var));
               p_vecteur_aux = p_vecteur_aux->succ;
        }
/*
        while (p_vecteur_aux != NULL) {
               fprintf(nom_fichier, " %s", (char *) p_vecteur_aux->var);
               p_vecteur_aux = p_vecteur_aux->succ;
        }
*/

        fprintf(nom_fichier, "))\n");
        return;
}


/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: ecrit_ligne                                                          */
/*                                                                            */
/* Parameters: p_vect: le vecteur a ecrire. Il doit PAS etre ordonne.         */
/*             p_sys_base: vecteur de base permet de connaitre l'ordre des    */
/*                         variables et de rajouter des 0 pour celles qui ne  */
/*                         sont pas dans la matrice creuse.                   */
/*             nom_ficher: pointeur sur le fichier destination.               */
/*             nb_var: permet de differencier les "nbvar" variables des       */
/*                     parametres dans la base.                               */
/*             eg_ineg: permet de distinger le cas dans lequel le vecteur     */
/*                      represente une egalite de celui ou il represente une  */
/*                      une inegalite.                                        */
/*                                                                            */
/* Result:  void                                                              */
/*                                                                            */
/* Aims:  ecrire un vecteur sur une ligne. Le vecteur doit etre ordonne       */
/*                                                                            */
/* Author: F Dumontet                                                         */
/*                                                                            */
/* Date of last change: AP, 18 oct 93. L'ordre dans le vecteur n'est plus     */
/*			important.                                            */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void ecrit_ligne(p_vect, p_sys_base, nom_fichier, nb_var, eg_ineg)
Pvecteur        p_vect;
Pbase           p_sys_base;
FILE            *nom_fichier;
int             nb_var,eg_ineg;
{
  Pvecteur        p_vect_aux = (Pvecteur) p_sys_base;
  int             aux = 0,      /* Compteur de variables deja vues */
		  first = 1;    /* Pas de blanc juste apres le #[ */

  fprintf(nom_fichier, "#[");

  for( ; (p_vect_aux != NULL) || (aux <= nb_var) ; aux++) {
    int val;

    if(aux == nb_var)
      val = (int) vect_coeff(TCST, p_vect);
    else {
      val = (int) vect_coeff(p_vect_aux->var, p_vect);
      p_vect_aux = p_vect_aux->succ;
    }

    if(first)
	first = 0;
    else
      fprintf(nom_fichier, " ");

    if(eg_ineg == EGALITE)
      fprintf(nom_fichier, "%i", val);
    else
      fprintf(nom_fichier, "%i", -val);
  }
  fprintf(nom_fichier, "]\n");
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: ecrit_tableau2.                                                      */
/*                                                                            */
/* Parameters: p_syst_tab: le Psysteme qui va etre ecrit sur le fichier. Ce   */
/*                          peut etre aussi bien un contexte q'un probleme.   */
/*             nb_var: le nombre de variables du Psysteme a differencier des  */
/*                     parametres.                                            */
/*             nom_fichier: fichier de destination.                           */
/*                                                                            */
/* Result: entier.                                                            */
/*                                                                            */
/* Aims: ecrire a partir d'un Psysteme, un tableau avec une ligne par vecteur */
/*       inegalite et deux pour un vecteur egalite ( opposes )                */
/*                                                                            */
/* Author: F Dumontet                                                         */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
int
ecrit_tableau2(p_syst_tab, nb_var, nom_fichier)
        Psysteme        p_syst_tab;
        int             nb_var;
        FILE            *nom_fichier;

{
        Pcontrainte     p_contr_tab_aux;

        fprintf(nom_fichier, "(");

        if (p_syst_tab != NULL)
            {
            p_contr_tab_aux = p_syst_tab->egalites;

            while (p_contr_tab_aux != NULL) {
                  ecrit_ligne(p_contr_tab_aux->vecteur, p_syst_tab->base,\
                              nom_fichier, nb_var, INEGALITE);
                  ecrit_ligne(p_contr_tab_aux->vecteur, p_syst_tab->base,\
                              nom_fichier, nb_var, EGALITE);
                  p_contr_tab_aux = p_contr_tab_aux->succ;
            }
            p_contr_tab_aux = p_syst_tab->inegalites;
      
            while (p_contr_tab_aux != NULL) {
                  ecrit_ligne(p_contr_tab_aux->vecteur, p_syst_tab->base, \
                              nom_fichier, nb_var, INEGALITE);
                  p_contr_tab_aux = p_contr_tab_aux->succ;
            }
        }
        fprintf(nom_fichier, ")\n");
        return OK;
}



/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: ecrit_probleme2.                                                     */
/*                                                                            */
/* Parameters: commentaire: chaine de caracteres contenant le texte du        */
/*                          commentaire.                                      */
/*             p_syst_tab: Psysteme devant etre resolu (il doit etre          */
/*                         ordonne).                                          */
/*             p_syst_cont: Psysteme contenant le contexte.                   */
/*             nb_var: le nombre de variables dans p_syst_tab.                */
/*             bg: numero d'ordre dans p_syst_tab du parametre devant etre    */
/*                 considere comme infiniment grand;                          */
/*             nb_var_cont: nbr de variables du contexte.                     */
/*                                                                            */
/* Result: entier indiquant si l'ecriture s'est bien deroulee.                */
/*                                                                            */
/* Aims: ecrire dans un fichier, au format d'entree reconnu par pip un        */
/*       probleme de programmation lineaire en nombre entiers. Il nous faut   */
/*       deux Psystemes. Le premier contient les variables, les parametres et */
/*       les constantes (dans l'ordre). Le second contient le contexte avec   */
/*       dans l'ordre ses variables et les constantes. L'odre est celui de la */
/*       base de chaque Psysteme. La distinction entre variables et parametres*/
/*       est effectuee au moyen de nb_var qui indique que les nb_var 1eres    */
/*       termes sont des variables.                                           */
/*                                                                            */
/* Author: F Dumontet                                                         */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
int
ecrit_probleme2(commentaire, p_syst_tab, p_syst_cont, nb_var, bg , nb_var_cont)
        char            *commentaire;
        Psysteme        p_syst_tab, p_syst_cont;
        int             nb_var, bg , nb_var_cont;

{
        Pvecteur        p_vecteur_aux;
        FILE            *nom_fichier;
        int             aux, aux1,nb_par;

        nom_fichier = fopen(PIP_IN_FILE, "w");
        if (nom_fichier == NULL)
            return errno;
        fprintf(nom_fichier,"(");
        ecrit_commentaire2(commentaire, p_syst_tab, nb_var, nom_fichier);
                              /* le nombre de variables */
        fprintf(nom_fichier, "%1d", nb_var);
        p_vecteur_aux = (Pvecteur) p_syst_tab->base;
        aux = 0;
        nb_par = p_syst_tab->dimension-nb_var;
                              /* le nombre de parametres */
        fprintf(nom_fichier, " %1d", nb_par);
        aux1 = (p_syst_tab->nb_eq * 2) + p_syst_tab->nb_ineq;
        fprintf(nom_fichier, " %1d", aux1);
        if (p_syst_cont != NULL)
            aux1 = (p_syst_cont->nb_eq * 2) + p_syst_cont->nb_ineq;
        else
            aux1 = 0;
                              /* le nombre de contraintes du systeme*/
                              /* le numero de la variable consideree*/
                              /* comme infiniment grande pour le max*/
                              /* l'indicateur de solution entiere.  */
        fprintf(nom_fichier, " %1d %1d 1\n", aux1, bg);

			/* ecriture du systeme a resoudre */
        ecrit_tableau2(p_syst_tab, nb_var, nom_fichier);

                        /* ecriture du contexte */
        ecrit_tableau2(p_syst_cont, nb_var_cont, nom_fichier);
        fprintf(nom_fichier, ")\n");

        return fclose(nom_fichier);
}


