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
/* Name     : solpip.c
 * Package  : pip
 * Author   : F. Dumontet
 * Date     : july 93
 * Historic :
 * - 17 nov 93, replace MakeBinaryCall() by make_op_exp(). AP
 *
 * Documents:
 *
 * Comments :
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"
#include "paf_ri.h"
#include "pip.h"

#define PREMIERE_VAR1_BASE premiere_var1 == 0
#define RESET_PREMIERE_VAR1 premiere_var1 = 0;
#define SET_PREMIERE_VAR1 premiere_var1 = 1;
#define PREMIERE_VAR_BASE premiere_var == 0
#define RESET_PREMIERE_VAR premiere_var = 0;
#define SET_PREMIERE_VAR premiere_var = 1;
#define OLD_VAR(x) x->var==NULL
#define MY_MIN 0
#define MY_MAX 1
#define VARSUPP "Variable_pour_max_en_min_q"


typedef struct baseaux
        {
        expression         var;
	entity		   ent;
        struct baseaux     *succ;
        } baseaux,*Pbaseaux;

typedef struct pileaux
        {
        Pbaseaux           pred;
        Pbaseaux           act;
        struct pileaux     *succ;
        } pileaux,*Ppileaux;

typedef struct pilepredicat
        {
        predicate           x;
        struct pilepredicat *succ;
        } pilepredicat, *Ppilepredicat;

typedef struct pilequast
        {
        quast              q;
        struct pilequast   *succ;
        } pilequast, *Ppilequast;

list        newparms_act;
list       solution_act;
static Ppilepredicat   predicate_act;
expression      expression_act;
quast           quast_false;
quast_value     quast_value_act;
conditional     conditional_act;
static Ppilequast      quast_true;
static Ppileaux        pile;
static Pbaseaux        new_base_ref;
static Pbaseaux        new_base_aux;
Pvecteur        vect_act, vect_aux, vect_var_aux;
Psysteme        psyst_act;
int             aux, premiere_var, premiere_var1,compteur_de_var;


/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: ajoute_constante                                                     */
/*                                                                            */
/* Parameters: ent : entier, valeur de la constante cree.                     */
/*                                                                            */
/* Side effect  : expression_act: est modifie, il lui est ajoute une          */
/*                constante.                                                  */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: ajouter une constante a expression_act:                              */
/*                         expression_act= expression_act+ent.                */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
ajoute_constante ( ent )

      int            ent;

{
        if (expression_act == NULL)
           {
           expression_act=int_to_expression(ent);
           return;
           }

        if (ent>=0)
           expression_act = make_op_exp(PLUS_OPERATOR_NAME, expression_act,
					int_to_expression( ent ));
        else
           expression_act = make_op_exp(MINUS_OPERATOR_NAME, expression_act,
                                	int_to_expression( -ent ));

        return;
}
/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: retire_par_de_pile.                                                  */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: new_base_ref: on en retire le dernier "newparm" ajoute.       */
/*              newparms_act: on retire le premier element de la liste.       */
/*              pile: on retire l'element du haut de la pile de gestion de    */
/*                    localite des "newparms".                                */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: retirer les parametres crees par le newparm correspondant:gestion de */
/*       la localite.                                                         */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void retire_par_de_pile()

{
       
        Ppileaux      pile1 = pile;

        newparms_act = CDR(newparms_act);
        if (pile->pred != NULL)
            pile->pred->succ = pile->act->succ;
        else
            new_base_ref = new_base_ref->succ;


        pile = pile->succ;
        free( pile1 );
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: init_quast.                                                          */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: expression_act: remis a 0.                                    */
/*              solution_act: remis a 0.                                      */
/*                                                                            */
/* Result: void                                                               */
/*                                                                            */
/* Aims: initialiser les variables necessaires a la creation d'un "quast".    */
/*                                                                            */
/* Author: F. Dumontet.                                                       */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
init_quast ()

{
extern  expression      expression_act;

        expression_act = NULL;
        solution_act = NIL;
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: creer_quast_value                                                    */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: quast_value_act: recoit le "quast_value" cree.                */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: creer un "quast_value" a partir d'une "solution".                    */
/*                                                                            */
/* Author: F. Dumontet                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
creer_quast_value ()

{
extern  quast_value      quast_value_act;

        if ( solution_act != NIL)
            quast_value_act = make_quast_value (is_quast_value_quast_leaf,\
                                 make_quast_leaf((char *) solution_act, 
						leaf_label_undefined) );
        else
            quast_value_act = quast_value_undefined;
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: creer_true_quast.                                                    */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: quast_true: recoit le true "quast_cree" sur le haut de la     */
/*                          pile.                                             */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: conserver le "true_quast" d'un IF dans la pile des "true_quast".     */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
creer_true_quast ()

{
      
        Ppilequast       new_quast;

        new_quast = (Ppilequast)malloc(sizeof(pilequast));
        new_quast->succ = quast_true;
        new_quast->q = quast_act;
        quast_true = new_quast;
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name:  creer_predicat.                                                     */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: predicat_act: recoit le "predicat" cree sur le haut de la     */
/*                            pile.                                           */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: creer un "predicat" a partir d'un "Psysteme", ce "predicat" est mis  */
/*       sur le haut de la pile des "predicats".                              */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
creer_predicat ()

{
			

        Ppilepredicat   new_predicate;

        new_predicate = (Ppilepredicat)malloc(sizeof(pilepredicat));
	if (psyst_act != NULL) {
		psyst_act->base = NULL;
		sc_creer_base( psyst_act );
	}
        new_predicate->x = make_predicate ( (char *) psyst_act);
        new_predicate->succ = predicate_act;
        predicate_act = new_predicate;
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: fait_quast.                                                          */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: quast_false: recoit le second "quast" d'un IF.                */
/*              quast_act: idem.                                              */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: creer un "quast" a partir d'un "quast_value" et d'un "newparms".     */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void fait_quast ()
{
extern  quast           quast_false, quast_act;
        list        newparms_aux;

        newparms_aux = gen_append(newparms_act,NIL);
        quast_act = make_quast ((char *) quast_value_act,\
                                (char *) newparms_aux);
        quast_false = quast_act;
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: fait_quast_value.                                                    */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: quast_value_act: recoit le "quast_value" cree.                */
/*              quast_true: on depile un element.                             */
/*              predicate_act: on depile un element.                          */
/*              conditional_act: recoit le "conditional" cree.                */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: creer un "quast_value" a partir d'un "predicat", d'un "true_quast",  */
/*       d'un "false_quast". le" true_quast" et le" predicat" sont consommes  */
/*       et donc retires de leur pile respective.                             */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
fait_quast_value ()
{

extern  quast_value     quast_value_act;
extern  conditional     conditional_act;
        Ppilepredicat   old_predicate = predicate_act;
        Ppilequast      old_quast = quast_true;

        conditional_act = make_conditional (predicate_act->x,\
                                            quast_true->q, quast_false);
        quast_value_act = make_quast_value (is_quast_value_conditional,\
                                            (char *) conditional_act);
        predicate_act = predicate_act->succ;
        free(old_predicate);
        quast_true = quast_true->succ;
        free(old_quast);
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: init_liste_vecteur.                                                  */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: vect_var_aux: initialisation, pointe sur le debut de la base  */
/*              des variables (au sens Psysteme) initiales du probleme.       */
/*              solution_act: mise a vide de la file.                         */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: initialiser les variables necessaires a la construction d'une        */
/*       "liste de vecteurs".                                                 */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
init_liste_vecteur ()
{
			
extern  Pvecteur        vect_var_aux;

        vect_var_aux = (Pvecteur) base_var_ref;
        solution_act = NIL;
      
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: init_vecteur.                                                        */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: expression_act: mise a vide.                                  */
/*              new_base_aux: pointe sur le premier element de la base locale */
/*                            c a d celle qui contient les "newparms".        */
/*              vect_aux: pointe sur la base initiale des parametres du pb.   */
/*              premiere_var: positionne a RESET.                             */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: initialiser les variables necessaires a l'ecriture d'un "vecteur"    */
/*       sous forme d'une "expression".                                       */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
init_vecteur ()
{

extern  expression      expression_act;
extern  Pvecteur        vect_aux;      
extern  int             premiere_var;

        expression_act = NULL;
        new_base_aux = new_base_ref;
        vect_aux = (Pvecteur) base_ref;
        RESET_PREMIERE_VAR
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: ecrit_coeff1.                                                        */
/*                                                                            */
/* Parameters: ent: entier, coefficient de la variable pointee par            */
/*                  "new_base_aux", a entrer dans le "Psysteme".              */
/*                                                                            */
/* Side effect: psyst_act: le "Psysteme" resultat avec une variable en plus   */
/*                         (dans la "base" et dans le "vecteur"). NON !!      */
/*              vect_aux: pointe sur la variable suivante a traiter.          */
/*              new_base_aux: pointe sur la variable suivante a traiter.      */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: ajouter le coefficient et sa variable correspondante dans le         */
/*       Psysteme a produire.                                                 */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
ecrit_coeff1 ( ent )
            int         ent;
{
    extern      Psysteme    psyst_act;
    extern      Pvecteur    vect_aux;

    if ((ind_min_max == MY_MAX) && (vect_aux != NULL) &&
	(vect_aux->var != NULL) &&
	(strcmp(entity_local_name((entity) vect_aux->var), VARSUPP) == 0)) {
	vect_aux = vect_aux->succ;
	new_base_aux = new_base_aux->succ;
	return;
    }
    if (new_base_aux == NULL) {
	/*vect_add_elem(&psyst_act->base, TCST, (value) 1);*/
	vect_add_elem(&psyst_act->inegalites->vecteur, (Variable) TCST,
		      (Value) -ent);
	return;
    }

    if (new_base_aux->var == NULL) {
	vect_add_elem(&psyst_act->base, vect_aux->var, (Value) 1);
	vect_add_elem(&psyst_act->inegalites->vecteur, vect_aux->var,
		      (Value) -ent);
	vect_aux = vect_aux->succ;
	psyst_act->dimension++;
    }
    else {
	vect_add_elem(&psyst_act->base,
		      (Variable) new_base_aux->ent, (Value) 1);      
	vect_add_elem(&psyst_act->inegalites->vecteur,
		      (Variable) new_base_aux->ent, (Value) -ent);
    }

    new_base_aux = new_base_aux->succ;
}

/*----------------------------------------------------------------------------*/
/*                  TEMPORAIRE                                                */
/* Name:                                                                      */
/*                                                                            */
/* Parameters:                                                                */
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
void
ecrit_resultat ()

{
      write_quast(stdout,quast_act);
}


/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: creer_Psysteme.                                                      */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: psyst_act: recoit le "Psysteme" cree.                         */
/*            vect_aux: pointe sur le premier parametre de la base initiale.  */
/*            new_base_aux: pointe sur le premier parametre de la base        */
/*                        contenant des les newparms.                         */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: initialiser le psysteme et les variables permettant de le construire.*/
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
creer_Psysteme()

{

extern  Psysteme      psyst_act;
extern  Pvecteur      vect_aux;
        Psysteme      p_syst;

        p_syst = sc_new();
        p_syst->dimension = 0;
        p_syst->nb_ineq = 1;
        p_syst->inegalites = contrainte_new();
        psyst_act = p_syst;
        vect_aux = (Pvecteur) old_base;/* a changer par base_ref*/
        new_base_aux = new_base_ref;

}


/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: init_new_base.                                                       */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: new_base_ref: recoit la nouvelle base initialisee.            */
/*              compteur de var: initialisation a 0.                          */
/*              newparms: initialisation de la liste a vide.                  */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: creer a partir de la base initiale , la nouvelle base contenant aussi*/
/*       les "newparms". Comme il s'agit de l'initialisation, il n'y a pas de */
/*       "newparms" donc tous les champs "var" sont a NULL. Les autres "var"  */
/*       permettant la construction future de cette base sont initialisees.   */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*	_ jan 5th 94, AP, new_base_ref MUST be initialized to NULL (when */
/*        base_ref is empty) */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void init_new_base ( )

{

extern  int            compteur_de_var;
        Pbaseaux       new_base_aux1 = (Pbaseaux) NULL, bbase_aux;
        Pvecteur       vect_aux1 = (Pvecteur) base_ref;
        int            premiere_var;

        compteur_de_var=0;
        RESET_PREMIERE_VAR
        newparms_act = NIL;
	new_base_ref = NULL;

        for ( ;vect_aux1 != NULL ; vect_aux1 =vect_aux1->succ)
            {
            bbase_aux = (Pbaseaux) malloc(sizeof(baseaux));
	    bbase_aux->var = NULL;
	    bbase_aux->ent = NULL;
	    bbase_aux->succ = NULL;

            if ( PREMIERE_VAR_BASE )
                  {
                  new_base_ref = bbase_aux;
                  new_base_aux1 = bbase_aux;
                  SET_PREMIERE_VAR
                  }
            else
                  {
                  new_base_aux1->succ = bbase_aux;
                  new_base_aux1 = new_base_aux1->succ;
                  }
            }      

         return;

}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: creer_nom_entite.                                                    */
/*                                                                            */
/* Parameters: module_name: le nom du module??  TOP_LEVEL_MODULE_NAME.        */
/*                                                                            */
/* Side effect: compteur_de_var: incrementation de 1.                         */
/*                                                                            */
/* Result: une "entite" dont le nom est cree automatiquement.                 */
/*                                                                            */
/* Aims: creer automatiquement des "entites" ayant pour nom nvarxxx.          */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
entity
creer_nom_entite(module_name)

char *module_name;

{
extern  int     compteur_de_var;
        string full_name;
        char *name;
        char *num;
        entity e ;
        basic b ;

        compteur_de_var++;
        num = (char*) malloc(64);
        name= (char*) malloc(70);
        (void) sprintf( name, "nvar%d" , compteur_de_var);
        (void) sprintf((char *) (num), "%d", compteur_de_var);

        debug(8,"make_scalar_integer_entity", "begin name=%s, module_name=%s\n",
              name, module_name);

        full_name = concatenate(module_name, MODULE_SEP_STRING, name, NULL);
        e = make_entity((char *) strdup(full_name),
                        type_undefined,
                        storage_undefined,
                        value_undefined);

        b = make_basic(is_basic_int, 4);

        entity_type(e) = (type) MakeTypeVariable(b, NIL);

        return (e);
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: ajoute_new_var.                                                      */
/*                                                                            */
/* Parameters: ent: valeur du diviseur.                                       */
/*            rang: position de la nouvelle variable dans le vecteur.         */
/*                                                                            */
/* Side effect: new_base_ref: lorsque la nouvelle variable est ajoutee en tete*/
/*              de la base, recoit la nouvelle base modifiee.                 */
/*              pile: ajout du nouveau parametre sur la pile pour la gestion  */
/*                    de la localite de ces variables.                        */
/*              newparms_act: ajout du nouveau parametre a la liste.          */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: ajouter un nouveau parametre dans la nouvelle base. Les piles de     */
/*       gestion de la localite et de reference entre "entite" et "expression"*/
/*       sont mises a jour.                                                   */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void ajoute_new_var (  ent , rang )

        int            ent, rang;

{


        Pbaseaux      new_var, new_base_aux1 = new_base_ref ;
        expression    expr;
        Ppileaux      new_pile;
        entity        enti_act;
        var_val       var_val_act;
        int           aux;

        expr = make_op_exp(DIVIDE_OPERATOR_NAME, expression_act,
                           int_to_expression( ent ));
        for (aux=0; aux<(rang-1) ; aux++)
            { new_base_aux1 = new_base_aux1->succ;}

        new_var=(Pbaseaux) malloc(sizeof(baseaux));
        new_var->var =  expr;

        enti_act = creer_nom_entite(TOP_LEVEL_MODULE_NAME);
	new_var->ent = enti_act;

        var_val_act = make_var_val(enti_act, expr);
        newparms_act = CONS(VAR_VAL,var_val_act,newparms_act);      
        new_pile = (Ppileaux) malloc(sizeof(pileaux));
        new_pile->act = new_var;
        new_pile->succ = pile;

        if ( rang > 0)
            {
            new_var->succ = new_base_aux1->succ;
            new_base_aux1->succ = new_var;
            new_pile->pred = new_base_aux1;
            }
        else
            {
            new_var->succ = new_base_ref;
            new_base_ref = new_var;
            new_pile->pred = NULL;
            }
        pile = new_pile;
        return ;
}


/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name: ecrit_liste_vecteur.                                                 */
/*                                                                            */
/* Parameters:                                                                */
/*                                                                            */
/* Side effect: solution_act: ajout d'un "vecteur" a la "liste de vecteurs".  */
/*            vect_var_aux: passe a la variable suivante.                     */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: mettre dans la "liste de vecteur" "l'expression" correspondant a un  */
/*       "vecteur".                                                           */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/
void
ecrit_liste_vecteur()


{
extern  Pvecteur      vect_var_aux;

        if ( vect_var_aux == NULL )
            return;

        solution_act = gen_nconc(solution_act, CONS(EXPRESSION, expression_act, NIL));
        vect_var_aux = vect_var_aux->succ;
        return;
}


void ecrit_une_var_neg(int ent);

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name:  ecrire_une_var.                                                     */
/*                                                                            */
/* Parameters: ent: valeur du parametre.                                      */
/*                                                                            */
/* Side effect: new_base_aux, vect_aux: pointent sur le prochain parametre    */
/*                  respectivement dans les bases initiale et actuelle.       */
/*            expression_act: l'expression obtenue.                           */
/*            premiere_ver: mis a set.                                        */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: ajouter un parametre et son coefficient a une "expression"           */
/*       correspondant a un vecteur. Les pointeurs de base sont actualises .  */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
ecrit_une_var ( ent ) 

        int            ent;

{

extern  Pvecteur        vect_aux;
extern  expression      expression_act;
extern  int             premiere_var;
        expression      e_act = expression_act;
        Pbaseaux        p_base1 = new_base_aux;

        if (ind_min_max == MY_MAX)
            {
            if ((vect_aux != NULL) && (vect_aux->var != NULL) &&\
                (strcmp(entity_local_name((entity) vect_aux->var), VARSUPP) == 0))
                {
                ent = 0;
                }
            else
                {
                if ( (p_base1 == NULL) || (OLD_VAR(p_base1)) )
                    {
                    ent = -ent; 
                    }
                else
                    {
                    ecrit_une_var_neg (ent);
                    return;
                    }
                }
            } 

        if ((p_base1 == NULL) || ((vect_aux != NULL)&&(vect_aux->var == TCST)))
            {
            ajoute_constante( ent );
            return;
            }

        if ( ent == 0 )
                {
                if (OLD_VAR(p_base1))
                        vect_aux = vect_aux->succ;
                new_base_aux = new_base_aux->succ;
                return;
                }



        if ( PREMIERE_VAR_BASE )
            {
            if ( OLD_VAR(p_base1) )
                  {
                  expression_act = make_factor_expression( ent, \
                        (entity) vect_aux->var);

                  vect_aux = vect_aux->succ;
                  }
            else
                  {
                  expression_act =  make_op_exp(MULTIPLY_OPERATOR_NAME,
                                        	int_to_expression( ent ),
                                        	(expression) p_base1->var);
                  }
            SET_PREMIERE_VAR
            }
        else
            {
            if ( OLD_VAR(p_base1) )
                  {
                  expression_act = make_op_exp(PLUS_OPERATOR_NAME, e_act,
                                       	       make_factor_expression(ent,
                                                 (entity) vect_aux->var));

                  vect_aux = vect_aux->succ;
                  }
            else
                  {
                   expression_act = make_op_exp(PLUS_OPERATOR_NAME, e_act,
                                        	make_op_exp(MULTIPLY_OPERATOR_NAME,
                                            		    int_to_expression( ent ),
                                            		    (expression) p_base1->var));
                  }
            }      
        new_base_aux = new_base_aux->succ;

        return ;
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name:  ecrit_une_var_neg.                                                  */
/*                                                                            */
/* Parameters: ent: valeur du parametre negatif.                              */
/*                                                                            */
/* Side effect: new_base_aux, vect_aux: pointent sur le prochain parametre    */
/*                      respectivement dans les bases initiale et actuelle.   */
/*              expression_act: l'expression obtenue.                         */
/*              premiere_ver: mis a set.                                      */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: ajouter un parametre et son coefficient a une "expression"           */
/*       correspondant a un vecteur. Les pointeurs de base sont actualises .  */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
ecrit_une_var_neg (ent)

        int             ent;

{


extern  expression      expression_act;
extern  int             premiere_var;
extern  int             ind_min_max;
        expression      e_act = expression_act;
        Pbaseaux        p_base1 = new_base_aux;


        if (ind_min_max == MY_MAX)
            {
            if ((vect_aux != NULL) && (vect_aux->var != NULL) &&\
                (strcmp(entity_local_name((entity) vect_aux->var), VARSUPP) == 0))
                {
                ent = 0;
                }
            else
                {
                ind_min_max = MY_MIN;
                ecrit_une_var(ent);
                ind_min_max = MY_MAX;
                return;
                }
            }

        if ((p_base1 == NULL) || ((vect_aux != NULL)&&(vect_aux->var == TCST)))
                {
                ajoute_constante( -ent );
                return;
                }

        if ( OLD_VAR(p_base1) )
            {
            ecrit_une_var(-ent);
            return;
            }
      else
            {
            expression_act = MakeUnaryCall(\
                                 gen_find_tabulated(\
                                          make_entity_fullname(\
                                               TOP_LEVEL_MODULE_NAME,\
                                               UNARY_MINUS_OPERATOR_NAME),\
                                          entity_domain),\
                                 int_to_expression( ent ));

            if ( PREMIERE_VAR_BASE )
                  {
                  expression_act =  make_op_exp(MULTIPLY_OPERATOR_NAME,
                                        	expression_act,
                                        	(expression) p_base1->var);
                  SET_PREMIERE_VAR
                  }
            else
                  {
                   expression_act = make_op_exp(PLUS_OPERATOR_NAME, e_act,
                                        	make_op_exp(MULTIPLY_OPERATOR_NAME,
                                            		    expression_act,
                                            		    (expression) p_base1->var));
                  }
            }
        new_base_aux = new_base_aux->succ;
        return;
}

/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name:  ecrit_coeff_neg2.                                                   */
/*                                                                            */
/* Parameters: ent: valeur du parametre.                                      */
/*                                                                            */
/* Side effect: new_base_aux, vect_aux: pointent sur le prochain parametre    */
/*                      respectivement dans les bases initiale et actuelle.   */
/*              expression_act: l'expression obtenue.                         */
/*              premiere_ver: mis a set.                                      */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: ajouter un parametre et son coefficient a une "expression"           */
/*       correspondant a un vecteur. Les pointeurs de base sont actualises .  */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
ecrit_coeff_neg2 ( ent )

        int             ent;
{

extern  int             ind_min_max;
        int             garde_min_max = MY_MIN;


        if (ind_min_max == MY_MAX)
            garde_min_max = MY_MAX; 
        ind_min_max = MY_MIN;
        if (new_base_aux == NULL)
            ajoute_constante ( -ent );
        else
            ecrit_une_var_neg( ent );
        if (garde_min_max == MY_MAX)
            ind_min_max = MY_MAX;
}
/*----------------------------------------------------------------------------*/
/*                                                                            */
/* Name:  ecrire_coeff2 .                                                     */
/*                                                                            */
/* Parameters: ent: valeur du parametre.                                      */
/*                                                                            */
/* Side effect: new_base_aux, vect_aux: pointent sur le prochain parametre    */
/*                      respectivement dans les bases initiale et actuelle.   */
/*              expression_act: l'expression obtenue.                         */
/*              premiere_ver: mis a set.                                      */
/*                                                                            */
/* Result: void.                                                              */
/*                                                                            */
/* Aims: ajouter un parametre et son coefficient a une "expression"           */
/*       correspondant a un vecteur. Les pointeurs de base sont actualises .  */
/*                                                                            */
/* Author: F Dumontet.                                                        */
/*                                                                            */
/* Date of last change:                                                       */
/*                                                                            */
/*----------------------------------------------------------------------------*/

void
ecrit_coeff2 ( ent )

        int             ent;
{
extern  int             ind_min_max;
        int             garde_min_max = MY_MIN;

        if (ind_min_max == MY_MAX)
            garde_min_max = MY_MAX;
        ind_min_max = MY_MIN;
        if (new_base_aux == NULL)
            ajoute_constante ( ent );
        else
            ecrit_une_var( ent );
        if (garde_min_max == MY_MAX)
            ind_min_max = MY_MAX;
}


