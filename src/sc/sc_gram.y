/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/


%{

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

extern int yyerror(char*);
extern int yylex(void);

bool yysyntax_error;

Psysteme ps_yacc;

Value fac;        /* facteur multiplicatif suivant qu'on analyse un terme*/
                      /* introduit par un moins (-1) ou par un plus (1) */

int sens;       /* indique le sens de l'inegalite
		   sens = -1  ==> l'operateur est soit > ,soit >=,
		   sens = 1   ==> l'operateur est soit <, soit <=   */
short int cote; /* booleen indiquant quel membre est en cours d'analyse*/

Value b1, b2; /* element du vecteur colonne du systeme donne par l'analyse*/
              /* d'une contrainte */

Pcontrainte eq;   /* pointeur sur l'egalite ou l'inegalite
                                courante */
   
extern Pvecteur cp ;   /* pointeur sur le membre courant             */ 

short int operat;    /* dernier operateur rencontre                 */


/*code des operateurs de comparaison */

#define OPINF 1
#define OPINFEGAL 2
#define OPEGAL 3
#define OPSUPEGAL 4
#define OPSUP 5
#define DROIT 1
#define GAUCHE 2
/* #define NULL 0 */
%}

%union {
    Value Value;
    Variable Variable;
}

/* explicit types: Value may be larger than a pointer (e.g. long long)
 */
%type <Value> const
%type <Variable> ident

%token ACCFERM		/* accolade fermante */ 
%token ACCOUVR		/* accolade ouvrante */ 
%term <Value> CONSTANTE	/* constante entiere sans signe  */ 
%token EGAL		/* signe == */
%term <Variable> IDENT		/* identificateur de variable */
%token INF		/* signe < */ 
%token INFEGAL		/* signe <= */
%token MOINS		/* signe - */
%token PLUS		/* signe + */
%token SUP		/* signe > */ 
%token SUPEGAL		/* signe >= */
%token VAR	     /* mot reserve VAR introduisant la liste de variables */
%token VIRG	     /* signe , */


%%
system	: inisys defvar ACCOUVR l_eq virg_opt ACCFERM
	; 

inisys	:
		{   /* initialisation des parametres du systeme */
                    /* et initialisation des variables */
                   
                       ps_yacc = sc_new();
		       init_globals();
		       yysyntax_error = false;
                }
	;

defvar	: VAR l_var
                 {   /* remise en ordre des vecteurs de base */
		     Pbase b;
		     
		     b = ps_yacc->base;
		     ps_yacc->base = base_reversal(b);
		     vect_rm(b);
		 }
	;

l_var	: newid
	| l_var VIRG newid
	;

l_eq	: eq
	| l_eq VIRG eq
	|
	;

eq	: debeq multi_membre op membre fin_mult_membre feq
	;

debeq	:
        {
	    fac = VALUE_ONE;
	    sens = 1;
	    cote = GAUCHE;
	    b1 = 0;
	    b2 = 0;
	    operat = 0;
	    cp = NULL;
	    eq = contrainte_new();
	}
	;

feq     :{ 

           contrainte_free(eq); 
        }
        ;

membre	: addop terme 
	| { fac = VALUE_ONE;} terme
	| membre addop terme
	;

terme	: const ident 
        {
	    if (cote==DROIT) fac = value_uminus(fac);
	    /* ajout du couple (ident,const) a la contrainte courante */
	    vect_add_elem(&(eq->vecteur),$2,value_mult(fac,$1));
	    /* duplication du couple (ident,const)
	       de la combinaison lineaire traitee           */ 
	    if (operat)
		vect_add_elem(&cp,(Variable) $2,
			      value_uminus(value_mult(fac,$1)));
	}
	| const
		{
		    Value p = value_mult(fac,$1);
		    if (cote==DROIT) {
			value_addto(b1, p);
			value_substract(b2, p);
		    } else {
			value_substract(b1, p);
			value_addto(b2, p);
		    }
		}     
	| ident
		{
		    if (cote==DROIT) fac = value_uminus(fac);
		    /* ajout du couple (ident,1) a la contrainte courante */
		    vect_add_elem (&(eq->vecteur),(Variable) $1,fac);
		    /* duplication du couple (ident,1) de la
		       combinaison lineaire traitee                    */
		    if (operat)
			vect_add_elem(&cp,(Variable) $1,value_uminus(fac));
		}
	;

ident	: IDENT { $$ = rec_ident(ps_yacc, $1); }
	;

newid	: IDENT	{ new_ident(ps_yacc,$1); }
	;

/* I'm pessimistic for long long here... 
 * should rather return a pointer to a Value stored somewhere...
 */
const	: CONSTANTE
		{ 
		    $$ = $1;
		}
	;

op	: INF
		{ cote = DROIT; 
                  sens = 1;
                  operat = OPINF;
                  cp = NULL;
                  b2 = VALUE_ZERO; }
	| INFEGAL
		{ cote = DROIT; 
                  sens = 1;
                  operat = OPINFEGAL;
                  cp = NULL;
                  b2 = VALUE_ZERO;}
	| EGAL
		{ cote = DROIT; 
                  sens = 1;
                  operat = OPEGAL; 
                  cp = NULL;
                  b2 = VALUE_ZERO;
                 }
	| SUP
		{ cote = DROIT; 
                  sens = -1;
                  operat = OPSUP; 
                  cp = NULL;
                  b2 = VALUE_ZERO;}
	| SUPEGAL
		{ cote = DROIT;	
                  sens = -1;
                  operat = OPSUPEGAL;
                  cp = NULL;
                  b2 = VALUE_ZERO;}
;

addop	: PLUS
        { fac = VALUE_ONE; }
	| MOINS
        { fac = VALUE_MONE; }
	;

multi_membre : membre
             | multi_membre op membre fin_mult_membre
             ;

fin_mult_membre :
                  {
                       vect_add_elem(&(eq->vecteur),TCST,value_uminus(b1));
			switch (operat) 
                        {
			case OPINF:
                                creer_ineg(ps_yacc,eq,sens);
                                vect_add_elem(&(eq->vecteur),TCST,VALUE_ONE);
				break;
			case OPINFEGAL:
                                creer_ineg(ps_yacc,eq,sens);
				break;
			case OPSUPEGAL:
                                creer_ineg(ps_yacc,eq,sens);
				break;
			case OPSUP:
				creer_ineg(ps_yacc,eq,sens);
                                vect_add_elem (&(eq->vecteur),TCST,VALUE_ONE);
                                break;
			case OPEGAL:
                                creer_eg(ps_yacc,eq);
				break;
			}

                    eq = contrainte_new();
                    eq->vecteur = cp;
                    b1 = b2;

                  }
                ;

virg_opt : VIRG
         |
         ; 
%%

