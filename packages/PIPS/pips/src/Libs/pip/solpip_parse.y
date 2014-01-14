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

/* Name     : solpip_parse.y
 * Package  : paf-util
 * Author   : F. Lamour, F. Dumontet
 * Date     : 25 march 1993
 * Historic : 2 august 93, moved into (package) paf-util, AP
 * Documents:
 *
 * Comments :
 * Grammaire Yacc pour interpreter une solution fournie par PIP en un "quast
 * newgen".
 */
 
%{
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>
#include <string.h>
#include <errno.h>

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
#include "matrix.h"

/* Pips includes        */
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "bootstrap.h"
#include "graph.h"
#include "paf_ri.h"
#include "paf-util.h"

%}

%union{
int valeur;
char identificateur;
char *blabla;
}

%start quast_sol
%token <valeur>  ENTIER
%token <blabla>  TEXTE
%token  LST LPAR RPAR LCRO RCRO DIES IF NEWPARM DIV DIV_OP MOINS_OP
%type <valeur> coefficient

%left DIV_OP 
%right MOINS_OP


%% 


/* Les regles de grammaire definissant une solution de PIP */
 
 
quast_sol          : LPAR LPAR commentaire RPAR  
		   {
		   init_new_base ();
		   }
		    super_quast RPAR
                  ;

commentaire       : 
		   |
		    TEXTE {}
                  |  
		    commentaire TEXTE
		  |
		   commentaire LPAR commentaire RPAR commentaire
                  ;


super_quast       : LPAR 
		   {
		   init_quast();
		   }
		    quast RPAR
		   {
		   fait_quast();
		   }

                  | nouveau_parametre super_quast
		   {
		   retire_par_de_pile();
		   }
                  ;


quast             : forme
		   {
		   creer_quast_value ();
		   } 
                  | IF vecteur1 
		   {
		   creer_predicat();
		   }
		    super_quast 
		   {
		   creer_true_quast ();
		   }
		    super_quast
		   {
		   fait_quast_value ();
		   }
                  ;


forme             :     
                  | LST 
		   {
		   init_liste_vecteur ();
		   }
		    liste_vecteur
                  ;
    

nouveau_parametre : NEWPARM ENTIER LPAR DIV vecteur2 ENTIER  RPAR RPAR
		   {
		   printf("nouveau_parametre1");
		   ajoute_new_var( $6 , $2 ); 
		   }
                  ;


liste_vecteur     : vecteur  
                  | liste_vecteur vecteur
                  ;


vecteur           : DIES LCRO 
		   {
		   init_vecteur ();
		   }
		    liste_coefficient RCRO 
		   {
		   ecrit_liste_vecteur();
		   }
                  ;
         
liste_coefficient : coefficient             {} 
                  | liste_coefficient coefficient 
                  ;


coefficient       : MOINS_OP ENTIER    
		   {
		    ecrit_une_var_neg( $2 );
		   }
                  | ENTIER                            
		   {
		   ecrit_une_var( $1 ); 
		   }
                  ;

vecteur2          : DIES LCRO
                   {
		   init_vecteur ();
                   }
                    liste_coefficient2 RCRO
                  ;

liste_coefficient2: coefficient2
                  | liste_coefficient2 coefficient2
                  ;


coefficient2      : MOINS_OP ENTIER
                   {
		   ecrit_coeff_neg2 ( $2 );
                   }
                  | ENTIER {}
                  ;

vecteur1          : DIES LCRO 
		   {
		   creer_Psysteme();
		   }
		    liste_coefficient1 RCRO 
                  ;

liste_coefficient1: coefficient1
                  | liste_coefficient1 coefficient1
                  ;


coefficient1      : MOINS_OP ENTIER
		   {
		   ecrit_coeff1 ( -$2 );
		   }	
                  | ENTIER
                   {
		   ecrit_coeff1 ( $1 );
                   }

		  ;

%% 

void yyerror(char* s)
{
    fputs(s,stderr);
	putc('\n',stderr);
}

    
