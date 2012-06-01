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

 /* Test de l'enveloppe convexe de deux systemes. L'enveloppe convexe
 * est faite par traduction des systemes lineaires en systemes
 * generateurs (par chernikova), puis par union des systemes
 * generateurs, enfin par la traduction du systeme generateur
 * resultant en systeme lineaire (toujours par chernikovva).  Cette
 * fonction utilise la bibliotheque fournie par l'IRISA.  On suppose
 * que les deux systemes fournis en entree ont la meme base
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
extern int fprintf();
extern int printf();
extern char * strdup();

#include "boolean.h"
#include "arithmetique.h"
#include"assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sc.h"
#include "sg.h"
#include "types.h"
#include "polyedre.h"


main(argc, argv)
     int argc;
     char **argv;
{
    FILE * f1;
    FILE * f2;
    char * filename = "stdin";
    Psysteme s1=sc_new(); 
    Psysteme s2=sc_new();
    Psysteme s=sc_new();
    Ptsg sg = sg_new();
    Ptsg sg1,sg2 ;
    Matrix *a;
    Polyhedron *A;
    if(argc!=3) {
	fprintf(stdout,"Usage: %s sc1 sc2\n",argv[0]);
	exit(1);
    }

    if((f1 = fopen(argv[1],"r")) == NULL) {
	fprintf(stdout,"Ouverture du fichier %s impossible\n",
		argv[1]);
	exit(4);
    }

    if((f2 = fopen(argv[2],"r")) == NULL) {
	fprintf(stdout,"Ouverture du fichier %s impossible\n",
		argv[2]);
	exit(4);
    }

    /* lecture du premier systeme */
    if(sc_fscan(f1,&s1)) {
	fprintf(stderr,"syntaxe correcte dans %s\n",argv[1]);
	sc_fprint(stdout, s1, *variable_default_name);
	assert(sc_weak_consistent_p(s1));
	
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",argv[1]);
	exit(1);
    }


    /* lecture du deuxieme systeme */
    if(sc_fscan(f2,&s2)) {
 	fprintf(stderr,"syntaxe correcte dans %s\n",argv[2]);
	sc_fprint(stdout, s2, *variable_default_name);
	assert(sc_weak_consistent_p(s2));
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",argv[2]);
	exit(1);
    }

    /* FI: commented out because prevents tests with empty systems */
    /*
    s1 = sc_normalize(s1);
    s2 = sc_normalize(s2);
    sc_fprint(stdout, s1, *variable_default_name);
    sc_fprint(stdout, s2, *variable_default_name);
    */

    assert(vect_size(s1->base) == vect_size(s2->base));

    s2 = sc_translate(s2,s1->base,  *variable_default_name);
    if (SC_RN_P(s2) || sc_rn_p(s2) || sc_dimension(s2)==0
	|| sc_empty_p(s1) || !sc_faisabilite_ofl(s1)) {
	Psysteme sc2 = sc_dup(s2);
	sc2 = sc_elim_redond(sc2);
	s = (SC_UNDEFINED_P(sc2)) ? sc_empty(base_dup(sc_base(s2))) : sc2;
    }
    else if (SC_RN_P(s1) ||sc_rn_p(s1) || sc_dimension(s1)==0   
	     || sc_empty_p(s2) || !sc_faisabilite_ofl(s2)) {
	Psysteme sc1 = sc_dup(s1);
	sc1 = sc_elim_redond(sc1);
	s = (SC_UNDEFINED_P(sc1)) ? sc_empty(base_dup(sc_base(s1))) : sc1;
    }
    else {
	/* calcul de l'enveloppe convexe */
	/* s = sc_new(); */
	/* s = sc_convex_hull(s1,s2); */
	s = sc_common_projection_convex_hull(s1,s2);
    }

    printf("systeme correspondant \n");
    sc_fprint(stdout, s, *variable_default_name);
    exit(0);
} /* main */
