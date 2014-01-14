/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
    Psysteme sc1=sc_new(); 
    Psysteme sc2=sc_new();
    Psysteme sc=sc_new();
    Ptsg sg = sg_new();
    Ptsg sg1 = sg_new();
    Ptsg sg2 = sg_new();

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
    if(sc_fscan(f1,&sc1)) {
	fprintf(stderr,"syntaxe correcte dans %s\n",argv[1]);
	sc_fprint(stdout, sc1, *variable_default_name);
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",argv[1]);
	exit(1);
    }

    /* lecture du deuxieme systeme */
    if(sc_fscan(f2,&sc2)) {
	fprintf(stderr,"syntaxe correcte dans %s\n",argv[2]);
	sc_fprint(stdout, sc2, *variable_default_name);
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",argv[2]);
	exit(1);
    }

    /* Homogeneiser les bases des systemes */
    assert(vect_size(sc1->base) == vect_size(sc2->base));
    sc2 = sc_translate(sc2,sc1->base,  *variable_default_name);

    assert(!SC_UNDEFINED_P(sc1) && !SC_UNDEFINED_P(sc2));

    if (SC_RN_P(sc2) || sc_rn_p(sc2) || sc_dimension(sc2)==0
	|| sc_empty_p(sc1) || !sc_faisabilite(sc1)) {
	Psysteme sc3 = sc_dup(sc2);
	sc3 = sc_elim_redond(sc3);
	sc = (SC_UNDEFINED_P(sc3)) ? sc_empty(base_dup(sc_base(sc2))) : sc3;
    }
    else if (SC_RN_P(sc1) ||sc_rn_p(sc1) || sc_dimension(sc1)==0   
	     || sc_empty_p(sc2) || !sc_faisabilite(sc2)) {
	Psysteme sc4 = sc_dup(sc1);
	sc4 = sc_elim_redond(sc4);
	sc = (SC_UNDEFINED_P(sc4)) ? sc_empty(base_dup(sc_base(sc1))) : sc4;
    }
    else {
	/* conversion en systemes generateurs */
	/*	printf("systemes initiaux \n");
		sc_dump(sc1);
		sc_dump(sc2); */
	sg1 = sc_to_sg_chernikova(sc1);
	sg2 = sc_to_sg_chernikova(sc2);

	/* calcul de l'enveloppe convexe */
	sg = sg_union(sg1, sg2);
	/*printf("union des systemes generateurs\n");
	  sg_fprint(stdout,sg,variable_dump_name);
	  */
	sc = sg_to_sc_chernikova(sg);
	/*	printf("systeme final \n");
		sc_dump(s);  */
    }

    printf("systeme correspondant \n");
    sc_fprint(stdout, sc, *variable_default_name);
}					/* main */
