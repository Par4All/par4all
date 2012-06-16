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
 * que les deux systemes fournis en entree ont la meme base */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sc.h"
#include "sg.h"
#include "polyedre.h"

Psysteme read_syst_from_file(char * name)
{
    FILE * f;
    Psysteme s = sc_new();
    
    if((f = fopen(name,"r")) == NULL) {
	fprintf(stderr,"Ouverture du fichier %s impossible\n",name);
	exit(2);
    }

    if(sc_fscan(f,&s)) {
	fprintf(stderr,"syntaxe correcte dans %s\n",name);
	/* sc_fprint(stderr, s, *variable_default_name); */
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",name);
	exit(3);
    }
    
    return s;
}

int main(int argc, char **argv)
{
    Psysteme sc1, sc2, sc;

    if(argc!=3) {
	fprintf(stdout,"Usage: %s sc1 sc2\n",argv[0]);
	exit(1);
    }

    sc1 = read_syst_from_file(argv[1]);
    sc2 = read_syst_from_file(argv[2]);

    sc1 = sc_normalize(sc1);
    sc2 = sc_normalize(sc2);

    sc_fprint(stdout, sc1, *variable_default_name);
    sc_fprint(stdout, sc2, *variable_default_name);

    assert(vect_size(sc1->base) == vect_size(sc2->base));

    sc2 = sc_translate(sc2, sc1->base, *variable_default_name);
    sc = sc_convex_hull(sc1,sc2); 

    printf("systeme correspondant \n");
    sc_fprint(stdout, sc, *variable_default_name);

    return 0;
}

/* that is all
 */
