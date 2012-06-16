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

/* test du calcul de l'enveloppe convexe de deux polyedres
 *
 * Francois Irigoin, Decembre 1989
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <string.h>

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"

#include "polyedre.h"

main(argc,argv)
int argc;
char * argv[];
{
    Psysteme s1;
    Psysteme s2;
    Ppoly p1;
    Ppoly p2;
    Ppoly p_env;
    FILE * f1;
    FILE * f2;

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
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",argv[1]);
	exit(1);
    }

    /* lecture du deuxieme systeme */
    if(sc_fscan(f2,&s2)) {
	fprintf(stderr,"syntaxe correcte dans %s\n",argv[2]);
	sc_fprint(stdout, s2, *variable_default_name);
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",argv[2]);
	exit(2);
    }

    /* mise de s2 dans la base de s1. On admet que la base de s1
       contient la base de s2 */
    s2 = sc_translate(s2, s1->base, variable_default_name);

    /* conversion en polyedres */
    p1 = sc_to_poly(s1);
    p2 = sc_to_poly(s2);

    p_env = env(p1, p2);
    poly_fprint(stdout, p_env, variable_default_name);
}

