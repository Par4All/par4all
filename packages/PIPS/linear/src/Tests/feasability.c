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

#include "boolean.h"
#include "assert.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

static void
test_system(Psysteme sc)
{
    CATCH(overflow_error) 
	fprintf(stdout, "*** Arithmetic error occured in simplex\n");
    TRY
	if (sc_feasibility_ofl_ctrl(sc, false, OFL_CTRL, true))
	    printf("Systeme faisable (soluble) en rationnels\n") ;
	else
	    printf("Systeme insoluble\n");
}

static void 
test_file(FILE * f, char * name)
{
    Psysteme sc=sc_new(); 
    printf("systeme initial \n");
    if(sc_fscan(f,&sc)) 
    {
	printf("syntaxe correcte dans %s\n",name);
	sc_fprint(stdout, sc, *variable_default_name);
	printf("Nb_eq %d , Nb_ineq %d, dimension %d\n",
	       sc->nb_eq, sc->nb_ineq, sc->dimension) ;
	test_system(sc);
    }
    else
    {
	fprintf(stderr,"erreur syntaxe dans %s\n",name);
	exit(1);
    }
}

int 
main(int argc, char *argv[])
{
    /*  Programme de test de faisabilite'
     *  d'un ensemble d'equations et d'inequations.
     */
    FILE * f1;
    int i; /* compte les systemes, chacun dans un fichier */

    initialize_sc(variable_default_name);
    
    /* lecture et test de la faisabilite' de systemes sur fichiers */

    if(argc>=2) 
    {
	for(i=1;i<argc;i++)
	{
	    if((f1 = fopen(argv[i],"r")) == NULL) {
		fprintf(stdout,"Ouverture fichier %s impossible\n", argv[i]);
		exit(4);
	    }
	    test_file(f1, argv[i]);
	    fclose(f1) ;
	}
    }
    else 
    {
	test_file(stdin, "standard input");
    }
    exit(0) ;
}
