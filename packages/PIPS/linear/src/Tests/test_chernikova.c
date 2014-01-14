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

 /* test de chernikovaa */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <string.h>

#include "boolean.h"
#include "arithmetique.h"

#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sc.h"
#include "sg.h"

typedef struct vector
{
  int size;
  int *p;
} vector;


typedef struct matrix
{
  int nbrows;
  int nbcolumns;
  int **p;
  int *p_init;
} matrix;


int main(
    int argc,
    char **argv)
{
    FILE * f;
    char * filename = "stdin";
    Psysteme sc=SC_EMPTY;
    Psysteme sc1;
    Ptsg sg;

    if(argc==1) {
	f = stdin;
	fprintf(stderr,"From stdin\n");
    }
    else if (argc==2) {
	filename = strdup(argv[1]);
	if((f = fopen(filename,"r")) == NULL) {
	    fprintf(stderr,"Cannot open file %s\n", filename);
	    return 4;
	}
    }
    else {
	fprintf(stderr,"Usage: test_chernikova [filename]\n");
	return 1;
    }

    sg = sg_new();
    if(sc_fscan(f,&sc)) {
	printf("Initial constraint system:\n");
	sc_fprint(stdout,sc,*variable_default_name);

	sg = sc_to_sg_chernikova(sc);
	printf("Generating system\n");
	sg_fprint(stdout,sg,*variable_default_name);
	
	sc1 = sc_new();
	sc1 = sg_to_sc_chernikova(sg);
	printf("Regenerated constraint system:\n");
	sc_fprint(stdout,sc,*variable_default_name);
    }
    else {
	fprintf(stderr,"syntax error in %s\n",filename);
    }
    return 0;
}					/* main */
