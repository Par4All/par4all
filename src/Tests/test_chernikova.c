 /* test de chernikovaa */

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
