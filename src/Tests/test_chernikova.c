 /* test de chernikovaa */

#include <stdio.h>
#include <malloc.h>
extern int fprintf();
extern int printf();
extern char * strdup();

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


main(argc, argv)
     int argc;
     char **argv;
{
    FILE * f;
    char * filename = "stdin";
    Psysteme sc=SC_EMPTY;
    Psysteme sc1;
    Ptsg sg;

    if(argc==1) {
	f = stdin;
	fprintf(stderr,"Lecture sur stdin\n");
    }
    else if (argc==2) {
	filename = strdup(argv[1]);
	if((f = fopen(filename,"r")) == NULL) {
	    fprintf(stderr,"Ouverture du fichier %s impossible\n",
		    filename);
	    exit(4);
	}
    }
    else {
	fprintf(stderr,"Usage: test_chernikova [filename]\n");
	exit(1);
    }

    sg = sg_new();
    if(sc_fscan(f,&sc)) {
	printf("Systeme a tester:\n");
	sc_fprint(stdout,sc,*variable_default_name);

	sg = sc_to_sg_chernikova(sc);
	printf("systeme generateur \n");
	sg_fprint(stdout,sg,*variable_default_name);
	
	sc1 = sc_new();
	sc1 = sg_to_sc_chernikova(sg);
	printf("Systeme lineaire:\n");
	sc_fprint(stdout,sc,*variable_default_name);
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",filename);
    }
    return 0;
}					/* main */
