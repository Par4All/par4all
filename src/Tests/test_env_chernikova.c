 /* Test de l'enveloppe convexe de deux systemes. L'enveloppe 
 * convexe est faite par traduction des systemes lineaires en 
 * systemes generateurs (par chernikova), puis par union des 
 * systemes generateurs, enfin par la traduction du systeme generateur 
 * resultant en systeme lineaire (toujours par chernikovva). 
 * Cette fonction utilise la bibliotheque fournie par l'IRISA. 
 * On suppose que les deux systemes fournis en entree ont la 
 * meme base */

#include <stdio.h>
#include <malloc.h>
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
    if(sc_fscan(f1,&sc1)) {
	fprintf(stderr,"syntaxe correcte dans %s\n",argv[1]);
	sc_fprint(stdout, sc1, *variable_default_name);
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",argv[1]);
	exit(1);
    }


    /* lecture du premier systeme */
    if(sc_fscan(f2,&sc2)) {
 	fprintf(stderr,"syntaxe correcte dans %s\n",argv[2]);
	sc_fprint(stdout, sc2, *variable_default_name);
    }
    else {
	fprintf(stderr,"erreur de syntaxe dans %s\n",argv[2]);
	exit(1);
    }

    sc1 = sc_normalize(sc1);
    sc2 = sc_normalize(sc2);
    sc_fprint(stdout, sc1, *variable_default_name);
    sc_fprint(stdout, sc2, *variable_default_name);

    assert(vect_size(sc1->base) == vect_size(sc2->base));

    sc2 = sc_translate(sc2,sc1->base,  *variable_default_name);
    sc=sc_convex_hull(sc1,sc2); 

    printf("systeme correspondant \n");
    sc_fprint(stdout, sc, *variable_default_name);
} /* main */
