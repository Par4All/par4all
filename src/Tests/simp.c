
/* test du simplex : ce test s'appelle par :
 *  programme fichier1.data fichier2.data ... fichiern.data
 * ou bien : programme<fichier.data
 * Si on compile grace a` "make sim" dans le directory
 *  /home/users/pips/C3/Linear/Development/polyedre.dir/test.dir
 * alors on peut tester l'execution dans le meme directory
 * en faisant : tests|more
 */

#include <stdio.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sc.h"
#include "sg.h"
#include "types.h"
#include "polyedre.h"

main(int argc, char *argv[])
{
/*  Programme de test de faisabilite'
 *  d'un ensemble d'equations et d'inequations.
 */
    FILE * f1;
    Psysteme sc=sc_new(); 
    int i; /* compte les systemes, chacun dans un fichier */

/* lecture et test de la faisabilite' de systemes sur fichiers */

    if(argc>=2) for(i=1;i<argc;i++){
        if((f1 = fopen(argv[i],"r")) == NULL) {
    	      fprintf(stdout,"Ouverture fichier %s impossible\n",
    		      argv[1]);
    	      exit(4);
        }
        printf("systeme initial \n");
        if(sc_fscan(f1,&sc)) {
            fprintf(stdout,"syntaxe correcte dans %s\n",argv[i]);
            sc_fprint(stdout, sc, *variable_default_name);
            printf("Nb_eq %d , Nb_ineq %d, dimension %d\n",
                NB_EQ, NB_INEQ, DIMENSION) ;
            if(feasible(sc)) printf("Systeme faisable (soluble) en rationnels\n") ;
            else printf("Systeme insoluble\n");
            fclose(f1) ;
        }
        else {
            fprintf(stderr,"erreur syntaxe dans %s\n",argv[1]);
            exit(1);
        }
    }
    else { f1=stdin ;

/* lecture et test de la faisabilite' du systeme sur stdin */

        printf("systeme initial \n");
        if(sc_fscan(f1,&sc)) {
	    fprintf(stdout,"syntaxe correcte dans %s\n",argv[1]);
	    sc_fprint(stdout, sc, *variable_default_name);
            printf("Nb_eq %d , Nb_ineq %d, dimension %d\n",
                NB_EQ, NB_INEQ, DIMENSION) ;
            if(feasible(sc)) printf("Systeme faisable (soluble) en rationnels\n") ;
            else printf("Systeme insoluble\n");
            exit(0) ;
        }
        else {
	    fprintf(stderr,"erreur syntaxe dans %s\n",argv[1]);
	    exit(1);
        }
    }
    exit(0) ;
}

