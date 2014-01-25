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

/* package sc */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

 /* Psysteme construit par sc_gram.y */
extern Psysteme ps_yacc;

 /* detection des erreurs de syntaxe par sc_gram.y */
extern bool syst_syntax_error;

 /* fichier lu par sc_lex.l */
extern FILE * syst_in;

/* Psysteme * sc_read(char * nomfic): construit un systeme d'inegalites
 * lineaires a partir d'une representation externe standard; les variables
 * utilisees dans le systeme doivent etre declarees dans la premiere ligne;
 * les inegalites sont separees par des virgules et regroupees entre
 * accolades; les egalites sont converties en paires d'inegalites
 *
 * Exemple:
 *     VAR x, y, z
 *     { x <= 1,
 *       y == 2, x + 2 z > y }
 *
 * Noter l'absence de signe "multiplier" entre les coefficients numeriques
 * et les noms de variables
 *
 * Cette fonction relit un fichier cree par la fonction sc_fprint()
 *
 * Ancien nom: fscsys_uf
 */
Psysteme * sc_read(char * nomfic)
{
	if ((syst_in = fopen(nomfic, "r")) == NULL) {
		(void) fprintf(stderr,
			       "Ouverture du fichier %s impossible\n",nomfic);
		exit(4);
	}
	sc_init_lex();
	syst_parse();
	return &ps_yacc;
}

/* bool sc_fscan(FILE * f, Psysteme * ps): construit un systeme d'inegalites
 * et d'egalites lineaires a partir d'une representation externe standard;
 *
 * Le systeme s est alloue et son ancienne valeur est perdue. Cette fonction
 * devrait donc etre appelee avec la precondition s==NULL. La variable s
 * n'est pas retournee pour respecter la syntaxe habituelle de scanf et
 * pour pouvoir retourner un code.
 *
 * Si la syntaxe du fichier f est correct, la valeur true est retournee;
 * false sinon.
 *
 * les variables utilisees dans le systeme doivent etre declarees dans
 * la premiere ligne precedees du mot-cle VAR et separees par des
 * virgules
 *
 * les inegalites et les egalites sont melangees entre elles, mais separees
 * par des virgules et regroupees entre accolades; les inegalites sont
 * toutes mises sous la forme :
 *
 * sum a x <= b
 *  i   i i
 *
 * ou b est le terme constant; noter que le terme constant est conserve
 * a droite du comparateur et la partie lineaire a gauche
 *
 * Exemple:
 *     VAR x, y, z
 *     { x <= 1,
 *       y == 2, x + 2 z > y }
 *
 * Noter l'absence de signe "multiplier" entre les coefficients numeriques
 * et les noms de variables
 *
 * Cette fonction peut relire un fichier cree par la fonction sc_fprint()
 */
bool sc_fscan(FILE * f, Psysteme * ps)
{
    syst_in = f;
    sc_init_lex();
    syst_restart(f);
    syst_parse();
    ps_yacc = sc_reversal(ps_yacc);
    *ps = ps_yacc;
    return !syst_syntax_error;
}

/* void sc_dump(Psysteme sc): dump to stderr
 *
 * Ancien nom: sc_fprint() (modifie a cause d'un conflit avec une autre
 * fonction sc_fprint d'un profil different)
 *
 * using variable_debug_name(); lost of original variables' names
 * now compatible with sc_fscan()
 * Better use sc_default_name(Psysteme)
 * DN(5/8/2002)
 */
void sc_dump(Psysteme sc)
{
  if(!SC_UNDEFINED_P(sc)) {

    (void) fprintf(stderr,"#DIMENSION: (%d)  ",sc->dimension);
    (void) fprintf(stderr,"INEGALITES (%d)  ",sc_nbre_inegalites(sc));
    (void) fprintf(stderr,"EGALITES (%d)  ",sc_nbre_egalites(sc));
    //(void) fprintf(stderr,"BASE (%p)  ", sc->base);
    //(void) fprintf(stderr,"LISTE INEGALITES (%p)  ", sc->inegalites);
    //(void) fprintf(stderr,"LISTE EGALITES (%p)  ", sc->egalites);
    (void) fprintf(stderr,"\nVAR ");
    base_fprint(stderr,sc->base, variable_debug_name);
    (void) fprintf(stderr,"  {\n");
    inegalites_fprint(stderr,sc->inegalites, variable_debug_name);
    egalites_fprint(stderr,sc->egalites, variable_debug_name);
    (void) fprintf(stderr,"  }\n");
  }
  else
    (void) fprintf(stderr, "SC_RN or SC_EMPTY or SC_UNDEFINED\n");
}

/* void sc_default_dump(Psysteme sc): dump to stderr
 *
 * sc_default_dump is now compatible with sc_fscan
 * using default_variable_to_string (stored by LINEAR. see sc_debug.c)
 * may fail in very few cases, because of variable names.
 * DN(5/8/2002) 
 */
void sc_default_dump(Psysteme sc)
{
    if(!SC_UNDEFINED_P(sc)) {
	(void) fprintf(stderr,"#DIMENSION: %d  ",sc->dimension);
	(void) fprintf(stderr,"INEGALITES (%d)  ",sc_nbre_inegalites(sc));
	(void) fprintf(stderr,"EGALITES (%d)  ",sc_nbre_egalites(sc));
	(void) fprintf(stderr,"\nVAR ");	
	base_fprint(stderr,sc->base, default_variable_to_string);
	(void) fprintf(stderr,"  {\n");
	inegalites_fprint(stderr,sc->inegalites, default_variable_to_string);
	egalites_fprint(stderr,sc->egalites, default_variable_to_string);	
	(void) fprintf(stderr,"  }\n");
    }
    else
	(void) fprintf(stderr, "SC_RN ou SC_EMPTY ou SC_UNDEFINED\n");
}



/* void sc_print() 
 *
 * Obsolete. Better use sc_default_dump()
 *
 */
void sc_print(Psysteme ps, get_variable_name_t nom_var)
{
    sc_fprint(stderr, ps, nom_var);
}

/* void sc_fprint(FILE * f, Psysteme ps, char * (*nom_var)()):
 * cette fonction imprime dans le fichier pointe par 'fp' la representation 
 * externe d'un systeme lineaire en nombres entiers, compatible avec la
 * fonction de lecture sc_fscan()
 *                                                                          
 * nom_var est un pointeur vers la fonction permettant d'obtenir le   
 * nom d'une variable (i.e. d'un des vecteurs de base)
 *
 * FI: 
 *  - le test ne devrait pas etre fait sur NULL;
 *  - il faudrait toujours faire quelque chose, ne serait-ce qu'imprimer
 *    un systeme nul sous la forme {}; et trouver quelque chose pour
 *    les systemes infaisables;
 *  - pourquoi n'utilise-t-on pas inegalites_fprint (et egalites_fprint)
 *    pour ne pas reproduire un boucle inutile? Sont-elles compatibles avec
 *    la routine de lecture d'un systeme?
 *
 * DN: better use sc_fprint_for_sc_fscan()
 *  - can add the information like #dimension, nb_eq, nb_ineq or label in the beginning
 *  - been implemented as sc_fprint_for_sc_fscan(), without infeasibility issue.
 */
void sc_fprint(FILE * fp,
	       Psysteme ps,
	       get_variable_name_t nom_var)
{
    register Pbase b;
    Pcontrainte peq;

    if (ps != NULL) {
      int count = 0;
	if (ps->dimension >=1) {
	    (void)fprintf(fp,"VAR %s",(*nom_var)(vecteur_var(ps->base)));

	    for (b=ps->base->succ; !VECTEUR_NUL_P(b); b = b->succ)
		(void)fprintf(fp,", %s",(*nom_var)(vecteur_var(b)));
	}
	(void) fprintf(fp," \n { \n");

	for (peq = ps->inegalites, count=0; peq!=NULL;
	     inegalite_fprint(fp,peq,nom_var),peq=peq->succ, count++);

	assert(count==ps->nb_ineq);

	for (peq = ps->egalites, count=0; peq!=NULL;
	     egalite_fprint(fp,peq,nom_var),peq=peq->succ, count++);

	assert(count==ps->nb_eq);

	(void) fprintf(fp," } \n");
    }
    else
	(void) fprintf(fp,"(nil)\n");
}

/* void sc_fprint_for_sc_fscan(FILE *f, Psysteme sc, char * (*nom_var)(Variable))
 *
 * compatible with sc_fscan. Replaced sc_default_dump_to_file (not really in use)
 *
 * should use default_variable_to_string
 * 
 */
void sc_fprint_for_sc_fscan(FILE * f, Psysteme sc, char * (*nom_var)(Variable))
{
      if(!SC_UNDEFINED_P(sc)) {
	(void) fprintf(f,"#DIMENSION (%d)  ",sc->dimension);
	(void) fprintf(f,"INEGALITES (%d)  ",sc_nbre_inegalites(sc));
	(void) fprintf(f,"EGALITES (%d)  ",sc_nbre_egalites(sc));
	(void) fprintf(f,"\nVAR ");
	base_fprint(f,sc->base, nom_var);
	(void) fprintf(f,"  {\n ");
	inegalites_fprint(f,sc->inegalites, nom_var);
	egalites_fprint(f,sc->egalites, nom_var);	
	(void) fprintf(f,"  }\n");
      }
      else {
	(void) fprintf(f, "SC_RN ou SC_EMPTY ou SC_UNDEFINED\n");
      }
}

/* void sc_default_dump_to_files(Psysteme sc, sc_nb,directory_name):
 *
 * Suitable for filtering purposes
 * Print the system of constraints into several output files in a directory with names given
 * Each file is 100% compatible with sc_fscan
 * print with name of variables from default_variable_to_string
 * overwrite if files exist
 * DN(10/2/2003) 
 *
 */
void sc_default_dump_to_files(sc, sc_nb,directory_name)
Psysteme sc;
int sc_nb;
char *directory_name;
{
  FILE * f;
  char fn[256],*filename;
  int d;
  
  filename = "_sc.out";
  if (directory_name==NULL) {directory_name = "SC_OUT_DEFAULT";}  
  d = chdir(directory_name);
  if (d) {
    mkdir(directory_name,S_IRWXU);
    d = chdir(directory_name);
  }   

  snprintf(fn,sizeof(fn),".0f%f%s",(double)sc_nb,filename);
   
  if ((f = fopen(fn,"w")) != NULL) {
    if(!SC_UNDEFINED_P(sc)) {
      (void) fprintf(f,"#DIMENSION (%d)  ",sc->dimension);
      (void) fprintf(f,"INEGALITES (%d)  ",sc_nbre_inegalites(sc));
      (void) fprintf(f,"EGALITES (%d)  ",sc_nbre_egalites(sc));
      (void) fprintf(f,"\nVAR ");

      base_fprint(f,sc->base, default_variable_to_string);
      (void) fprintf(f,"  {\n ");
      inegalites_fprint(f,sc->inegalites, default_variable_to_string);
      egalites_fprint(f,sc->egalites, default_variable_to_string);
      (void) fprintf(f,"  }\n");
    }
    else {
      (void) fprintf(f, "SC_RN ou SC_EMPTY ou SC_UNDEFINED\n");
    }
    fclose(f);
  } else {
    fprintf(stderr,"Ouverture du fichier %s impossible\n",fn);
  }
  if (chdir("..")) // just to avoid a gcc warning
    fprintf(stderr, "chdir(\"..\") failed\n");
}
