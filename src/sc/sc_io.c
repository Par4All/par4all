 /* package sc */

#include <stdio.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

 /* Psysteme construit par sc_gram.y */
extern Psysteme ps_yacc;

 /* detection des erreurs de syntaxe par sc_gram.y */
extern boolean syst_syntax_error;

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
Psysteme * sc_read(nomfic)
char * nomfic;
{
	if ((syst_in = fopen(nomfic,"r")) == NULL) {
		(void) fprintf(stderr,
			       "Ouverture du fichier %s impossible\n",nomfic);
		exit(4);
	}
	sc_init_lex();
	syst_parse();
	return(&ps_yacc);
}

/* boolean sc_fscan(FILE * f, Psysteme * ps): construit un systeme d'inegalites
 * et d'egalites lineaires a partir d'une representation externe standard;
 * 
 * Le systeme s est alloue et son ancienne valeur est perdue. Cette fonction
 * devrait donc etre appelee avec la precondition s==NULL. La variable s
 * n'est pas retournee pour respecter la syntaxe habituelle de scanf et
 * pour pouvoir retourner un code.
 *
 * Si la syntaxe du fichier f est correct, la valeur TRUE est retournee;
 * FALSE sinon.
 *
 * les variables utilisees dans le systeme doivent etre declarees dans
 * la premiere ligne precedees du mot-cle VAR et separees par des
 * virgules
 *
 * les inegalites et les egalites sont melangees entre elles, mais separees
 * par des virgules et regroupees entre accolades; les inegalites sont
 * toutes mises sous la forme 
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
 * Cette fonction peut relir un fichier cree par la fonction sc_fprint()
 */
boolean sc_fscan(f,ps)
FILE * f;
Psysteme * ps;
{
    syst_in = f;
    sc_init_lex();
    syst_restart(f);
    syst_parse();
    ps_yacc = sc_reversal(ps_yacc);
    *ps = ps_yacc;
    return(!syst_syntax_error);
}

/* void sc_dump(Psysteme sc): impression d'un systeme
 * de contraintes sans passer de fonction pour calculer le nom
 * des variables (cf. sc_fprint()).
 *
 * Ancien nom: sc_fprint() (modifie a cause d'un conflit avec une autre
 * fonction sc_fprint d'un profil different)
 *
 */
void sc_dump(sc)
Psysteme sc;
{
    if(!SC_UNDEFINED_P(sc)) {
	(void) fprintf(stderr,"DIMENSION: %d\n",sc->dimension);
	base_fprint(stderr,sc->base, variable_dump_name);
	(void) fprintf(stderr,"INEGALITES (%d)\n",sc_nbre_inegalites(sc));
	inegalites_fprint(stderr,sc->inegalites,variable_dump_name);
	(void) fprintf(stderr,"EGALITES (%d)\n",sc_nbre_egalites(sc));
	egalites_fprint(stderr,sc->egalites,variable_dump_name);
    }
    else
	(void) fprintf(stderr, "SC_RN ou SC_EMPTY ou SC_UNDEFINED\n");
}

/* void sc_print() */
void sc_print(ps, nom_var)
Psysteme ps;
char * (*nom_var)();
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
 */
void sc_fprint(fp, ps, nom_var)
FILE *fp;
Psysteme ps;
char * (*nom_var)(Variable);
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





