/* package sur les polyedres
 *
 * Francois Irigoin
 */

#include <stdio.h>
#include <malloc.h>

#include "assert.h"

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"

#include "polyedre.h"

#define MALLOC(s,t,f) malloc(s)
#define FREE(p,t,f) free(p)

/* void poly_rm(Ppoly p): desallocation d'un polyedre
 *
 * Ancien nom: elim_poly();
 *
 * Modifications:
 *  - appel a sc_rm() (FI, 3/1/90)
 *
 * Bugs:
 *  - disymmetrie entre l'appel a sc_rm() et celui a sg_rm()
 *  - est-il grave d'avoir p->sc undefined?
 */
void poly_rm(p)
Ppoly p;
{	
    if (p==NULL) return;

    if(!SC_UNDEFINED_P(p->sc)) {
	sc_rm(p->sc);
    }
    else
	/*
	(void) fprintf(stderr,"poly_rm: warning sc NULL\n");
	*/

    sg_rm(p->sg);

    FREE((char *)p,POLY,"poly_rm");
}

/* void poly_fprint(FILE * f, Ppoly p, char * (*nom_var)()):
 * impression d'un polyedre
 */
void poly_fprint(f,p, nom_var)
FILE * f;
Ppoly p;
char * (*nom_var)();
{
    Psysteme syst;

    if (p==NULL) {
	(void) fprintf(f,"POLY NEUTRE\n");
	return;
    }

    syst = p->sc;
    if (syst!=NULL) 
	sc_fprint(f, syst, nom_var);
    else { 
	(void) fprintf(f,"POLY VIDE\n"); 
	return;
    }

    if (p->sg!=NULL) sg_fprint(f, p->sg, nom_var);
}

/* void poly_dump(Ppoly p): impression de debug d'un polyedre, sur stderr
 * et avec variable_dump_name
 */
void poly_dump(p)
Ppoly p;
{
    poly_fprint(stderr, p, variable_dump_name);
}

/* fprint_ineq_sat: impression d'une inequation avec les saturations 
 * temporairement associees; elles sont relatives a un systeme generateur
 * implicite dont le nombre de sommets est nb_s et le nombre de rayons
 * est nb_r
 */
void fprint_ineq_sat(f,ineq,nb_s,nb_r)
FILE * f;
Pcontrainte ineq;
int nb_s;
int nb_r;
{
    int s, r;

    inegalite_fprint(f,ineq,variable_dump_name);
    for(s=0;s<nb_s;s++) 
	(void) fprintf(f,"sommet %d: %d,",s,ineq->s_sat[s]);
    for(r=0;r<nb_r;r++) 
	(void) fprintf(f,"rayon %d: %d,",r,ineq->r_sat[r]);
    (void) fprintf(f,"\n");
}

/* fprint_lineq_sat: impression d'une liste d'inequations avec saturations 
 * les saturations ont ete calculees par rapport a un systeme generateur
 * implicite dont on passe les nombres de sommets (nb_s) et de rayons
 * (nb_r).
 */
void fprint_lineq_sat(f,lineq,nb_s,nb_r)
FILE * f;
Pcontrainte lineq;
int nb_s;
int nb_r;
{
    Pcontrainte e;

    for (e = lineq; e != NULL; e = e->succ) {
	fprint_ineq_sat(f,e,nb_s,nb_r);
    }
    (void) fprintf(f,"\n");
}

/* Ppoly sc_to_poly(Psysteme syst): alloue et construit un polyedre
 * contenant un systeme generateur correct
 * a partir d'un systeme de contraintes
 *
 * Le systeme de contraintes n'est pas duplique. Il fait partie integrante
 * du polyedre au retour.
 *
 * Il faudrait distinguer les cas SC_RN, SC_EMPTY et SC_UNDEFINED. Pour le
 * moment, on prend NULL==SC_RN
 */
Ppoly sc_to_poly(syst)
Psysteme syst;
{
    Ppoly p;

    p = (Ppoly ) MALLOC(sizeof(Spoly),POLY,"sc_to_poly");
    p->sg = sc_to_sg(syst);
    if (SC_RN_P(syst)) {
	p->sc = sc_new();
    }
    else if(SC_EMPTY_P(syst)) {
	assert(FALSE);
    }
    else if(SC_UNDEFINED_P(syst)) {
	assert(FALSE);
    }
    else 
	p->sc = syst;
    return(p);
}

/* Ppoly sc_to_poly_chernikova(Psysteme syst): alloue et construit 
 * un polyedre contenant un systeme generateur correct a partir d'un 
 * systeme de contraintes.  L'algorithme de Chernikova est utilise.
 *
 * Le systeme de contraintes n'est pas duplique. Il fait partie integrante
 * du polyedre au retour.
 *
 * Il faudrait distinguer les cas SC_RN, SC_EMPTY et SC_UNDEFINED. Pour le
 * moment, on prend NULL==SC_RN
 */
Ppoly sc_to_poly_chernikova(syst)
Psysteme syst;
{
    Ppoly p;
    /* Ptsg sg = sg_new(); */

    p = (Ppoly ) MALLOC(sizeof(Spoly),POLY,"sc_to_poly");
    p->sg = sc_to_sg_chernikova(syst);
    /* p->sg= sg; */
    if (SC_RN_P(syst)) {
	p->sc = sc_new();
    }
    else if(SC_EMPTY_P(syst)) {
	assert(FALSE);
    }
    else if(SC_UNDEFINED_P(syst)) {
	assert(FALSE);
    }
    else 
	p->sc = syst;
    return(p);
}

/* Ppoly poly_dup(Ppoly p): allocation et copie d'un polyedre
 *
 * Ancien nom: cp_poly()
 */
Ppoly poly_dup(p)
Ppoly p;
{
    Ppoly cp;

    if (p==NULL) return((Ppoly )NULL);
    else {
	cp = (Ppoly ) MALLOC(sizeof(Spoly),POLY,"poly_dup");
	cp->sc = sc_dup(p->sc);
	cp->sg = sg_dup(p->sg);
	return (cp);
    }
}

/* boolean soms_verif_sc(Psommet listsoms, Psysteme sc):
 * test d'appartenance d'un ensemble de sommets a un polyedre
 * decrit par son systeme de contrainte
 *
 * Cette fonction est rangee au niveau polyedre, bien qu'elle n'utilise
 * pas le type polyedre, pour ne pas introduire de sommets dans les
 * systemes de contraintes ou de systemes de contraintes dans les sommets
 * C'est au niveau polyedre qu'est exploitee la dualite entre systeme
 * generateur et systeme de contraintes
 */
boolean soms_verif_sc(listsoms,sc)
Psommet listsoms;
Psysteme sc;
{
	int all_verif;
	Pcontrainte  eg;
	Psommet s;
	all_verif = 0;
	for (eg = sc->egalites; eg !=NULL && all_verif==0 ; eg = eg->succ) {
		for ( s=listsoms; s!=NULL && all_verif==0; s=s->succ)
			all_verif = satur_som(s,eg);
	}
	if (all_verif!=0) return(0);
	for (eg = sc->inegalites; eg !=NULL && all_verif<=0 ; eg = eg->succ) {
		for ( s=listsoms; s!=NULL && all_verif<=0; s=s->succ)
			all_verif = satur_som(s,eg);
	}
	if (all_verif<=0) return(1);
	return(0);
}

/* boolean polyedre_egal(Ppoly p1, Ppoly p2): test d'egalite de polyedre base
 * sur les systemes generateurs
 *
 * Utilise pour savoir si un polyedre est a reevaluer lors de l'analyse
 */
boolean polyedre_egal(p1,p2)
Ppoly p1, p2;
{
    if (p1==p2) return(TRUE);
    if (p1==NULL) return(FALSE);
    if (p2==NULL) return(FALSE);
    return( sg_egal(p1->sg,p2->sg) );
}

/* Ppoly proj(Ppoly p, Variable v): projection du polyedre p selon la
 * variable v
 *
 * p est modifie par effet de bord et retourne
 *
 * Ce n'est pas vraiment une projection mais le cylindre construit
 * a partir de la projection avec la direction v comme axe.
 * La dimension du polyedre ne change donc pas.
 *
 * Si le polyedre est vide et si on s'en rend compte lors de la
 * projection du systeme de contraintes, on renvoie un polyedre
 * dont le systeme generateur ne contient aucun sommet. Cependant
 * le systeme de contraintes retourne est VIDE sans cependant
 * representer l'espace tout entier.
 */
Ppoly poly_proj(p,v)
Ppoly p;
Variable v;
{
    if (p==NULL) {
	/* polyedre initial de la resolution: Rn, l'espace tout entier
	 reste l'espace tout entier */
	return(NULL);
    }
    if (poly_nbre_sommets(p)==0)
	/* polyedre non faisable: l'ensemble vide reste l'ensemble vide */
	return(p);

    if ((p->sc=sc_projection(p->sc,v))!=NULL)
	/* le systeme generateur doit etre complete par une droite */
	p->sg=ajout_dte(p->sg,v);

    else {				
	/* le polyedre est vide puisque son systeme de contrainte est
	   non faisable; il n'a donc aucun sommets; on elimine tous ses
	   sommets */
	/* Remarque: pourquoi n'elimine-t-on pas le reste?
	             pourquoi n'utilise-t-on pas une fonction sur les sg's? */
	Psommet s,s1;
	if (poly_nbre_sommets(p)!=0) {
	    for (s=poly_sommets(p);s!=NULL;s=s1) {
		s1 = s->succ;
		sommet_rm(s);
	    }
	}
	poly_nbre_sommets(p) = 0;
	poly_sommets(p) = NULL;
    }
    return(p);
}
