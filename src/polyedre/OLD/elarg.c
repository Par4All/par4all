 /* package sg */
 
 /* operateur d'elargissement
  *
  * deux versions existent, conservant plus ou moins d'information
  *
  * Malik Imadache
  *
  * Modifications par Francois Irigoin:
  *  - remplacement et minimisation des includes 3B5
  *  - reformattages divers
  *  - ajout d'un troisieme arguments aux MALLOC
  *  - modification de elarg() pour p1 non-faisable
  *  - modification de mk_m1 pour suivre la definition de M1 donnee
  *    par Halbwachs page 57 section 5.3.3; Malik exigeait que tous
  *    les elements du systeme generateur de P2 SATURE la contrainte i
  *    de P1 pour la garder; Halbwachs ne semble demander que le RESPECT
  *    de cette contrainte par tous les elements du systeme generateur
  *  - modification de elarg; suppression de l'ajout des contraintes de M2
  *    et utilisation exclusive de M1 selon la these de Cousot page (5)-60
  *    section 5.8.3; la demonstration de convergence donnee par Halbwachs
  *    page 59 ne semble pas prevoir cet ajout de contraintes M2... qui devait
  *    cependant rendre l'algorithme d'elargissement independant de la 
  *    representation de P1, propriete qu'on perd; donne un resultat correct
  *    sur le test malikp.f
  *  - remodification de elarg: conservation de M2 et modification du moteur
  *    pour atteindre un point fixe avant d'arreter les iterations
  *    decroissantes (auparavant, on ne faisait qu'une iteration decroissante)
  *  - modification de elarg pour prendre en compte les droites comme elements
  *    des systemes generateurs; calcul de deux sg locaux, sg1 pour p1 et sg2
  *    pour p2, ou les droites sont remplacees par 2 rayons; pas d'elimination
  *    de redondance du tout pour le moment;
  */

#include <stdio.h>
#include <string.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"

#include "polyedre.h"

#include "saturation.h"

#define MALLOC(s,t,f) malloc((unsigned)(s))
#define FREE(p,t,f) free((char*)(p))

#include "liste-table.h"

/* sat_elarg2: la fonction suivante eclate toute egalite en deux inegalites et
 * calcule toutes les saturations du systeme donne; 	
 * 
 * le systeme donne perd toutes ses contraintes qui se retrouvent dans la
 * liste fournie en resultat;
 */
Pcontrainte sat_elarg2(sc,sg)                    
Psysteme sc;
Ptsg sg;
{
    int nb_s,nb_r,i,*ssat,*rsat;
    Pcontrainte eg,eg_soeur,reste_eg,lt=NULL;
    Psommet s_p;
    Pray_dte r_p;

    (void) fprintf(stderr,"sat_elarg2: begin\n");
    (void) fprintf(stderr,"sc=\n");
    sc_fprint(stderr, sc, variable_dump_name);
    (void) fprintf(stderr,"sg=\n");
    sg_fprint(stderr, sg, variable_dump_name);

    nb_s = sg_nbre_sommets(sg);
    nb_r = sg_nbre_rayons(sg);
    for (eg = sc->egalites; eg!=NULL ; eg = reste_eg) {
	reste_eg = eg->succ;
	ssat = (int *) MALLOC(nb_s*sizeof(int),SAT_TAB,"sat_elarg2");
	rsat = (int *) MALLOC(nb_r*sizeof(int),SAT_TAB,"sat_elarg2");
	for (i=0,s_p = sg_sommets(sg); s_p != NULL; i++,s_p = s_p->succ)
	    ssat[i] = satur_som(s_p,eg);
	for (i=0,r_p = sg_rayons(sg); r_p != NULL; i++,r_p = r_p->succ)
	    rsat[i] = satur_vect(r_p->vecteur,eg);
	eg->s_sat = ssat;
	eg->r_sat = rsat;
	eg_soeur = soeur(eg,nb_s,nb_r,-1);
	eg_soeur->succ = eg;
	eg->succ = lt;
	lt = eg_soeur;
    }
    sc->egalites = NULL;
    for (eg = sc->inegalites;eg!=NULL ; eg=reste_eg) {
	reste_eg = eg->succ;
	ssat = (int *) MALLOC(nb_s*sizeof(int),SAT_TAB,"sat_elarg2");
	rsat = (int *) MALLOC(nb_r*sizeof(int),SAT_TAB,"sat_elarg2");
	for (i=0,s_p = sg_sommets(sg); s_p != NULL; i++,s_p = s_p->succ)
	    ssat[i] = satur_som(s_p,eg);
	for (i=0,r_p = sg_rayons(sg); r_p != NULL; i++,r_p = r_p->succ)
	    rsat[i] = satur_vect(r_p->vecteur,eg);
	eg->s_sat = ssat;
	eg->r_sat = rsat;
	eg->succ = lt;
	lt = eg;
    }
    sc->inegalites = NULL;

    (void) fprintf(stderr,"resultat:\n");
    fprint_lineq_sat(stderr,lt,nb_s,nb_r);
    (void) fprintf(stderr,"sat_elarg2: end\n");

    return(lt);
}

/* sat_elarg1: duplique un systeme de contrainte et calcule toutes
 * ses saturations par rapport au systeme generateur donne
 */
Pcontrainte sat_elarg1(sc,sg)                    
Psysteme sc;
Ptsg sg;
{
    Psysteme sc_copy;
    Pcontrainte lt=NULL;

    (void) fprintf(stderr,"sat_elarg1: begin\n");

    /* simplification des inegalites en utilisant les inegalites */
    red_egs_inegs(sc);

    sc_copy = sc_dup(sc);
    lt = sat_elarg2(sc_copy, sg);
    FREE((char *) sc_copy, SYSTEME,"sat_elarg1");

    (void) fprintf(stderr,"sat_elarg1: end\n");
    return(lt);
}

/* mk_m1: construit une liste de contraintes qui sont des copies des
 * contraintes de inegs1 qui sont verifiees (saturent en version Malik)
 * tout le systeme generateur sg2
 *
 * Voir these Halbwachs, p. 57, section 5.3.3
 */
Pcontrainte mk_m1(inegs1,sg2)
Pcontrainte inegs1;
Ptsg sg2;
{
    int all_verif;
    Pcontrainte eg_soeur,eg,result;
    Psommet s;
    Pray_dte rd;

    (void) fprintf(stderr,"mk_m1: begin\n");
    (void) fprintf(stderr,"mk_m1: inegs1\n");
    inegalites_fprint(stderr,inegs1, variable_dump_name);
    (void) fprintf(stderr,"mk_m1: sg2\n");
    sg_fprint(stderr, sg2, variable_dump_name);

    result = NULL;
    for (eg = inegs1; eg !=NULL ; eg = eg->succ) {
	all_verif = 1;
	for ( s=sg_sommets(sg2); s!=NULL && all_verif; s=s->succ)
	    all_verif = satur_som(s,eg) <= 0;
	for ( rd=sg_rayons(sg2); rd!=NULL && all_verif; rd=rd->succ)
	    all_verif = satur_vect(rd->vecteur,eg) <= 0;
	for ( rd=sg_droites(sg2); rd!=NULL && all_verif; rd=rd->succ)
	    all_verif = satur_vect(rd->vecteur,eg) == 0;
	if (all_verif) {
	    eg_soeur = contrainte_dup(eg);
	    eg_soeur->succ = result;
	    result = eg_soeur;
	}
    }
    (void) fprintf(stderr,"mk_m1: result\n");
    inegalites_fprint(stderr,result, variable_dump_name);
    (void) fprintf(stderr,"mk_m1: end\n");
    return(result);
}

/* meme_tab: compare deux tableaux et renvoie 1 s'ils ont des 0 aux memes
 * indices (sert a detecter une redondance mutuelle pour m2; cf Halbwachs
 * page 57, section 5.3.4)
 */
int meme_tab(t1,t2,l)
int *t1, *t2;
int l;	/* longueur des tableaux */
{
    int i,ident=1;
    for(i=0;i<l && ident==1;i++) {
	if ( (t1[i]==0 && t2[i]!=0) || (t1[i]!=0 && t2[i]==0) ) {
	    ident = 0;
	    break;
	}
    }
    return(ident);
}

/* mk_m2: contruit une liste de contraintes contenant celles de la liste
 * m1 et toutes les inegalites de inegs2 mutuellement redondantes avec une
 * inegalite de inegs1; les saturations etant dans les champs prevus a cet
 * effet pour chaque inegalite.  toute inegalite non retenue est eliminee
 *
 * Voir Halbwachs, page 57, section 5.3.4
 * 
 * Les contraintes de la liste retournee n'ont plus de saturations.
 * eg2 est modifie.
 *
 * nbs1, nbr1: nombre d'elements generateurs pour le test des saturations
 *             (valables pour inegs1 et inegs2 dont les saturations ont
 *             dues etre calculees au prealable par rapport au meme
 *             systeme generateur)
 */
Pcontrainte mk_m2(m1,inegs1,inegs2,nbs1,nbr1)
Pcontrainte m1,inegs1,inegs2;
int nbs1,nbr1;    
{
    Pcontrainte eg1,eg2,result,resteg;

    (void) fprintf(stderr,"mk_m2: begin\n");
    (void) fprintf(stderr,"mk_m2: m1\n");
    inegalites_fprint(stderr,m1, variable_dump_name);
    (void) fprintf(stderr,"mk_m2: inegs1\n");
    fprint_lineq_sat(stderr,inegs1,nbs1,nbr1);
    (void) fprintf(stderr,"mk_m2: inegs2\n");
    fprint_lineq_sat(stderr,inegs2,nbs1,nbr1);

    result = m1;
    for( eg2=inegs2; eg2!=NULL ; eg2=resteg) {
	resteg = eg2->succ;

	/* recherche d'une inegalite de inegs1 mutuellement redondante
	   avec eg2 */

	for( eg1=inegs1; eg1!=NULL && 
	    (! (meme_tab(eg1->s_sat,eg2->s_sat,nbs1) && 
		meme_tab(eg1->r_sat,eg2->r_sat,nbr1)));
	    eg1=eg1->succ);

	/* menage inconditionnel sur les saturations de eg2 */

	if (eg2->s_sat!=NULL) FREE((char *)eg2->s_sat,SAT_TAB,"mk_m2");
	if (eg2->r_sat!=NULL) FREE((char *)eg2->r_sat,SAT_TAB,"mk_m2");
	eg2->s_sat = NULL;
	eg2->r_sat = NULL;

	/* on a trouve une redondance mutuelle */

	if (eg1!=NULL) {
	    eg2->succ = result;
	    result = eg2;
	}

	/* pas de redondance mutuelle => on elimine eg2 */

	else  CONTRAINTE_RM(eg2,"mk_m2");
    }

    (void) fprintf(stderr,"mk_m2: result\n");
    inegalites_fprint(stderr, result, variable_dump_name);
    (void) fprintf(stderr,"mk_m2: end\n");
    return(result);
}

/* mk_egalites: construction d'un systeme a partir d'une liste d'inegalites,
 * i.e. trouver les egalites cachees
 */
Psysteme mk_egalites(listineg)
Pcontrainte listineg;
{
    Pcontrainte ineg1,ineg2,restineg1,restineg2,ineg3;
    Psysteme sc;
    int a_soeur;

    /* creation et initialisation d'un Psysteme */

    sc = sc_new();
    /*
     * sc = (Psysteme) MALLOC(sizeof(Ssysteme),SYSTEME,"mk_egalites");
     * sc->egalites = sc->inegalites = NULL;
     * sc_nbre_inegalites(sc) = sc_nbre_egalites(sc) = 0;
     */

    /* pour chaque inegalite */

    for (ineg1=listineg; ineg1!=NULL; ineg1=restineg1) {
	restineg1 = ineg1->succ;
	ineg1->succ = NULL;
	a_soeur = 0;

	/* la comparer avec les autres pour trouver une egalite */

	for (ineg2=ineg3=restineg1;ineg2!=NULL;ineg3=ineg2,ineg2=restineg2) {
	    restineg2 = ineg2->succ;
	    if ((a_soeur = contrainte_oppos(ineg1,ineg2))) {

		/* ineg1 et ineg2 forment une egalite */

		if (ineg2==ineg3) restineg1=restineg1->succ;
		else ineg3->succ = ineg2->succ;
		CONTRAINTE_RM(ineg2,"mk_egalites");
		break;
	    }
	}
	if (a_soeur) {

	    /* on sait que ineg1 est en fait une egalite */
	    /* => l'ajouter au systeme                   */

	    ineg1->succ = sc->egalites;
	    sc->egalites = ineg1;
	    sc_nbre_egalites(sc)++;
	}
	else {

	    /* ineg1 reste une inegalite => ajout au Psysteme */

	    ineg1->succ = sc->inegalites;
	    sc->inegalites = ineg1;
	    sc_nbre_inegalites(sc)++;
	}
    }
    /* initialisation de la base de sc */
    sc_creer_base(sc);
    return(sc);
}

/* elarg: realisation de l'elargissement de p1 et de p2 (technique d'Halbwachs)
 * fonction non destructrice de p1 et donnant le resultat dans p2
 *
 * Francois Irigoin:
 *  - le resultat ne semble pas toujours retourne dans p2
 *  - il ne me semble pas evident de savoir si p1 et p2 peuvent etre
 *    desalloues sans risque dans la procedure appelante
 *
 * Modifications:
 *  - suppression du parametre "dim"
 */
Ppoly elarg(p1,p2)
Ppoly p1,p2;
{
    Pcontrainte inegs1,inegs2,m1,m2;
    /* version temporaire sans lignes des systemes generateurs de p1 et de p2*/
    Ptsg sg1, sg2;

    (void) fprintf(stderr,"elarg: begin\n");

    (void) fprintf(stderr,"poly p1:\n");
    poly_fprint(stderr,p1, variable_dump_name);
    (void) fprintf(stderr,"poly p2:\n");
    poly_fprint(stderr,p2, variable_dump_name);

    /* rappel : un polyedre NULL est un polyedre avec la valeur initiale de */
    /*		resolution iterative                                    */

    if (p1==NULL) {
	(void) fprintf(stderr,"elarg: p1=NULL return=p2\n");
	return(p2);
    }
    if (p2==NULL) {
	(void) fprintf(stderr,"elarg: p2=NULL return=copie(p1)\n");
	return(poly_dup(p1));
    }
    if (poly_nbre_sommets(p2)==0) {		
	/* systeme p2 non faisable */
	(void) fprintf(stderr,"elarg: p2 non faisable\n");
	poly_rm(p2);
	return(poly_dup(p1));
    }
    if (poly_nbre_sommets(p1)==0) {
	/* systeme p1 non faisable */
	(void) fprintf(stderr,"elarg: p1 non faisable\n");
	/* code de Malik:
	 * poly_rm(p2);
	 * return(poly_dup(p1));
	 */
	poly_rm(p1);
	return(p2);
    }

    /* transformation des droites en paires de rayons pour p1 et pour p2 */
    sg1 = sg_without_line(p1->sg);
    sg2 = sg_without_line(p2->sg);

    /* calcul des saturations et duplication des contraintes a conserver */

    (void) fprintf(stderr,"elarg: calcul des saturations\n");
    inegs1 = sat_elarg1(p1->sc,sg1);
    inegs2 = sat_elarg2(p2->sc,sg1);

    /* calcul de la liste d'inegalites de p1 a conserver: elles sont
       satisfaites par tous les elements du systeme
       generateur sg de p2 (Halbwachs, section 5.3.3, page 57 */

    m1 = mk_m1(inegs1,sg2);
    (void) fprintf(stderr,"elarg: liste des inegalites de p1 a conserver\n");
    inegalites_fprint(stderr, m1, variable_dump_name);

    /* calcul de la liste des inegalites de p2 a ajouter aux precedentes:

       une inegalite i2 de p2 est conservee s'il existe une inegalite i1 de p1
       telle les saturations de i2 et de i1 par rapport au systeme generateur
       de p1 sont nulles pour les memes elements

       et construction de la liste finale des inegalites voulues par 
       concatenation a m1 */

    /* non-ajout eventuelle des inegalites provenant de M2: les ajouts
       d'inegalites dus a M2 ne verifiraient plus la propriete de decroissance
       utilisee dans la deuxieme partie de la preuve, section 5.3.6, page 59.

       Le premier elargissement avec M2 ne fait rien pour le programme:
                  C = 0
		  DO 100 I = 1, N
		     C = 1 - C
       On termine avec un point fixe faux quand on ne fait qu'une iteration
       decroissante
       */

      m2 = mk_m2(m1,inegs1,inegs2,poly_nbre_sommets(p1),poly_nbre_rayons(p1));
      (void) fprintf(stderr,
		     "elarg: apres ajout des inegalites de p2 a conserver\n");
      inegalites_fprint(stderr, m2, variable_dump_name);

    /* Si on ne veut pas ajouter M2:
     * m2 = m1;
     */

    /* reste a faire le menage */

    sg_rm(p2->sg);
    sg_rm(sg1);
    sg_rm(sg2);

    /* et a reformer les egalites possibles ainsi que le systeme de contraintes
       finales
       */

    p2->sc = mk_egalites(m2);

    /* a ce systeme de contraintes associer le systeme generateur afin
       d'obtenir le polyedre resultant */

    p2->sg = sc_to_sg(p2->sc);

    /* eliminer les objets crees localement et inutiles dans la suite */

    contraintes_free(inegs1);

    (void) fprintf(stderr,"elarg: resultat\n");
    poly_fprint(stderr,p2, variable_dump_name);
    (void) fprintf(stderr,"elarg: end\n");
    return(p2);
}


/* AUTRE VERSION QUI GARDE TOUTES LES INEGALITES DE p1 ou n'apparaissent */
/*	pas les variables de listvar					 */
/* on ne veut elargir que dans les directions associees aux variables    */
/* modifiees dans la boucle analysee                                     */
static Ppoly elarg_new(p1,p2,dim,listvar,nbvar)
Ppoly p1,p2;
int dim;
int nbvar;
Pcons *listvar;
{
    Pcontrainte inegs1,inegs2,m1,m2;
    Psysteme cp_sc2;
    int i;

    if (p1==NULL) return(p2);
    if (p2==NULL) return(poly_dup(p1));
    if (poly_nbre_sommets(p2)==0) {
	poly_rm(p2);
	return(poly_dup(p1));
    }
    if (poly_nbre_sommets(p1)==0) {
	poly_rm(p2);
	return(poly_dup(p1));
    }
    inegs1 = sat_elarg1(p1->sc,p1->sg);
    cp_sc2 = sc_dup(p2->sc);
    for (i=0;i<nbvar;i++) 
	cp_sc2 = sc_projection(cp_sc2,CONSVAL(listvar[i],Variable));
    /* on garde toute information sur les variables n'apparaissant pas
       dans listvar par projections successives
       */
    inegs2 = sat_elarg2(p2->sc,p1->sg);
    m1 = mk_m1(inegs1,p2->sg);
    m2 = mk_m2(m1,inegs1,inegs2,poly_nbre_sommets(p1),poly_nbre_rayons(p1));
    sg_rm(p2->sg);
    p2->sc = mk_egalites(m2);
    /* il faut recuperer toutes les informations conservees dans
       cp_sc2
       */
    i=0;
    if (cp_sc2->egalites!=NULL)
	for(inegs2=cp_sc2->egalites;inegs2->succ!=NULL;
	    inegs2=inegs2->succ,i++);
    if (i++!=0) {
	inegs2->succ=p2->sc->egalites;
	p2->sc->egalites=cp_sc2->egalites;
	poly_nbre_egalites(p2)+=i;
    }
    i=0;
    if (cp_sc2->inegalites!=NULL)
	for(inegs2=cp_sc2->inegalites;inegs2->succ!=NULL;
	    inegs2=inegs2->succ,i++);
    if (i++!=0) {
	inegs2->succ=p2->sc->inegalites;
	p2->sc->inegalites=cp_sc2->inegalites;
	poly_nbre_inegalites(p2)+=i;
    }
    FREE((char *)cp_sc2,SYSTEME,"elarg_new");
    p2->sg = sc_to_sg(p2->sc);
    return(p2);
}

