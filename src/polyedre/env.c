 /* Envelope convexe de deux polyedres
  * Algorithme decrit dans la these de Nicolas Halbwachs
  *
  * Malik Imadache
  *
  * Modifie par Francois Irigoin, le 24 mars 1989:
  *  - reprise des includes
  *  - reformattages divers
  *  - ajout d'un troisieme argument aux MALLOCs
  *  - modification des appels a init_eq(); suppression des appels
  *    prealables a creer_eq();
  *  - reprise de env suite a la detection d'une erreur
  *  - remplacement des appels a mk_couple par des appels a vect_chg_coeff
  *    (mk_couple n'etait plus qu'un appel a vect_chg_coeff)
  *  - modification de l'elimination et/ou de la promotion en egalite
  *    d'une inegalite dans init_sat_env(); Avril 1991
  */

#include <stdio.h>
#include <malloc.h>

#include "assert.h"

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

static int convex_hull_debug_level = 0;
#define ifdebug(n) if((n)<convex_hull_debug_level)

/* soeur: pour une equation donnee eg, contruit une equation eg
 * qui est la meme multipliee par x; l'equation donnee a ses saturations
 * calculees et on en deduit les saturations de l'egalites creee
 *
 * aussi utilise sur les inequations pour deriver deux inequations
 * a partir d'une equation en la multipliant par 1 et -1
 */
Pcontrainte soeur(eg,nb_s,nb_r,x)
Pcontrainte eg;
int nb_s,nb_r;
int x;			/* x=1  si c'est pour dupliquer l'equation */
			/* x=-1 si on veut l'opposee de l'equation */
{
    Pcontrainte eg_soeur;
    int i;

    eg_soeur = contrainte_new();
    eg_soeur->s_sat = (int *) MALLOC(nb_s*sizeof(int),SAT_TAB,"soeur");
    eg_soeur->r_sat = (int *) MALLOC(nb_r*sizeof(int),SAT_TAB,"soeur");
    for(i=0;i<nb_s;i++) (eg_soeur->s_sat)[i] = x*(eg->s_sat)[i];
    for(i=0;i<nb_r;i++) (eg_soeur->r_sat)[i] = x*(eg->r_sat)[i];
    /* probleme potentiel avec vect_mult:
     * eg_soeur->vecteur = vect_multiply(vect_dup(eg->vecteur),x);
     */
    eg_soeur->vecteur = vect_dup(eg->vecteur);
    if (x==-1) vect_chg_sgn(eg_soeur->vecteur);
    return(eg_soeur);
}

/* analyse_eg:
 * toute  egalite saturee  par tous les elements  generateurs
 * doit etre  "recollee"  au  polyedre  et  ne pas etre prise
 * en compte dans la suite car elle fait parti de l'enveloppe
 *
 * lt: est l'adresse a laquelle il faut rendre la liste des inegalites
 * a traiter dans l'enveloppe convexe
 *
 * type: indique si l'enveloppe est avec un sommet (type=0)
 * ou un rayon ou une droite (type=1)
 */
void analyse_eg(eg,type,nb_s,nb_r,nb_s_total,nb_r_total,lt,end_lt,sc)
Pcontrainte eg,*lt,*end_lt;
Psysteme sc;
int nb_s,nb_r,nb_s_total,nb_r_total;
int type;
{
    int all_0=1;
    Pcontrainte eg_soeur;
    if (type == 0)	if ((eg->s_sat)[nb_s-1] != 0) all_0 = 0;
    if (type == 1)	if ((eg->r_sat)[nb_r-1] != 0) all_0 = 0;
    if (all_0) {
	/* cette egalite sature tout les elements generateurs du polyedre
	   => la conserver afin de la prendre comme contrainte de l'env.
	   */
	sc_add_eg(sc, eg);
    }
    else {
	/* cette egalite ne sature pas tous les elements generateurs =>
	   l'eclater en 2 inegalites a ajouter dans l'analyse de l'enveloppe
	   */
	eg_soeur = soeur(eg,nb_s_total,nb_r_total,-1);
	eg_soeur->succ = eg;
	if (*lt == NULL) (*end_lt)=eg;
	eg->succ = *lt;
	*lt = eg_soeur;
    }
}

/* s_sat_env: enveloppe d'un polyedre p avec un sommet s;
 * calcul de la saturation de ce sommet s pour
 * chaque egalite et inegalite afin de construire la liste des
 * inegalites a combiner pour obtenir l'enveloppe;
 *
 * inclus: indique en retour si le sommet s est inclus dans p
 *
 * rappel : pour chaque contrainte un tableau de saturations est alloue
 *	pour les sommets et un pour les rayons, a chaque enveloppe cvx
 *	avec un element generateur on ajoute la saturation le concernant
 */
Pcontrainte s_sat_env(p,s,nb_s_total,nb_r_total,inclus)
Ppoly p;
Psommet s;
int nb_s_total,nb_r_total;
int *inclus;    
{
    int nb_s,nb_r;
    int satur_tout=1;	/* ce booleen detecte si le sommet est inclus */
    /* dans le polyedre avec lequel on fait       */
    /* l'enveloppe auquel cas on n'a rien a faire */
    Pcontrainte eg,eg1,egs,lt=NULL,end_lt=lt;

    ifdebug(8) (void) fprintf(stderr,"s_sat_env: begin\n");

    nb_s = poly_nbre_sommets(p);
    nb_r = poly_nbre_rayons(p);
    egs = poly_egalites(p);

    /* calcul de la saturation du sommet pour chaque egalite */
    for (eg = egs; eg!=NULL ; eg = eg1) {
	eg1 = eg->succ;
	if (((eg->s_sat)[nb_s] = satur_som(s,eg)) != 0)
	    satur_tout = 0;
    }

    /* calcul de la saturation du sommet pour chaque inegalite
       aucune inegalite saturant tous les elements generateurs ne doit
       exister ici, ceci est garanti par les eliminations de redondances
       effectuees au cours des propagations;
       DERNIERE MODIFICATION : la fonction init_sat_env permet dans la
       version actuelle d'eviter ce probleme, ainsi cette enveloppe
       convexe est tout a fait generale
       */
    for (eg = poly_inegalites(p); eg!=NULL ; eg=eg->succ) {
	if (((eg->s_sat)[nb_s] = satur_som(s,eg)) > 0)
	    satur_tout = 0;
    }

    *inclus = satur_tout;
    if (satur_tout) {
	ifdebug(8) {
	    (void) fprintf(stderr,"s_sat_env: inclus=%d\n",*inclus);
	    (void) fprintf(stderr,"s_sat_env: end\n");
	}
	return((Pcontrainte)NULL);
    }

    /* toute egalite saturee par tous les elements generateurs est recuperee
       en fin de traitement de l'enveloppe, les autres sont eclatees en 2
       inegalites
       */

    egs = poly_egalites(p);
    poly_egalites(p) = NULL;
    poly_nbre_egalites(p) = 0;
    for (eg = egs; eg!=NULL ; eg = eg1) {
	eg1 = eg->succ;
	analyse_eg(eg,0,nb_s+1,nb_r,nb_s_total,nb_r_total,&lt,&end_lt,p->sc);
    }

    /* les inegalites sont toutes mises dans la liste de travail; les egalites
       a utiliser y ont ete mises lors de l'appel a analyse_eg
       */

    if (lt == NULL)
	lt = poly_inegalites(p);
    else
	end_lt->succ = poly_inegalites(p);
    poly_inegalites(p) = NULL;
    poly_nbre_inegalites(p) = 0;

    ifdebug(8) (void) fprintf(stderr,"s_sat_env: end\n");
    return(lt);
}

/* rd_sat_env: enveloppe d'un polyedre p avec un rayon ou une droite r;
 * calcul de la saturation de ce rayon ou de cette droite r pour
 * chaque egalite et inegalite de p afin de construire la liste des
 * inegalites a combiner pour obtenir l'enveloppe;
 *
 * type: 0 => rayon
 *       1 => droite
 * inclus: indique en retour si le rayon ou la droite r est inclus dans p
 *
 * Note: je ne comprends pas comment Malik arrive a traiter les rayons et
 *       les droites de la meme maniere; d'apres moi, il faut un
 *       r_sat_env et un d_sat_env distinct car le calcul de satur_tout
 *       est a coup sur faux; j'ajoute un parametre pour savoir si
 *       j'ai affaire a un rayon ou a une droite
 *
 * rappel : pour chaque contrainte un tableau de saturations est alloue
 *	pour les sommets et un pour les rayons, a chaque enveloppe cvx
 *	avec un element generateur on ajoute la saturation le concernant
 *
 * Commentaire initial de Malik:
 * rd_sat_env: identique a la precedente; traitera des rayons et
 * droites avec lesquels seront fait des enveloppes convexes
 */
Pcontrainte rd_sat_env(p,r,type,nb_s_total,nb_r_total,inclus)
Ppoly p;
Pray_dte r;
int type;
int nb_s_total,nb_r_total;
int *inclus;
{
    int satur_tout=1; 
    int nb_s,nb_r;
    Pcontrainte eg,eg1,egs,lt=NULL,end_lt=lt;

    ifdebug(8) (void) fprintf(stderr,"rd_sat_env: begin\n");

    nb_s = poly_nbre_sommets(p);
    nb_r = poly_nbre_rayons(p);
    egs = poly_egalites(p);

    /* calcul de la saturation du rayon pour chaque egalite */
    for (eg = egs; eg!=NULL ; eg = eg1) {
	eg1 = eg->succ;
	if (((eg->r_sat)[nb_r] = satur_vect(r->vecteur,eg))!= 0)
	    satur_tout = 0;
    }

    /* calcul de la saturation du rayon pour chaque inegalite */
    for (eg = poly_inegalites(p); eg!=NULL ; eg=eg->succ) {
	(eg->r_sat)[nb_r] = satur_vect(r->vecteur,eg);
	if (type == 0 && (eg->r_sat)[nb_r] > 0)
	    satur_tout = 0;
	else if (type == 1 && (eg->r_sat)[nb_r] != 0)
	    satur_tout = 0;
    }
  
    *inclus=satur_tout;
    if(satur_tout) {
	ifdebug(8) {
	    (void) fprintf(stderr,"rd_sat_env: inclus=%d\n",*inclus);
	    (void) fprintf(stderr,"rd_sat_env: end\n");
	}
	return((Pcontrainte)NULL);
    }

    egs = poly_egalites(p);
    poly_egalites(p) = NULL;
    for (eg = egs; eg!=NULL ; eg = eg1) {
	eg1 = eg->succ;
	analyse_eg(eg,1,nb_s,nb_r+1,nb_s_total,nb_r_total,&lt,&end_lt,p->sc);
    }

    if(lt==NULL)
	lt = poly_inegalites(p);
    else
	end_lt->succ = poly_inegalites(p);
    poly_inegalites(p) = NULL;

    ifdebug(8) {
/*	(void) fprintf(stderr,"rd_sat_env: inclus=%d\n"); */
	(void) fprintf(stderr,"rd_sat_env: end\n");
    }
    return(lt);
}

/* init_sat_env:
 *
 * au debut de l'enveloppe convexe de deux polyedres avec elimination de
 * redondance, il faut calculer toutes les saturations des contraintes
 * vis a vis des sommets et rayons pour p1
 *
 * env(p1,p2) effectue l'enveloppe convexe de p1 avec chaque element
 * generateur de p2
 */
void init_sat_env(p,nb_s_total,nb_r_total)
Ppoly p;
int nb_s_total,nb_r_total;
{
    int i,*ssat,*rsat,satur_tout;
    Pcontrainte eg,egs,eg_pred;
    Psommet s_p;
    Pray_dte r_p,d;

    ifdebug(8) (void) fprintf(stderr,"init_sat_env: begin\n");

    /* calcul des saturations pour chaque egalite */
    egs = poly_egalites(p);
    for (eg = egs; eg!=NULL ; eg = eg->succ) {
	ssat = (int *) MALLOC(nb_s_total*sizeof(int),SAT_TAB,"init_sat_env");
	rsat = (int *) MALLOC(nb_r_total*sizeof(int),SAT_TAB,"init_sat_env");
	for (i=0,s_p = poly_sommets(p); s_p != NULL; i++,s_p = s_p->succ){
	    ssat[i] = satur_som(s_p,eg);
	}
	for (i=0,r_p = poly_rayons(p); r_p != NULL; i++,r_p = r_p->succ){
	    rsat[i] = satur_vect(r_p->vecteur,eg);
	}
	eg->s_sat = ssat;
	eg->r_sat = rsat;
    }

    /* calcul des saturations pour chaque inegalite
       toute inegalite saturant tous les elements generateurs est transformee
       en egalite
       */
    eg_pred = NULL;
    egs = poly_inegalites(p);
    for (eg = egs; eg!=NULL ; ) {
	satur_tout = 1;
	ssat = (int *) MALLOC(nb_s_total*sizeof(int),SAT_TAB,"init_sat_env");
	rsat = (int *) MALLOC(nb_r_total*sizeof(int),SAT_TAB,"init_sat_env");
	for (i=0,s_p = poly_sommets(p); s_p != NULL; i++,s_p = s_p->succ){
	    if ((ssat[i] = satur_som(s_p,eg))!=0) satur_tout=0;
	}
	for (i=0,r_p = poly_rayons(p); r_p != NULL; i++,r_p = r_p->succ){
	    if ((rsat[i] = satur_vect(r_p->vecteur,eg))!=0)
		satur_tout=0;
	}
	for (d=poly_droites(p); d!=NULL && satur_tout ; d=d->succ){
	    if ((satur_vect(d->vecteur,eg))!=0) satur_tout=0;
	}
	eg->s_sat = ssat;
	eg->r_sat = rsat;
	if (satur_tout) {		/* c'est une egalite */
	    if (eg_pred==NULL) {
		poly_inegalites(p) = eg->succ;
		poly_nbre_inegalites(p)--;
		if (egalite_in_liste(eg,poly_egalites(p)))
		    CONTRAINTE_RM(eg,"init_sat_env");
		else {
		    eg->succ = poly_egalites(p);
		    poly_egalites(p) = eg;
		    poly_nbre_egalites(p)++;
		}
		eg = poly_inegalites(p);
	    }
	    else {
		eg_pred->succ = eg->succ;
		if (egalite_in_liste(eg,poly_egalites(p)))
		    CONTRAINTE_RM(eg,"init_sat_env");
		else {
		    eg->succ = poly_egalites(p);
		    poly_egalites(p) = eg;
		    poly_nbre_egalites(p)++;
		}
		poly_nbre_inegalites(p)--;
		eg = eg_pred->succ;
	    }
	}
	else {
	    eg_pred = eg;
	    eg = eg->succ;
	}
    }
    ifdebug(8) (void) fprintf(stderr,"init_sat_env: end\n");
}

/* Psysteme mk_env_egs(Pcontrainte listinegs, Pray_dte dtes, int nb_s,
 *                     int nb_r):
 * allocation et initialisation d'un systeme contenant les contraintes
 * de listinegs qui sont en fait des egalites parce que toutes leurs
 * saturations par rapport au systeme generateur correspondant sont nulles
 *
 * Les egalites du Psysteme sont sharees avec les inegalites de listinegs
 */
Psysteme mk_env_egs(listinegs,dtes,nb_s,nb_r)
Pcontrainte listinegs;
Pray_dte dtes;
int nb_s,nb_r;
{
    int i,satur_tout;
    Psysteme syst;
    Pcontrainte eg,egs;
    Pray_dte d;

    /* creation d'un systeme */

    /*
     * syst = (Psysteme) MALLOC(sizeof(Ssysteme),SYSTEME,"mk_env_egs");
     * syst->egalites = syst->inegalites = (Pcontrainte) NULL;
     * syst->nb_eq = syst->nb_ineq = 0;
     */
    syst = sc_new();

    /* toute inegalite saturant tous les elements generateurs est transformee
       en egalite */

    for (eg = listinegs; eg!=NULL ; eg = egs ) {
	egs = eg->succ;
	eg->succ = NULL;
	satur_tout = TRUE;
	for (i=0; i< nb_s && satur_tout ; i++){
	    if ((eg->s_sat[i])!=0) satur_tout=FALSE;
	}
	for (i=0; i< nb_r && satur_tout ; i++){
	    if ((eg->r_sat[i])!=0) satur_tout=FALSE;
	}
	for (d=dtes; d!=NULL && satur_tout ; d=d->succ){
	    if ((satur_vect(d->vecteur,eg))!=0) satur_tout=FALSE;
	}
	if (satur_tout) {		/* c'est une egalite */
	    eg->succ = syst->egalites;
	    syst->egalites = eg;
	    (syst->nb_eq) ++;
	}
	else {
	    eg->succ = syst->inegalites;
	    syst->inegalites = eg;
	    (syst->nb_ineq) ++;
	}
    }
    return(syst);
}

/* Pcontrainte mk_sigma1(Pcontrainte lt, int type, int indice,
 *                       int nb_s_tot, int nb_r_tot):
 * construction de sigma1 qui est la premiere partie du sc de l'enveloppe
 *
 * lt: liste de travail -> inegalites avec saturations
 * type:  0 ~ sommet  /  1 ~ rayon  /  2 ~ droite
 * indice: indice dans  les  tableaux  de  saturation  de
 *    l'element avec lequel  l'enveloppe est faite
 * nbre total d'elements : sert si on copie une eq
 */
Pcontrainte mk_sigma1(lt,type,indice,nb_s_tot,nb_r_tot)
Pcontrainte *lt;
int type;
int indice;
int nb_s_tot,nb_r_tot;
{
    Pcontrainte sigma1, ineg, eg_copie, reste_lt;
    int vl;

    sigma1 = NULL;
    ineg = *lt;
    *lt = NULL;
    for (; ineg!=NULL; ineg=reste_lt) {
	reste_lt = ineg->succ;
	ineg->succ = NULL;
	vl = (type==0 ? (ineg->s_sat)[indice] : (ineg->r_sat)[indice]);

	/* toute inegalite sature par le nouvel element generateur est retenue
	   comme contrainte de l'enveloppe convexe
	   */

	if (vl==0) {
	    ineg->succ = sigma1;
	    sigma1 = ineg;
	}
	else {
	    /* si ce nouvel element generateur est un sommet ou un rayon
	       on retient aussi toutes les contraintes de saturation negative
	       */
	    if (vl<0 && type!=2) {
		/* on effectue une copie ici car d'autre
		   contraintes de l'enveloppe seront issues
		   de combinaisons les utilisant
		   */
		eg_copie = soeur(ineg,nb_s_tot,nb_r_tot,1);
		eg_copie->succ = sigma1;
		sigma1 = eg_copie;
	    }
	    ineg->succ = (*lt);
	    (*lt) = ineg;
	}
    }
    return(sigma1);
}

/* mk_sigma2: construction de sigma2 d'ou sont issues par combinaisons
 * d'autres contraintes de l'enveloppe convexe.
 *
 * voir dans mk_sigma1 la semantique de ces variables.
 *
 * dim : dimension de l'espace
 * den: si on a un sommet, c'est le denominateur
 *
 * Modifications:
 *  - suite au passage du terme constant a gauche, mise a jour
 *    du terme constant par -vl (FI, 20/12/89)
 */
Pcontrainte mk_sigma2(lt,type,den,indice,nb_s,nb_r,nb_s_tot,nb_r_tot,dim)
Pcontrainte *lt;
int type;
int indice;
int nb_s;
int nb_r;
Value den;
int nb_s_tot;
int nb_r_tot;
int dim;
{
    Pcontrainte ineg;
    Value vl;
    int i;

    /* il faut construire les contraintes a combiner a partir des contraintes
       retenues (voir la these d'Halbwachs) */

    for (ineg=(*lt); ineg!=NULL; ineg=ineg->succ) {
	vl = (type==0 ? (ineg->s_sat)[indice] : (ineg->r_sat)[indice]);
	if (type==0) {
	    /* pour un sommet il faut faire attention au denominateur
	       pour construire les contraintes a combiner */
	    (void) vect_multiply(ineg->vecteur,den);
	    vect_chg_coeff(&(ineg->vecteur),TCST,
			   value_minus(COEFF_CST(ineg),vl));
	}
	/* mk_couple(dim+1,vl,ineg); FI: probleme pour la creation d'une
	   nouvelle variable! Mieux vaut utiliser variable_dump_name
	   ensuite... */
	vect_chg_coeff(&(ineg->vecteur),(Variable) (dim+1),vl);
    }

    /* pour une enveloppe avec un sommet il faut creer une contrainte
       supplementaire pour faire les combinaisons */

    if (type==0) {
	ineg = contrainte_new();
	/* mk_couple(dim+1,-1,ineg); */
	vect_chg_coeff(&(ineg->vecteur),(Variable)(dim+1),-1);
	ineg->s_sat = (int *) MALLOC(nb_s_tot*sizeof(int),SAT_TAB,"mk_sigma2");
	ineg->r_sat = (int *) MALLOC(nb_r_tot*sizeof(int),SAT_TAB,"mk_sigma2");
	for(i=0;i<=nb_s;i++) (ineg->s_sat)[i] = -1;
	for(;i<nb_s_tot;i++) (ineg->s_sat)[i] = 0;
	for(i=0;i<=nb_r;i++) (ineg->r_sat)[i] = 0;
	for(;i<nb_r_tot;i++) (ineg->r_sat)[i] = 0;
	ineg->succ = (*lt);
	(*lt) = ineg;
    }
    return(*lt);
}

 /* les contraintes issues de sigma2 seront combinees par les deux fonctions
  * suivantes
  */

/* Pcontrainte envcomb(Pcontrainte posit, Pcontrainte negat, Variable v,
 *                     int nb_s, int nb_r, int nb_s_tot, int nb_r_tot):
 * combine deux contraintes pour en construire une troisieme
 *
 * elle opere comme une combinaison simple (voir la projection) avec
 * en plus une combinaison des saturations
 *
 * combinaison des deux inegalites
 * par la variable v par construction
 * d'une nouvelle egalite.	
*/
Pcontrainte envcomb(posit,negat,v,nb_s,nb_r,nb_s_tot,nb_r_tot)
Pcontrainte posit, negat;
Variable v;
int nb_s,nb_r,nb_s_tot,nb_r_tot;
{
    int i,cv_p, cv_n;
    Pcontrainte ineg;

    cv_p = vect_coeff(v,posit->vecteur);
    cv_n = vect_coeff(v,negat->vecteur);
    i = pgcd(abs(cv_p),abs(cv_n));
    cv_p/=i;
    cv_n/=i;

    /* combinaison des inegalites */

    ineg = contrainte_new();
    ineg->vecteur = vect_cl2(abs(cv_n),posit->vecteur,cv_p,negat->vecteur);

    /* combinaison des saturations */

    (ineg->s_sat) = (int *) MALLOC(nb_s_tot*sizeof(int),SAT_TAB,"envcomb");
    (ineg->r_sat) = (int *) MALLOC(nb_r_tot*sizeof(int),SAT_TAB,"envcomb");

    for(i=0;i<nb_s;i++) 
	(ineg->s_sat)[i] = cv_p*((negat->s_sat)[i]) - cv_n*((posit->s_sat)[i]);
    for(i=0;i<nb_r;i++) 
	(ineg->r_sat)[i] = cv_p*((negat->r_sat)[i]) - cv_n*((posit->r_sat)[i]);

    return(ineg);
}

/* elim_env_sat: sert a eliminer les champs saturations des inegalites
 * a detruire
 *
 * Modifications:
 *  - FI: set the pointers to the freed zones to NULL
 */
void elim_env_sat(inegs)
Pcontrainte inegs;
{
    Pcontrainte ineg;
    for(ineg=inegs;ineg!=NULL;ineg=ineg->succ) {
	if (ineg->s_sat!=NULL) {
	    FREE((char *)ineg->s_sat,SAT_TAB,"elim_env_sat");
	    ineg->s_sat = NULL;
	}
	if (ineg->r_sat!=NULL) {
	    FREE((char *)ineg->r_sat,SAT_TAB,"elim_env_sat");
	    ineg->r_sat = NULL;
	}
    }
}

struct prtri_struct {Pcontrainte pos,neg,cnul;};
/* void tri():
 * pour combiner les inegalites afin d'eliminer une variable, il faut
 * trier les inegalites en trois classes; celles ou la variable a
 * un coefficient positif, celles ou il est negatif et enfin celles ou il
 * est nul; la fonction tri fait cela en modifiant une structure a trois
 * champs qui seront les pointeurs des listes des inegalites triees
 *
 * FI: j'ai recupere cette fonction chez malik dans proj.c; je l'avais
 * auparavant eliminee de l'algo de projection; je dois pouvoir en faire
 * autant ici;
 */
static void tri(inegs,v,prtri)
Pcontrainte inegs;
Variable v;
struct prtri_struct *prtri;
{
    Pcontrainte ineg,ineg1;
    int c;

    prtri->pos = NULL;
    prtri->neg = NULL;
    prtri->cnul = NULL;
    if (inegs!=NULL) {
	for(ineg=inegs,ineg1=ineg->succ;ineg!=NULL;) {
	    if ((c = vect_coeff(v,ineg->vecteur)) > 0) {
		ineg->succ = prtri->pos;
		prtri->pos = ineg;
	    }
	    else {
		if (c < 0) {
		    ineg->succ = prtri->neg;
		    prtri->neg = ineg;
		}
		else {
		    ineg->succ = prtri->cnul;
		    prtri->cnul = ineg;
		}
	    }
	    ineg=ineg1;
	    if (ineg1!=NULL) ineg1=ineg1->succ;
	}
    }
}
	
/* env_combiner: construction de la liste des contraintes de l'enveloppe
 * convexe issues de combinaisons
 *
 * dans sigma2, elimine la variable v des inegalites par combinaisons
 */
Pcontrainte env_combiner(sigma2,v,nb_s,nb_r,nb_s_tot,nb_r_tot,type,d)
Pcontrainte sigma2;
Variable v;
int nb_s,nb_r,nb_s_tot,nb_r_tot;
int type;			/* type d'env. (sommet=>0, r ou dte=>1 ou 2) */
int d;				/* si type == 0 => denominateur du sommet  */
{
    struct {Pcontrainte pos,neg,cnul;} rtri;
    Pcontrainte posit, negat;
    Pcontrainte ineg,inegs=sigma2;

    /* il faut trier les contraintes en trois listes, celles ou le coefficient
       de la variable a eliminer est positif, ou bien negatif et enfin nul
       */

    tri(inegs,v,&rtri);

    /* il suffit maintenant de faire chacune des combinaisons */

    for(posit=rtri.pos;posit!=NULL;posit=posit->succ) {
	for(negat=rtri.neg;negat!=NULL;negat=negat->succ) {

	    ineg=envcomb(posit,negat,v,nb_s,nb_r,nb_s_tot,nb_r_tot);

	    if (contrainte_constante_p(ineg)) {
		if (contrainte_verifiee(ineg,FALSE)) {
		    CONTRAINTE_RM(ineg,"combiner");
		    /* combinaison => 1 termcst >= 0 */
		    /* inutile de la garder          */
		}
		else {
		    (void) printf("COMBINAISON DE ENV");
		    (void) printf(" SYSTEME NON FAISABLE\n");
		    abort();
		    /* systeme non faisable */
		    /* ceci ne doit jamais  */
		    /* arriver avec cousot  */
		    /* de toute facon, au   */
		    /* debut de l'enveloppe */
		    /* on teste que les 2   */
		    /* polyedres donnes sont*/
		    /* non vides            */
		}
	    }
	    else {
		/* normaliser la contrainte obtenue */
		env_norm_eq(ineg,nb_s,nb_r,type,d);
		ineg->succ = rtri.cnul;
		rtri.cnul = ineg;
		/* combinaison simple reussie => 1ineg en + */
	    }
	}
    }

    /* apres les combinaisons eliminer les elements devenus inutiles */

    /* is directly performed by contraintes_free and contrainte_free:
     * elim_env_sat(rtri.pos);
     * elim_env_sat(rtri.neg);
     */
    contraintes_free(rtri.pos);
    contraintes_free(rtri.neg);

    return(rtri.cnul);
}

/* env_s: enveloppe convexe d'un polyedre avec un sommet;
 * chaque contrainte (eg. et ineg.) du polyedre contient les informations
 * de saturation vis a vis des elements generateurs
 */
Ppoly env_s(p,s,nb_s_tot,nb_r_tot,dim)
Ppoly p;
Psommet s;
int nb_s_tot,nb_r_tot,dim;
{
    Psysteme syst;
    Psommet s1;
    int inclus;	/* cet entier sera positionne dans s_sat_env */
    /* pour indiquer si s est inclus dans p      */
    int nb_s,nb_r;
    Pcontrainte lt,sigma1,sigma2;

    ifdebug(8) {
	fprintf(stderr,"env_s: begin\n");
	fprintf(stderr,"env_s: p=\n");
	poly_dump(p);
	fprintf(stderr,"env_s: s=\n");
	sommet_dump(s);
    }

    sommet_normalize(s);
    nb_s = poly_nbre_sommets(p);
    nb_r = poly_nbre_rayons(p);

    /* calcul des saturations pour le sommet  */

    lt = s_sat_env(p,s,nb_s_tot,nb_r_tot,&inclus);
    if(inclus) {
	ifdebug(8) {
	    fprintf(stderr,"env_s: return p=\n");
	    poly_dump(p);
	    fprintf(stderr,"env_s: end\n");
	}
	return(p);
    }

    /* calcul de la premiere partie des contraintes de l'enveloppe */

    sigma1 = mk_sigma1(&lt,0,nb_s,nb_s_tot,nb_r_tot);

    /* calcul et  formation des contraintes a combiner pour obtenir */
    /* les dernieres contraintes de l'enveloppe convexe             */

    sigma2 = mk_sigma2(&lt,0,sommet_denominateur(s),
		       nb_s,nb_s,nb_r,nb_s_tot,nb_r_tot,dim);

    /* combinaison de ces contraintes */

    sigma2=env_combiner(sigma2,(Variable)(dim+1),nb_s+1,nb_r,nb_s_tot,
			nb_r_tot,0,sommet_denominateur(s));

    /* construction d'un systeme de contraintes a partir du resultat des   */
    /* combinaisons par formation possible d'egalites                      */

    syst = mk_env_egs(sigma2,poly_droites(p),nb_s+1,nb_r);

    /* on rajoute au systeme construit les premieres contraintes retenues  */
    /* sans ajouter d'inegalites redondantes                               */

    complete_env(syst->inegalites,sigma1,p->sc,nb_r,nb_s+1);

    /* on y rajoute aussi toutes les egalites initiales qui saturent tous  */
    /* les elements generateurs                                            */
    /* et on met toutes ces contraintes dans le polyedre                   */

    if (syst->egalites!=NULL) {
	if (p->sc->egalites!=NULL) {
	    for (lt=p->sc->egalites;lt->succ!=NULL;lt=lt->succ);
	    lt->succ = syst->egalites;
	}
	else p->sc->egalites = syst->egalites;
	p->sc->nb_eq = nb_elems_list(p->sc->egalites);
    }

    /* destruction des elements de travail locaux */

    FREE((char *)syst,SYSTEME,"env_s");

    /* on n'oublie pas d'ajouter le sommet au systeme generateur pour obtenir
       le polyedre final
       */

    if (poly_sommets(p)!=NULL) {
	for(s1=poly_sommets(p);s1->succ!=NULL;s1=s1->succ);
	s1->succ=s;
	s->succ = NULL;
    }
    else {
	s->succ = NULL;
	poly_sommets(p) = s;
    }
    poly_nbre_sommets(p)++;

    ifdebug(8) {
	fprintf(stderr,"env_s: return p=\n");
	poly_dump(p);
	fprintf(stderr,"env_s: end\n");
    }

    return(p);
}

/* env_r: enveloppe convexe d'un polyedre avec un rayon
 * le processus est le meme que precedemment
 */
Ppoly env_r(p,r,nb_s_tot,nb_r_tot,dim)
Ppoly p;
Pray_dte r;
int nb_s_tot,nb_r_tot,dim;
{
    int inclus;	/* cet entier sera positionne dans s_sat_env */
    /* pour indiquer si s est inclus dans p      */
    Psysteme syst;
    Pray_dte r1;
    int nb_s,nb_r;
    Pcontrainte lt,sigma1,sigma2;

    nb_s = poly_nbre_sommets(p);
    nb_r = poly_nbre_rayons(p);
    lt = rd_sat_env(p,r,0,nb_s_tot,nb_r_tot,&inclus);
    if (inclus) return(p);
    sigma1 = mk_sigma1(&lt,1,nb_r,nb_s_tot,nb_r_tot);
    sigma2 = mk_sigma2(&lt,1,1,nb_r,nb_s,nb_r,nb_s_tot,nb_r_tot,dim);
    sigma2 = env_combiner(sigma2,(Variable)(dim+1),nb_s,nb_r+1,nb_s_tot,
			  nb_r_tot,1,1);
    syst = mk_env_egs(sigma2,poly_droites(p),nb_s,nb_r+1);
    complete_env(syst->inegalites,sigma1,p->sc,nb_r+1,nb_s);
    if (syst->egalites!=NULL) {
	if (p->sc->egalites!=NULL) {
	    for (lt=p->sc->egalites;lt->succ!=NULL;lt=lt->succ);
	    lt->succ = syst->egalites;
	}
	else p->sc->egalites = syst->egalites;
    }
    FREE((char *)syst,SYSTEME,"env_r");
    if (poly_rayons(p)!=NULL) {
	for(r1=poly_rayons(p);r1->succ!=NULL;r1=r1->succ);
	r1->succ=r;
	r->succ = NULL;
    }
    else {
	r->succ = NULL;
	poly_rayons(p) = r;
    }
    poly_nbre_rayons(p)++;
    return(p);
}

/* env_d: enveloppe convexe d'un polyedre p avec une droite d; le
 * polyedre p est modifie; la droite d perd son successeur;
 */
Ppoly env_d(p,d,nb_s_tot,nb_r_tot,dim)
Ppoly p;
Pray_dte d;
int nb_s_tot,nb_r_tot,dim;
{
    int inclus;
    Psysteme syst;
    Pray_dte d1;
    int nb_s,nb_r;
    Pcontrainte lt,sigma1,sigma2;

    ifdebug(8) (void) fprintf(stderr,"env_d: begin\n");

    /* la droite d est-elle inclue dans p? */
    lt = rd_sat_env(p,d,1,nb_s_tot,nb_r_tot,&inclus);
    if (inclus) {
	ifdebug(8) (void) fprintf(stderr,"env_d: end\n");
	return(p);
    }

    nb_s = poly_nbre_sommets(p);
    nb_r = poly_nbre_rayons(p);
    sigma1 = mk_sigma1(&lt,2,nb_r,nb_s_tot,nb_r_tot);
    sigma2 = mk_sigma2(&lt,2,1,nb_r,nb_s,nb_r,nb_s_tot,nb_r_tot,dim);
    sigma2 = env_combiner(sigma2,(Variable)(dim+1),nb_s,nb_r+1,nb_s_tot,
			  nb_r_tot,2,1);

    /* ajouter la droite d, passee en argument, au polyedre p */
    if (poly_droites(p)!=NULL) {
	for(d1=poly_droites(p);d1->succ!=NULL;d1=d1->succ)
	    ;
	d1->succ=d;
	d->succ = NULL;
    }
    else {
	d->succ = NULL;
	poly_droites(p) = d;
    }
    poly_nbre_droites(p)++;

    syst = mk_env_egs(sigma2,poly_droites(p),nb_s,nb_r);
    complete_env(syst->inegalites,sigma1,p->sc,nb_r,nb_s);
    /* ajouter une liste d'egalites apres avoir calcule des inegalites !?! */
    if (syst->egalites!=NULL) {
	if (p->sc->egalites!=NULL) {
	    for (lt=p->sc->egalites;lt->succ!=NULL;lt=lt->succ);
	    lt->succ = syst->egalites;
	}
	else p->sc->egalites = syst->egalites;
    }
    FREE((char *)syst,SYSTEME,"env_d");

    ifdebug(8) (void) fprintf(stderr,"env_d: end\n");
    return(p);
}

/* Ppoly env(Ppoly a, Ppoly b): enveloppe convexe de deux polyedres a et b;
 * ces deux polyedres doivent etre exprimes dans la meme base
 *
 * ATTENTION FONCTION DESTRUCTRICE DE Pa ET Pb
 *
 * Modifications:
 *  - il faudrait ecrire deux fonctions d'enveloppe convexe;
 *    la premiere aurait la meme interface que env et se
 *    contenterait de traiter les cas triviaux et d'appeler
 *    la seconde; la seconde recevrait un polyedre et un
 *    systeme generateur pour renvoyer un systeme generateur
 *  - les elements du systeme generateur de p2 semblent etre
 *    "dechaines" sans etre desalloues
 */
Ppoly env(pa,pb)
Ppoly pa,pb;
{
    Psommet s;
    Pray_dte rd;
    Ppoly p1,p2;
    int i,nb_s_tot,nb_r_tot,nbs2=0,nbr2,nbd2,nbs1=0,nbr1,nbd1;
    int dim;

    ifdebug(8) {
	(void) fprintf(stderr,"env: begin\n");
	(void) fprintf(stderr,"polyedre a:\n");
	poly_fprint(stderr, pa, variable_dump_name);
	(void) fprintf(stderr,"polyedre b:\n");

    }

    /* rappel : un polyedre est NULL si c'est le polyedre initial de la
       propagation (Malik Imadache)
       */
    /* Mein Gott! Ce n'est plus vrai avec l'interface intf! 
       (Francois Irigoin)
       */

    /* rappel : si le nombre de sommet est 0, le polyedre est vide
       (Goot sei dank! C'est encore vrai. (Francois Irigoin)
       */

    if (pa==NULL) 
	p1 = pb;
    else if (pb==NULL)
	p1 = pa;
    else if ((nbs1=poly_nbre_sommets(pa))==0) {
	poly_rm(pa);
	p1 = pb;
    }
    else if((nbs2 = poly_nbre_sommets(pb))==0) {
	poly_rm(pb);
	p1 = pa;
    }

    if(pa==NULL || pb==NULL || nbs1==0 || nbs2==0) {
	ifdebug(8) (void) fprintf(stderr,"env: end\n");
	return p1;
    }

    assert(pa->sc->dimension==pb->sc->dimension);
    dim = pa->sc->dimension;

    nbr2 = poly_nbre_rayons(pb);
    nbd2 = poly_nbre_droites(pb);
    nbr1 = poly_nbre_rayons(pa);
    nbd1 = poly_nbre_droites(pa);

    /* Sachons tirer profit du fait que l'enveloppe convexe
       de 2 polyedres est commutative. On fait les calculs
       sur p1, le polyedre qui a le plus d'elements generateurs,
       avec les elements du systeme generateur de p2.
       */

    if (nbs1 + nbr1 + nbd1 < nbs2 + nbr2 + nbd2) { 
	p1 = pb ; p2 = pa;
	nbs2 = nbs1; nbr2 = nbr1; nbd2 = nbd1;
    }
    else { p1 = pa ; p2 = pb; }

    /* le nombre de sommets et rayons finals est au plus la somme du nombre
       de sommets d'une part et de rayons d'autre part des 2 polyedres
       */

    nb_s_tot = poly_nbre_sommets(p1) + nbs2 ;
    nb_r_tot = poly_nbre_rayons(p1) + nbr2 + 1;

    /*
     *  norm_syst(p1->sc);
     */

    /* initialisation des saturations pour les contraintes de p1 vis a vis  */
    /* de son systeme generateur                                            */

    init_sat_env(p1,nb_s_tot,nb_r_tot);

    /* pour chaque droite de p2 affecter a p1 l'enveloppe convexe de p1 et de
       cette droite */

    for (i=0;i<nbd2;i++) {
	rd = poly_droites(p2);
	poly_droites(p2) = rd->succ;
	(void) env_d(p1,rd,nb_s_tot,nb_r_tot,dim);
    }

    /* pour chaque rayon de p2 affecter a p1 l'enveloppe convexe de p1 et de */
    /* ce rayon                                                              */

    for (i=0;i<nbr2;i++) {
	rd = poly_rayons(p2);
	poly_rayons(p2) = rd->succ;
	(void) env_r(p1,rd,nb_s_tot,nb_r_tot,dim);
    }

    /* pour chaque sommet de p2 affecter a p1 l'enveloppe convexe de p1 et de
       ce sommet */

    for (i=0;i<nbs2;i++) {
	s = poly_sommets(p2);
	poly_sommets(p2) = s->succ;
	(void) env_s(p1,s,nb_s_tot,nb_r_tot,dim);
    }

    /* eliminer les derniere traces de p2 (sc) */

    poly_rm(p2);

    ifdebug(8) (void) fprintf(stderr,"env: end\n");
    return(p1);
}

void env_norm_eq(nr,nb_s,nb_r,type,d)
Pcontrainte nr;
int nb_s,nb_r;
int type,d;		/* si env. avec sommet => type==0 et d == sommet_denominateur(s) */
{
    int i;
    int div = vect_pgcd_all(nr->vecteur);
    vect_normalize(nr->vecteur);
    if (type==0 && div%d==0) div/=d;
    for (i=0;i<nb_s;i++) (nr->s_sat)[i]/=div;
    for (i=0;i<nb_r;i++) (nr->r_sat)[i]/=div;
}
