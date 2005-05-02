 /* elimination de redondance
  *
  * Malik Imadache
  *
  * modifications par Francois Irigoin:
  *  - reprise des includes
  *  - reformattages divers
  *  - ajout d'un troisieme argument a MALLOC
  *  - elim_red(): modification du traitement des inegalites qui s'averent
  *    etre des egalites; Avril 1991
  */

#include <stdio.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"

#include "polyedre.h"

static int elim_red_debug_level = 0;
#define ifdebug(n) if((n)<elim_red_debug_level)

#define MALLOC(s,t,f) malloc((unsigned)(s))
#define FREE(p,t,f) free((char *)(p))

/* red_s_inter: elimination dans une liste de sommets contenant les
 * informations de redondances, de tous les sommets redondants; on donne
 * aussi l'adresse de la variables ou l'on compte le nombre de sommets
 * existants; Cette version d'elimination de redondance est appelee lors
 * des fonctions d'intersection de polyedres avec des demi-espaces ou des
 * hyperplans (versions non redondantes)
 */
Psommet red_s_inter(listsoms,ad_nb_soms,nb_eq)
Psommet listsoms;
int *ad_nb_soms,nb_eq;
{
    Psommet result,s1,s1_1,s2,s3;
    int red,elim;

    result = listsoms;
    s1_1 = NULL;

    /* pour chaque sommet  */

    for (s1=listsoms;s1!=NULL;) {
	elim = 0;

	/* comparaison avec chacun des autres sommets */

	for(s2=s1->succ,s3=s1;s2!=NULL;) {
	    if ((red=redondance(s1->eq_sat,s2->eq_sat,nb_eq))==2 || 
		red ==3) {

		/* le sommet avec lequel on compare s1 est redondant => l'eliminer */
		/* continuer la boucle                                             */

		s3->succ = s2->succ;
		SOMMET_RM(s2,"red_s_inter"); 
		s2 = s3->succ;
		(*ad_nb_soms)--;
	    } 
	    else {
		if (red==1) {

		    /* s1 est redondant par rapport au sommet avec lequel on le compare */
		    /* => marquer s1 elimine , l'eliminer et arreter la boucle           */

		    elim = 1;
		    (*ad_nb_soms)--;
		    if (result==s1) {
			/* s1 etait en tete de liste */
			result = s1->succ;
			SOMMET_RM(s1,"red_s_inter");
			s1 = result;
			break;
		    }
		    else {
			/* s1 est en milieu de liste avec s1_1 */
			/* comme predecesseur                  */
			s1_1->succ = s1->succ;
			SOMMET_RM(s1,"red_s_inter");
			s1 = s1_1->succ;
			break;
		    }
		} else {

		    /* pas de redondance entre s1 et l'element avec lequel on l'a comparer */
		    /* continuer les comparaisons avec les suivant                         */

		    s2 = s2->succ;
		    s3 = s3->succ;
		}
	    }
	}
	if (elim==0) {

	    /* si s1 n'a pas ete elimine passer au successeur en notant le predecesseur */
	    /* s1_1; si s1 a ete elimine s1 et s1_1 ont deja leur valeur                */

	    s1_1 = s1;
	    if (s1 != NULL) s1 = s1->succ;
	}
		
    }
    return(result);
}

Psommet env_red_s(listsoms,ad_nb_soms,nb_eq)
Psommet listsoms;
int *ad_nb_soms,nb_eq;
{
    return(red_s_inter(listsoms,ad_nb_soms,nb_eq-1));
}

/* red_r_inter: meme fonctionnalite que precedemment mais sur des rayons */
Pray_dte red_r_inter(listrays,ad_nb_rays,nb_eq)
Pray_dte listrays;
int *ad_nb_rays,nb_eq;
{
    Pray_dte result,s1,s1_1,s2,s3;
    int red,elim;

    result = listrays;
    s1_1 = NULL;

    /* pour chaque rayon */

    for (s1=listrays;s1!=NULL;) {
	elim = 0;

	/* comparaison avec chacun des autres rayons */

	for(s2=s1->succ,s3=s1;s2!=NULL;) {
	    if ((red=redondance(s1->eq_sat,s2->eq_sat,nb_eq))==2 || 
		red ==3) {

		/* le rayon avec lequel on compare s1 est redondant: l'eliminer
		   et continuer la boucle */

		s3->succ = s2->succ;
		RAY_DTE_RM(s2,"red_s_inter"); 
		s2 = s3->succ;
		(*ad_nb_rays)--;
	    } 
	    else {
		if (red==1) {

		    /* s1 est redondant par rapport au rayon avec lequel on le
		       compare: marquer s1 elimine, l'eliminer et arreter la
		       boucle */

		    elim = 1;
		    (*ad_nb_rays)--;
		    if (result==s1) {
			/* s1 etait en tete de liste */
			result = s1->succ;
			RAY_DTE_RM(s1,"red_s_inter");
			s1 = result;
			break;
		    }
		    else {
			/* s1 est en milieu de liste avec s1_1 */
			/* comme predecesseur                  */
			s1_1->succ = s1->succ;
			RAY_DTE_RM(s1,"red_s_inter");
			s1 = s1_1->succ;
			break;
		    }
		} else {

		    /* pas de redondance entre s1 et l'element avec lequel on l'a comparer */
		    /* continuer les comparaisons avec les suivant                         */

		    s2 = s2->succ;
		    s3 = s3->succ;
		}
	    }
	}
	if (elim==0) {

	    /* si s1 n'a pas ete elimine passer au successeur en notant le predecesseur */
	    /* s1_1; si s1 a ete elimine s1 et s1_1 ont deja leur valeur                */

	    s1_1 = s1;
	    if (s1 != NULL) s1 = s1->succ;
	}
		
    }
    return(result);
}

/* env_red_r: identique a la precedente; la seule difference est que
 * tous les rayons proportionnels a des droites ont un entier valant 1 dans
 * leur case associee du tableau proport_ray; de tels rayons sont alors a
 * eliminer; cette fonction est necessaire lors d'enveloppes convexes
 * successives
 *
 * rappel : l'indice associe a un rayon est son numero d'ordre dans la
 * la liste des rayons
 */
Pray_dte env_red_r(listrays,ad_nb_rays,nb_eq,proport_ray)
Pray_dte listrays;
int *ad_nb_rays,nb_eq,*proport_ray;
{
    Pray_dte s1,s1_1,s2;
    int i;
    s1_1 = listrays;

    /* elimination des rayons proportionnels a des droites */

    for (s2=s1=listrays,i=0;s1!=NULL;i++) {
	if (proport_ray[i]==1) {
	    /* tester si le rayon est en tete de liste */
	    if (s2!=s1) {
		s2->succ = s1->succ;
		RAY_DTE_RM(s1,"env_red_r");
		s1 = s2->succ;
	    }
	    else {
		s2 = s1->succ;
		RAY_DTE_RM(s1,"env_red_r");
		s1 = s2;
		s1_1 = s1;
	    }
	    (*ad_nb_rays)--;
	}
	else {
	    s2 = s1;
	    s1 = s1->succ;
	}
    }

    /* elimination des rayons redondants entre eux */

    return(red_r_inter(s1_1,ad_nb_rays,nb_eq-1));
}

/**** fonctions de test de redondance ****/

/*** dans l'enveloppe convexe ***/

/* entre inegalites */

int env_red(eq1,eq2,nb_s,nb_r)
Pcontrainte eq1,eq2;
int nb_r,nb_s;
{
    int reds = redondance(eq1->s_sat,eq2->s_sat,nb_s-1);
    int redr = redondance(eq1->r_sat,eq2->r_sat,nb_r-1);
    if (nb_r<=0) return(reds);
    if (redr == reds) return(reds);
    if (reds == 3) return(redr);
    if (redr == 3) return(reds);
    return(0);
}

/* redondance: compare deux tableaux et renvoie 1 si un des tableaux
 * a des 0 sur au moins toutes les cases ou l'autre en a .
 */
int redondance(t1,t2,l)
int *t1, *t2;
int l;	/* longueur des tableaux */
{
    int i,ident=1;
    int more_0=0;
    for(i=0;i<=l && ident==1;i++) {
	if ( (t1[i]==0 && t2[i]!=0) || (t1[i]!=0 && t2[i]==0) ) {
	    ident = 0;
	    more_0 = (t1[i]==0 ? 1 : 2);
	}
    }
    if (ident==1) return(3);
    if (more_0 == 1) 
	for(;i<=l;i++) {
	    if (t1[i]!=0 && t2[i]==0) return(0);
	}
    else
	for(;i<=l;i++) {
	    if (t2[i]!=0 && t1[i]==0) return(0);
	}
    if (more_0 == 1) return(2);
    else return(1);
}

/******************************************************************************/
/**** ici on complete sigma1, a l'aide des inegalites inegs, toutes ces ineg. */
/**** ont leurs saturations calculees,  on ajoute l'inegalite testee a celles */
/**** de sigma1     ou on l'elimine ou on l'echange avec celles qui saturent  */
/**** moins d'elements generateurs                                            */
/******************************************************************************/

Pcontrainte complete1(inegs,sigma1,nb_r,nb_s)
Pcontrainte inegs,sigma1;
int nb_r,nb_s;
{
    Pcontrainte result,ltpred,lt,ineg,reste_inegs;
    int red,elim;
    result = ltpred = lt = sigma1;

    /* pour chacune des inegalites a ajouter */

    for (ineg=inegs;ineg!=NULL;ineg=reste_inegs) {
	reste_inegs = ineg->succ;
	ineg->succ = NULL;
	elim = 0;

	/* faire une comparaison avec chacune des inegalites deja retenues */

	for(;lt!=NULL;) {
	    if ((red = env_red(ineg,lt,nb_s,nb_r))==1 || red==3) {

		/* si l'inegalite a ajouter est redondante => l'eliminer */

		elim = 1;
		/* directly performed by contrainte_free
		 * FREE((char *)ineg->s_sat,SAT_TAB,"complete1");
		 * FREE((char *)ineg->r_sat,SAT_TAB,"complete1");
		 */
		CONTRAINTE_RM(ineg,"complete1");
		break;
	    }
	    else {
		if (red==2) {
		    /* si des inegalites deja retenues sont redondantes
		       par rapport a la presente, les eliminer */

		    if (lt == result) {
			/* en tete de liste */
			result = result->succ;
			/*
			 * FREE((char *)lt->s_sat,SAT_TAB,"complete_env");
			 * FREE((char *)lt->r_sat,SAT_TAB,"complete_env");
			 */
			CONTRAINTE_RM(lt,"complete_env");
			lt = result;
			ltpred = result;
		    }
		    else {
			/* en milieu de liste */
			ltpred->succ = lt->succ;
			/*
			  * FREE((char *)lt->s_sat,SAT_TAB,"complete_env");
			  * FREE((char *)lt->r_sat,SAT_TAB,"complete_env");
			  */
			CONTRAINTE_RM(lt,"complete_env");
			lt = ltpred->succ;
		    }
		}
		else {
		    /* pas de redondance, tester avec la contrainte
		       retenue suivante */
		    ltpred = lt;
		    lt = lt->succ;
		}
	    }
	}

	/* maintenant passer a l'element a ajouter suivant en 
	   preparant les varpour le tour de boucle suivant */

	if (elim!=1) {
	    /* si l'inegalite a ajouter n'est pas redondante => l'ajouter */
	    ineg->succ = result;
	    lt = ltpred = result = ineg;
	}
	else ltpred = lt = result;
    }
    return(result);
}

/* complete_env: pour l'enveloppe convexe, on a besoin de completer un systeme
 * de contraintes avec elimination de redondances et de connaitre le
 * nombre de contraintes resultantes
 */
void complete_env(inegs,sigma1,sc,nb_r,nb_s)
Pcontrainte inegs,sigma1;
Psysteme sc;
int nb_r,nb_s;
{
    sc->inegalites = complete1(inegs,sigma1,nb_r,nb_s);
    sc->nb_ineq = nb_elems_list(sc->inegalites);
    sc->nb_eq = nb_elems_list(sc->egalites);
}

/* env_sg_red: la fonction suivante elimine les elements generateurs redondants
 * sachant que l'on connait les saturations des contraintes
 * par rapport a chacun d'eux
 */
void env_sg_red(p)
Ppoly p;
{
    Psommet s;
    Pray_dte r,d;
    Pcontrainte ineq;
    int i,j,proport=0;
    int *proport_ray;

    /* recopie des saturations des contraintes dans les saturations des sommets */

    for(s=poly_sommets(p),i=0;s!=NULL;s=s->succ,i++) {
	s->eq_sat = 
	    (int *) MALLOC(poly_nbre_inegalites(p)*sizeof(int),SAT_TAB,"env_sg_red");
	for (ineq=poly_inegalites(p),j=0;ineq!=NULL;ineq=ineq->succ,j++) {
	    (s->eq_sat)[j] = (ineq->s_sat)[i];
	}
    }

    /* recuperations des saturations pour les rayons                         */
    /* et test de l'eventuelle proportionnalite d'un rayon par rapport a une */
    /* droite (le resultat est dans le tableau proport_ray)                  */

    proport_ray = (int *)MALLOC(poly_nbre_rayons(p)*sizeof(int),TINT,"env_sg_red");
    for(r=poly_rayons(p),i=0;r!=NULL;r=r->succ,i++) {
	r->eq_sat = 
	    (int *) MALLOC(poly_nbre_inegalites(p)*sizeof(int),SAT_TAB,"env_sg_red");

	/* pour chaque rayon test de la proportionnalite a une droite */

	proport = 0;
	proport_ray[i] = 0;
	for (d=poly_droites(p);d!=NULL;d=d->succ) 
	    if (vect_proport(r->vecteur,d->vecteur)) {
		proport_ray[i] = 1;
		proport = 1;
		break;
	    }
	if (!proport) {

	    /* en cas de non proportionnalite, recopie des saturations */

	    for (ineq=poly_inegalites(p),j=0;ineq!=NULL;ineq=ineq->succ,j++) {
		(r->eq_sat)[j] = (ineq->r_sat)[i];
	    }
	}

	/* en cas de proportionnalite, on donne ce rayon comme ne saturant aucune */
	/* aucune contrainte => elimination                                       */

	else
	    for(j=0;j<poly_nbre_inegalites(p);j++) (r->eq_sat)[j] = 1;
    }

    /* appels effectifs aux fonctions d'elimination de redondance */

    poly_sommets(p)=env_red_s(poly_sommets(p),&poly_nbre_sommets(p),poly_nbre_inegalites(p));
    poly_rayons(p)=env_red_r(poly_rayons(p),&poly_nbre_rayons(p),poly_nbre_inegalites(p),
			   proport_ray);
}

/* elim_egs_id: elimine les egalites identiques d'un polyedre,
 * les eliminations de redondances ne portant pas sur les egalites
 */
void elim_egs_id(p)
Ppoly p;
{
    Pcontrainte egs,eg1;
    egs = poly_inegalites(p);
    poly_nbre_egalites(p) = 0;
    poly_inegalites(p) = NULL;
    for (eg1 = egs; eg1!=NULL; eg1=egs) {
	egs = egs->succ;

	/* test d'appartenance d'une egalite a une liste; 
	   teste l'egalite de tous
	   les coefficients ou bien l'opposition de tous les coeff. mais pas la
	   proportionnalite
         */

	if (egalite_in_liste(eg1,poly_inegalites(p)))
	    CONTRAINTE_RM(eg1,"elim_egs_id");
	else {
	    eg1->succ = poly_inegalites(p);
	    poly_inegalites(p) = eg1;
	    poly_nbre_egalites(p)++;
	}
    }
}

/* elim_red: elinmination de redondance generale au sein d'un polyedre */
void elim_red(p)
Ppoly p;
{
    int nb_s,nb_r,i,*ssat,*rsat,satur_tout;
    Pcontrainte eg,eg_pred;
    Psommet s_p;
    Pray_dte r_p,d;

    nb_s = poly_nbre_sommets(p);
    nb_r = poly_nbre_rayons(p);
    eg_pred = NULL;

    /* calcul de toutes les saturations */

    for (eg = poly_inegalites(p); eg!=NULL ; ) {
	satur_tout = 1;
	ssat = (int *) MALLOC(nb_s*sizeof(int),SAT_TAB,"elim_red");
	rsat = (int *) MALLOC(nb_r*sizeof(int),SAT_TAB,"elim_red");
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
	if (satur_tout) {		/* c'est une egalite */
	    if (eg_pred==NULL) {
		poly_inegalites(p) = eg->succ;
		poly_nbre_inegalites(p)--;
		if (egalite_in_liste(eg,poly_egalites(p)))
		    CONTRAINTE_RM(eg,"elim_red");
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
		    CONTRAINTE_RM(eg,"elim_red");
		else {
		    eg->succ = poly_egalites(p);
		    poly_egalites(p) = eg;
		    poly_nbre_egalites(p)++;
		}
		poly_nbre_inegalites(p)--;
		eg = eg_pred->succ;
	    }
	    FREE((char *)ssat,SAT_TAB,"elim_red");
	    FREE((char *)rsat,SAT_TAB,"elim_red");
	}
	else {
	    eg->s_sat = ssat;
	    eg->r_sat = rsat;
	    eg_pred = eg;
	    eg = eg->succ;
	}
    }

    /* elimination des egalites identiques */

    elim_egs_id(p);

    /* elimination des inegalites redondantes */

    complete_env(poly_inegalites(p),(Pcontrainte)NULL,p->sc,
		 poly_nbre_rayons(p),poly_nbre_sommets(p));

    /* elimination des elements generateurs redondants */

    env_sg_red(p);
}

/*****************************************************************************/
/* ELIMINATION DES INEGALITES REDONDANTES VIS A VIS DES EGALITES.            */
/* par substitution successives des egalites dans les inegalites             */
/*****************************************************************************/

void elim_triviales(sc)
Psysteme sc;
{
    Pcontrainte eg,eg1;
    int elim=0;
    for (eg=sc->egalites,eg1=NULL;eg!=NULL;) {
	elim=0;
	if (eg->vecteur!=NULL) {
	    if (eg->vecteur->var==0 && eg->vecteur->succ==NULL)
		elim=1;
	} else elim=1;
	if (elim) {
	    if (eg1==NULL) {
		sc->egalites=eg->succ;
		CONTRAINTE_RM(eg,"elim_triviales");
		eg=sc->egalites;
		sc_nbre_egalites(sc)--;
	    }
	    else {
		eg1->succ=eg->succ;
		CONTRAINTE_RM(eg,"elim_triviales");
		sc_nbre_egalites(sc)--;
		eg=eg1->succ;
	    }
	} else {
	    eg1=eg;
	    eg=eg->succ;
	}
    }
    for (eg=sc->inegalites,eg1=NULL;eg!=NULL;) {
	elim=0;
	if (eg->vecteur!=NULL) {
	    if (eg->vecteur->var==0 && eg->vecteur->succ==NULL)
		elim=1;
	} else elim=1;
	if (elim) {
	    if (eg1==NULL) {
		sc->inegalites=eg->succ;
		CONTRAINTE_RM(eg,"elim_triviales");
		eg=sc->inegalites;
		sc_nbre_inegalites(sc)--;
	    }
	    else {
		eg1->succ=eg->succ;
		CONTRAINTE_RM(eg,"elim_triviales");
		sc_nbre_inegalites(sc)--;
		eg=eg1->succ;
	    }
	} 
	else {
	    eg1=eg;
	    eg=eg->succ;
	}
    }
}

/* red_egs_inegs: resolution d'un systeme d'egalite et utilisation du
 * resultat pour simplifier un systeme d'inegalite
 *
 * ce module ne profite pas du travail de Corinne et n'effectue pas une
 * resolution en nombres entiers qui permettrait peut-etre d'affiner
 * les termes constants de inegalites
 */
void red_egs_inegs(sc)
Psysteme sc;
{
    Pcontrainte eg,eg1;
    Variable v;

    ifdebug(8) {
	(void) fprintf(stderr,"red_egs_inegs: begin\n");
	(void) fprintf(stderr,"red_egs_inegs: sc\n");
	sc_fprint(stderr, sc, variable_dump_name);
    }

    /* pour toutes les egalites */
    for (eg=sc->egalites;eg!=NULL;eg=eg->succ) 
	if (eg->vecteur!=NULL) 
	    /* est-ce une VRAIE equation? */
	    if ((v=eg->vecteur->var)!=TCST||eg->vecteur->succ!=NULL) {
		/* recherche d'une variable pivot, la premiere */
		if (v==TCST) 
		    v=eg->vecteur->succ->var;
		/* mise a jour des egalites suivantes */
		for (eg1=eg->succ;eg1!=NULL;eg1=eg1->succ)
		    (void) contrainte_subst(v,eg,eg1,TRUE);
		/* mise a jour de toutes les inegalites */
		for (eg1=sc->inegalites;eg1!=NULL;eg1=eg1->succ)
		    (void) contrainte_subst(v,eg,eg1,FALSE);
	    }
    elim_triviales(sc);

    ifdebug(8) {
	(void) fprintf(stderr,"red_egs_inegs: resultat\n");
	sc_fprint(stderr, sc, variable_dump_name);
	(void) fprintf(stderr,"red_egs_inegs: end\n");
    }
}
