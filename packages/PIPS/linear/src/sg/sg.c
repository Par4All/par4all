/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

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

 /* package pour la structure de donnees tsg (systeme generateur) et pour les
  * deux structures de donnees inclues tsg_vects et tsg_soms qui representent
  * les listes de rayons, de droites et de sommets d'un systeme generateur
  *
  * Francois Irigoin
  */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"

static int sg_debug_level = 0;
#define ifdebug(n) if((n)<sg_debug_level)

#define MALLOC(s,t,f) malloc(s)
#define FREE(p,t,f) free(p)

/* Ptsg sg_new(): allocation d'un systeme generateur et initialisation a la
 * valeur ensemble vide
 */
Ptsg sg_new()
{
    Ptsg sg;

    sg = (Ptsg) MALLOC(sizeof(Stsg),TSG,"sg_new");
    sg->soms_sg.nb_s = 0;
    sg->soms_sg.ssg = NULL;
    sg->rays_sg.nb_v = 0;
    sg->rays_sg.vsg = NULL;
    sg->dtes_sg.nb_v = 0;
    sg->dtes_sg.vsg = NULL;
    sg->base = NULL;
    return sg;
}

/* Ptsg sg_dup(Ptsg sg_in): allocation d'un systeme generateur sg_out
 * et copie sans sharing des ensembles de rayons, de droites et de
 * sommets de l'argument sg_in
 *
 * sg_in n'est pas modifie
 *
 * Autre nom (obsolete): cp_sg()
 *
 * Extension: duplicate undefined generating systems (FI, 22 May 2009)
 *
 * sc_dup() has been replaced by sc_copy() which maintains the order
 * in the data structure lists. Maybe we need the same change for
 * generating systems.
 */
Ptsg sg_dup(Ptsg sg_in)
{
  Ptsg sg_out = SG_UNDEFINED;

  if(!SG_UNDEFINED_P(sg_in)) {
    Pray_dte rd;
    Pray_dte rd_new;
    Psommet s;
    Psommet s_new;

    sg_out = sg_new();

    /* duplication de la liste des sommets */
    sg_out->soms_sg.nb_s = sg_in->soms_sg.nb_s;
    for(s = sg_in->soms_sg.ssg; s != NULL; s = s->succ) {
      s_new = sommet_dup(s);
      s_new->succ = sg_out->soms_sg.ssg;
      sg_out->soms_sg.ssg = s_new;
    }

    /* duplication de la liste des rayons */
    sg_out->rays_sg.nb_v = sg_in->rays_sg.nb_v;
    for(rd = sg_in->rays_sg.vsg; rd != NULL; rd = rd->succ) {
      rd_new = ray_dte_dup(rd);
      rd_new->succ = sg_out->rays_sg.vsg;
      sg_out->rays_sg.vsg = rd_new;
    }

    /* duplication de la liste des droites */
    sg_out->dtes_sg.nb_v = sg_in->dtes_sg.nb_v;
    for(rd = sg_in->dtes_sg.vsg; rd != NULL; rd = rd->succ) {
      rd_new = ray_dte_dup(rd);
      rd_new->succ = sg_out->dtes_sg.vsg;
      sg_out->dtes_sg.vsg = rd_new;
    }

    sg_out->base = base_dup(sg_in->base);
  }
  return sg_out;
}

/* Ptsg sg_without_line(Ptsg sg_in): allocation d'un systeme generateur
 * et copie d'un systeme generateur dont on transforme les lignes en rayons
 *
 * Aucune elimination de redondance pour le moment. On peut en envisager trois
 * niveaux: egalite entre rayons, proportionalite entre rayons et
 * combinaison lineaire positive de rayons.
 */
Ptsg sg_without_line(sg_in)
Ptsg sg_in;
{
    Ptsg sg_out;
    Pray_dte rd;
    Pray_dte rd_new;
    Pray_dte rd_prec;

    ifdebug(8) {
	(void) fprintf(stderr,"sg_without_line: begin\n");
	(void) fprintf(stderr,"sg_without_line: sg_in\n");
	sg_fprint(stderr, sg_in, variable_dump_name);
    }

    sg_out = sg_dup(sg_in);

    /* ajout des l'oppose des vecteurs directeur des droites a la liste
       des rayons */
    for(rd_prec=rd=sg_out->dtes_sg.vsg; rd!=NULL; rd_prec=rd, rd=rd->succ) {
	/* allocation et calcul du vecteur opppose */
	rd_new = ray_oppose(rd);
	/* chainage en tete de la liste des rayons */
	rd_new = sg_out->rays_sg.vsg;
	sg_out->rays_sg.vsg = rd_new;
    }

    /* ajout (sans duplication en memoire) de la liste des vecteurs directeurs
       des droites a la liste des rayons, si l'ensemble des droites n'est
       pas vide */
    if(rd_prec!=NULL) {
	/* ajout de l'ensemble des rayons a l'ensemble des droites */
	rd_prec->succ = sg_out->rays_sg.vsg;
	/* transfert de la liste des droites a la liste des rayons */
	sg_out->rays_sg.vsg = sg_out->dtes_sg.vsg;
	sg_out->dtes_sg.vsg = NULL;
	/* mise a jour des nombres de droites et de vecteurs */
	sg_out->rays_sg.nb_v += 2*(sg_out->dtes_sg.nb_v);
	sg_out->dtes_sg.nb_v = 0;
    }

    ifdebug(8) {
	(void) fprintf(stderr,"sg_without_line: sg_out\n");
	sg_fprint(stderr, sg_out, variable_dump_name);
	(void) fprintf(stderr,"sg_without_line: end\n");
    }

    return sg_out;
}

/* Ptsg sg_add_ray(Ptsg sg, Pray_dte ray): ajout d'un rayon a un syteme
 * generateur; aucune allocation n'est affectuee; il y a donc risque
 * de sharing pour le rayon ray; l'argument sg est modifie
 *
 * Aucune elimination de redondance pour le moment. On peut en envisager trois
 * niveaux: egalite entre rayons, proportionalite entre rayons et
 * combinaison lineaire positive de rayons.
 */
Ptsg sg_add_ray(sg,ray)
Ptsg sg;
Pray_dte ray;
{
    /* chainage en tete de la liste des rayons */
    ray = sg->rays_sg.vsg;
    sg->rays_sg.vsg = ray;

    return sg;
}

/* void sg_rm_sommets(Ptsg sg): desallocation d'une liste de sommets
 * d'un systeme generateur
 *
 * Ex-fonction elim_tt_som() qui prenait en argument une liste de sommets
 */
void sg_rm_sommets(sg)
Ptsg sg;
{
    Psommet s,s1;

    for (s=sg->soms_sg.ssg; s!=NULL;) {
	s1 = s->succ;
	SOMMET_RM(s,"sg_rm_sommets");
	s = s1;
    }
    sg->soms_sg.ssg = NULL;
    sg->soms_sg.nb_s = 0;
}

/* void sg_rm_rayons(Ptsg sg): desallocation d'une liste de rayons d'un systeme
 * generateur
 */
void sg_rm_rayons(sg)
Ptsg sg;
{
    elim_tt_rd(sg->rays_sg.vsg);
    sg->rays_sg.vsg = NULL;
    sg->rays_sg.nb_v = 0;
}

/* void sg_rm_droites(Ptsg sg): desallocation d'une liste de droites
 * d'un systeme generateur
 */
void sg_rm_droites(sg)
Ptsg sg;
{
    elim_tt_rd(sg->rays_sg.vsg);
    sg->rays_sg.vsg = NULL;
    sg->rays_sg.nb_v = 0;
}

/* void sg_rm(Ptsg sg): liberation de l'espace memoire occupe par
 * un systeme generateur
 *
 * Ne pas oublier de remettre le pointeur correspondant a NULL dans le
 * programme appelant
 *
 * Ancien nom: elim_sg()
 */
void sg_rm(sg)
Ptsg sg;
{
    sg_rm_sommets(sg);
    sg_rm_rayons(sg);
    sg_rm_droites(sg);
    base_rm(sg->base);
    FREE((char *)sg,TSG,"sg_rm");
}

/* void sg_fprint(FILE * f, Ptsg sg, char * (*nom_var)()):
 * impression d'un systeme generateur
 */
void sg_fprint(FILE * f, Ptsg sg, char * (*nom_var)(Variable))
{
    (void) fprintf(f,"Generating system:\n");
    (void) fprintf(f,"%d Vert%s \n",sg_nbre_sommets(sg),
		   sg_nbre_sommets(sg)==1? "ex" : "ices");
    fprint_lsom(f, sg_sommets(sg), nom_var);
    if(sg_nbre_rayons(sg)>0) {
	(void) fprintf(f,"\n%d Ray%s \n",sg_nbre_rayons(sg),
		       sg_nbre_rayons(sg)==1? "" : "s");
	fprint_lray_dte(f, sg_rayons(sg), nom_var);
    }
    if(sg_nbre_droites(sg)>0) {
	(void) fprintf(f,"\n%d Line%s \n",sg_nbre_droites(sg),
		       sg_nbre_rayons(sg)==1? "" : "s");
	fprint_lray_dte(f, sg_droites(sg), nom_var);
    }
    (void) fprintf(f,"\nEnd of generating system ****\n");
}

/* For debugging */
void sg_print(Ptsg sg, char * (*nom_var)(Variable))
{
  sg_fprint(stderr, sg, nom_var);
}

/* For debugging */
void sg_dump(Ptsg sg)
{
  sg_print(sg, variable_dump_name);
}

/* void sg_fprint_as_dense(FILE * f, Ptsg sg):
 * impression d'un systeme generateur
 *
 * Note: b could/should be sg->base
 */
void sg_fprint_as_dense(f, sg, b)
FILE * f;
Ptsg sg;
Pbase b;
{
    (void) fprintf(f,"Generating system:\n");

    (void) fprintf(f,"%d Vert%s \n",sg_nbre_sommets(sg),
		   sg_nbre_sommets(sg)==1? "ex" : "ices");
    fprint_lsom_as_dense(f, sg_sommets(sg), b);

    if(sg_nbre_rayons(sg)>0) {
	(void) fprintf(f,"\n%d Ray%s \n",sg_nbre_rayons(sg),
		       sg_nbre_rayons(sg)==1? "" : "s");
	fprint_lray_dte_as_dense(f, sg_rayons(sg), b);
    }

    if(sg_nbre_droites(sg)>0) {
	(void) fprintf(f,"\n%d Line%s \n",sg_nbre_droites(sg),
		       sg_nbre_rayons(sg)==1? "" : "s");
	fprint_lray_dte_as_dense(f, sg_droites(sg), b);
    }

    (void) fprintf(f,"\nEnd of generating system ****\n");
}

/* Print a generating system as a dependence direction vector */

#define LESSER_DIRECTION 4
#define GREATER_DIRECTION 1
#define ZERO_DIRECTION 2
#define ANY_DIRECTION 7
#define NO_DIRECTION 0

static int vertex_to_direction[3] = {LESSER_DIRECTION, ZERO_DIRECTION, GREATER_DIRECTION};

static int ray_to_direction[3] = {LESSER_DIRECTION, NO_DIRECTION, GREATER_DIRECTION};

static int line_to_direction[3] = {ANY_DIRECTION, NO_DIRECTION, ANY_DIRECTION};

static char * direction_to_representation[8] = {"?", "<", "=", "<=", ">", "*", ">=", "*"};

void
sg_fprint_as_ddv(
    FILE * fd,
    Ptsg sg)
{
    Pbase c = BASE_NULLE;
    int ddv = 0;

    fprintf(fd, "ddv(");

    /* For each coordinate */
    for(c=sg->base; !BASE_NULLE_P(c); c = c->succ) {
	Psommet v;
	Pray_dte r;
	Pray_dte l;

	/* For all vertices */
	for(v=sg_sommets(sg); v!= NULL; v = v->succ) {
	    ddv |= vertex_to_direction[1+value_sign(vect_coeff(vecteur_var(c), v->vecteur))];
	}

	/* For all rays */
	for(r=sg_rayons(sg); r!= NULL; r = r->succ) {
	    ddv |= ray_to_direction[1+value_sign(vect_coeff(vecteur_var(c), r->vecteur))];
	}

	/* For all lines */
	for(l=sg_droites(sg); l!= NULL; l = l->succ) {
	    ddv |= line_to_direction[1+value_sign(vect_coeff(vecteur_var(c), l->vecteur))];
	}

	fprintf(fd, c==sg->base? "%s" : ",%s", direction_to_representation[ddv]);
    }

    fprintf(fd, ")");
}

//void sg_dump(sg)
//Ptsg sg;
//{
//    /* char * (*variable_dump_name)(); */
//
//    sg_fprint(stderr, sg, variable_dump_name);
//}


/* bool egal_soms(Ptsg_soms sgs1, Ptsg_soms sgs2): test de l'egalite
 * de deux listes de sommets associees a un systeme generateur
 *
 * Malik: on notera que deux polyedres non faisables (nb sommets = 0)
 * seront identiques quelque soit la representation du SC 
 *
 * Francois: comme la structure de donnees tsg_soms n'existe jamais "libre"
 * mais n'est toujours qu'une partie de la structure tsg, je ne pense pas
 * qu'il soit necessaire de rendre cette routine visible de l'exterieur
 */
static bool egal_soms(sgs1,sgs2)
Ptsg_soms sgs1,sgs2;
{
	if (sgs1->nb_s != sgs2->nb_s) return(false);
	if (sgs1->nb_s == 0) return(true);
	return(egaliste_s(sgs1->ssg,&(sgs2->ssg)));
}
		
/* bool egal_rd(Ptsg_vects rdgs1, Ptsg_vects rdgs2): 
 * test de de l'egalite des deux objets representes par des structures
 * tsg_vects
 * 
 * Ne doit servir que dans sg_egal()
 */
static bool egal_rd(rdgs1,rdgs2)
Ptsg_vects rdgs1,rdgs2;
{
	int result;
	if (rdgs1->nb_v != rdgs2->nb_v) return(0);
	result = egaliste_rd(rdgs1->vsg,&(rdgs2->vsg));
	return(result);
}

/* bool sg_egal(Ptsg sg1, Ptsg sg2): test de l'egalite de deux systemes
 * generateur sg1 et sg2
 * 
 * Ce test suppose que les deux systemes ont ete prealablement normalises:
 *  - reduction des coordonnees des sommets, des rayons et des droites
 *    par leur PGCD
 *  - suppression des doublons dans les listes
 *  - suppression des rayons qui sont des droites
 *  - representation unique des vecteurs directeurs des droites
 *    (vecteur lexicopositif par exemple)
 *  - suppression des vecteurs nuls comme rayon ou droite
 *  - absence d'elements redondants dans les systemes generateurs
 *
 * Ancien nom: sg_egal()
 */
bool sg_egal(sg1,sg2)
Ptsg sg1,sg2;
{
    if (! egal_soms(&(sg1->soms_sg) ,&(sg2->soms_sg) ) ) return(false);
    if (sg_nbre_sommets(sg1) == 0) return(true);
    if (! egal_rd(&(sg1->rays_sg),&(sg2->rays_sg))) return(false);
    if (! egal_rd(&(sg1->dtes_sg),&(sg2->dtes_sg))) return(false);
    return(true);
}

/* Ptsg mk_rn(Pbase b): construction du systeme generateur du polyedre
 * qui represente l'espace entier; le systeme de contraintes correspondant
 * est vide: il ne contient ni egalite ni inegalite non triviale;
 *
 * Modifications:
 *  - remplacement du parametre entier dim par une base (FI, 18/12/89)
 */
Ptsg mk_rn(b)
Pbase b;
{
    Ptsg sg;
    Pray_dte d;

    ifdebug(8) (void) fprintf(stderr,"mk_rn: begin\n");

    sg = sg_new();

    /* un sommet a l'origine */
    (sg->soms_sg).nb_s = 1;
    (sg->soms_sg).ssg = sommet_make(VALUE_ONE, VECTEUR_NUL);

    /* il n'y a aucun rayon */
    (sg->rays_sg).nb_v = 0;
    (sg->rays_sg).vsg = NULL;

    /* enfin, une droite par dimension */
    (sg->dtes_sg).nb_v = base_dimension(b);
    (sg->dtes_sg).vsg = NULL;
    for (;!VECTEUR_NUL_P(b); b = b->succ) {
	d = ray_dte_make(vect_new(vecteur_var(b), VALUE_ONE));
	d->succ = (sg->dtes_sg).vsg;
	(sg->dtes_sg).vsg = d;
    }
    ifdebug(8) {
	sg_fprint(stderr, sg, variable_dump_name);
	(void) fprintf(stderr,"mk_rn: end\n");
    }
    return(sg);
}

/* Ptsg ajout_dte(Ptsg sg, Variable v): ajoute une droite dans la direction
 * correspondant a la variable v
 */
Ptsg ajout_dte(sg,v)	
Ptsg sg;
Variable v;
{
    Pray_dte dte;

    /* creation */

    dte = (Pray_dte ) MALLOC(sizeof(Sray_dte),RAY_DTE,"ajout_dte");
    dte->vecteur = vect_new(v, VALUE_ONE);

    /* ajout */

    dte->succ = sg_droites(sg);
    sg_droites(sg) = dte;
    sg_nbre_droites(sg)++;
    return(sg);
}


bool sommet_in_sg_p(som,sg)
Psommet som;
Ptsg sg;
{
    Psommet ps;
    bool trouve = false;

    if (!SG_UNDEFINED_P(sg)) {
	for (ps = sg->soms_sg.ssg; ps != NULL && !trouve ; ps=ps->succ) {
	    trouve = trouve || vect_equal(som->vecteur,ps->vecteur);
	}
    }
    return (trouve);
}

bool ray_in_sg_p(ray,sg)
Pray_dte ray;
Ptsg sg;
{
    Pray_dte pr;
    bool trouve = false;

    if (!SG_UNDEFINED_P(sg)) {
	for (pr = sg->rays_sg.vsg;  pr != NULL; pr=pr->succ) {
	    trouve = trouve || vect_equal(ray->vecteur,pr->vecteur);
	}
    }
    return (trouve);
}

bool dte_in_sg_p(dte,sg)
Pray_dte dte;
Ptsg sg;
{
    Pray_dte pr;
    bool trouve = false;

    if (!SG_UNDEFINED_P(sg)) {
	for (pr = sg->dtes_sg.vsg; pr != NULL; pr=pr->succ) {
	    trouve = trouve || vect_equal(dte->vecteur,pr->vecteur);
	}
    }
    return (trouve);
}

Ptsg sg_union(sg1,sg2)
Ptsg sg1,sg2;
{
    Ptsg sg;
    bool newsommet = (sg_nbre_sommets(sg1)== 0) ? true : false;
    bool newray = (sg_nbre_rayons(sg1)== 0) ? true : false;
    bool newdte = (sg_nbre_droites(sg1)== 0) ? true : false;
    Psommet ps, ps_tmp;
    Pray_dte pr,pr_tmp;

    if (sg1 == NULL)	
	return (sg_dup(sg2));
    if (sg2 == NULL)
	return (sg_dup(sg1));

    sg = sg_dup(sg1); 
    sg->soms_sg.nb_s = sg_nbre_sommets(sg1);
    sg->rays_sg.nb_v =sg_nbre_rayons(sg1) ;
    sg->dtes_sg.nb_v = sg_nbre_droites(sg1);
    /* union des sommets */

    for (ps = sg->soms_sg.ssg, ps_tmp = ps; ps != NULL ;  ps_tmp = ps, ps=ps->succ)
	;
    for (ps = sg2->soms_sg.ssg; ps != NULL; ps=ps->succ) {
	if (newsommet) {
	    sg->soms_sg.ssg = ps_tmp=sommet_dup(ps);
	    newsommet = false;
	    sg->soms_sg.nb_s++;
	} else {
	    if (!sommet_in_sg_p(ps,sg)) {
		ps_tmp->succ=sommet_dup(ps);
		ps_tmp = ps_tmp->succ; sg->soms_sg.nb_s++;
	    }
	}
    }

    /* union des rayons */
    for (pr = sg->rays_sg.vsg, pr_tmp=pr; pr!= NULL; pr_tmp = pr, pr = pr->succ)
	;
    for (pr = sg2->rays_sg.vsg; pr != NULL; pr=pr->succ) {
	if (newray) {
	    sg->rays_sg.vsg = pr_tmp=ray_dte_dup(pr);
	    newray = false;
	    sg->rays_sg.nb_v++;
	} else {
	    if (!ray_in_sg_p(pr,sg)) {
		pr_tmp->succ=ray_dte_dup(pr);
		pr_tmp = pr_tmp->succ;
		sg->rays_sg.nb_v++;
	    }
	}
    }

    /*union des droites */
    for (pr = sg->dtes_sg.vsg, pr_tmp=pr; pr!= NULL; pr_tmp = pr,pr=pr->succ)
	;
    for (pr = sg2->dtes_sg.vsg; pr != NULL; pr=pr->succ) {
	if (newdte) {
	    sg->dtes_sg.vsg = pr_tmp=ray_dte_dup(pr);
	    newdte = false;
	    sg->dtes_sg.nb_v++;
	} else {
	    if (!dte_in_sg_p(pr,sg)) {
		pr_tmp->succ=ray_dte_dup(pr);
		pr_tmp = pr_tmp->succ;
		sg->dtes_sg.nb_v++;
	    }
	}
    }
    base_rm(sg->base);
    sg->base = base_union(sg1->base,sg2->base);
    return (sg); 
}
