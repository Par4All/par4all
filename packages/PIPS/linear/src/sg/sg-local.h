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

/* package sur les systemes generateur sg
 *
 * Francois Irigoin, Mai 1989
 *
 * packages a inclure: boolean.h, arithmetique.h, variable.h, vecteur.h,
 * ray_dte.h et sommet.h
 * 
 * package utilisateur: polyedre.h
 */

#ifndef TSG
#define TSG	101
#define TSGSOMS 103
#define TSGVECTS 104

/* Representation d'un ensemble de sommets */
typedef struct ttsg_soms {
	int nb_s;
	struct	typ_som *ssg;
	} *Ptsg_soms,Stsg_soms;

/* Representation d'un ensemble de droites */
typedef struct ttsg_vects {
	int nb_v;
	Pray_dte vsg;
	} *Ptsg_vects,Stsg_vects;

/* Representation d'un systeme generateur par trois ensembles de sommets
 * de rayons et de droites
 *
 * L'ensemble vide est represente par un systeme generateur n'ayant
 * aucun element dans ces trois ensembles (soms_sg.nb_s==0 &&
 * rays_sg.nb_v == 0 && dtes_sg.nb_v == 0)
 *
 * L'espace tout entier Rn est represente par n droites et un sommet.
 * Par convention ce sommet est l'origine.
 *
 * La dimension de l'espace contenant le polyedre genere n'est pas
 * accessible directement. Il faut parcourir tous les elements
 * generateurs et chercher leurs coordonnees non nulles.
 */
typedef struct type_sg {
	Stsg_soms soms_sg;
	Stsg_vects rays_sg;
	Stsg_vects dtes_sg;
	Pbase base;
	} *Ptsg,Stsg;

#define SG_UNDEFINED ((Ptsg) NULL)
#define SG_UNDEFINED_P(sg) ((sg)==(SG_UNDEFINED))

/* vieilles definitions des fonctions d'impression
 * void sg_fprint();
 * #define print_sg(sg) sg_fprint(stdout,sg)
 */

/* macros d'acces aux champs */

/* acces au premier sommet de la liste des sommets d'un systeme generateur
   defini par un pointeur: sg_sommets(Ptsg) */
#define sg_sommets(sg) ((sg)->soms_sg.ssg)

/* acces au premier rayon de la liste des rayons d'un systeme generateur
   defini par un pointeur: sg_rayons(Ptsg) */
#define sg_rayons(sg) ((sg)->rays_sg.vsg)

/* acces a la premiere droite de la liste des droites d'un systeme generateur
   defini par un pointeur: sg_droites(Ptsg) */
#define sg_droites(sg) ((sg)->dtes_sg.vsg)

/* nombre de sommets: int sg_nbre_sommets(Ptsg) */
#define sg_nbre_sommets(sg) ((sg)->soms_sg.nb_s)

/* nombre de rayons: int sg_nbre_rayons(Ptsg) */
#define sg_nbre_rayons(sg) ((sg)->rays_sg.nb_v)

/* nombre de droites: int sg_nbre_droites(Ptsg) */
#define sg_nbre_droites(sg) ((sg)->dtes_sg.nb_v)

/* Basis used for the generating system */
#define sg_base(sg) ((sg)->base)

/* Test for an empty generating system, which corresponds to an empty set */
#define sg_empty(sg) \
  ((sg)->soms_sg.nb_s==0 &&(sg)-> rays_sg.nb_v == 0 && (sg)->dtes_sg.nb_v == 0)

/* Obsolete macros of Malik Imadache
 * #define soms_of_sg(sg) (((sg).soms_sg).ssg)
 * #define rays_of_sg(sg) (((sg).rays_sg).vsg)
 * #define dtes_of_sg(sg) (((sg).dtes_sg).vsg)
 */

#endif /* TSG */



