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

/* package ray_dte: structure de donnees representant les rayons et les
 * droites d'un systeme generateur; elle contient le vecteur correspondant,
 * un eventuel tableau de saturation, et le chainage vers les autres rayons
 * ou droites.
 *
 * Francois Irigoin, Mai 1989
 *
 * Voir poly.h
 *
 * A terme, poly.h devrait exploser et la definition de ray_dte etre remise
 * dans ce fichier; a moins qu'on ne mette plutot ray_dte.h dans sg.h
 * pour eviter une explosion des .h
 */

#ifndef RAY_DTE
/* numero du type de donnees */
#define RAY_DTE 105

typedef struct rdte   {
	int *eq_sat;
	struct	Svecteur *vecteur;
	struct rdte   *succ;
	} *Pray_dte,Sray_dte;

#define print_rd(s) ray_dte_fprint(stdout,s)

#define print_lray_dte(lv) fprint_lray_dte(stdout,lv)

#ifndef VERSION_FINALE
#define RAY_DTE_RM(rd,f) dbg_ray_dte_rm(rd,f)
#else
#define RAY_DTE_RM(rd,f) ray_dte_rm(rd)
#endif

#endif /* RAY_DTE */
