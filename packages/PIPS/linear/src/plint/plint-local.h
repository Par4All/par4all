/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

/* package plint: programmation lineaire en nombres entiers
 *
 * Corinne Ancourt
 *
 * utilise les packages:
 *  arithmetique.h
 *  vecteur.h
 *  contrainte.h
 *  sc.h
 *  ray_dte.h
 *  sommet.h
 *  sg.h
 *  polyedre.h (indirectement, pour les fonctions de conversion de sc en
 *              liste de sommets et reciproquement)
 *  matrice.h 
 */

/*
 * Representation d'une solution d'un systeme lineaire
 *
 * Pourrais-tu etre plus precise, Corinne? FI
 */

/* constante associee a une solution (Comment est-elle associee? FI)    */
#define SOLUTION 0

typedef struct Ssolution{
    /* variable du systeme */
    Variable var;
    /* valeur de la variable */
    Value val;
    /* denominateur de la valeur de la variable */
    Value denominateur;
    /* pointeur vers la variable suivante */
    struct Ssolution *succ;
} *Psolution,Ssolution;

