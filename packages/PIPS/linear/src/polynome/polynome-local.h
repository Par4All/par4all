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

/*
 *
 * SYMBOLIC POLYNOMIALS     P. Berthomier  02-08-90
 *
 *                          L. Zhou        Apr.8,91
 *
 *
 * This package contains some routines manipulating symbolic polynomials
 * with any number of variables. It uses the packages vecteur and arithmetique.
 *
 * There are routines for allocation/duplication/destruction, input/output,
 * "factorization", variable substitution, integer power exponentiation,
 * addition, multiplication, integration of polynomials.
 *
 * The package lacks: division by a one-variable polynomial, comparison
 * of two polynomials (when possible), ...
 *
 *
 * POLYNOMIAL STRUCTURE
 *
 *
 * We use a Pvecteur structure to represent a sort of "normalized monomial":
 * a product of powers of variables without scalar coefficient.
 *
 * The field "Variable" of the Pvecteur is a pointer to the variable
 * We only use pointer equality, so if two different pointers
 * refer to the same variable, the polynomials won't be normalized:
 * things such as 2*N + 3*N may appear.
 * The field "Value" contains the integer exponent of that unknown
 * in that monomial.
 *
 * We add a floating-point coefficient to form a complete monomial
 * structure: a *Pmonome.
 *
 * We define a polynomial as an unsorted list of monomial structure.
 *
 *
 *
 * Example of Pmonome:     2 * X^2.Y.Z^3
 * (the "@" below means: beginning of a pointer)
 *
 *
 *                    -----------
 * Pmonome @------>   |    2    | (coeff:float)
 *                    |---------|
 *                    |    @    | (term:Pvecteur)
 *                    ---- | ----
 *                         |
 *                         V
 *          /         -----------
 *          |         |  Y , 1  |
 *          |         -----------
 * Pvecteur |         |  Z , 3  |
 *          |         -----------
 *          |         |  X , 2  |
 *          \         -----------
 *
 *
 *
 * Example of Ppolynome:    6 * N^3.k + 3 * N.k - 2
 *
 *                  -----------    -----------    -----------    
 * Ppolynome:@----> |    @-------> |    @-------> |    /    | (succ:Ppolynome)
 *                  |---------|    |---------|    |---------|    
 *                  |    @    |    |    @    |    |    @    | (monome:Pmonome)   
 *                  ---- | ----    ---- | ----    ---- | ----    
 *                       |              |              |         
 *                       V              V              V         
 *          /       -----------    -----------    -----------    
 *          |       |    6    |    |    3    |    |   -2    |
 * Pmonomes |       -----------    -----------    -----------    
 *          |       |    @    |    |    @    |    |    @    |    
 *          \       ---- | ----    ---- | ----    ---- | ----    
 *                       |              |              |         
 *                       V              V              V         
 *          /       -----------    -----------    -----------    
 *          |       |  N , 3  |    |  N , 1  |    | TCST, 1 |    
 * Pvecteurs|       -----------    -----------    -----------    
 *          |       |  k , 1  |    |  k , 1  |
 *          \       -----------    -----------    
 *
 * Nul polynomials are represented by a fixed pointer: POLYNOME_NUL.
 * Undefined polynomials are represented by POLYNOME_UNDEFINED.
 *
 * The package is cut into ten files:
 *
 *   pnome-types.h
 *   polynome.h         (This file. Automatically generated from pnome-types.h)
 *
 *   pnome-alloc.c
 *   pnome-bin.c
 *   pnome-error.c
 *   pnome-io.c
 *   pnome-private.c
 *   pnome-reduc.c
 *   pnome-scal.c
 *   pnome-unaires.c
 *
 * 
 */


#ifndef POLYNOME_INCLUDED
#define POLYNOME_INCLUDED

typedef struct Smonome {
    float coeff;
    Pvecteur term;
} Smonome, *Pmonome;

typedef struct Spolynome {
    Pmonome monome;
    struct Spolynome *succ;
} Spolynome, *Ppolynome;


/* Macros definitions */
#define monome_coeff(pm) ((pm)->coeff)
#define monome_term(pm) ((pm)->term)
#define polynome_monome(pp) ((pp)->monome)
#define polynome_succ(pp) ((pp)->succ)
/*
#define is_single_monome(pp) ((POLYNOME_NUL_P(pp)) || (POLYNOME_NUL_P(polynome_succ(pp))))
*/
#define is_single_monome(pp) ((!POLYNOME_NUL_P(pp)) && (POLYNOME_NUL_P(polynome_succ(pp))))

#define monome_constant_new(coeff) make_monome(coeff, TCST, 1)
#define monome_power1_new(coeff, var) make_monome(coeff, var, 1)

/* Null/undefined, monomial/polynomial definitions */
#define MONOME_NUL ((Pmonome) -256)
#define MONOME_NUL_P(pm) ((pm)==MONOME_NUL)
#define MONOME_UNDEFINED ((Pmonome) -252)
#define MONOME_UNDEFINED_P(pm) ((pm)==MONOME_UNDEFINED)
#define MONOME_CONSTANT_P(pm) (term_cst((pm)->term))

#define POLYNOME_NUL ((Ppolynome) NULL)
#define POLYNOME_NUL_P(pp) ((pp)==POLYNOME_NUL)
#define POLYNOME_UNDEFINED ((Ppolynome) -248)
#define POLYNOME_UNDEFINED_P(pp) ((pp)==POLYNOME_UNDEFINED)

#define MONOME_COEFF_MULTIPLY_SYMBOL "*"
#define MONOME_VAR_MULTIPLY_SYMBOL "."
#define POLYNOME_NUL_SYMBOL "0"
#define POLYNOME_UNDEFINED_SYMBOL "<polynome undefined>"
#define MONOME_NUL_SYMBOL "<monome nul>"
#define MONOME_UNDEFINED_SYMBOL "<monome undefined>"

#define MAX_NAME_LENGTH 50

#define PNOME_MACH_EPS 1E-8          /* below this value, a float is null      */
#define PNOME_FLOAT_N_DECIMALES 2    /* nb of figures after point for coeffs   */
#define PNOME_FLOAT_TO_EXP_LEVEL 1E8 /* numbers >1E8 are printed with exponent */
#define PNOME_FLOAT_TO_FRAC_LEVEL 9  /* print 1/2..1/9 rather than 0.50..0.11  */

#endif /* POLYNOME_INCLUDED */

