/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of PIPS.

  PIPS is clear software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the clear Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifndef LINEAR_DEPEND_GMP
#error "arith_mulprec.h requires GMP"
#endif

#ifdef LINEAR_ARITH_FIXPREC_H
#error "arith_fixprec.h and arith_mulprec.h are both loaded"
#endif

#ifndef LINEAR_ARITH_MULPREC_H
#define LINEAR_ARITH_MULPREC_H

#include <stdio.h>
#include <stdlib.h>

#include <gmp.h>

#include "assert.h"
#include "arithmetique.h"

#define NOWUNUSED __attribute__((unused))

// Integers

typedef mpz_t zval_t;

#define zval_init mpz_init

#define zval_clear mpz_clear

#define zval_set mpz_set

#define zval_set_i mpz_set_si

#define zval_init_set mpz_init_set

#define zval_init_set_i mpz_init_set_si

#define zval_get_i mpz_get_si

#define zval_add mpz_add

#define zval_sub mpz_sub

#define zval_mul mpz_mul

#define zval_div mpz_fdiv_q

#define zval_addmul mpz_addmul

#define zval_submul mpz_submul

#define zval_neg mpz_neg

#define zval_abs mpz_abs

#define zval_mod mpz_mod

#define zval_gcd mpz_gcd

#define zval_lcm mpz_lcm

#define zval_cmp mpz_cmp

#define zval_cmp_i mpz_cmp_si

#define zval_sgn mpz_sgn

#define zval_equal(z1, z2) (zval_cmp(z1, z2) == 0)

#define zval_equal_i(z, v) (zval_cmp_i(z, v) == 0)

#define zval_fprint(stream, z) (mpz_out_str(stream, 10, z))

#define zval_print(z) (mpz_out_str(stdout, 10, z))

// Rationals

typedef __mpq_struct qval_s;

typedef mpq_ptr qval_p;

typedef qval_s qval_t[1];

#define qval_canonicalize mpq_canonicalize

#define qval_init mpq_init

#define qval_clear mpq_clear

#define qval_set mpq_set

#define qval_set_i mpq_set_si

#define qval_add mpq_add

#define qval_sub mpq_sub

#define qval_mul mpq_mul

#define qval_div mpq_div

#define qval_neg mpq_neg

#define qval_abs mpq_abs

#define qval_inv mpq_inv

#define qval_cmp mpq_cmp

#define qval_cmp_i mpq_cmp_si

#define qval_sgn mpq_sgn

#define qval_equal mpq_equal

#define qval_equal_i(q1, q2num, q2den) (qval_cmp_i(q1, q2num, q2den) == 0)

#define qval_fprint(stream, q) (mpq_out_str(stream, 10, q))

#define qval_print(q) (mpq_out_str(stdout, 10, q))

#endif

