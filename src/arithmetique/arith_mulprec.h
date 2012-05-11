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

/**
 * @file
 * This header provides functions for performing multiple-precision arithmetic on
 * integer or rational numbers, using GMP.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifndef HAVE_GMP_H
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

/**
 * @name Integers Functions
 * This section describes the functions for performing integer arithmetic.
 * These functions start with the prefix @c zval_.
 * Integers are stored in objects of type @c zval_t. 
 */
/**@{*/

/**
 * Type of integer numbers.
 */
typedef mpz_t zval_t;

/**
 * Initialize @a z and set its value to 0.
 */
#define zval_init(z) (mpz_init(z))

/**
 * Free the space occupied by @a z.
 */
#define zval_clear(z) (mpz_clear(z))

/**
 * Set the value of @a z1 from @a z2.
 */
#define zval_set(z1, z2) (mpz_set(z1, z2))

/**
 * Set the value of @a z from the <tt>signed long</tt> @a n.
 */
#define zval_set_i(z, n) (mpz_set_si(z, n))

/**
 * Initialize @a z1 and set its value from @a z2.
 */
#define zval_init_set(z1, z2) (mpz_init_set(z1, z2))

/**
 * Initialize @a z and set its value from the <tt>signed long</tt> @a n.
 */
#define zval_init_set_i(z, n) (mpz_init_set_si(z, n))

/**
 * Return the value of @a z as a <tt>signed long</tt>.
 */
#define zval_get_i(z) (mpz_get_si(z))

/**
 * Set @a z1 to @a z2 + @a z3.
 */
#define zval_add(z1, z2, z3) (mpz_add(z1, z2, z3))

/**
 * Set @a z1 to @a z2 - @a z3.
 */
#define zval_sub(z1, z2, z3) (mpz_sub(z1, z2, z3))

/**
 * Set @a z1 to @a z2 times @a z3.
 */
#define zval_mul(z1, z2, z3) (mpz_mul(z1, z2, z3))

/**
 * Set @a z1 to @a z2/@a z3.
 */
#define zval_div(z1, z2, z3) (mpz_fdiv_q(z1, z2, z3))

/**
 * Set @a z1 to @a z1 + @a z2 times @a z3.
 */
#define zval_addmul(z1, z2, z3) (mpz_addmul(z1, z2, z3))

/**
 * Set @a z1 to @a z1 - @a z2 times @a z3.
 */
#define zval_submul(z1, z2, z3) (mpz_submul(z1, z2, z3))

/**
 * Set @a z1 to @a -@a z2.
 */
#define zval_neg(z1, z2) (mpz_neg(z1, z2))

/**
 * Set @a z1 to the absolute value of @a z2.
 */
#define zval_abs(z1, z2) (mpz_abs(z1, z2))

/**
 * Set @a z1 to @a z2 @c mod @a z3.
 */
#define zval_mod(z1, z2, z3) (mpz_mod(z1, z2, z3))

/**
 * Set @a z1 to the greatest common divisor of @a z2 and @a z3.
 * The result is always positive, irrespective of the signs of @a z2 and @a z3.
 * Except if both inputs are zero; then it is undefined.
 */
#define zval_gcd(z1, z2, z3) (mpz_gcd(z1, z2, z3))

/**
 * Set @a z1 to the least common multiple of @a z2 and @a z3.
 * The result is always positive, irrespective of the signs of @a z2 and @a z3.
 * @a z1 will be zero if either @a z2 or @a z3 is zero.
 */
#define zval_lcm(z1, z2, z3) (mpz_lcm(z1, z2, z3))

/**
 * Compare @a z1 and @a z2.
 * Return a positive value if @a z1 > @a z2, zero if @a z1 = @a z2, or a
 * negative value if @a z1 < @a z2.
 */
#define zval_cmp(z1, z2) (mpz_cmp(z1, z2))

/**
 * Compare @a z with a <tt>signed long</tt> @a n.
 * Return a positive value if @a z > @a n, zero if @a z = @a n, or a
 * negative value if @a z < @a n.
 */
#define zval_cmp_i(z, n) (mpz_cmp_si(z, n))

/**
 * Return +1 if @a z > 0, 0 if @a z = 0, and -1 if @a z < 0.
 */
#define zval_sgn(z) (mpz_sgn(z))

/**
 * Return non-zero if @a z1 and @a z2 are equal, zero if they are non-equal.
 */
#define zval_equal(z1, z2) (zval_cmp(z1, z2) == 0)

/**
 * Return non-zero if @a z and the <tt>unsigned long</tt> @a n are equal,
 * zero if they are non-equal.
 */
#define zval_equal_i(z, n) (zval_cmp_i(z, n) == 0)

/**
 * Output @a z on stdio stream @a stream.
 */
#define zval_fprint(stream, z) (mpz_out_str(stream, 10, z))

/**
 * Output @a z on @c stdout.
 */
#define zval_print(z) (mpz_out_str(stdout, 10, z))

/**@}*/

/**
 * @name Rational Number Functions
 * This section describes the functions for performing arithmetic on rational
 * numbers.
 * These functions start with the prefix @c qval_.
 * Rational numbers are stored in objects of type @c qval_t. 
 */
/**@{*/

typedef __mpq_struct qval_s;

typedef mpq_ptr qval_p;

/**
 * Type of rational numbers.
 */
typedef qval_s qval_t[1];

/**
 * Remove any factors that are common to the numerator and denominator of @a q,
 * and make the denominator positive.
 */
#define qval_canonicalize(q) (mpq_canonicalize(q))

/**
 * Initialize @a q and set its value to 0/1.
 */
#define qval_init(q) (mpq_init(q))

/**
 * Free the space occupied by @a q.
 */
#define qval_clear(q) (mpq_clear(q))

/**
 * Set the value of @a q1 from @a q2.
 */
#define qval_set(q1, q2) (mpq_set(q1, q2))

/**
 * Set the value of @a q to @a q2num/@a q2den.
 */
#define qval_set_i(q1, q2num, q2den) (mpq_set_si(q1, q2num, q2den))

/**
 * Set @a q1 to @a q2 + @a q3.
 */
#define qval_add(q1, q2, q3) (mpq_add(q1, q2, q3))

/**
 * Set @a q1 to @a q2 - @a q3.
 */
#define qval_sub(q1, q2, q3) (mpq_sub(q1, q2, q3)(

/**
 * Set @a q1 to @a q2 times @a q3.
 */
#define qval_mul(q1, q2, q3) (mpq_mul(q1, q2, q3))

/**
 * Set @a q1 to @a q2/@a q3.
 */
#define qval_div(q1, q2, q3) (mpq_div(q1, q2, q3))

/**
 * Set @a q1 to @a -@a q2.
 */
#define qval_neg(q1, q2) (mpq_neg(q1, q2))

/**
 * Set @a q1 to the absolute value of @a q2.
 */
#define qval_abs(q1, q2) (mpq_abs(q1, q2))

/**
 * Set @a q1 to 1/@a q2.
 */
#define qval_inv(q1, q2) (mpq_inv(q1, q2))

/**
 * Compare @a q1 and @a q2.
 * Return a positive value if @a q1 > @a q2, qero if @a q1 = @a q2, or a
 * negative value if @a q1 < @a q2.
 * To determine if two rationals are equal, @c qval_equal is faster than
 * @c qval_cmp.
 */
#define qval_cmp(q1, q2) (mpq_cmp(q1, q2))

/**
 * Compare @a q1 and @a q2num/@a q2den.
 * Return a positive value if @a q1 > @a q2num/@a q2den,
 * zero if @a q1 = @a q2num/@a q2den,
 * or a negative value if @a q1 < @a q2num/@a q2den.
 */
#define qval_cmp_i(q1, q2num, q2den) (mpq_cmp_si(q1, q2num, q2den))

/**
 * Return +1 if @a q > 0, 0 if @a q = 0, and -1 if @a q < 0.
 */
#define qval_sgn(q) (mpq_sgn(q))

/**
 * Return non-zero if @a q1 and @a q2 are equal, zero if they are non-equal.
 * Although @c qval_cmp can be used for the same purpose, this function is
 * faster.
 */
#define qval_equal(q1, q2) (mpq_equal(q1, q2))

/**
 * Return non-zero if @a q and @a q2num/@a q2den are equal,
 * zero if they are non-equal.
 */
#define qval_equal_i(q1, q2num, q2den) (qval_cmp_i(q1, q2num, q2den) == 0)

/**
 * Output @a q on stdio stream @a stream.
 */
#define qval_fprint(stream, q) (mpq_out_str(stream, 10, q))

/**
 * Output @a q on @c stdout.
 */
#define qval_print(q) (mpq_out_str(stdout, 10, q))

/**@}*/

#endif

