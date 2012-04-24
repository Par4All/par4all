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

#ifdef LINEAR_ARITH_MULPREC_H
#error "arith_fixprec.h and arith_mulprec.h are both loaded"
#endif

#ifndef LINEAR_ARITH_FIXPREC_H
#define LINEAR_ARITH_FIXPREC_H

#include <stdio.h>
#include <stdlib.h>

#include "assert.h"
#include "arithmetique.h"

#define NOWUNUSED __attribute__((unused))

// Integers

typedef long int zval_t;

#define zval_init(z) ((z) = 0)

#define zval_clear(z)

#define zval_set(z1, z2) ((z1) = (z2))

#define zval_set_i(z, n) ((z) = (n))

#define zval_init_set(z1, z2) ((z1) = (z2))

#define zval_init_set_i(z, n) ((z) = (n))

#define zval_get_i(z) (z)

#define zval_add(z1, z2, z3) ((z1) = (z2) + (z3))

#define zval_sub(z1, z2, z3) ((z1) = (z2) - (z3))

#define zval_mul(z1, z2, z3) ((z1) = value_protected_mult(z2, z3))

#define zval_div(z1, z2, z3) ((z1) = (z2) / (z3))

#define zval_addmul(z1, z2, z3) ((z1) += value_protected_mult(z2, z3))

#define zval_submul(z1, z2, z3) ((z1) -= value_protected_mult(z2, z3))

#define zval_neg(z1, z2) ((z1) = -(z2))

#define zval_abs(z1, z2) ((z1) = ABS(z2))

#define zval_mod(z1, z2, z3) ((z1) = (z2) % (z3))

#define zval_gcd(z1, z2, z3) ((z1) = pgcd(z2, z3))

#define zval_lcm(z1, z2, z3) ((z1) = ppcm(z2, z3))

#define zval_cmp(z1, z2) ((z1) - (z2))

#define zval_cmp_i(z, n) ((z) - (n))

#define zval_sgn(z) (value_sign(z))

#define zval_equal(z1, z2) ((z1) == (z2))

#define zval_equal_i(z, n) ((z) == (n))

#define zval_fprint(stream, z) (fprintf(stream, "%li", z))

#define zval_print(z) (printf("%li", z))

// Rationals

typedef struct {
	zval_t num;
	zval_t den;
} qval_s, *qval_p;

typedef qval_s qval_t[1];

static void qval_canonicalize_unsafe(qval_p q) {
	if (zval_cmp_i(q->num, 0) == 0) {
		zval_set_i(q->den, 1);
		return;
	}
	else {
		zval_t gcd; zval_init(gcd);
		zval_abs(gcd, q->num);
		zval_gcd(gcd, gcd, q->den);
		zval_div(q->num, q->num, gcd);
		zval_div(q->den, q->den, gcd);
		zval_clear(gcd);
	}
}

static void NOWUNUSED qval_canonicalize(qval_p q) {
	if (zval_cmp_i(q->den, 0) < 0) {
		zval_neg(q->num, q->num);
		zval_neg(q->den, q->den);
	}
	qval_canonicalize_unsafe(q);
}

static void NOWUNUSED qval_init(qval_p q) {
	q->num = 0;
	q->den = 1;
}

#define qval_clear(q)

static void NOWUNUSED qval_set(qval_p q1, qval_p q2) {
	zval_set(q1->num, q2->num);
	zval_set(q1->den, q2->den);
}

static void NOWUNUSED qval_set_i(qval_p q1, Value q2num, Value q2den) {
	assert(zval_cmp_i(q2den, 0) != 0);
	zval_set_i(q1->num, q2num);
	zval_set_i(q1->den, q2den);
	qval_canonicalize(q1);
}

static void NOWUNUSED qval_add(qval_p q1, qval_p q2, qval_p q3) {
	zval_t q3num; zval_init(q3num); zval_set(q3num, q3->num);
	zval_t lcm; zval_init(lcm); zval_lcm(lcm, q2->den, q3->den);
	zval_t tmp; zval_init(tmp);
	zval_div(tmp, lcm, q2->den);
	zval_mul(q1->num, q2->num, tmp);
	zval_div(tmp, lcm, q3->den);
	zval_addmul(q1->num, q3num, tmp);
	zval_set(q1->den, lcm);
	zval_clear(q3num); zval_clear(lcm); zval_clear(tmp);
	qval_canonicalize_unsafe(q1);
}

static void NOWUNUSED qval_sub(qval_t q1, qval_t q2, qval_t q3) {
	zval_t q3num; zval_init(q3num); zval_set(q3num, q3->num);
	zval_t lcm; zval_init(lcm); zval_lcm(lcm, q2->den, q3->den);
	zval_t tmp; zval_init(tmp);
	zval_div(tmp, lcm, q2->den);
	zval_mul(q1->num, q2->num, tmp);
	zval_div(tmp, lcm, q3->den);
	zval_submul(q1->num, q3num, tmp);
	zval_set(q1->den, lcm);
	zval_clear(q3num); zval_clear(lcm); zval_clear(tmp);
	qval_canonicalize_unsafe(q1);
}

static void NOWUNUSED qval_mul(qval_t q1, qval_t q2, qval_t q3) {
	zval_mul(q1->num, q2->num, q3->num);
	zval_mul(q1->den, q2->den, q3->den);
	qval_canonicalize_unsafe(q1);
}

static void NOWUNUSED qval_div(qval_t q1, qval_t q2, qval_t q3) {
	zval_t q3num; zval_init(q3num); zval_set(q3num, q3->num);
	assert(zval_cmp_i(q3num, 0) != 0);
	zval_mul(q1->num, q2->num, q3->den);
	zval_mul(q1->den, q2->den, q3num);
	zval_clear(q3num);
	qval_canonicalize(q1);
}

static void NOWUNUSED qval_neg(qval_t q1, qval_t q2) {
	zval_neg(q1->num, q2->num);
	zval_set(q1->den, q2->den);
}

static void NOWUNUSED qval_abs(qval_t q1, qval_t q2) {
	zval_abs(q1->num, q2->num);
	zval_set(q1->den, q2->den);
}

static void NOWUNUSED qval_inv(qval_t q1, qval_t q2) {
	zval_t q2num; zval_init(q2num); zval_set(q2num, q2->num);
	assert(zval_cmp_i(q2num, 0) != 0);
	zval_set(q1->num, q2->den);
	zval_set(q1->den, q2num);
	zval_clear(q2num);
	qval_canonicalize(q1);
}

static int NOWUNUSED qval_cmp(qval_t q1, qval_t q2) {
	zval_t lcm; zval_init(lcm); zval_lcm(lcm, q1->den, q2->den);
	zval_t z1; zval_init(z1);
	zval_t z2; zval_init(z2);
	zval_t tmp; zval_init(tmp);
	zval_div(tmp, lcm, q1->den);
	zval_mul(z1, q1->num, tmp);
	zval_div(tmp, lcm, q2->den);
	zval_mul(z2, q2->num, tmp);
	int res = zval_cmp(z1, z2);
	zval_clear(lcm); zval_clear(z1); zval_clear(z2); zval_clear(tmp);
	return res;
}

static int NOWUNUSED qval_cmp_i(qval_t q1, Value q2num, Value q2den) {
	zval_t lcm; zval_init(lcm); zval_lcm(lcm, q1->den, q2den);
	zval_t z1; zval_init(z1);
	zval_t z2; zval_init(z2);
	zval_t tmp; zval_init(tmp);
	zval_div(tmp, lcm, q1->den);
	zval_mul(z1, q1->num, tmp);
	zval_div(tmp, lcm, q2den);
	zval_mul(z2, q2num, tmp);
	int res = zval_cmp(z1, z2);
	zval_clear(lcm); zval_clear(z1); zval_clear(z2); zval_clear(tmp);
	return res;
}

static int NOWUNUSED qval_sgn(qval_t q) {
	return zval_sgn(q->num);
}

static int NOWUNUSED qval_equal(qval_t q1, qval_t q2) {
	return zval_cmp(q1->den, q2->den) == 0 && zval_cmp(q1->num, q2->num) == 0;
}

#define qval_equal_i(q1, q2num, q2den) (qval_cmp_i(q1, q2num, q2den) == 0)

static int NOWUNUSED qval_fprint(FILE* stream, qval_t q) {
	int c;
	c = zval_fprint(stream, q->num);
	if (zval_cmp_i(q->den, 1)) {
		c += fprintf(stream, "/");
		c += zval_fprint(stream, q->den);
	}
	return c;
}

#define qval_print(q) (qval_fprint(stdout, q))

#endif

