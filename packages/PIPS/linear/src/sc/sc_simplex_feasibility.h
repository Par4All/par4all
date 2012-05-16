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
 * This header provides functions to test whether a constraint system is
 * feasible, using the simplex method.
 * It can be used with any of the headers @c arith_fixprec.c or @c
 * arith_mulprec.c, this in fixed- or multiple-precision.
 */

#ifndef LINEAR_SC_SIMPLEX_FEASIBILITY_H
#define LINEAR_SC_SIMPLEX_FEASIBILITY_H

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <strings.h>
#include <limits.h>

#include "boolean.h"
#include "assert.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

// debugging
#ifndef LINEAR_DEBUG_SIMPLEX
#define DEBUG(code) {}
#else
#define DEBUG(code) {code}
#endif

// memory management
static void* safe_malloc(size_t size)
{
	void* m = malloc(size);
	if (m == NULL) {
		fprintf(stderr, "[" __FILE__ "] out of memory\n");
		abort();
	}
	return m;
}

/**
 * @name Variables
 * Each variable is represented by a non-negative, integer value.
 */
/**@{*/

/**
 * Type of variables.
 */
typedef int var_t;

/**
 * A special value used to represent the absence of variable.
 */
#define VAR_NULL (-1)

/**
 * Maximum number of variables.
 */
#define VAR_MAXNB (1971)

/**
 * Output @a var on stdio stream @a stream.
 */
#define var_fprint(stream, var) (fprintf(stream, "x%d", var))

/**
 * Output @a var on @c stdout.
 */
#define var_print(var) (var_fprint(stdout, var))

/**@}*/

/**
 * @name Vectors
 * A vector is a structure to map each variable to a rational value.
 */
/**@{*/

/**
 * Type of vectors.
 * Typically, most of values are equal to zero, so a sparse, linked-list
 * structure is used.
 * Internally, variables are ordered by decreasing order.
 */
typedef struct vec_s {
	var_t var; /**< mapped variable */
	qval_t coeff; /**< value associated to the variable */
	struct vec_s* succ; /**< rest of the vector */
} vec_s, *vec_p;

/**
 * The empty vector.
 */
#define VEC_NULL NULL

/**
 * Set @a vec to be the empty vector.
 */
#define vec_init(vec) (vec = VEC_NULL)

/**
 * Free the space occupied by @a pvec.
 */
static void vec_clear(vec_p* pvec)
{
	while (*pvec != VEC_NULL) {
		vec_p succ = (*pvec)->succ;
		qval_clear((*pvec)->coeff);
		free(*pvec);
		*pvec = succ;
	}
}

/**
 * Copy @a vec into @a pvec.
 */
static void vec_set(vec_p* pvec, vec_p vec)
{
	vec_clear(pvec);
	bool first = true;
	vec_p* last = NULL;
	for (; vec != VEC_NULL; vec = vec->succ) {
		*last = safe_malloc(sizeof(vec_s));
		(*last)->var = vec->var;
		qval_init((*last)->coeff);
		qval_set((*last)->coeff, vec->coeff);
		if (first) {
			*pvec = *last;
			first = false;
		}
		last = &(*last)->succ;
	}
	if (!first) {
		*last = VEC_NULL;
	}
}

/**
 * Add the element (@a var, @coeff) to @a pvec, in first position.
 * @a var must be greater to any variable in @a pvec s.t. @a pvec remains
 * consistent.
 */
static void vec_append_atfirst(vec_p* pvec, var_t var, qval_t coeff)
{
	assert(*pvec == VEC_NULL || (*pvec)->var < var);
	vec_p hd = safe_malloc(sizeof(vec_s));
	hd->var = var;
	qval_init(hd->coeff); qval_set(hd->coeff, coeff);
	hd->succ = *pvec;
	*pvec = hd;
}

/**
 * Add the element (@a var, @a coeff) to @a pvec.
 */
static void vec_append(vec_p* pvec, var_t var, qval_t coeff)
{
	if (!qval_equal_i(coeff, 0, 1)) {
		while (*pvec != VEC_NULL && (*pvec)->var > var) {
			pvec = &(*pvec)->succ;
		}
	}
	vec_append_atfirst(pvec, var, coeff);
}

/**
 * Get the value associated to @a var in @a vec, and store it into @a coeff.
 */
static void vec_get_coeff(qval_t coeff, vec_p vec, var_t var)
{
	while (vec != VEC_NULL && vec->var > var) {
		vec = vec->succ;
	}
	if (vec != VEC_NULL && vec->var == var) {
		qval_set(coeff, vec->coeff);
	}
	else {
		qval_set_i(coeff, 0, 1);
	}
}

/**
 * Set @a pvec to @a pvec + @a vec.
 */
static void vec_iadd(vec_p* pvec, vec_p vec)
{
	for (; vec != VEC_NULL; vec = vec->succ) {
		var_t var = vec->var;
		while (*pvec != VEC_NULL && (*pvec)->var > var) {
			pvec = &(*pvec)->succ;
		}
		if (*pvec == VEC_NULL || (*pvec)->var < var) {
			// a new variable is added
			vec_p hd = safe_malloc(sizeof(vec_s));
			hd->var = var;
			qval_init(hd->coeff); qval_set(hd->coeff, vec->coeff);
			hd->succ = *pvec;
			*pvec = hd;
			pvec = &(*pvec)->succ;
		}
		else {
			// the variable is present in the original vector
			assert((*pvec)->var == var);
			qval_add((*pvec)->coeff, (*pvec)->coeff, vec->coeff);
			if (qval_equal_i((*pvec)->coeff, 0, 1)) {
				vec_p tl = (*pvec)->succ;
				qval_clear((*pvec)->coeff);
				free(*pvec);
				*pvec = tl;
			}
			else {
				pvec = &(*pvec)->succ;
			}
		}
	}
}

/**
 * Set @a pvec to @a coeff times @a pvec.
 */
static void vec_imul(vec_p* pvec, qval_t coeff)
{
	if (qval_equal_i(coeff, 0, 1)) {
		vec_clear(pvec);
	}
	else {
		vec_p vec;
		for (vec = *pvec; vec != VEC_NULL; vec = vec->succ) {
			qval_mul(vec->coeff, vec->coeff, coeff);
		}
	}
}

/**
 * Set @a pvec to @a pvec + @a coeff times @a vec.
 */
static void vec_iaddmul(vec_p* pvec, qval_t coeff, vec_p vec)
{
	if (qval_equal_i(coeff, 0, 1)) {
		return;
	}
	else {
		qval_t tmp;
		qval_init(tmp);
		for (; vec != VEC_NULL; vec = vec->succ) {
			var_t var = vec->var;
			while (*pvec != VEC_NULL && (*pvec)->var > var) {
				pvec = &(*pvec)->succ;
			}
			if (*pvec == VEC_NULL || (*pvec)->var < var) {
				// a new variable is added
				vec_p hd = safe_malloc(sizeof(vec_s));
				hd->var = var;
				qval_init(hd->coeff); qval_mul(hd->coeff, coeff, vec->coeff);
				hd->succ = *pvec;
				*pvec = hd;
				pvec = &(*pvec)->succ;
			}
			else {
				// the variable is present in the original vector
				assert((*pvec)->var == var);
				qval_mul(tmp, coeff, vec->coeff);
				qval_add((*pvec)->coeff, (*pvec)->coeff, tmp);
				if (qval_equal_i((*pvec)->coeff, 0, 1)) {
					vec_p tl = (*pvec)->succ;
					qval_clear((*pvec)->coeff);
					free(*pvec);
					*pvec = tl;
				}
				else {
					pvec = &(*pvec)->succ;
				}
			}
		}
		qval_clear(tmp);
	}
}

/**
 * Set @a vec to -@a vec.
 */
static void vec_ineg(vec_p vec)
{
	for (; vec != VEC_NULL; vec = vec->succ) {
		qval_neg(vec->coeff, vec->coeff);
	}
}

/**
 * Output @a vec on stdio stream @a stream.
 */
static int NOWUNUSED vec_fprint(FILE* stream, vec_p vec)
{
	if (vec == VEC_NULL) {
		return fprintf(stream, "0");
	}
	else {
		int c = qval_fprint(stream, vec->coeff);
		c += fprintf(stream, " ");
		c += var_fprint(stream, vec->var);
		for (vec = vec->succ; vec != VEC_NULL; vec = vec->succ) {
			if (qval_cmp_i(vec->coeff, 0, 1) > 0) {
				c += fprintf(stream, " + ");
				c += qval_fprint(stream, vec->coeff);
			}
			else {
				c += fprintf(stream, " - ");
				qval_neg(vec->coeff, vec->coeff);
				c += qval_fprint(stream, vec->coeff);
				qval_neg(vec->coeff, vec->coeff);
			}
			c += fprintf(stream, " ");
			c += var_fprint(stream, vec->var);
		}
		return c;
	}
}

/**
 * Output @a vec on @c stdout.
 */
#define vec_print(vec) (vec_fprint(stdout, vec))

/**@}*/

/**
 * @name Linear Constraints
 * A linear constraint is either a linear equality
 * (@a a1 @a x1 + ... + @a an @a xn = @a b) or a linear inequality
 * (@a a1 @a x1 + ... + @a an @a xn <= @a b).
 */
/**@{*/

/**
 * Type of linear constraint relations.
 */
typedef enum
{
	CONSTR_EQ, /**< equality */
	CONSTR_LE /**< inequality */
} constrrel_t;

/**
 * Type of linear constraint.
 */
typedef struct constr_s {
	vec_p vec; /**< coefficients of variables */
	constrrel_t rel; /**< equality or inequality relation */
	qval_t cst; /**< constant term */
} constr_s, *constr_p;

/**
 * Type of linear constraint.
 */
typedef constr_s constr_t[1];

/**
 * Initialize @a constr and set it to 0 = 0.
 */
static void constr_init(constr_p constr)
{
	vec_init(constr->vec);
	constr->rel = CONSTR_EQ;
	qval_init(constr->cst);
}

/**
 * Free the space occupied by @a constr.
 */
static void constr_clear(constr_p constr)
{
	vec_clear(&constr->vec);
	qval_clear(constr->cst);
}

/**
 * Copy @a constr2 into @a constr1.
 */
static void NOWUNUSED constr_set(constr_p constr1, constr_p constr2)
{
	vec_set(&constr1->vec, constr2->vec);
	constr1->rel = constr2->rel;
	qval_set(constr1->cst, constr2->cst);
}

/**
 * Get the coefficient of @a var in @a constr, and store it into @a coeff.
 */
#define constr_get_coeff(coeff, constr, var) \
	(vec_get_coeff(coeff, (constr)->vec, var))

/**
 * Set @a constr1 to @a constr1 + @a constr2.
 * @a constr2 must be an equality.
 */
static void constr_iadd(constr_p constr1, constr_p constr2)
{
	// general case is not needed
	assert(constr2->rel == CONSTR_EQ);
	vec_iadd(&constr1->vec, constr2->vec);
	qval_add(constr1->cst, constr1->cst, constr2->cst);
}

/**
 * Set @a constr to @a coeff times @a constr.
 * @a constr must be an equality or @a coeff a non-negative value.
 */
static void constr_imul(constr_p constr, qval_t coeff)
{
	vec_imul(&constr->vec, coeff);
	qval_mul(constr->cst, constr->cst, coeff);
}

/**
 * Set @a constr1 to @a constr1 + @a coeff times @a constr2.
 * @a constr2 must be an equality or @a coeff a non-negative value.
 */
static void constr_iaddmul(constr_p constr1, qval_t coeff, constr_p constr2)
{
	if (!qval_equal_i(coeff, 0, 1)) {
		vec_iaddmul(&constr1->vec, coeff, constr2->vec);
		qval_t tmp;
		qval_init(tmp);
		qval_mul(tmp, coeff, constr2->cst);
		qval_add(constr1->cst, constr1->cst, tmp);
		qval_clear(tmp);
	}
}

/**
 * Turn @a constr into an equivalent constraint whose constant term is
 * non-negative.
 * @a constr must be an equality.
 */
static void constr_makepos(constr_p constr)
{
	assert(constr->rel == CONSTR_EQ);
	if (qval_cmp_i(constr->cst, 0, 1) < 0) {
		vec_ineg(constr->vec);
		qval_neg(constr->cst, constr->cst);
	}
}

/**
 * Make @a pivot_var coefficient null in @a constr, using @a pivot_constr.
 * @a pivot_constr must be an equality.
 */
static void constr_apply_pivot(constr_p constr, var_t pivot_var,
		constr_p pivot_constr)
{
	qval_t coeff;
	qval_init(coeff);
	constr_get_coeff(coeff, constr, pivot_var);
	qval_neg(coeff, coeff);
	constr_iaddmul(constr, coeff, pivot_constr);
	qval_clear(coeff);
}

/**
 * Output @a constr on stdio stream @a stream.
 */
static int NOWUNUSED constr_fprint(FILE* stream, constr_p constr)
{
	int c = vec_fprint(stream, constr->vec);
	if (constr->rel == CONSTR_EQ) {
		c += fprintf(stream, " == ");
	}
	else {
		c += fprintf(stream, " <= ");
	}
	c += qval_fprint(stream, constr->cst);
	return c;
}

/**
 * Output @a constr on @c stdout.
 */
#define constr_print(constr) (constr_fprint(stdout, constr))

/**@}*/

/**
 * @name Simplex Tableau
 * A simplex tableau consists in a list of constraint, an objective function to
 * minimize and its current value. Those data evolve when the simplex is
 * running.
 */
/**@{*/

/**
 * Type of simplex tableau.
 * The objective function and the associated value are stored into a constraint
 * @c obj, other constraints are in the constraint table @c constr.
 */
typedef struct table_s {
	constr_t obj; /**< objective function and value */
	constr_t* constrs; /**< constraints */
	int nbvars; /**< number of variables */
	int nbconstrs; /**< number of constraints */
} table_s, *table_p;

/**
 * Type of simplex tableau.
 */
typedef table_s table_t[1];

/**
 * Initialize @a tbl, with room for @a nbconstrs constraints.
 */
static void table_init(table_p tbl, int nbconstrs)
{
	constr_init(tbl->obj);
	tbl->constrs = safe_malloc(nbconstrs * sizeof(constr_t));
	int i;
	for (i = 0; i < nbconstrs; i++) {
		constr_init(tbl->constrs[i]);
	}
	tbl->nbvars = 0;
	tbl->nbconstrs = nbconstrs;
}

/**
 * Free the space occupied by @a tbl.
 */
static void table_clear(table_p tbl)
{
	constr_clear(tbl->obj);
	int i;
	for (i = 0; i < tbl->nbconstrs; i++) {
		constr_clear(tbl->constrs[i]);
	}
	free(tbl->constrs);
}

/**
 * Number of variables in @a tbl.
 */
#define table_get_nbvars(tbl) ((tbl)->nbvars)

/**
 * Number of constraints in @a tbl.
 */
#define table_get_nbconstrs(tbl) ((tbl)->nbconstrs)

/**
 * Turn constraints into @a tbl into equivalent ones, whose constant term is
 * non-negative.
 * All constraints must be equalities.
 */
static void table_makepos(table_p tbl)
{
	int i;
	for (i = 0; i < tbl->nbconstrs; i++) {
		constr_makepos(tbl->constrs[i]);
	}
}

/**
 * Ensure that all variables in @a tbl are non-negative.
 * Each occurrence of a unknown-sign variable @a a @a x is turned into
 * @a a @a x+ - @a a @a x-, where @x+ and @x- are new, non-negative variables.
 * Each occurrence of a negative variable @a a @a x is turned into -@a a @a x',
 * where @a x' is a new, non-negative variable.
 * We try to do this only when necessary, to limit the amount of newly created
 * variables.
 */
static void table_addsignvars(table_p tbl)
{
	// first, we try to determine the sign of variables
	static int vars_info[VAR_MAXNB]; // -1 negative or null, +1 positive or null, 0 unknown
	int i;
	for (i = 0; i < VAR_MAXNB; i++) {
		vars_info[i] = 0;
	}
	for (i = 0; i < tbl->nbconstrs; i++) {
		constr_p constr = tbl->constrs[i];
		vec_p vec = constr->vec;
		if (vec != VEC_NULL && vec->succ == VEC_NULL) {
			var_t var = vec->var;
			int lsign = value_sign(qval_cmp_i(vec->coeff, 0, 1));
			assert(lsign != 0);
			int rsign = value_sign(qval_cmp_i(constr->cst, 0, 1));
			int sign;
			if (constr->rel == CONSTR_EQ) {
				sign = rsign == 0 ? 1 : lsign * rsign;
			}
			else {
				sign = rsign == 0 ? -lsign : (rsign == 1 ? 0 : lsign * rsign);
			}
			if (sign != 0 && vars_info[var] == 0) {
				vars_info[var] = sign;
			}
		}
	}
	// negative variables are inverted
	for (i = 0; i < tbl->nbconstrs; i++) {
		constr_p constr = tbl->constrs[i];
		vec_p vec;
		for (vec = constr->vec; vec != VEC_NULL; vec = vec->succ) {
			var_t var = vec->var;
			if (vars_info[var] < 0) {
				qval_neg(vec->coeff, vec->coeff);
			}
		}
	}
	// unknown sign variables are split
	var_t var;
	for (var = tbl->nbvars - 1; var >= 0; var--) {
		if (vars_info[var] == 0) {
			vars_info[var] = tbl->nbvars;
			tbl->nbvars++;
		}
		else {
			vars_info[var] = VAR_NULL;
		}
	}
	// and then, injected into constraints
	qval_t coeff;
	qval_init(coeff);
	for (i = 0; i < tbl->nbconstrs; i++) {
		constr_p constr = tbl->constrs[i];
		vec_p vec;
		for (vec = constr->vec; vec != VEC_NULL; vec = vec->succ) {
			var_t var = vec->var;
			if (vars_info[var] != VAR_NULL) {
				qval_neg(coeff, vec->coeff);
				vec_append_atfirst(&constr->vec, vars_info[var], coeff);
			}
		}
	}
	qval_clear(coeff);
}

/**
 * Ensure that all constraints in @a tbl are equalities.
 * A new offset variable is introduced into inequalities to turn them in
 * equalities. E.g., inequality @a a1 @a x1 + ... + @a an @a xn <= @a b
 * becomes: @a a1 @a x1 + ... + @a an @a xn + y = @a b.
 */
static void table_addofsvars(table_p tbl)
{
	qval_t one;
	qval_init(one); qval_set_i(one, 1, 1);
	int i;
	for (i = 0; i < tbl->nbconstrs; i++) {
		constr_p constr = tbl->constrs[i];
		if (constr->rel == CONSTR_LE) {
			vec_append_atfirst(&constr->vec, tbl->nbvars, one);
			tbl->nbvars++;
			constr->rel = CONSTR_EQ;
		}
	}
	qval_clear(one);
}

static int NOWUNUSED table_fprint(FILE*, table_p);

/**
 * Canonicalize @a tbl: ensure that all variables are non-negative, and all
 * constraints are equalities whose constant term is non-negative.
 */
static void table_canonicalize(table_p tbl)
{
	DEBUG(
		fprintf(stderr, "Initial system:\n");
		table_fprint(stderr, tbl);
	);
	table_addsignvars(tbl);
	DEBUG(
		fprintf(stderr, "\nWith signed variables:\n");
		table_fprint(stderr, tbl);
	);
	table_addofsvars(tbl);
	DEBUG(
		fprintf(stderr, "\nWith offset variables:\n");
		table_fprint(stderr, tbl);
	);
	table_makepos(tbl);
	DEBUG(
		fprintf(stderr, "\nWith positive constants:\n");
		table_fprint(stderr, tbl);
	);
}

/**
 * Initialize the objective function and value from constraints in @a tbl.
 */
static void table_set_obj(table_p tbl)
{
	constr_p obj = tbl->obj;
	int i;
	for (i = 0; i < tbl->nbconstrs; i++) {
		constr_p constr = tbl->constrs[i];
		constr_iadd(obj, constr);
	}
}

/**
 * Add objective variables to each constraint of @tbl whose constant term is
 * not zero.
 */
static void table_addobjvars(table_p tbl)
{
	qval_t one;
	qval_init(one); qval_set_i(one, 1, 1);
	int i;
	for (i = 0; i < tbl->nbconstrs; i++) {
		constr_p constr = tbl->constrs[i];
		assert(qval_cmp_i(constr->cst, 0, 1) >= 0);
		if (!qval_equal_i(constr->cst, 0, 1)) {
			var_t var = tbl->nbvars;
			vec_append_atfirst(&constr->vec, var, one);
			tbl->nbvars++;
		}
	}
	qval_clear(one);
}

/**
 * Canonicalize @a tbl, set the objective and add objective variables.
 */
static void table_prepare(table_p tbl)
{
	table_canonicalize(tbl);
	table_set_obj(tbl);
	table_addobjvars(tbl);
	DEBUG(
		fprintf(stderr, "\nWith objective:\n");
		table_fprint(stderr, tbl);
		fprintf(stderr, "\n");
	);
}

/**
 * Get the next pivot variable in @tbl.
 * Bland's rule is used to ensure termination: pivot variable is the
 * lowest-numbered variable whose objective coefficient is negative.
 */
static var_t table_get_pivotvar(table_p tbl)
{
	var_t var = VAR_NULL;
	vec_p vec;
	for (vec = tbl->obj->vec; vec != VEC_NULL; vec = vec->succ) {
		if (qval_cmp_i(vec->coeff, 0, 1) > 0) {
			var = vec->var;
		}
	}
	return var;
}

/**
 * Retrieve the variable associated with the row (i.e. constraint) @a row in @a
 * tbl.
 */
static var_t table_get_assocvar(table_p tbl, int row)
{
	if (row != -1) {
		qval_t tmp;
		qval_init(tmp);
		constr_p constr = tbl->constrs[row];
		vec_p vec;
		for (vec = constr->vec; vec != VEC_NULL; vec = vec->succ) {
			if (qval_equal_i(vec->coeff, 1, 1)) {
				var_t var = vec->var;
				bool unitcol = true;
				int i;
				for (i = 0; i < tbl->nbconstrs; i++) {
					if (i != row) {
						constr_get_coeff(tmp, tbl->constrs[i], var);
						if (!qval_equal_i(tmp, 0, 1)) {
							unitcol = false;
							break;
						}
					}
				}
				if (unitcol) {
					return var;
				}
			}
		}
		qval_clear(tmp);
	}
	return VAR_NULL;
}

/**
 * Get the next pivot row (i.e. constraint) in @tbl, corresponding to pivot
 * variable @var.
 * Bland's rule is used to ensure termination: among the rows with the best
 * ratio, choose the one whose associated variable index is the smaller.
 */
static int table_get_pivotrow(table_p tbl, var_t var)
{
	int pivot_row = -1;
	if (var != VAR_NULL) {
		var_t pivot_assoc = VAR_NULL;
		qval_t pivot_ratio, ratio;
		qval_init(pivot_ratio); qval_init(ratio);
		int i;
		for (i = 0; i < tbl->nbconstrs; i++) {
			constr_p constr = tbl->constrs[i];
			constr_get_coeff(ratio, constr, var);
			if (qval_cmp_i(ratio, 0, 1) > 0) {
				assert(qval_cmp_i(constr->cst, 0, 1) >= 0);
				qval_div(ratio, constr->cst, ratio);
				if (pivot_row == -1 || qval_cmp(ratio, pivot_ratio) < 0) {
					pivot_row = i;
					pivot_assoc = table_get_assocvar(tbl, pivot_row);
					qval_set(pivot_ratio, ratio);
				}
				else if (qval_cmp(ratio, pivot_ratio) == 0) {
					var_t assoc = table_get_assocvar(tbl, i);
					if (assoc < pivot_assoc) {
						pivot_row = i;
						pivot_assoc = assoc;
					}
				}
			}
		}
		qval_clear(pivot_ratio); qval_clear(ratio);
	}
	return pivot_row;
}

/**
 * Get the next pivot variable and row in @tbl, and store them in @a pvar and
 * @a prow respectively.
 */
static bool table_get_pivot(table_p tbl, var_t* pvar, int* prow)
{
	*pvar = table_get_pivotvar(tbl);
	*prow = table_get_pivotrow(tbl, *pvar);
	return *prow != -1;
}

/**
 * Apply pivot (@a var, @a row) in @a tbl.
 */
static void table_apply_pivot(table_p tbl, var_t var, int row)
{
	constr_p constr = tbl->constrs[row];
	// pivot constraint is normalized s.t. its coefficient on var is 1
	qval_t coeff;
	qval_init(coeff);
	constr_get_coeff(coeff, constr, var);
	qval_inv(coeff, coeff);
	constr_imul(constr, coeff);
	qval_clear(coeff);
	// then is substracted from other constraints s.t. the coefficient on var is 0
	constr_apply_pivot(tbl->obj, var, constr);
	int i;
	for (i = 0; i < tbl->nbconstrs; i++) {
		if (i != row) {
			constr_apply_pivot(tbl->constrs[i], var, constr);
		}
	}
}

/**
 * Run simplex algorithm on @tbl.
 */
static void table_run_simplex(table_p tbl)
{
	int i = 0;
	while (true) {
		var_t var;
		int row;
		if (table_get_pivot(tbl, &var, &row)) {
			table_apply_pivot(tbl, var, row);
			i++;
			DEBUG(
				fprintf(stderr, "After iteration %d:\n", i);
				fprintf(stderr, "Pivot variable: ");
				var_fprint(stderr, var);
				fprintf(stderr, "\nPivot row: %d\nTable:\n", row);
				table_fprint(stderr, tbl);
				fprintf(stderr, "\n");
				if (i > 200) {
					fprintf(stderr, "Seems to long, exiting...\n");
					exit(42);
				}
			);
		}
		else {
			break;
		}
	}
}

/**
 * Determine whether the constraint system described in @a tbl is feasible,
 * using simplex method.
 */
static bool table_get_feasibility(table_p tbl)
{
	table_prepare(tbl);
	table_run_simplex(tbl);
	return qval_equal_i(tbl->obj->cst, 0, 1);
}

/**
 * Output @a tbl on stdio stream @a stream.
 */
static int NOWUNUSED table_fprint(FILE* stream, table_p tbl)
{
	int c = constr_fprint(stream, tbl->obj);
	c += fprintf(stream, "\n");
	int i;
	for (i = 0; i < 40; i++) {
		c += fprintf(stream, "-");
	}
	c += fprintf(stream, "\n");
	if (tbl->nbconstrs == 0) {
		c += fprintf(stream, "true\n");
	}
	else {
		for (i = 0; i < tbl->nbconstrs; i++) {
			c += fprintf(stream, "(%2d: ", i);
			c += var_fprint(stream, table_get_assocvar(tbl, i));
			c += fprintf(stream, ") ");
			c += constr_fprint(stream, tbl->constrs[i]);
			c += fprintf(stream, "\n");
		}
	}
	for (i = 0; i < 40; i++) {
		c += fprintf(stream, "-");
	}
	c += fprintf(stream, "\n");
	return c;
}

/**
 * Output @a tbl on @c stdout.
 */
#define table_print(tbl) (table_fprint(stdout, tbl))

/**@}*/

/**
 * @name Datatype Conversion
 * Here are utility functions to convert Linear types (@c Variable, @c Vecteur,
 * @c Contrainte, @c Systeme) to datatypes used in this files.
 * In most of conversion functions, a variable table @a vartbl is passed, to
 * help translating Linear named variables into indices.
 */
/**@{*/

/**
 * Type of variable tables.
 */
typedef struct {
	int nbvars; /**< number of variables */
	Variable names[VAR_MAXNB]; /** names of variables */
} vartbl_s, *vartbl_p;

/**
 * Type of variable tables.
 */
typedef vartbl_s vartbl_t[1];

/**
 * Initialize @vartbl to an empty table.
 */
static void vartbl_init(vartbl_p vartbl)
{
	vartbl->nbvars = 0;
}

/**
 * Free the space occupied by @a vartbl.
 */
#define vartbl_clear(vartbl)

/**
 * Get the variable index associated to @a name, creating a new one if
 * necessary.
 */
static var_t vartbl_find(vartbl_p vartbl, Variable name)
{
	int i;
	for (i = 0; i < vartbl->nbvars; i++) {
		if (vartbl->names[i] == name) {
			return i;
		}
	}
	vartbl->names[vartbl->nbvars] = name;
	return vartbl->nbvars++;
}

/**
 * Copy @a vec into @a pvec.
 */
static void vec_set_vecteur(vartbl_t vartbl, vec_p* pvec, Pvecteur vec)
{
	vec_clear(pvec);
	qval_t coeff;
	qval_init(coeff);
	for (; vec != NULL; vec = vec->succ) {
		if (vec->var != TCST) {
			var_t var = vartbl_find(vartbl, vec->var);
			qval_set_i(coeff, vec->val, 1);
			vec_append(pvec, var, coeff);
		}
	}
	qval_clear(coeff);
}

/**
 * Copy @a constr2 into @a constr1.
 */
static void constr_set_contrainte(vartbl_t vartbl,
		constr_p constr1, Pcontrainte constr2, bool is_ineq)
{
	vec_set_vecteur(vartbl, &constr1->vec, constr2->vecteur);
	constr1->rel = is_ineq ? CONSTR_LE : CONSTR_EQ;
	Pvecteur vec;
	for (vec = constr2->vecteur; vec != NULL; vec = vec->succ) {
		if (vec->var == TCST) {
			qval_set_i(constr1->cst, -vec->val, 1);
			break;
		}
	}
}

/**
 * Initialize @a tbl from @a sys.
 */
static void table_init_set_systeme(table_p tbl, Psysteme sys)
{
	vartbl_t vartbl;
	vartbl_init(vartbl);
	table_init(tbl, sys->nb_eq + sys->nb_ineq);
	int i = 0;
	Pcontrainte constr;
	for (constr = sys->egalites; constr != NULL; constr = constr->succ) {
		constr_set_contrainte(vartbl, tbl->constrs[i], constr, false);
		i++;
	}
	for (constr = sys->inegalites; constr != NULL; constr = constr->succ) {
		constr_set_contrainte(vartbl, tbl->constrs[i], constr, true);
		i++;
	}
	tbl->nbvars = vartbl->nbvars;
	vartbl_clear(vartbl);
}

/**@}*/

/**
 * Main Function
 */
/**@{*/

/**
 * Determine whether a system @a sys of equations and inequations is feasible.
 * Parameter @a ofl_ctrl indicates whether an overflow control is performed
 * (possible values: @c NO_OFL_CTRL, @c FWD_OFL_CTRL).
 */
static inline bool sc_get_feasibility(Psysteme sys, int ofl_ctrl)
{
	if (ofl_ctrl != FWD_OFL_CTRL) {
		fprintf(stderr, "[sc_simplexe_feasibility] "
			"should not (yet) be called with control %d...\n", ofl_ctrl);
	}
	volatile table_p tbl = safe_malloc(sizeof(table_t));
	table_init_set_systeme(tbl, sys);
	CATCH(simplex_arithmetic_error | timeout_error | overflow_error) {
		table_clear(tbl);
		free(tbl);
		if (ofl_ctrl == FWD_OFL_CTRL) {
			RETHROW();
		}
		return true;
	}
	bool feasible = table_get_feasibility(tbl);
	table_clear(tbl);
	free(tbl);
	UNCATCH(simplex_arithmetic_error | timeout_error | overflow_error);
	return feasible;
}

/**@}*/

#endif

