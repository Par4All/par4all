/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

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

#ifndef LINEAR_CHERNIKOVA_H
#define LINEAR_CHERNIKOVA_H

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
#include <unistd.h>

#include "assert.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"
#include "polyedre.h"
#include "polynome.h"

// Debugging
// ---------

#define FPRINT_NBDIGITS 6

#define _STRINGOF(str) #str
#define STRINGOF(str) _STRINGOF(str)

// Memory management
// -----------------

static void* NOWUNUSED safe_malloc(size_t size) {
	void* ptr = malloc(size);
	if (ptr == NULL) {
		fprintf(stderr, "[" __FILE__ "] out of memory space\n");
		abort();
	}
	return ptr;
}

static void* NOWUNUSED safe_realloc(void* ptr, size_t size) {
	ptr = realloc(ptr, size);
	if (ptr == NULL) {
		fprintf(stderr, "[" __FILE__ "] out of memory space\n");
		abort();
	}
	return ptr;
}

static void NOWUNUSED safe_free(void* ptr) {
	if (ptr != NULL) {
		free(ptr);
	}
}

#define BSWAP(a, b, l) { \
	char* t = (char*) safe_malloc(l * sizeof(char)); \
	memcpy((t), (char*) (a), (int) (l)); \
	memcpy((char*) (a), (char*) (b), (int) (l)); \
	memcpy((char*) (b), (t), (int) (l)); \
	free(t); \
}

#define SWAP(a, b, t) { \
	(t) = (a); \
	(a) = (b); \
	(b) = (t); \
}

// Alarms
// ------

// set timeout with signal and alarm
// based on environment variable LINEAR_CONVEX_HULL_TIMEOUT
#include <signal.h>

// name of environment variable to trigger local timeout
#define TIMEOUT_ENV "LINEAR_CONVEX_HULL_TIMEOUT"

static bool sc_convex_hull_timeout = false;

static int get_linear_convex_hull_timeout() {
	static int linear_convex_hull_timeout = -1;
	// initialize once from environment
	if (linear_convex_hull_timeout == -1) {
		char* env = getenv(TIMEOUT_ENV);
		if (env) {
			// fprintf(stderr, "setting convex hull timeout from: %s\n", env);
			linear_convex_hull_timeout = atoi(env);
		}
		// set to 0 in case of
		if (linear_convex_hull_timeout < 0) {
			linear_convex_hull_timeout = 0;
		}
	}
	return linear_convex_hull_timeout;
}

static void catch_alarm_sc_convex_hull(int sig NOWUNUSED) {
	fprintf(stderr, "CATCH alarm sc_convex_hull !!!\n");
	//put inside CATCH(any_exception_error) alarm(0); //clear the alarm
	sc_convex_hull_timeout = true;
	THROW(timeout_error);
}

// Real vectors
// ------------

static zval_t* zvec_alloc(unsigned nbvals) {
	zval_t* zvec = safe_malloc(nbvals * sizeof(zval_t));
	int i;
	for (i = 0; i < (int) nbvals; i++) {
		zval_init(zvec[i]);
	}
	return zvec;
}

static void zvec_free(zval_t* zvec, unsigned nbvals) {
	int i;
	for (i = 0; i < (int) nbvals; i++) {
		zval_clear(zvec[i]);
	}
	free(zvec);
}

static void NOWUNUSED zvec_safe_free(zval_t* zvec, unsigned nbvals) {
	if (zvec != NULL) {
		zvec_free(zvec, nbvals);
	}
}

static void zvec_swap(zval_t* zvec1, zval_t* zvec2, unsigned nbvals) {
	zval_t tmp;
	zval_init(tmp);
	unsigned i;
	for (i = 0; i < nbvals; i++) {
		zval_set(tmp, zvec1[i]);
		zval_set(zvec1[i], zvec2[i]);
		zval_set(zvec2[i], tmp);
	}
	zval_clear(tmp);
}

static void zvec_set_i(zval_t* zvec, long val, unsigned nbvals) {
	unsigned i;
	for (i = 0; i < nbvals; i++) {
		zval_set_i(zvec[i], val);
	}
}

static void zvec_copy(zval_t* zvec1, zval_t* zvec2, unsigned nbvals) {
	zval_t* cv1, * cv2;
	cv1 = zvec1;
	cv2 = zvec2;
	unsigned i;
	for (i = 0; i < nbvals; i++)  {
		zval_set(*cv2++,*cv1++);
	}
}

// set zvec3 to lambda * zvec1 + mu * zvec2
static void zvec_combine(zval_t* zvec1, zval_t* zvec2, zval_t* zvec3,
		zval_t lambda, zval_t mu, unsigned nbvals) {
	zval_t tmp;
	zval_init(tmp);
	unsigned i;
	for (i = 0; i < nbvals; i++) {
		zval_mul(tmp, lambda, zvec1[i]);
		zval_addmul(tmp, mu, zvec2[i]);
		zval_set(zvec3[i], tmp);
	}
	zval_clear(tmp);
}

// set zvec2 to 1/lambda * zvec1
static void zvec_antiscale(zval_t* zvec1, zval_t* zvec2,
		zval_t lambda, unsigned nbvals) {
	unsigned i;
	for (i = 0; i < nbvals; i++) {
		zval_div(zvec2[i], zvec1[i], lambda);
	}
}

static void zvec_neg(zval_t* zvec1, zval_t* zvec2, unsigned nbvals) {
	unsigned i;
	for (i = 0; i < nbvals; i++) {
		zval_neg(zvec2[i], zvec1[i]);
	}
}

static int zvec_index_first_notzero(zval_t* zvec, unsigned nbvals) { 
	zval_t *cv;
	cv = zvec;
	int i;
	for (i = 0; i < (int) nbvals; i++) {
		if (!zval_equal_i(*cv, 0)) {
			break;
		}
		cv++;
	}
	return i == (int) nbvals ? -1 : i;
}

// find the position and the index of the smallest (in absolute value) element
// different from 0
static void zvec_min_notzero(zval_t* zvec, unsigned nbvals,
		int* pindex, zval_t* pmin) {
	zval_t aux;
	int i = zvec_index_first_notzero(zvec, nbvals);
	if (i == -1) {
		zval_set_i(*pmin, 1);
		return;
	}
	*pindex = i;
	zval_abs(*pmin, zvec[i]);
	zval_init(aux);
	for (i = i + 1; i < (int) nbvals; i++) {
		if (zval_equal_i(zvec[i], 0)) {
			continue;
		}
		zval_abs(aux, zvec[i]);
		if (zval_cmp(aux, *pmin) < 0) {
			zval_set(*pmin, aux);
			*pindex = i;
		}  
	}
	zval_clear(aux);
}

// compute the gcd of the values in zvec
static void zvec_gcd(zval_t* zvec, unsigned nbvals, zval_t *pgcd) {
	zval_t* vals = safe_malloc(nbvals * sizeof(zval_t));
	zval_t* cw, * cv;
	int i, index = 0;
	bool notzero;
	for (i = 0; i < (int) nbvals; i++) {
		zval_init(vals[i]);
	}
	cv = zvec;
	for (cw = vals, i = 0; i < (int) nbvals; i++) {
		zval_abs(*cw, *cv);
		cw++;
		cv++;
	}
	do {
		zvec_min_notzero(vals, nbvals, &index, pgcd);
		if (!zval_equal_i(*pgcd, 1)) {
			cw = vals;
			notzero = false;
			for (i = 0; i < (int) nbvals; i++, cw++) {
				if (i != index) {
					zval_mod(*cw, *cw, *pgcd);
					notzero |= !zval_equal_i(*cw, 0);
				}
			}
		}
		else {
			break;
		}
	}
	while (notzero);
	for (i = 0; i < (int) nbvals; i++) {
		zval_clear(vals[i]);
	}
	free(vals);
}

// divide values in zvec s.t. the gcd is 1
static void zvec_normalize(zval_t* zvec, unsigned nbvals) {
	zval_t gcd;
	zval_init(gcd);
	zvec_gcd(zvec, nbvals, &gcd);
	if (!zval_equal_i(gcd, 1)) {
		zvec_antiscale(zvec, zvec, gcd, nbvals);
	}
	zval_clear(gcd);
}

static int NOWUNUSED zvec_fprint(FILE* stream, zval_t* zvec, unsigned nbvals) {
	int c = fprintf(stream, "[");
	if (nbvals != 0) {
		c += zval_fprint(stream, zvec[0]);
	}
	unsigned i;
	for (i = 1; i < nbvals; i++) {
		c += fprintf(stream, ", ");
		c += zval_fprint(stream, zvec[i]);
	}
	c += fprintf(stream, "]");
	return c;
}

// Real matrices
// -------------

typedef struct {
	unsigned nbrows;
	unsigned nbcols;
	zval_t** vals;
	zval_t* rawvals;
	unsigned nbvals;
} zmat_s, *zmat_p;

typedef zmat_s zmat_t[1];

static zmat_p zmat_alloc(unsigned nbrows, unsigned nbcols) {
	zmat_p zmat = safe_malloc(sizeof(zmat_s));
	zmat->nbrows = nbrows;
	zmat->nbcols = nbcols;
	if (nbrows == 0 || nbcols == 0) {
		zmat->vals = (zval_t**) 0;
		zmat->rawvals = (zval_t*) 0;
		zmat->nbvals = 0;
	}
	else {
		zmat->nbvals = nbrows * nbcols;
		zval_t* rawvals = zvec_alloc(zmat->nbvals);
		zval_t** vals = safe_malloc(nbrows * sizeof(zval_t*));
		zmat->rawvals = rawvals;
		zmat->vals = vals;
		int i;
		for (i = 0; i < (int) nbrows; i++) {
			*vals++ = rawvals;
			rawvals += nbcols;
		}
	}
	return zmat;
}

static void zmat_free(zmat_p zmat) {
	if (zmat->rawvals) {
		zvec_free(zmat->rawvals, zmat->nbvals);
	}
	if (zmat->vals) {
		free(zmat->vals);
	}
	free(zmat);
}

static void NOWUNUSED zmat_safe_free(zmat_p zmat) {
	if (zmat != NULL) {
		zmat_free(zmat);
	}
}

// add rows to the matrix
static void zmat_extend(zmat_p zmat, unsigned nbrows) {
	zmat->vals = safe_realloc(zmat->vals, nbrows * sizeof(zval_t*));
	int i;
	if (zmat->nbvals < nbrows * zmat->nbcols) {
		zmat->rawvals = safe_realloc(zmat->rawvals, nbrows * zmat->nbcols * sizeof(zval_t));
		zvec_set_i(zmat->rawvals + zmat->nbrows * zmat->nbcols, 0,
				 zmat->nbvals - zmat->nbrows * zmat->nbcols);
		for (i = (int) zmat->nbvals; i < (int) (zmat->nbcols * nbrows); i++) {
			zval_init(zmat->rawvals[i]);
		}
		zmat->nbvals = zmat->nbcols*nbrows;
	}
	else {
		zvec_set_i(zmat->rawvals + zmat->nbrows * zmat->nbcols, 0,
				 (nbrows - zmat->nbrows) * zmat->nbcols);
	}
	for (i = 0; i < (int) nbrows; i++) {
		zmat->vals[i] = zmat->rawvals + (i * zmat->nbcols);
	}
	zmat->nbrows = nbrows;
}

static void NOWUNUSED zmat_fprint(FILE* stream, zmat_p zmat) {
	fprintf(stream, "%d %d\n", zmat->nbrows, zmat->nbcols);
	if (zmat->nbcols == 0) {
		fprintf(stream, "\n");
		return;
	}
	int i, j;
	for (i = 0; i < (int) zmat->nbrows; i++) {
		zval_t* vals = *(zmat->vals+i);
		for (j = 0; j < (int) zmat->nbcols; j++) {
			fprintf(stream, "%" STRINGOF(FPRINT_NBDIGITS) "li", zval_get_i(*vals++));
		}
		fprintf(stream, "\n");
	}
}

// Boolean misc
// ------------

#define WSIZE (8 * sizeof(int))

#define MSB ((unsigned) (((unsigned) 1) << (WSIZE - 1))) 

#define NEXT(j ,b) { \
	if (!((b) >>= 1)) { \
		(b) = MSB; \
		(j)++; \
	} \
}

// Boolean vectors
// ---------------

// store the logical `or' the bvec1 and bvec2 into bvec3
static void bvec_lor(int* bvec1, int* bvec2, int* bvec3, unsigned nbvals) {
	int* cp1, * cp2, * cp3;
	cp1 = bvec1;
	cp2 = bvec2;
	cp3 = bvec3;
	int i;
	for (i = 0; i < (int) nbvals; i++) {
		*cp3 = *cp1 | *cp2;
		cp3++;
		cp1++;
		cp2++;
	}
}

#define bvec_copy(bvec1, bvec2, nbvals) \
	(memcpy((char*) (bvec2), (char*) (bvec1), (int) ((nbvals) * sizeof(int))))

#define bvec_init(bvec, nbvals) \
	(memset((char*) (bvec), 0, (int) ((nbvals) * sizeof(int))))

// Boolean matrices (used as saturation matrices)
// ----------------

typedef struct {
	unsigned nbrows;
	unsigned nbcols;
	int **vals;
	int *rawvals;
} bmat_s, *bmat_p;

typedef bmat_s bmat_t[1];

static bmat_p bmat_alloc(int nbrows, int nbcols) {
	int** vals;
	int* rawvals;
	bmat_p bmat = safe_malloc(sizeof(bmat_s));
	bmat->nbrows = nbrows;
	bmat->nbcols = nbcols;
	if (nbrows == 0 || nbcols == 0) {
		bmat->vals = NULL;
		return bmat;
	}
	bmat->vals = vals = safe_malloc(nbrows * sizeof(int*));
	bmat->rawvals = rawvals = safe_malloc(nbrows * nbcols * sizeof(int));
	int i;
	for (i = 0; i < nbrows; i++) {
		*vals++ = rawvals;
		rawvals += nbcols;
	}
	return bmat;
}

static void bmat_free(bmat_p bmat) {
	if (bmat->vals) {
		free(bmat->rawvals);
		free(bmat->vals);
	}
	free(bmat);
}

static void NOWUNUSED bmat_safe_free(bmat_p bmat) {
	if (bmat != NULL) {
		bmat_free(bmat);
	}
}

static void bmat_extend(bmat_p bmat, zmat_p zmat, unsigned nbrows) {
	unsigned nbcols = (zmat->nbrows - 1) / WSIZE + 1;
	bmat->vals = safe_realloc(bmat->vals, nbrows * sizeof(int*));
	bmat->rawvals = safe_realloc(bmat->rawvals, nbrows * nbcols * sizeof(int));
	int i;
	for (i = 0; i < (int) nbrows; i++) {
		bmat->vals[i] = bmat->rawvals + (i * nbcols);
	}
	bmat->nbrows = nbrows;
}

static void NOWUNUSED bmat_fprint(FILE* stream, bmat_p bmat) {
	int *vals;
	fprintf(stream, "%d %d\n", bmat->nbrows, bmat->nbcols);
	int i, j;
	for (i = 0; i < (int) bmat->nbrows; i++) {
		vals = *(bmat->vals + i);
		for (j = 0; j < (int) bmat->nbcols; j++) {
			fprintf(stream, " %10X ", *vals++);
		}
		fprintf(stream, "\n");
	}
}

// Chernikova's algorithm
// ----------------------

// Chernikova's algorithm allows to switch between constraint and ray
// representations of a polyhedron. Both are stored into matrices. A (boolean)
// saturation matrix is also used to speed up computations.
//
// Constraints are represented by lines in the matrix `constrs', as follows:
// ----------------------------
// | ... |       ...    | ... |
// |  e  | a1 a2 ... an |  b  |
// | ... |       ...    | ... |
// ----------------------------
// |  1  |  0  0 ...  0 |  1  | <- l
// ----------------------------
// where
// * e is the type of constraint: 0 for an equation ax = b,
//   1 for an inequation ax >= b;
// * a1, ..., an are the coefficients in a;
// * b is the constant term.
// Equations are stored first, followed by inequations.
// An additional last line l is also present.
//
// Rays, lines and vertices are represented by lines in the matrix `rays':
// ----------------------------
// | ... |       ...    | ... |
// |  r  | a1 a2 ... an |  d  |
// | ... |       ...    | ... |
// ----------------------------
// where
// * r and d allow to determine the data type: line (r = d = 0),
//   ray (r = 1, d = 0) or vertex (r = 1, d != 0);
// * a1, ..., an are the coefficients;
// * only for vertices, d is the denominator.
// Lines are stored first, then rays and finally vertices.

static void combine(zval_t* zvec1, zval_t* zvec2, zval_t* zvec3,
		int pos, unsigned nbvals) {
	zval_t a1, a2, gcd;
	zval_t abs_a1, abs_a2, neg_a1;
	zval_init(a1); zval_init(a2); zval_init(gcd);
	zval_init(abs_a1); zval_init(abs_a2); zval_init(neg_a1);
	zval_set(a1, zvec1[pos]);
	zval_set(a2, zvec2[pos]);
	zval_abs(abs_a1, a1);
	zval_abs(abs_a2, a2);
	zval_gcd(gcd, abs_a1, abs_a2);
	zval_div(a1, a1, gcd);
	zval_div(a2, a2, gcd);
	zval_neg(neg_a1, a1);
	zvec_combine(zvec1 + 1, zvec2 + 1, zvec3 + 1, a2, neg_a1, nbvals);
	zvec_normalize(zvec3 + 1, nbvals);
	zval_clear(a1); zval_clear(a2); zval_clear(gcd);
	zval_clear(abs_a1); zval_clear(abs_a2); zval_clear(neg_a1);
}

static void sort_rays(zmat_p rays, bmat_p sat, int nbbrays, int nbrays,
		int *equal_bound, int *sup_bound, unsigned nbcols1,
		unsigned nbcols2, unsigned bx, unsigned jx) {
	int inf_bound;
	zval_t** uni_eq, ** uni_sup, ** uni_inf;
	int ** inc_eq, ** inc_sup, ** inc_inf;
	*sup_bound = *equal_bound = nbbrays;
	uni_sup = uni_eq = rays->vals + nbbrays;
	inc_sup = inc_eq = sat->vals + nbbrays;
	inf_bound = nbrays;
	uni_inf = rays->vals + nbrays;
	inc_inf = sat->vals + nbrays;
	while (inf_bound > *sup_bound) {
		if (zval_equal_i(**uni_sup, 0)) {
			if (inc_eq != inc_sup) {
				zvec_swap(*uni_eq, *uni_sup, nbcols1);
				BSWAP(*inc_eq, *inc_sup, nbcols2);
			}
			(*equal_bound)++;
			uni_eq++;
			inc_eq++;
			(*sup_bound)++;
			uni_sup++;
			inc_sup++;
		}
		else {
			*((*inc_sup) + jx) |= bx;
			if (zval_cmp_i(**uni_sup, 0) < 0) {
				inf_bound--;
				uni_inf--;
				inc_inf--;
				if (inc_inf != inc_sup) {
					zvec_swap(*uni_inf, *uni_sup, nbcols1);
					BSWAP(*inc_inf, *inc_sup, nbcols2);
				}
			}
			else {
				(*sup_bound)++;
				uni_sup++;
				inc_sup++;
			}
		}
	}
}

static int chernikova(zmat_p constrs, zmat_p rays, bmat_p sat, unsigned nbbrays,
			unsigned nbmaxrays, bool dual) {
	unsigned nbconstrs = constrs->nbrows;
	unsigned nbrays = rays->nbrows;
	unsigned nbdims = constrs->nbcols - 1;
	unsigned sat_nbcols = sat->nbcols;
	unsigned nbcols1 = nbdims + 1;
	unsigned nbcols2 = sat_nbcols * sizeof(int);
	int sup_bound, equal_bound, bound;
	int i, j, k, l;
	unsigned bx, m, jx;
	bool redundant, rayonly;
	int aux, nbcommonconstrs, index;
	zval_t* vals1, * vals2, * vals3;
	int* iv1, * iv2;
	int* tmp = safe_malloc(nbcols2);
	CATCH(any_exception_error) {
		free(tmp);
		RETHROW();
	}
	TRY {
		jx = 0;
		bx = MSB;
		for (k = 0; k < (int) nbconstrs; k++) {
			index = nbrays;
			for (i = 0; i < (int) nbrays; i++) {
				vals1 = rays->vals[i] + 1;
				vals2 = constrs->vals[k] + 1;
				vals3 = rays->vals[i];
				zval_mul(*vals3, *vals1, *vals2);
				vals1++;
				vals2++;
				for (j = 1; j < (int) nbdims; j++) {
					zval_addmul(*vals3, *vals1, *vals2);
					vals1++;
					vals2++;
				}
				if (!zval_equal_i(*vals3, 0) && i < index) {
					index = i;
				}
			}
			if (index < (int) nbbrays) {
				nbbrays--;
				if ((int) nbbrays != index) {
					zvec_swap(rays->vals[index], rays->vals[nbbrays], nbcols1);
				}
				for (i = 0; i < (int) nbbrays; i++) {
					if (!zval_equal_i(rays->vals[i][0], 0)) {
						combine(rays->vals[i], rays->vals[nbbrays], rays->vals[i],
								0, nbdims);
					}
				}
				if (zval_cmp_i(rays->vals[nbbrays][0], 0) < 0) {
					vals1 = rays->vals[nbbrays];
					for (j = 0; j < (int) nbdims + 1; j++) {
						zval_neg(*vals1, *vals1);
						vals1++;
					}
				}
				for (i = (int) nbbrays + 1; i < (int) nbrays; i++) {
					if (!zval_equal_i(rays->vals[i][0], 0)) {
						combine(rays->vals[i], rays->vals[nbbrays], rays->vals[i],
								0, nbdims);
					}
				}
				if (!zval_equal_i(constrs->vals[k][0], 0)) {
					for (j = 0; j < (int) sat_nbcols; j++) {
						sat->vals[nbbrays][j] = 0;
					}
					sat->vals[nbbrays][jx] |= bx;
				}
				else {
					if (--nbrays != nbbrays) {
						zvec_copy(rays->vals[nbrays], rays->vals[nbbrays], nbdims + 1);
						bvec_copy(sat->vals[nbrays], sat->vals[nbbrays], sat_nbcols);
					}
				}
			}
			else {
				sort_rays(rays, sat, nbbrays, nbrays, &equal_bound, &sup_bound,
					 nbcols1, nbcols2, bx, jx);
				bound = nbrays;
				for (i = equal_bound; i < sup_bound; i++) {
					for (j = sup_bound; j < bound; j++) {
						nbcommonconstrs = 0;
						for (l = 0; l < (int) jx; l++) {
							aux = tmp[l] = sat->vals[i][l] | sat->vals[j][l];
							for (m = MSB; m != 0; m >>= 1) {
								if(!(aux & m)) {
									nbcommonconstrs++;
								}
							}
						}
						aux = tmp[jx] = sat->vals[i][jx] | sat->vals[j][jx];
						for (m = MSB; m != bx; m >>= 1) {
							if (!(aux & m)) {
								nbcommonconstrs++;
							}
						}
						rayonly = zval_equal_i(rays->vals[i][nbdims], 0) &&
							 zval_equal_i(rays->vals[j][nbdims], 0) &&
							 dual == false;
						if (rayonly) {
							nbcommonconstrs++;
						}
						if (nbcommonconstrs + nbbrays >= nbdims - 2) {
							redundant = false;
							for (m = nbbrays; (int) m < bound; m++) {
								if (((int) m != i) && ((int) m != j)) {
									if (rayonly && !zval_equal_i(rays->vals[m][nbdims], 0)) {
										continue;
									}
									iv1 = tmp;
									iv2 = sat->vals[m];
									for (l = 0; l <= (int) jx; l++, iv2++, iv1++) {
										if (*iv2 & ~*iv1) {
											break;
										}
									}
									if (l > (int) jx) {
										redundant = true;
										break;
									}
								}
							}
							if (!redundant) {
								if (nbrays == nbmaxrays) {
									nbmaxrays *= 2;
									rays->nbrows = nbrays;
									zmat_extend(rays, nbmaxrays);
									bmat_extend(sat, constrs, nbmaxrays);
								}
								combine(rays->vals[j], rays->vals[i], rays->vals[nbrays],
										0, nbdims);
								bvec_lor(sat->vals[j], sat->vals[i], sat->vals[nbrays],
										sat_nbcols);
								sat->vals[nbrays][jx] &= ~bx;
								nbrays++;
							}
						}
					}
				}
				j = !zval_equal_i(constrs->vals[k][0], 0) ? sup_bound : equal_bound;
				i = nbrays;
				while (j < bound && i > bound) {
					i--;
					zvec_copy(rays->vals[i], rays->vals[j], nbdims + 1);
					bvec_copy(sat->vals[i], sat->vals[j], sat_nbcols);
					j++;
				}
				if (j == bound) {
					nbrays = i;
				}
				else {
					nbrays = j;
				}
			}
			NEXT(jx, bx);
		}
		rays->nbrows = nbrays;
		sat->nbrows = nbrays;
	}
	UNCATCH(any_exception_error);
	free(tmp);
	return 0;
}

static int gauss(zmat_p zmat, int nbeqs, int nbdims) {
	zval_t** vals = zmat->vals;
	int nbrows = zmat->nbrows;
	int i, j, k, pivot, rank;
	int* indexes = safe_malloc(nbdims * sizeof(int));
	zval_t gcd;
	zval_init (gcd);
	rank = 0;
	CATCH(any_exception_error) {
		safe_free(indexes);
		zval_clear(gcd);
		RETHROW();
	}
	TRY {
		for (j = 1; j <= nbdims; j++) {
			for (i = rank; i < nbeqs; i++) {
				if (!zval_equal_i(vals[i][j], 0)) {
					break;
				}
			}
			if (i != nbeqs) {
				if (i != rank) {
					zvec_swap(vals[rank] + 1, vals[i] + 1, nbdims);
				}
				zvec_gcd(vals[rank] + 1, nbdims, &gcd);
				if (zval_cmp_i(gcd, 2) >= 0) {
					zvec_antiscale(vals[rank] + 1, vals[rank] + 1, gcd, nbdims);
				}
				if (zval_cmp_i(vals[rank][j], 0) < 0) {
					zvec_neg(vals[rank] + 1, vals[rank] + 1, nbdims);
				}
				pivot = i;
				for (i = pivot + 1; i < nbeqs; i++) {
					if (!zval_equal_i(vals[i][j], 0)) {
						combine(vals[i], vals[rank], vals[i], j, nbdims);
					}
				}
				indexes[rank] = j;
				rank++;
			}
		}
		for (k = rank - 1; k >= 0; k--) {
			j = indexes[k];
			for (i = 0; i < k; i++) {
				if (!zval_equal_i(vals[i][j], 0)) {
					combine(vals[i], vals[k], vals[i], j, nbdims);
				}
			}
			for (i = nbeqs; i < nbrows; i++) {
				if (!zval_equal_i(vals[i][j], 0)) {
					combine (vals[i], vals[k], vals[i], j, nbdims);
				}
			}
		}
	}
	UNCATCH(any_exception_error);
	free(indexes);
	zval_clear(gcd);
	return rank;
}

static void remove_redundants(zmat_p* pconstrs, zmat_p* prays, bmat_p sat) {
	zmat_p constrs = *pconstrs, rays = *prays;
	zmat_p constrs2 = NULL, rays2 = NULL;
	unsigned nbbrays, nburays, nbeqs, nbineqs;
	unsigned nbbrays2 = -1, nburays2 = -1, nbeqs2 = -1, nbineqs2 = -1;
	unsigned nbdims = constrs->nbcols - 1;
	unsigned nbrays = rays->nbrows;
	unsigned sat_nbcols = sat->nbcols;
	unsigned nbconstrs = constrs->nbrows;
	unsigned nbcols2 = sat_nbcols * sizeof(int);
	int i, j, k;
	int aux;
	bool redundant;
	unsigned status;
	unsigned* trace = NULL, * bx = NULL, * jx = NULL, dimrayspace, b;
	zval_t* tmp = zvec_alloc(nbdims + 1);;
	zval_t one/*, mone*/;
	zval_init(one);/* zval_init(mone);*/
	zval_set_i(one, 1);/* zval_set_i(mone, -1);*/
	bx = safe_malloc(nbconstrs * sizeof(unsigned));
	jx = safe_malloc(nbconstrs * sizeof(unsigned));
	CATCH(any_exception_error) {
		zvec_free(tmp, nbdims + 1);
		zval_clear(one); zval_clear(mone);
		safe_free(bx);
		safe_free(jx);
		safe_free(trace);
		zmat_safe_free(constrs2);
		zmat_safe_free(rays2);
		RETHROW();
	}
	TRY {
		i = 0;
		b = MSB;
		for (j = 0; j < (int) nbconstrs; j++) {
			jx[j] = i;
			bx[j] = b;
			NEXT(i, b);
		}
		aux = 0;
		for (i = 0; i < (int) nbrays; i++) {
			zval_set_i(rays->vals[i][0], 0);
			if (!zval_equal_i(rays->vals[i][nbdims], 0)) {
				aux++;
			}
		}
		if (!aux) {
			goto empty;
		}
		nbeqs = 0;
		for (j = 0; j < (int) nbconstrs; j++) {
			zval_set_i(constrs->vals[j][0], 0);
			i = zvec_index_first_notzero(constrs->vals[j] + 1, nbdims - 1);
			if (i == -1) {
				for (i = 0; i < (int) nbrays; i++) {
					if (!(sat->vals[i][jx[j]] & bx[j])) {
						zval_add(constrs->vals[j][0], constrs->vals[j][0], one);
					}
				}
				if (zval_cmp_i(constrs->vals[j][0], nbrays) == 0 &&
						!zval_equal_i(constrs->vals[j][nbdims], 0)) {
					goto empty;
				}
				nbconstrs--;
				if (j == (int) nbconstrs) {
					continue;
				}
				zvec_swap(constrs->vals[j], constrs->vals[nbconstrs], nbdims + 1);
				SWAP(jx[j], jx[nbconstrs], aux);
				SWAP(bx[j], bx[nbconstrs], aux);
				j--;
				continue;
			}
			for (i = 0; i < (int) nbrays; i++) {
				if (!(sat->vals[i][jx[j]] & bx[j])) {
					zval_add(constrs->vals[j][0], constrs->vals[j][0], one);
					zval_add(rays->vals[i][0], rays->vals[i][0], one);
				}
			}
			if (zval_cmp_i(constrs->vals[j][0], nbrays) == 0) {
				nbeqs++;
			}
		}
		constrs->nbrows = nbconstrs;
		nbbrays = 0;
		for (i = 0; i < (int) nbrays; i++) {
			if (zval_equal_i(rays->vals[i][nbdims], 0)) {
				zval_add(rays->vals[i][0], rays->vals[i][0], one);
			}
			if (zval_cmp_i(rays->vals[i][0], nbconstrs + 1) == 0) {
				nbbrays++;
			}
		}
		for (i = 0; i < (int) nbeqs; i++) {
			if (zval_cmp_i(constrs->vals[i][0], nbrays) != 0) {
				for (k = i + 1; zval_cmp_i(constrs->vals[k][0], nbrays) != 0 &&
						k < (int) nbconstrs; k++);
				if (k == (int) nbconstrs) {
					break;
				}
				zvec_copy(constrs->vals[k], tmp, nbdims + 1);
				aux = jx[k];
				j = bx[k];
				for (; k > i; k--) {
					zvec_copy(constrs->vals[k - 1], constrs->vals[k], nbdims + 1);
					jx[k] = jx[k - 1];
					bx[k] = bx[k - 1];
				}
				zvec_copy(tmp, constrs->vals[i], nbdims + 1);
				jx[i] = aux;
				bx[i] = j;
			}
		}
		nbeqs2 = gauss(constrs, nbeqs, nbdims);
		if (nbeqs2 >= nbdims) {
			goto empty;
		}
		for (i = 0, k = nbrays; i < (int) nbbrays && k > i; i++) {
			if (zval_cmp_i(rays->vals[i][0], nbconstrs + 1) != 0) {
				while (--k > i && zval_cmp_i(rays->vals[k][0], nbconstrs + 1) != 0);
				zvec_swap(rays->vals[i], rays->vals[k], nbdims + 1);
				BSWAP(sat->vals[i], sat->vals[k], nbcols2);
			}
		}
		nbbrays2 = gauss(rays, nbbrays, nbdims);
		if (nbbrays2 >= nbdims) {
			goto empty;
		}
		dimrayspace = nbdims - 1 - nbeqs2 - nbbrays2;
		nbineqs = 0;
		for (j = 0; j < (int) nbconstrs; j++) {
			i = zvec_index_first_notzero(constrs->vals[j] + 1, nbdims - 1);
			if (i == -1) {
				if (zval_cmp_i(constrs->vals[j][0], nbrays) == 0 &&
						!zval_equal_i(constrs->vals[j][nbdims], 0)) {
					goto empty;
				}
				zval_set_i(constrs->vals[j][0], 2);
				continue;
			}
			status = zval_get_i(constrs->vals[j][0]);
			if (status == 0) {
				zval_set_i(constrs->vals[j][0], 2);
			}
			else if (status < dimrayspace) {
				zval_set_i(constrs->vals[j][0], 2);
			}
			else if (status == nbrays) {
				zval_set_i(constrs->vals[j][0], 0);
			}
			else {
				nbineqs++;
				zval_set_i(constrs->vals[j][0], 1);
			}
		}
		nburays = 0;
		for (j = 0; j < (int) nbrays; j++) {
			status = zval_get_i(rays->vals[j][0]);
			if (status < dimrayspace) {
				zval_set_i(rays->vals[j][0], 2);
			}
			else if (status == nbconstrs + 1) {
				zval_set_i(rays->vals[j][0], 0);
			}
			else {
				nburays++;
				zval_set_i(rays->vals[j][0], 1);
			}
		}
		constrs2 = zmat_alloc(nbineqs + nbeqs2 + 1, nbdims + 1);
		rays2 = zmat_alloc(nburays + nbbrays2, nbdims + 1);
		if (nbbrays2) {
			zvec_copy(rays->vals[0], rays2->vals[0], (nbdims + 1) * nbbrays2);
		}
		if (nbeqs2) {
			zvec_copy(constrs->vals[0], constrs2->vals[0], (nbdims + 1) * nbeqs2);
		}
		trace = safe_malloc(sat_nbcols * sizeof(unsigned));
		nbineqs2 = 0;
		for (j = nbeqs; j < (int) nbconstrs; j++) {
			if (zval_equal_i(constrs->vals[j][0], 1)) {
				for (k = 0; k < (int) sat_nbcols; k++) {
					trace[k] = 0;
				}
				for (i = nbbrays; i < (int) nbrays; i++) {
					if (zval_equal_i(rays->vals[i][0], 1)) {
						if (!(sat->vals[i][jx[j]] & bx[j])) {
							for (k = 0; k < (int) sat_nbcols; k++) {
								trace[k] |= sat->vals[i][k];
							}
						}
					}
				}
				redundant = false;
				for (i = nbeqs; i < (int) nbconstrs; i++) {
					if (zval_equal_i(constrs->vals[i][0], 1) && (i != j) &&
							!(trace[jx[i]] & bx[i])) {
						redundant = true;
						break;
					}
				}
				if (redundant) {
					zval_set_i(constrs->vals[j][0], 2);
				}
				else {
					zvec_copy(constrs->vals[j], constrs2->vals[nbeqs2 + nbineqs2],
							nbdims + 1);
					nbineqs2++;
				}
			}
		}
		free(trace);
		trace = safe_malloc(nbrays * sizeof(unsigned));
		nburays2 = 0;
		aux = 0;
		for (i = nbbrays; i < (int) nbrays; i++) {
			if (zval_equal_i(rays->vals[i][0], 1)) {
				if (!zval_equal_i(rays->vals[i][nbdims], 0)) {
					for (k = nbbrays; k < (int) nbrays; k++) {
						trace[k] = 0;
					}
				}
				else {
					for (k = nbbrays; k < (int) nbrays; k++) {
						trace[k] = !zval_equal_i(rays->vals[k][nbdims], 0);
					}
				}
				for (j = nbeqs; j < (int) nbconstrs; j++) {
					if (zval_equal_i(constrs->vals[j][0], 1)) {
						if (!(sat->vals[i][jx[j]] & bx[j])) {
							for (k = nbbrays; k < (int) nbrays; k++) {
								trace[k] |= sat->vals[k][jx[j]] & bx[j];
							}
						}
					}
				}
				redundant = false;
				for (j = nbbrays; j < (int) nbrays; j++) {
					if (zval_equal_i(rays->vals[j][0], 1) && (i != j) && !trace[j]) {
						redundant = true;
						break;
					}
				}
				if (redundant) {
					zval_set_i(rays->vals[i][0], 2);
				}
				else {
					zvec_copy(rays->vals[i], rays2->vals[nbbrays2 + nburays2],
							nbdims + 1);
					nburays2++;
					if (zval_equal_i(rays->vals[i][nbdims], 0)) {
						aux++;
					}
				}
			}
		}
		if (aux >= (int) dimrayspace) {
			zvec_set_i(constrs2->vals[nbeqs2 + nbineqs2], 0, nbdims + 1);
			zval_set_i(constrs2->vals[nbeqs2 + nbineqs2][0], 1);
			zval_set_i(constrs2->vals[nbeqs2 + nbineqs2][nbdims], 1);
			nbineqs2++;
		}
	}
	UNCATCH(any_exception_error);
	free(trace);
	free(bx);
	free(jx);
	constrs2->nbrows = nbeqs2 +nbineqs2;
	rays2->nbrows = nbbrays2 + nburays2;
	zvec_free(tmp, nbdims + 1);
	zval_clear(one); zval_clear(mone);
	zmat_free(constrs);
	zmat_free(rays);
	*pconstrs = constrs2;
	*prays = rays2;
	return;
empty:
	zvec_free(tmp, nbdims + 1);
	zval_clear(one); zval_clear(mone);
	safe_free(bx);
	safe_free (jx);
	UNCATCH(any_exception_error);
	constrs2 = zmat_alloc(nbdims, nbdims + 1);
	rays2 = zmat_alloc(0, nbdims + 1);
	for (i = 0; i < (int) nbdims; i++) {
		zval_set_i(constrs2->vals[i][i + 1], 1);
	}
	zmat_free(constrs);
	zmat_free(rays);
	*pconstrs = constrs2;
	*prays = rays2;
	return;
}

static bmat_p transpose_sat(zmat_p constrs, zmat_p rays, bmat_p sat) {
	int sat_nbcols;
	if (constrs->nbrows != 0) {
		sat_nbcols = (constrs->nbrows - 1) / WSIZE + 1;
	}
	else {
		sat_nbcols = 0;
	}
	bmat_p tsat = bmat_alloc(rays->nbrows, sat_nbcols);
	bvec_init(tsat->rawvals, rays->nbrows * sat_nbcols);
	unsigned i, j;
	unsigned jx1, jx2, bx1, bx2;
	for (i = 0, jx1 = 0, bx1 = MSB; i < rays->nbrows; i++) {
		for (j = 0, jx2 = 0, bx2 = MSB; j < constrs->nbrows; j++) {
			if (sat->vals[j][jx1] & bx1) {
				tsat->vals[i][jx2] |= bx2;
			}
			NEXT(jx2, bx2);
		}
		NEXT(jx1, bx1);
	}
	return tsat;
}

// convert from constrs to rays
static zmat_p rays_of_constrs(zmat_p* pconstrs, unsigned nbmaxrays) {
	zmat_p constrs = *pconstrs;
	unsigned nbdims = constrs->nbcols - 1;
	nbmaxrays = nbmaxrays < nbdims ? nbdims : nbmaxrays;
	zmat_p rays = zmat_alloc(nbmaxrays, nbdims + 1);
	unsigned i;
	for (i = 0; i < nbmaxrays * (nbdims + 1); i++) {
		zval_set_i(rays->rawvals[i], 0);
	}
	for (i = 0; i < nbdims; i++) {
		zval_set_i(rays->vals[i][i + 1], 1);
	}
	zval_set_i(rays->vals[nbdims - 1][0], 1);
	rays->nbrows = nbdims;
	int nbcols = (constrs->nbrows - 1) / WSIZE + 1;
	bmat_p sat = bmat_alloc(nbmaxrays, nbcols);
	bvec_init(sat->rawvals, nbdims * nbcols);
	sat->nbrows = nbdims;
	chernikova(constrs, rays, sat, nbdims - 1, nbmaxrays, false);
	remove_redundants(pconstrs, &rays, sat);
	bmat_free(sat);
	return rays;
}

// convert from rays to constrs
static zmat_p constrs_of_rays(zmat_p* prays, unsigned nbmaxconstrs) {
	zmat_p rays = *prays;
	unsigned nbdims = rays->nbcols - 1;
	nbmaxconstrs = nbmaxconstrs < nbdims ? nbdims : nbmaxconstrs;
	zmat_p constrs = zmat_alloc(nbmaxconstrs, nbdims + 1);
	unsigned i;
	for (i = 0; i < nbmaxconstrs * (nbdims + 1); i++) {
		zval_set_i(constrs->rawvals[i], 0);
	}
	for (i = 0; i < nbdims; i++) {
		zval_set_i(constrs->vals[i][i + 1], 1);
	}
	constrs->nbrows = nbdims;
	int nbcols = (rays->nbrows - 1) / WSIZE + 1;
	bmat_p tsat = bmat_alloc(nbmaxconstrs, nbcols);
	bvec_init(tsat->rawvals, nbdims * nbcols);
	tsat->nbrows = nbdims;
	chernikova(rays, constrs, tsat, nbdims, nbmaxconstrs, true);
	bmat_p sat = transpose_sat(constrs, rays, tsat);
	bmat_free(tsat);
	remove_redundants(&constrs, prays, sat);
	bmat_free(sat);
	return constrs;
}

// Data conversion
// ---------------

static void zmat_set_row(zmat_p zmat, unsigned i,
		int first, Pvecteur pv, int last, Pbase base) {
	Pvecteur coord = base;
	zval_set_i(zmat->vals[i][0], first);
	int j = 1;
	for (; coord != NULL; coord = coord->succ, j++) {
		zval_set_i(zmat->vals[i][j], vect_coeff(vecteur_var(coord), pv));
	}
	zval_set_i(zmat->vals[i][j], last);
}

static Pvecteur vecteur_of_zvec(zval_t* zvec, Pbase base) {
	int i;
	Pvecteur pv = NULL, coord;
	for (i = 0, coord = base; coord != NULL; i++, coord = coord->succ) {
		if (!zval_equal_i(zvec[i], 0)) {
			if (pv == NULL) {
				pv = vect_new(vecteur_var(coord), zval_get_i(zvec[i]));
			}
			else {
				vect_add_elem(&pv, vecteur_var(coord), zval_get_i(zvec[i]));
			}
		}
	}
	return pv;
}

// build a constraint matrix from a constraint system
static zmat_p constrs_of_sc(Psysteme sc) {
	unsigned nbrows = sc->nb_eq + sc->nb_ineq + 1; // extra constraint
	unsigned nbcols = sc->dimension + 2; // constraint type + extra var
	zmat_p constrs = zmat_alloc(nbrows, nbcols);
	unsigned i, j;
	i = 0;
	for (; (int) i < sc->nb_eq; i++) {
		zval_set_i(constrs->vals[i][0], 0);
	}
	for (; i < nbrows; i++) {
		zval_set_i(constrs->vals[i][0], 1);
	}
	i = 0;
	Pcontrainte pc;
	Pvecteur coord;
	for (pc = sc->egalites; pc != NULL; pc = pc->succ, i++) {
		for (j = 1, coord = sc->base; coord != NULL; coord = coord->succ, j++) {
			zval_set_i(constrs->vals[i][j], -vect_coeff(vecteur_var(coord), pc->vecteur));
		}
		zval_set_i(constrs->vals[i][j], -vect_coeff(TCST, pc->vecteur));
	}
	for (pc = sc->inegalites; pc != NULL; pc = pc->succ, i++) {
		for (j = 1, coord = sc->base; coord != NULL; coord = coord->succ, j++) {
			zval_set_i(constrs->vals[i][j], -vect_coeff(vecteur_var(coord), pc->vecteur));
		}
		zval_set_i(constrs->vals[i][j], -vect_coeff(TCST, pc->vecteur));
	}
	for (j = 1; j < nbcols - 1; j++) {
		zval_set_i(constrs->vals[nbrows - 1][j], 0);
	}
	zval_set_i(constrs->vals[nbrows - 1][nbcols - 1], 1);
	return constrs;
}

// build a constraint system from a constraint matrix
static Psysteme sc_of_constrs(zmat_p constrs, Pbase base) {
	Psysteme sc = sc_new();
	sc->base = base_copy(base);
	sc->dimension = base_dimension(base);
	sc->nb_eq = 0;
	sc->egalites = NULL;
	sc->nb_ineq = 0;
	sc->inegalites = NULL;
	Pcontrainte pce = NULL, pci = NULL;
	unsigned i;
	for (i = 0; i < constrs->nbrows; i++) {
		Pvecteur pv = vecteur_of_zvec(constrs->vals[i] + 1, base);
		int cst = zval_get_i(constrs->vals[i][constrs->nbcols - 1]);
		vect_add_elem(&pv, TCST, cst);
		vect_chg_sgn(pv);
		if (zval_equal_i(constrs->vals[i][0], 0)) {
			sc->nb_eq++;
			if (!pce) {
				sc->egalites = pce = contrainte_make(pv);
			}
			else {
				pce->succ = contrainte_make(pv);
				pce = pce->succ;
			}
		}
		else if (zval_equal_i(constrs->vals[i][0], 1)) {
			sc->nb_ineq++;
			if (!pci) {
				sc->inegalites = pci = contrainte_make(pv);
			}
			else {
				pci->succ = contrainte_make(pv);
				pci = pci->succ;
			}
		}
		else {
			assert(false);
		}
	}
	sc_normalize(sc);
	return sc;
}

// build a ray matrix from a generator system
static zmat_p rays_of_sg(Ptsg sg) {
	unsigned nbrows = sg_nbre_sommets(sg) + sg_nbre_rayons(sg) +
			sg_nbre_droites(sg);
	unsigned nbcols = base_dimension(sg->base) + 2;
	zmat_p rays = zmat_alloc(nbrows, nbcols);
	Psommet ps; Pray_dte pr;
	int i = 0;
	for (pr = sg->dtes_sg.vsg; pr != NULL; pr = pr->succ, i++) {
		zmat_set_row(rays, i, 0, pr->vecteur, 0, sg->base);
	}
	for (pr = sg->rays_sg.vsg; pr != NULL; pr = pr->succ, i++) {
		zmat_set_row(rays, i, 1, pr->vecteur, 0, sg->base);
	}
	for (ps = sg->soms_sg.ssg; ps != NULL; ps = ps->succ, i++) {
		zmat_set_row(rays, i, 1, ps->vecteur, ps->denominateur, sg->base);
	}
	return rays;
}

// build a generator system from a ray matrix
static Ptsg sg_of_rays(zmat_p rays, Pbase base) {
	Ptsg sg = sg_new();
	sg->base = base_copy(base);
	sg->soms_sg.nb_s = 0;
	sg->soms_sg.ssg = NULL;
	sg->rays_sg.nb_v = 0;
	sg->rays_sg.vsg = NULL;
	sg->dtes_sg.nb_v = 0;
	sg->dtes_sg.vsg = NULL;
	Psommet verts = NULL; Pray_dte urays = NULL, brays = NULL;
	Pvecteur pv;
	unsigned i;
	for (i = 0; i < rays->nbrows; i++) {
		pv = vecteur_of_zvec(rays->vals[i] + 1, base);
		if (zval_equal_i(rays->vals[i][0], 0)) {
			sg->dtes_sg.nb_v++;
			if (!brays) {
				sg->dtes_sg.vsg = brays = ray_dte_make(pv);
			}
			else {
				brays->succ = ray_dte_make(pv);
				brays = brays->succ;
			}
		}
		else if (zval_equal_i(rays->vals[i][0], 1)) {
			if (zval_equal_i(rays->vals[i][rays->nbcols - 1], 0)) {
				sg->rays_sg.nb_v++;
				if (!urays) {
					sg->rays_sg.vsg = urays = ray_dte_make(pv);
				}
				else {
					urays->succ = ray_dte_make(pv);
					urays = urays->succ;
				}
			}
			else {
				sg->soms_sg.nb_s++;
				long ratio = zval_get_i(rays->vals[i][rays->nbcols - 1]);
				if (!verts) {
					sg->soms_sg.ssg = verts = sommet_make(ratio, pv);
				}
				else {
					verts->succ = sommet_make(ratio, pv);
					verts = verts->succ;
				}
			}
		}
		else {
			assert(false);
		}
	}
	return sg;
}

static zmat_p NOWUNUSED rays_of_sc(Psysteme sc, unsigned nbmaxrays) {
	zmat_p constrs = constrs_of_sc(sc);
	zmat_p rays = rays_of_constrs(&constrs, nbmaxrays);
	zmat_free(constrs);
	return rays;
}

static Psysteme NOWUNUSED sc_of_rays(zmat_p* prays, Pbase base, unsigned nbmaxconstrs) {
	zmat_p constrs = constrs_of_rays(prays, nbmaxconstrs);
	Psysteme sc = sc_of_constrs(constrs, base);
	zmat_free(constrs);
	return sc;
}

static zmat_p NOWUNUSED constrs_of_sg(Ptsg sg, unsigned nbmaxconstrs) {
	zmat_p rays = rays_of_sg(sg);
	zmat_p constrs = constrs_of_rays(&rays, nbmaxconstrs);
	zmat_free(rays);
	return constrs;
}

static Ptsg NOWUNUSED sg_of_constrs(zmat_p* pconstrs, Pbase base, unsigned nbmaxrays) {
	zmat_p rays = rays_of_constrs(pconstrs, nbmaxrays);
	Ptsg sg = sg_of_rays(rays, base);
	zmat_free(rays);
	return sg;
}

// Main functions
// --------------

#define NBMAXRAYS    (20000)
#define NBMAXCONSTRS (20000)

static inline Ptsg sg_of_sc(Psysteme sc) {
	zmat_p rays = NULL;
	Ptsg sg = NULL;
	CATCH(any_exception_error) {
		zmat_safe_free(rays);
		if (sg != NULL) {
			sg_rm(sg);
		}
		RETHROW();
	}
	TRY {
		rays = rays_of_sc(sc, NBMAXRAYS);
		sg = sg_of_rays(rays, sc->base);
		zmat_free(rays);
	}
	UNCATCH(any_exception_error);
	return sg;
}

static inline Psysteme sc_of_sg(Ptsg sg) {
	zmat_p constrs = NULL;
	Psysteme sc = NULL;
	CATCH(any_exception_error) {
		zmat_safe_free(constrs);
		if (sc != NULL) {
			sc_rm(sc);
		}
		RETHROW();
	}
	TRY {
		constrs = constrs_of_sg(sg, NBMAXCONSTRS);
		sc = sc_of_constrs(constrs, sg->base);
		zmat_free(constrs);
	}
	UNCATCH(any_exception_error);
	return sc;
}

static inline Psysteme sc_union(Psysteme sc1, Psysteme sc2) {
	assert(bases_strictly_equal_p(sc1->base, sc2->base));
	zmat_p rays1 = NULL, rays2 = NULL, rays = NULL;
	Psysteme sc = NULL;
	CATCH(any_exception_error) {
		zmat_safe_free(rays1);
		zmat_safe_free(rays2);
		zmat_safe_free(rays);
		if (sc != NULL) {
			sc_rm(sc);
		}
		// clear the alarm if necessary		
		if (get_linear_convex_hull_timeout()) {
			alarm(0);
		}
		if (sc_convex_hull_timeout) {
			sc_convex_hull_timeout = false;
		}
		RETHROW();
	}
	TRY {
		// start the alarm
		if (get_linear_convex_hull_timeout()) {
			signal(SIGALRM, catch_alarm_sc_convex_hull);
			alarm(get_linear_convex_hull_timeout());
		}
		rays1 = rays_of_sc(sc1, NBMAXRAYS);
		rays2 = rays_of_sc(sc2, NBMAXRAYS);
		unsigned nbrows1 = rays1->nbrows;
		unsigned nbrows2 = rays2->nbrows;
		unsigned nbrows = nbrows1 + nbrows2;
		unsigned nbcols = rays1->nbcols;
		rays = zmat_alloc(nbrows, nbcols);
		// merge rays1 and rays2, preserving ray order
		unsigned i = 0, i1 = 0, i2 = 0, j;
		while (i1 < nbrows1 && zval_equal_i(rays1->vals[i1][0], 0) &&
				zval_equal_i(rays1->vals[i1][nbcols - 1], 0)) {
			for (j = 0; j < nbcols; j++) {
				zval_set(rays->vals[i][j], rays1->vals[i1][j]);
			}
			i++, i1++;
		}
		while (i2 < nbrows2 && zval_equal_i(rays2->vals[i2][0], 0) &&
				zval_equal_i(rays2->vals[i2][nbcols - 1], 0)) {
			for (j = 0; j < nbcols; j++) {
				zval_set(rays->vals[i][j], rays2->vals[i2][j]);
			}
			i++, i2++;
		}
		while (i1 < nbrows1 && zval_equal_i(rays1->vals[i1][0], 1) &&
				zval_equal_i(rays1->vals[i1][nbcols - 1], 0)) {
			for (j = 0; j < nbcols; j++) {
				zval_set(rays->vals[i][j], rays1->vals[i1][j]);
			}
			i++, i1++;
		}
		while (i2 < nbrows2 && zval_equal_i(rays2->vals[i2][0], 1) &&
				zval_equal_i(rays2->vals[i2][nbcols - 1], 0)) {
			for (j = 0; j < nbcols; j++) {
				zval_set(rays->vals[i][j], rays2->vals[i2][j]);
			}
			i++, i2++;
		}
		while (i1 < nbrows1 && zval_equal_i(rays1->vals[i1][0], 1) &&
				!zval_equal_i(rays1->vals[i1][nbcols - 1], 0)) {
			for (j = 0; j < nbcols; j++) {
				zval_set(rays->vals[i][j], rays1->vals[i1][j]);
			}
			i++, i1++;
		}
		while (i2 < nbrows2 && zval_equal_i(rays2->vals[i2][0], 1) &&
				!zval_equal_i(rays2->vals[i2][nbcols - 1], 0)) {
			for (j = 0; j < nbcols; j++) {
				zval_set(rays->vals[i][j], rays2->vals[i2][j]);
			}
			i++, i2++;
		}
		assert(i == nbrows);
		zmat_free(rays1); rays1 = NULL;
		zmat_free(rays2); rays2 = NULL;
		sc = sc_of_rays(&rays, sc1->base, NBMAXCONSTRS);
		zmat_free(rays);
	}
	// Clear the alarm if necessary
	if (get_linear_convex_hull_timeout()) {
		alarm(0);
	}
	UNCATCH(any_exception_error);
	return sc;
}

#endif

