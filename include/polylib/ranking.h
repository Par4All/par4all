// Tools to compute the ranking function of an iteration J : the number of integer points in P that are lexicographically inferior to J
// (C) B. Meister 6/2005
// LSIIT, UMR 7005 CNRS Université Louis Pasteur
// HiPEAC Network
// ICPS

#ifndef __BM_POLYLIB_RANKING_H__
#define __BM_POLYLIB_RANKING_H__
#include <polylib/polylib.h>

/*
 * Returns a list of polytopes needed to compute
 * the number of points in P that are lexicographically
 * smaller than a given point in D.
 * When P == D, this is the conventional ranking function.
 * P and D are assumed to have the same parameter domain C.
 *
 * The first polyhedron in the list returned is the
 * updated context: a combination of D and C.
 */
Polyhedron *RankingPolytopes(Polyhedron *P, Polyhedron *D, Polyhedron *C, 
			     unsigned MAXRAYS);

/*
 * Returns the number of points in P that are lexicographically
 * smaller than a given point in D.
 * When P == D, this is the conventional ranking function.
 * P and D are assumed to have the same parameter domain C.
 * The variables in the Enumeration correspond to the variables
 * in D followed by the parameter of D (the variables of C).
 */
Enumeration *Polyhedron_Ranking(Polyhedron *P, Polyhedron *D, Polyhedron *C, 
				unsigned MAXRAYS);

#endif // __BM_POLYLIB_RANKING_H__
