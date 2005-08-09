// Tools to compute the ranking function of an iteration J : the number of integer points in P that are lexicographically inferior to J
// (C) B. Meister 6/2005
// LSIIT, UMR 7005 CNRS Université Louis Pasteur
// HiPEAC Network
// ICPS

#ifndef __BM_POLYLIB_RANKING_H__
#define __BM_POLYLIB_RANKING_H__
#include <polylib/polylib.h>


// given the constraints of a polyhedron P (the matrix Constraints), returns the number of points that are lexicographically stricly lesser than a point I of P,
// defined by I = M.(J N 1)^T
// J are the n' first variables of the returned Ehrhart polynomial.
// If M is NULL, I = J is taken by default.
Enumeration *Ranking(Matrix * Constraints, Matrix * C, Matrix * M, unsigned MAXRAYS);

#endif // __BM_POLYLIB_RANKING_H__
