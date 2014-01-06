/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#ifndef EXPRESSION_H_
#define EXPRESSION_H_
/*
 * for partial_eval.c:
 *
 * EFORMAT: the expression format used in recursiv evaluations. 
 * = ((ICOEF * EXPR) + ISHIFT)
 * it is SIMPLER when it is interesting to replace initial expression by the
 * one generated from eformat.
 */
struct eformat {
    expression expr;
    int icoef;
    int ishift;
    bool simpler;
};

typedef struct eformat eformat_t;
#endif
