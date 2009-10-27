/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

#define SIGN_EQ(a,b) ((((a)>0 && (b)>0) || ((a)<0 && (b)<0)) ? TRUE : FALSE)
#define FORTRAN_DIV(n,d) (SIGN_EQ((n),(d)) ? ABS(n)/ABS(d) : -(ABS(n)/ABS(d)))
#define FORTRAN_MOD(n,m) (SIGN_EQ((n),(m)) ? ABS(n)%ABS(m) : -(ABS(n)%ABS(m)))

#define OUTLINE_PRAGMA "outline this"
#define OUTLINE_IGNORE "outline_ignore"

/* What is returned by dead_test_filter : */
enum dead_test { nothing_about_test, then_is_dead, else_is_dead };
typedef enum dead_test dead_test;


enum region_to_dma_switch { dma_load, dma_store, dma_allocate, dma_deallocate };
/* Add NewGen-like methods: */
#define dma_load_p(e) ((e) == dma_load )
#define dma_store_p(e) ((e) == dma_store )
#define dma_allocate_p(e) ((e) == dma_allocate )
#define dma_deallocate_p(e) ((e) == dma_deallocate )


enum range_to_expression_mode{
    range_to_distance,
    range_to_nbiter
} ;
#define range_to_distance_p(e) ((e) == range_to_distance)
#define range_to_nbiter_p(e) ((e) == range_to_nbiter)
