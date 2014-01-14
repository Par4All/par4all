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
/* macros */
#define VERTEX_ENCLOSING_SCC(v) \
    sccflags_enclosing_scc(dg_vertex_label_sccflags((dg_vertex_label) \
	vertex_vertex_label(v)))


/* a macro to insert an element at the end of a list. c is the element
to insert.  bl and el are pointers to the begining and the end of the
list. */

#define INSERT_AT_END(bl, el, c) \
    { cons *_insert_ = c; if (bl == NIL) bl = _insert_; else CDR(el) = _insert_; el = _insert_; CDR(el) = NIL; }


/* external variables. see declarations in kennedy.c */
extern graph dg;
extern bool rice_distribute_only;
extern int Nbrdoall;
