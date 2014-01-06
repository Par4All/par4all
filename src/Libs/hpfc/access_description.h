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

#ifndef ACCESS_DESCRIPTION_H
#define ACCESS_DESCRIPTION_H

/*
 * just something looking like Newgen Domains for homogeneity
 *
 * should use normalized somewhere, somehow...
 */

#define access			int

#define	access_undefined        (-1)
#define	local_constant		 (1)
#define local_shift		 (2)
#define local_star		 (3)
#define local_affine		 (4)
#define aligned_constant	 (5)
#define aligned_shift		 (6)
#define aligned_star		 (7)
#define aligned_affine		 (8)
#define not_aligned		 (9)
#define local_form_cst          (10)

#define	access_undefined_p(a)		(a == access_undefined)

#define access_tag(a)			(a)
#define make_access(a)			(a)

#endif

