/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_GMP_H

#include "../arithmetique/arith_mulprec.h"
#include "chernikova.h"

Ptsg sc_to_sg_chernikova_mulprec(Psysteme sc) {
	return sg_of_sc(sc);
}

Psysteme sg_to_sc_chernikova_mulprec(Ptsg sg) {
	return sc_of_sg(sg);
}

Psysteme sc_convex_hull_mulprec(Psysteme sc1, Psysteme sc2) {
	return sc_union(sc1, sc2);
}

#endif

