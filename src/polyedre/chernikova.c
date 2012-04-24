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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>

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

#define WNOGMP (fprintf(stderr, "[" __FILE__ "] linear was compiled without GMP support\n"))

static bool usegmp() {
	char* env = getenv("LINEAR_USE_GMP");
	return env && atoi(env) != 0;
}

Ptsg sc_to_sg_chernikova(Psysteme sc) {
	Ptsg sg;
	if (usegmp()) {
#ifdef LINEAR_DEPEND_GMP
		sg = sc_to_sg_chernikova_mulprec(sc);
#else
		WNOGMP;
		sg = sc_to_sg_chernikova_fixprec(sc);
#endif
	}
	else {
		sg = sc_to_sg_chernikova_fixprec(sc);
	}
	return sg;
}

Psysteme sg_to_sc_chernikova(Ptsg sg) {
	Psysteme sc;
	if (usegmp()) {
#ifdef LINEAR_DEPEND_GMP
		sc = sg_to_sc_chernikova_mulprec(sg);
#else
		WNOGMP;
		sc = sg_to_sc_chernikova_fixprec(sg);
#endif
	}
	else {
		sc = sg_to_sc_chernikova_fixprec(sg);
	}
	return sc;
}

Psysteme sc_convex_hull(Psysteme sc1, Psysteme sc2) {
	Psysteme sc;
	if (usegmp()) {
#ifdef LINEAR_DEPEND_GMP
		sc = sc_convex_hull_mulprec(sc1, sc2);
#else
		WNOGMP;
		sc = sc_convex_hull_fixprec(sc1, sc2);
#endif
	}
	else {
		sc = sc_convex_hull_fixprec(sc1, sc2);
	}
	return sc;
}

