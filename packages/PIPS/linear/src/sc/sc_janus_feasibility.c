/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#include <string.h>
#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sc-private.h"

/**
 * Fallback function to compute feasibility with Janus, used when no custom
 * function is set.
 */
bool sc_janus_feasibility_fallback(Psysteme sc __attribute__((unused))) 
{ 
	return true;
}

/**
 * Internal pointer to a custom Janus feasibility function. Is null when none
 * is set.
 */
static bool (*sc_janus_feasibility_ptr)(Psysteme) = NULL;

/**
 * Compute feasibility, using custom Janus function if set, fallback function
 * otherwise.
 */
bool sc_janus_feasibility(Psysteme sc)
{
	if (sc_janus_feasibility_ptr != NULL) {
		return sc_janus_feasibility_ptr(sc);
	}
	else {
		return sc_janus_feasibility_fallback(sc);
	}
}

/**
 * Set custom Janus feasibility function.
 */
void set_sc_janus_feasibility(bool (*sc_janus_feasibility_fct)(Psysteme))
{
	sc_janus_feasibility_ptr = sc_janus_feasibility_fct;
}

