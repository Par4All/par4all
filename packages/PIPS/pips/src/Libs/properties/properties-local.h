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
#include "property.h"

#define ONE_TRIP_DO "ONE_TRIP_DO"

#define PRETTYPRINT_TRANSFORMER "PRETTYPRINT_TRANSFORMER"
#define PRETTYPRINT_EXECUTION_CONTEXT "PRETTYPRINT_EXECUTION_CONTEXT"
#define PRETTYPRINT_EFFECTS "PRETTYPRINT_EFFECTS"
#define PRETTYPRINT_PARALLEL "PRETTYPRINT_PARALLEL"
#define PRETTYPRINT_REVERSE_DOALL "PRETTYPRINT_REVERSE_DOALL"
#define PRETTYPRINT_REGION "PRETTYPRINT_REGION"

#define SEMANTICS_FLOW_SENSITIVE "SEMANTICS_FLOW_SENSITIVE"
#define SEMANTICS_INTERPROCEDURAL "SEMANTICS_INTERPROCEDURAL"
#define SEMANTICS_INEQUALITY_INVARIANT "SEMANTICS_INEQUALITY_INVARIANT"
#define SEMANTICS_FIX_POINT "SEMANTICS_FIX_POINT"
#define SEMANTICS_DEBUG_LEVEL "SEMANTICS_DEBUG_LEVEL"
#define SEMANTICS_STDOUT "SEMANTICS_STDOUT"

#define PARALLELIZE_USE_EXECUTION_CONTEXT "PARALLELIZE_USE_EXECUTION_CONTEXT"

#define DEPENDENCE_TEST "DEPENDENCE_TEST"
#define RICEDG_PROVIDE_STATISTICS "RICEDG_PROVIDE_STATISTICS"

/* for upwards compatibility with Francois's modified version */
#define pips_flag_p(p) get_bool_property(p)
#define pips_flag_set(p) set_bool_property((p), true)
#define pips_flag_reset(p) set_bool_property((p), false)
#define pips_flag_fprint(fd) fprint_properties(fd)

bool too_many_property_errors_pending_p(void);
void reset_property_error(void);
