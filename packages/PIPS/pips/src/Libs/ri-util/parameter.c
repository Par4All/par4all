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
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"

/*
 * Functions for effective parameters
 */

/**
 * Display a parameter on stderr, useful for debugging
 * @param p is the parameter to display
 */
void print_parameter(parameter p) {
  if(p == parameter_undefined) {
    fprintf(stderr, "PARAMETER UNDEFINED\n");
  } else {
    fprintf(stderr, "type = ");
    print_type(parameter_type(p));
    fprintf(stderr, "\nmode= ");
    print_mode(parameter_mode(p));
    fprintf(stderr, "\ndummy= ");
    print_dummy(parameter_dummy(p));
    fprintf(stderr, "\n");
  }
}

/**
 * Display a parameter on stderr, useful for debugging
 * @param lp is the list of parameters to display
 */
void print_parameters(list lp) {
  FOREACH(PARAMETER, p , lp)
  {
    print_parameter(p);
  }
}



/**
 * Display a "mode" on stderr, useful for debugging
 * @param p is the mode to display
 */
void print_mode(mode m) {
  if(!mode_defined_p(m)) {
    fprintf(stderr, "MODE UNDEFINED\n");
  } else if( mode_value_p(m)) {
    fprintf(stderr, "value");
  } else if( mode_reference_p(m)) {
    fprintf(stderr, "reference");
  } else {
    fprintf(stderr, "unknown");
  }
}


/**
 * Display a "dummy" on stderr, useful for debugging
 * @param d is the dummy to display
 */
void print_dummy(dummy d) {
  if(!dummy_defined_p(d)) {
    fprintf(stderr, "DUMMY UNDEFINED\n");
  } else {
    fprintf(stderr, "%s",entity_name(dummy_identifier(d)));
  }
}
