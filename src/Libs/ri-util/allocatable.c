/*

 $Id$

 Copyright 1989-2010 MINES ParisTech
 Copyright 2009-2010 HPC Project

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
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "misc.h"
#include "ri.h"

#include "ri-util.h"

/***
 * Handling of ALLOCATABLE (Fortran95)
 *
 * Allocatable are represented internally as structure, for instance :
 *
 * integer, dimension (:,:,:), allocatable :: myarray
 *
 * would be represented as (pseudo code display here) :
 *
 * struct __pips_allocatable__2D myarray;
 *
 * with
 *
 * struct __pips_allocatable__2D {
 *  int lowerbound1;
 *  int upperbound1;
 *  int lowerbound2;
 *  int upperbound2;
 *  int data[lowerbound1:upperbound1][lowerbound2:upperbound2]
 * }
 *
 * The structure is dependent of the number of dimension and is created
 * dynamically when encounting an allocatable declaration.
 *
 * The prettyprint recognize the structure based on the special prefix and
 * display it as an allocatable array in Fortran95.
 *
 */

/**
 * @brief Helper for creating an allocatable structure. Here we create the
 * field corresponding to the data array.
 */
static entity make_data_field(basic b, const char *struct_name, list dimensions) {
  entity data;
  string name = strdup(concatenate(TOP_LEVEL_MODULE_NAME,
                                   MODULE_SEP_STRING,
                                   struct_name,
                                   MEMBER_SEP_STRING,
                                   "data",
                                   NULL));

  pips_assert("Trying to create data for an already existing struct ?",
      gen_find_tabulated( name, entity_domain ) == entity_undefined );

  type type = make_type_variable(make_variable(b, dimensions, NULL));

  storage rom = make_storage_rom();
  value initial = make_value_unknown();

  data = make_entity(name, type, rom, initial);

  return data;
}

static entity make_bound(const char *struct_name, const char *lname, int suffix) {
  entity bound;

  // Create the name
  string name;
  pips_assert("asprintf !",
      asprintf( &name,
          "%s" MEMBER_SEP_STRING "%s%d",
          struct_name,
          lname,
          suffix ));

  pips_assert("Trying to create lower bound but already existing ?",
      gen_find_tabulated( name, entity_domain ) == entity_undefined );

  bound = find_or_create_entity(concatenate(TOP_LEVEL_MODULE_NAME,
                                            MODULE_SEP_STRING,
                                            name,
                                            NULL));

  entity_type(bound) = make_type_variable(make_variable(make_basic_int(4),
                                                        NULL,
                                                        NULL));
  entity_storage(bound) = make_storage_rom();
  entity_initial(bound) = make_value_unknown();

  free(name);
  return bound;
}

/**
 * @brief This function try to find the allocatable structure corresponding to
 * the number of dimensions requested, and create it if necessary.
 */
entity find_or_create_allocatable_struct(basic b, int ndim) {
  printf("Creating allocatable struct for dim %d\n", ndim);

  // Create the entity name according to the number of dims
  string name;
  pips_assert("asprintf !", asprintf( &name, ALLOCATABLE_PREFIX "%dD", ndim));

  // Here is the internal PIPS name, there is a prefix for struct
  string prefixed_name = strdup(concatenate(STRUCT_PREFIX, name, NULL));

  // Let's try to localize the structure
  entity struct_entity = FindEntity(TOP_LEVEL_MODULE_NAME, name);

  // Localization failed, let's create it
  if(struct_entity == entity_undefined) {
    list fields = NULL;
    list dimensions = NULL;
    for (int dim = ndim; dim >= 1; dim--) {
      entity lower = make_bound(name, "lbound", dim);
      entity upper = make_bound(name, "ubound", dim);

      // Field for struct
      fields = CONS(ENTITY,lower,fields);
      fields = CONS(ENTITY,upper,fields);

      // Dimensions for the data array
      dimension d = make_dimension(make_expression_from_entity(lower),
                                   make_expression_from_entity(upper));
      dimensions = CONS(DIMENSION,d,dimensions );
    }

    // Create data holder
    fields = CONS(ENTITY,make_data_field(b, name, dimensions),fields);

    // Create the struct
    struct_entity = find_or_create_entity(concatenate(TOP_LEVEL_MODULE_NAME,
                                                      MODULE_SEP_STRING,
                                                      STRUCT_PREFIX,
                                                      name,
                                                      NULL));
    entity_type(struct_entity) = make_type_struct(fields);
    entity_storage(struct_entity) = make_storage_rom();
    entity_initial(struct_entity) = make_value_unknown();
  }

  free(prefixed_name);
  free(name);

  return struct_entity;
}

