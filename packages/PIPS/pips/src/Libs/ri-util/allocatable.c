/*

 $Id$

 Copyright 1989-2014 MINES ParisTech
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
 * @brief Check if an entity is an allocatable
 */
bool entity_allocatable_p(entity e) {
  type t = entity_type(e);
  if(!type_variable_p(t)) {
    return false;
  }
  variable v = type_variable(t);
  if(!basic_derived_p(variable_basic(v))) {
    return false;
  }
  entity allocatable_struct = basic_derived(variable_basic(v));

  if(!same_stringn_p(entity_local_name(allocatable_struct),
      STRUCT_PREFIX ALLOCATABLE_PREFIX,
      sizeof(STRUCT_PREFIX ALLOCATABLE_PREFIX)-1)) {
    return false;
  }

  return true;
}

/**
 * @brief Check if an expression is a reference to an allocatable array
 */
bool expression_allocatable_data_access_p(expression e) {
  // This must be a call
  if(!expression_call_p(e)) {
    return false;
  }

  entity field_call = call_function(expression_call(e));
  list args_list = call_arguments(expression_call(e));

  // This must be a call to "." and we must have args
  if(!ENTITY_FIELD_P(field_call) || ENDP(args_list)) {
    return false;
  }

  // Check that we deal with an allocatable
  expression allocatable_exp = CAR(args_list).e;
  entity allocatable =
      reference_variable(expression_reference(allocatable_exp));
  if(!entity_allocatable_p(allocatable)) {
    return false;
  } else {
    pips_assert("Allocatable shouldn't have any indices !",
        ENDP(reference_indices(expression_reference(allocatable_exp))));
  }

  // Check that it is the data field
  expression field_exp = CAR(CDR(args_list)).e;
  pips_assert("Allocatable field shouldn't have any indices !",
      ENDP(reference_indices(expression_reference(field_exp))));
  entity field = reference_variable(expression_reference(field_exp));
  if(same_stringn_p(entity_user_name(field),
      ALLOCATABLE_LBOUND_PREFIX,
      strlen(ALLOCATABLE_LBOUND_PREFIX))
      || same_stringn_p(entity_user_name(field),
          ALLOCATABLE_UBOUND_PREFIX,
          strlen(ALLOCATABLE_UBOUND_PREFIX))) {
    return false;
  }

  return true;
}

/**
 * @brief This function produce an expression that is an access to the array
 * inside the allocatable structure.
 */
expression get_allocatable_data_expr(entity e) {
  pips_assert("Entity isn't an allocatable", entity_allocatable_p(e));

  entity data_field = get_allocatable_data_entity(e);

  // Construct the expression e.data
  return MakeBinaryCall(CreateIntrinsic(FIELD_OPERATOR_NAME),
                        entity_to_expression(e),
                        entity_to_expression(data_field));

}

/**
 * @brief Get the entity inside the struct corresponding to the array,
 * mostly for correct prettyprint
 */
entity get_allocatable_data_entity(entity e) {
  pips_assert("Entity isn't an allocatable", entity_allocatable_p(e));

  // Get the data field inside the allocatable struct
  variable v = type_variable(entity_type(e));
  entity allocatable_struct = basic_derived(variable_basic(v));
  entity data_field = CAR(type_struct(entity_type(allocatable_struct))).e;
  return data_field;
}

/**
 * @brief Helper for creating an allocatable structure. Here we create the
 * field corresponding to the data array.
 */
static entity make_data_field(basic b,
                              const char *struct_name,
                              const char *name,
                              list dimensions) {

  string field ;
  asprintf(&field, "%s" MEMBER_SEP_STRING"%s", struct_name,name);

  pips_assert("Trying to create data for an already existing struct ?",
      FindEntity(TOP_LEVEL_MODULE_NAME,field) == entity_undefined );

  entity data = FindOrCreateTopLevelEntity(field);
  free(field);
  entity_type(data) = make_type_variable(make_variable(b, dimensions, NULL));
  entity_storage(data) = make_storage_rom();
  entity_initial(data) = make_value_unknown();

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

  bound = FindOrCreateTopLevelEntity(name);

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
 * @param name is the name of the array (prettyprint name)
 */
entity find_or_create_allocatable_struct(basic b, string name, int ndim) {
  printf("Creating allocatable struct for dim %d\n", ndim);

  // Create the entity name according to the number of dims
  string struct_name;
  string b_str = STRING(CAR(words_basic(b,NULL)));
  pips_assert("asprintf !",
      asprintf( &struct_name, ALLOCATABLE_PREFIX"%s_%s_%dD", name, b_str,ndim));

  // Here is the internal PIPS name, there is a prefix for struct
  string prefixed_name = strdup(concatenate(STRUCT_PREFIX, struct_name, NULL));

  // Let's try to localize the structure
  entity struct_entity = FindEntity(TOP_LEVEL_MODULE_NAME, prefixed_name);

  // Localization failed, let's create it
  if(struct_entity == entity_undefined) {
    list fields = NULL;
    list dimensions = NULL;
    for (int dim = ndim; dim >= 1; dim--) {
      entity lower = make_bound(struct_name, ALLOCATABLE_LBOUND_PREFIX, dim);
      entity upper = make_bound(struct_name, ALLOCATABLE_UBOUND_PREFIX, dim);

      // Field for struct
      fields = CONS(ENTITY,lower,fields);
      fields = CONS(ENTITY,upper,fields);

      // Dimensions for the data array
      dimension d = make_dimension(entity_to_expression(lower),
                                   entity_to_expression(upper));
      dimensions = CONS(DIMENSION,d,dimensions );
    }

    // Create data holder
    fields
        = CONS(ENTITY,make_data_field(b, struct_name, name, dimensions),fields);

    // Create the struct
    string field;
    asprintf(&field,STRUCT_PREFIX "%s",struct_name);
    struct_entity = FindOrCreateTopLevelEntity(field);
    free(field);
    entity_type(struct_entity) = make_type_struct(fields);
    entity_storage(struct_entity) = make_storage_rom();
    entity_initial(struct_entity) = make_value_unknown();
  }

  free(prefixed_name);
  free(struct_name);

  return struct_entity;
}

