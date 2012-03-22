/* This file contains unused functions initially designed to implement
 * Emami's algorithm.
 */


/* From input points-to pts_to_set, compute and return pt_out after
   the assignment "m.a = n.a;" where a is of type pointer.

   Parameter pts_to_set is modified by side effect and returned.

   m.a and n.a are assumed simple enough not to generate multiple
   write effects.

   AM: written for Emami's algorithm, currently unused, superseded by
   a more general function, points_to_assignment()
   
 */
set struct_double_pointer(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
                                    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
                                        points_to_rank);
  effect e1 = effect_undefined, e2 = effect_undefined;
  reference r1 = reference_undefined;
  reference r2 = reference_undefined;
  set s = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);

  /* Make sure the input meets the function precondition. Note that it
     is not fully checked: the check below is "m = n;" */
  pips_assert("lhs and rhs are references",
              expression_reference_p(lhs) && expression_reference_p(rhs));

  /* init the effect's engine. */
  set_methods_for_proper_simple_effects();
  /* Compute the lhs location in effect e1 */
  list l_ef = NIL;
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
                                                                 &l_ef, true);
  e1 = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
  gen_free_list(l_ef); /* free the spine */

  /* Compute the rhs location in effect e2 */
  l_ef = NIL;
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
                                                                 &l_ef, false);
  e2 = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
  gen_free_list(l_ef); /* free the spine */

  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();

  /* Build the cells needed for points-to information */
  r1 = effect_any_reference(e1);
  cell source = make_cell_reference(r1);
  r2 = effect_any_reference(e2);
  cell sink = make_cell_reference(r2);
    
  /* FI: I am lost. I do not understand how pt_in is used to interpret
     source and sink... */
  SET_FOREACH(points_to, i, pts_to_set){
    if(locations_equal_p(points_to_source(i), sink))
      s = set_add_element(s, s, (void*)i);
  }
    
  /* Compute the gen set */
  SET_FOREACH(points_to, j, s){
    cell new_sink = copy_cell(points_to_sink(j));
    approximation rel = points_to_approximation(j);
    points_to pt_to = make_points_to(source,
                                     new_sink,
                                     rel,
                                     make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
                                 (void*) pt_to );
  }
    
  /* Compute the kill set */
  SET_FOREACH(points_to, k, pts_to_set){
    if(locations_equal_p(points_to_source(k), source))
      written_pts_to = set_add_element(written_pts_to,
                                       written_pts_to, (void *)k);
  }

  /* Apply the dataflow equation: out = (in - kill) U gen */
  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);

  ifdebug(1)
    print_points_to_set("Points to for the case x is a double pointer to struct\n",
                        pts_to_set);

  set_clear(s);
  set_clear(gen_pts_to);
  set_clear(written_pts_to);
  set_free(s);
  set_free(gen_pts_to);
  set_free(written_pts_to);

  return pts_to_set;
}


/*  compute m.a = n.a where a is of pointer type
 *
 * FI: same comment as for previous function
 *
 * Code seems almost cut-and-pasted from previous function (or vice-versa)
 */
set struct_pointer(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
                                    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
                                        points_to_rank);
  effect e1 = effect_undefined, e2 = effect_undefined;
  reference r1 = reference_undefined;
  reference r2 = reference_undefined;
  list l_ef = NIL;

  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs, &l_ef,
                                                                 true);
  e1 = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
  gen_free_list(l_ef); /* free the spine */
  l_ef = NIL;
  list l2 = generic_proper_effects_of_complex_address_expression(rhs, &l_ef,
                                                                 false);
  e2 = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
  gen_free_list(l_ef); /* free the spine */

  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  r1 = effect_any_reference(e1);
  cell source = make_cell_reference(r1);
  // add the points_to relation to the set generated
  // by this assignement
  r2 = effect_any_reference(e2);
  cell sink = make_cell_reference(r2);
  set s = set_generic_make(set_private, points_to_equal_p, points_to_rank);

  SET_FOREACH(points_to, i, pts_to_set){
      if(locations_equal_p(points_to_source(i), sink))
        s = set_add_element(s, s, (void*)i);
    }
  
  SET_FOREACH(points_to, j, s){
      cell new_sink = copy_cell(points_to_sink(j));
      approximation rel = points_to_approximation(j);
      points_to pt_to = make_points_to(source,
                                       new_sink,
                                       rel,
                                       make_descriptor_none());

      gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
                                   (void*) pt_to );
    }

  SET_FOREACH(points_to, k, pts_to_set) {
    if(locations_equal_p(points_to_source(k), source))
      written_pts_to = set_add_element(written_pts_to,
                                       written_pts_to, (void *)k);
  }

  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
  ifdebug(1)
    print_points_to_set("Points to for the case <x = y>\n",
                        pts_to_set);
  set_clear(s);
  set_clear(gen_pts_to);
  set_clear(written_pts_to);
  set_free(s);
  set_free(gen_pts_to);
  set_free(written_pts_to);
  return pts_to_set;
}

/* Assuming in==pt_in, compute pt_out after the assignment "m = n;"
 * where both m and n have type struct.
 *
 * The result should be equivalent to the assignment of each pointer
 * field, "m.field1 = n.field1;..." A.M
 */
set struct_decomposition(expression lhs, expression rhs, set pt_in) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
                                points_to_rank);
  entity e1 = expression_to_entity(lhs);
  entity e2 = expression_to_entity(rhs);
  type t1 = entity_type(e1);
  type t2 = entity_type(e2);
  variable v1 = type_variable(t1);
  variable v2 = type_variable(t2);
  basic b1 = variable_basic(v1);
  basic b2 = variable_basic(v2);
  entity ent1 = basic_derived(b1);
  entity ent2 = basic_derived(b2);
  type tt1 = entity_type(ent1);
  type tt2 = entity_type(ent2);
  list l1 = type_struct(tt1);
  list l2 = type_struct(tt2);

  /* FI: Should not we assert that tt1==tt2 and hence l1==l2? */

  FOREACH(ENTITY, i, l1) {
    if( expression_pointer_p(entity_to_expression(i)) ) { // memory leak
      ent2 = ENTITY (CAR(l2));
      expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
                                      lhs,
                                      entity_to_expression(i));
      expression ex2 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
                                      rhs,
                                      entity_to_expression(ent2));
      expression_consistent_p(ex1);
      expression_consistent_p(ex1);
      pt_out = set_union(pt_out, pt_out,
                         struct_pointer(pt_in,
                                        ex1,
                                        ex2));
    }
    l2 = CDR(l2);
  }

  return pt_out;
}

/* FI: */
list points_to_init_pointer_to_typedef(entity e)
{
  list l = NIL;
  type t1 = ultimate_type(entity_type(e));
 
  if (type_variable_p(t1)) {
    basic b = variable_basic(type_variable(entity_type(e)));
    if(basic_pointer_p(b)){
      type t2 = basic_pointer(variable_basic(type_variable(t1)));
      if(typedef_type_p(t2)){
        basic b1 = variable_basic(type_variable(t2));
          if(basic_typedef_p(b1)){
            entity e2  = basic_typedef(b1);
            if(entity_variable_p(e2)){
              basic b2 = variable_basic(type_variable(entity_type(e2)));
              if(basic_derived_p(b2)){
                entity e3 = basic_derived(b2);
                l = points_to_init_pointer_to_derived(e, e3);
              }
            }
          }
        }
      else if(derived_type_p(t2)){
        entity e3 = basic_derived(variable_basic(type_variable(t2)));
        l = points_to_init_pointer_to_derived(e, e3);
      }
    }
  }

  return l;
}

/* FI: */
list points_to_init_pointer_to_derived(entity e, entity ee){
  list l = NIL;
  type tt = entity_type(ee);

  if(type_struct_p(tt))
   l = points_to_init_pointer_to_struct(e, ee);
  else if(type_union_p(tt))
    pips_user_warning("union case not handled yet \n");
  else if(type_enum_p(tt))  // FI: something to be done for enum?
    pips_user_warning("enum case not handled yet \n");
  return l;
}

/* FI: */
list  points_to_init_pointer_to_struct(entity e, entity ee)
{
  list l = NIL;
  bool  eval = true;
  type tt = entity_type(ee);
  expression ex = entity_to_expression(e);

  if(  type_struct_p(tt) ) {
    list l1 = type_struct(tt);
    if(  !array_argument_p(ex) ) {
      FOREACH( ENTITY, i, l1 ) {
        expression ef = entity_to_expression(i);
        if( expression_pointer_p(ef) ) {
          expression ex1 =
            MakeBinaryCall(entity_intrinsic(POINT_TO_OPERATOR_NAME),
                           ex,
                           ef);
          cell c = get_memory_path(ex1, &eval);
          l = gen_nconc(CONS(CELL, c, NIL),l);
        }
      }
    }
    else
      l = points_to_init_array_of_struct(e, ee);
  }

  return l;
}


/* FI: */
list points_to_init_typedef(entity e)
{
  list l = NIL;
  type t1 = entity_type(e);
    
  if(type_variable_p(t1)) { // FI: real test or assert?
    basic b = variable_basic(type_variable(entity_type(e)));
    tag tt = basic_tag(b);
    switch(tt){
    case is_basic_int:;
      break;
    case is_basic_float:;
      break;
    case is_basic_logical:;
      break;
    case is_basic_overloaded:;
      break;
    case is_basic_complex:;
      break;
    case is_basic_string:;
      break;
    case is_basic_bit:;
      break;
    case is_basic_pointer:;
      break;
    case is_basic_derived:
      {
      bool  eval = true;
      type t = entity_type(e);
      expression ex = entity_to_expression(e);
        if( type_struct_p(t) )
          {
        list l1 = type_struct(t);
            FOREACH( ENTITY, i, l1 ) {
          expression ef = entity_to_expression(i);
              if( expression_pointer_p(ef) )
                {
            expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
                                            ex,
                                            ef);
            cell c = get_memory_path(ex1, &eval);
            l = gen_nconc(CONS(CELL, c, NIL),l);
        }
        }
          }
        else if( type_enum_p(t) )
        pips_user_warning("enum case not handled yet \n");
        else if( type_union_p(t) )
        pips_user_warning("union cas not handled yet \nx");
    }
      break;
    case is_basic_typedef:
      {
        entity e1  = basic_typedef(b);
        type t1 = entity_type(e1);
        if( entity_variable_p(e1) )
          {
          basic b2 =  variable_basic(type_variable(t1));
          if( basic_derived_p(b2) )
            {
            entity e2  = basic_derived(b2);
            l = points_to_init_derived(e, e2);
          }
        }
      }
      break;
    default: pips_internal_error("unexpected tag %d\n", tt);
      break;
    }
  }
  return l;
}

