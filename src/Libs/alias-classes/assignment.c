list points_to_call_sinks(statement s, expression rhs, expression lhs, set in)
{
  list sinks = NIL;
  entity e = call_function(c);
  cons* pc = call_arguments(c);
  tag tt;
  switch (tt = value_tag(entity_initial(e))) {
  case is_value_code:
    sinks = points_to_user_call_sinks(s, rhs, lhs, in);
    break;
  case is_value_symbolic:
    break;
  case is_value_constant:
    {
      if( expression_equal_integer_p(rhs, 0 ) )
	sinks = points_to_null_sinks();
    }
    break;
  case is_value_unknown:
    pips_internal_error("function %s has an unknown value\n",
                        entity_name(e));
    break;
  case is_value_intrinsic: 
    sinks = points_to_intrinsic_sinks(s, rhs, lhs, in);
    break;
  default:
    pips_internal_error("unknown tag %d\n", tt);
    break;
  }

return sinks
  }

list points_to_intrinsic_sinks(statement s, expression rhs, expression lhs, set in)
{
  list sinks = NIL;
  if( operator_expression_p(rhs) )
    sinks = points_to_operator_sinks(s, rhs, lhs, in);    
  else if (ENTITY_MALLOC_SYSTEM_P(expression_to_entity(rhs)) ||
	   ENTITY_CALLOC_SYSTEM_P(expression_to_entity(rhs)))
    sinks = points_to_malloc_sinks(s, rhs, lhs, in); 
  else 
    pips_user_error("Not implemented yet \n");

  return sinks;
}

list points_to_operator_sinks(statement s, expression rhs, expression lhs, set in)
{
if( assignment_expression_p(rhs) )
    {
    }
 else if( comma_expression_p(rhs) ) 
   {
     
   }
 else if( address_of_expression_p(rhs) )
   {
     
   }
 else if( operator_expression_p(rhs, POINT_TO_OPERATOR_NAME) )
  {
    
  }
 else if( expression_field_p(rhs) )
   {
     
   }
  else if( operator_expression_p(rhs, DEREFERENCING_OPERATOR_NAME) )
 {
   
 }
 else if( operator_expression_p(rhs, C_AND_OPERATOR_NAME) )
    {
    /* case && operator */
  }
  else if( operator_expression_p(rhs, C_OR_OPERATOR_NAME) )
 {
    /* case || operator */
  }
  else if( operator_expression_p(rhs, CONDITIONAL_OPERATOR_NAME) )
 {
   
 }
 return sinks;
}

list points_to_null_sinks();
{
  entity ne = entity_null_locations();
  reference nr = make_reference(ne, NIL);
  cell nc = make_cell_reference(nr);
  list sinks = CONS(CELL, nc, NIL);
  return sinks;
}

list points_to_reference_sinks(statement s, expression rhs, set in)
{
  list sinks = NIL;
  if(expression_reference_p(rhs)){
    if (array_argument_p(rhs)) {
      sinks = array_to_constant_paths(rhs, in);
    }
    else {
      reference r = expression_reference(rhs);
      entity e = reference_variable(r);
      /* scalar case, rhs is already a lvalue */
      /* add points-to relations in demand for global pointers */
      if( pointer_type_p(ultimate_type(entity_type(e)) )) {
	if( top_level_entity_p(e) || formal_parameter_p(e) ) {
	  cell nc = make_cell_reference(r);
	  if (!source_in_set_p(nc, in)) {
              set tmp = formal_points_to_parameter(nc);
              in = set_union(in, in, tmp);
              set_clear(tmp);
              set_free(tmp);
	  }
	}
	sinks = expression_to_constant_paths(s, rhs, in);
      }
    }
  }
  return sinks;
}

list points_to_user_call_sinks(statement s, expression rhs, expression lhs, set in)
{
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
  type t = entity_type(call_function(expression_call(rhs)));
  entity ne = entity_undefined;
  if(type_sensitive_p)
    ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
  else
    ne = entity_all_xxx_locations(ANYWHERE_LOCATION);
  
  reference nr = make_reference(ne, NIL);
  cell nc = make_cell_reference(nr);
  list sinks = CONS(CELL, nc, NIL);
  return sinks;
}

list points_to_cast_sinks(statement s, expression rhs, expression lhs, set in)
{
  expression nrhs = cast_expression(expression_cast(rhs));
  list sinks = points_to_expression_sinks(s, nrhs, lhs, in);
  return sinks;
}

list points_to_sizeofexpression_sinks(statement s, expression rhs, expression lhs, set in)
{
  list sinks = NIL;
  sizeofexpression soe = syntax_sizeofexpression(expression_syntax(rhs));
  if( sizeofexpression_expression_p(soe) ){
    expression ne = sizeofexpression_expression(soe);
    sinks = points_to_expression_sinks(s, ne, lhs, in);
  }
  return sinks;
}

list points_to_malloc_sinks(statement s, expression rhs, expression lhs, set in)
{
  expression sizeof_exp = EXPRESSION (CAR(call_arguments(expression_call(rhs))));
  type t = expression_to_type(lhs);
  reference nr = original_malloc_to_abstract_location(lhs,
						      t,
						      type_undefined,
						      sizeof_exp,
						      get_current_module_entity(),
						      s);
  cell nc = make_cell_reference(nr);
  list sinks  = CONS(CELL, nc, NIL);
  return sinks;
}

list points_to_application_sinks(statement s, expression rhs, expression lhs, set in)
{
  application a = syntax_application(expression_syntax(rhs));
  expression f = application_function(a);
  list args = application_arguments(a);
  type t = expression_to_type(f);
  entity ne = entity_undefined;
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
  pips_user_warning("Case application is not correctly handled\n");
  if(type_sensitive_p)
    ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
  else
    ne = entity_all_xxx_locations(ANYWHERE_LOCATION);

  reference nr = make_reference(ne, NIL);
  cell nc = make_cell_reference(nr);
  list sinks = CONS(CELL, nc, NIL);
  return sinks;

}


list points_to_va_arg_sinks(statement s, expression rhs, expression lhs, set in)
{
  list sinks = NIL;
  list l_varg = syntax_va_arg(expression_syntax(rhs));
  
  FOREACH(sizeofexpression, soe, l_varg) {
    expression ne = make_expression(make_syntax_sizeofexpression(soe),normalized_undefined);
    list soe_l = points_to_sizeofexpression_sinks(s, ne, lhs, in);
    sinks = gen_nconc(soe_l, sinks);
  }
  return sinks;

}


list points_to_subscript_sinks(expression rhs, set in)
{
  list sinks = NIL;
  pips_user_warning("Not implemented yet \n");
  return sinks;
}

list points_to_range_sinks(statement s, expression rhs, expression lhs, set in)
{
  list sinks = NIL;
  pips_user_warning("Not implemented yet \n");
  return sinks;
}

list points_to_expression_sinks(statement st, expression rhs, expression lhs, set in)
{
  /*reference + range + call + cast + sizeofexpression + subscript + application*/
  tag tt ;
  list sinks = NIL;
  syntax s = expression_syntax(rhs);
  switch (tt = syntax_tag(s)) {
  case is_syntax_reference:
    sinks = points_to_reference_sinks(st, rhs, in);
    break;
  case is_syntax_range:
    sinks = points_to_range_sinks(st, rhs, lhs, in);
    break;
  case  is_syntax_call:
    sinks = points_to_call_sinks(st, rhs, lhs, in);
    break;
  case  is_syntax_cast:
    sinks = points_to_cast_sinks(st, rhs, lhs, in);
    break;
  case  is_syntax_sizeofexpression:
    sinks = points_to_sizeofexpression_sinks(st, rhs, lhs, in);
    break;
  case  is_syntax_subscript:
    sinks = points_to_subscript_sinks(rhs, in);
    break;
  case  is_syntax_application:
    sinks = points_to_application_sinks(st, rhs, lhs, in);
  case  is_syntax_va_arg:
    sinks = points_to_va_arg_sinks(st, rhs, lhs, in);
    break;
  default:
    pips_internal_error("unknown tag %d\n", tt);
    break;
  }
  return sinks;
}




/* Using points-to "in", compute the new points-to set "out" for any
 *  assignment "lhs = rhs;" that meets one of Emami's patterns.
 *
 * If the assignment cannot be analyzed according to Emami's rules,
 * returns an empty set. So the assignment can be treated by
 *  points_to_general_assignment().
 *
 * To be able to apply Emami rules we have to test the lhs and the rhs:
 * are they references, fields of structs, &operator...
 *
 * "lhs" and "rhs" can be any one of Emami enum's fields.
 *
 * FI: 1) the default value could be set_undefined. 2) why do you need
 * a special function for special cases if you have a more general
 * function? 3) this function is much too long; if it is still useful,
 * it should be broken down into several functions, one for each
 * special Emami's case.
  */
set points_to_pointer_assignment(statment current,
				 expression lhs, 
				 expression rhs,
				 set in) 
{
  set cur = set_generic_make(set_private, points_to_equal_p,
                                points_to_rank);
  set incur = set_generic_make(set_private, points_to_equal_p,
                                points_to_rank);
  set in_may = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set in_must = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set kill_must = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set kill_may = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set gen_must = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set gen_may = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set out = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set kill = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set gen = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set out1 = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set out2 = set_generic_make(set_private, points_to_equal_p,
                           points_to_rank);
  set tmp = set_generic_make(set_private, points_to_equal_p,
                              points_to_rank);
  list L = NIL, R = NIL, args = NIL;
  bool address_of_p = false;
   
  /* Take into account the possible points-to side effects of
     expressions "rhs" and "lhs". E.g. "p = q = r;" */
  cur = points_to_expression(rhs, in, true);
  incur = points_to_expression(lhs, cur, true);
 
  /* Generate dummy targets for formal parameters and global
     variables. This is part of the input points-to context assumed
     for the current module. FI: I am not too sure I really understand
     what goes on here. */
  if( expression_reference_p(lhs) ) {
    entity e = expression_to_entity(lhs);
    if( !entity_undefined_p(e) && entity_variable_p(e)  ) {
      if(top_level_entity_p(e) || formal_parameter_p(e) ) {
        reference nr = make_reference(e, NIL);
        cell nc = make_cell_reference(nr);
        if (!source_in_set_p(nc, incur)) {
          tmp = formal_points_to_parameter(nc);
          incur = set_union(incur, incur, tmp);
          set_clear(tmp);
          set_free(tmp);
        }
      }
    }
  }
 
  /*Change the "lhs" into a contant memory path using points-to
    information "incur" */
  L = expression_to_constant_paths(current, lhs, incur);

  /* rhs should be a lvalue since we assign pointers; if not, we
     should transform it into a lvalue or call the adequate function
     according to its type */
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  /* Disjunction on the rhs */

  if( array_argument_p(rhs) ) {
    R = array_to_constant_paths(rhs, incur);
    if( !expression_pointer_p(rhs) )
      address_of_p = true;
  }
  else if(expression_reference_p(rhs)){
    if (array_argument_p(rhs)) {
      R = array_to_constant_paths(rhs, incur);
      if(!expression_pointer_p(rhs))
        address_of_p = true;
    }
    /* scalar case, rhs is already a lvalue */
    entity e = expression_to_entity(rhs);
    if( same_string_p(entity_local_name(e),"NULL") ) { 
      entity ne = entity_null_locations();
      reference nr = make_reference(ne, NIL);
      cell nc = make_cell_reference(nr);
      R = CONS(CELL, nc, NIL);
      address_of_p = true;
    }
    else if( ! entity_undefined_p(e) && entity_variable_p(e) ) {
      /* add points-to relations in demand for global pointers */
      if( entity_pointer_p(e) ) {
        if( top_level_entity_p(e) || formal_parameter_p(e) ) {
          reference nr = make_reference(e, NIL);
          cell nc = make_cell_reference(nr);
          if (!source_in_set_p(nc, incur)) {
              tmp = formal_points_to_parameter(nc);
              incur = set_union(incur, incur, tmp);
              set_clear(tmp);
              set_free(tmp);
          }
        }
      }
      R = expression_to_constant_paths(current, rhs, incur);
    }
  }
  else if ( expression_cast_p(rhs) )
 {
    expression nrhs = cast_expression(expression_cast(rhs));
    return points_to_assignment(current, lhs, nrhs, incur);
  }
  else if( expression_equal_integer_p(rhs, 0 ))
    {
    entity ne = entity_null_locations();
    reference nr = make_reference(ne, NIL);
    cell nc = make_cell_reference(nr);
    R = CONS(CELL, nc, NIL);
    address_of_p = true;
  }
  else if( assignment_expression_p(rhs) )
    {
    call c = expression_call(rhs);
    args = call_arguments(c);
    expression nrhs = EXPRESSION(CAR(args));
    return  points_to_assignment(current, lhs, nrhs, incur);
  }
  else if( comma_expression_p(rhs) )
    {
    /* comma case, lhs should point to the same location as the last
       pointer which appears into comma arguments*/
    call c = expression_call(rhs);
    args = call_arguments(c);
    expression nrhs = expression_undefined;
    FOREACH(expression, ex, args){
        incur = points_to_expression(ex, incur, true);
      nrhs = copy_expression(ex);
    }
    return  points_to_assignment(current, lhs, nrhs, incur);
  }
  else if( address_of_expression_p(rhs) )
 {
    /* case & opeator */
    call c = expression_call(rhs);
    args = call_arguments(c);
    expression nrhs = EXPRESSION(CAR(args));
   if( array_argument_p(nrhs) )
      R = array_to_constant_paths(nrhs,incur);
    else
      R = expression_to_constant_paths(current, nrhs,incur);
    address_of_p = true;
  }
  else if( subscript_expression_p(rhs) )
 {
    /* case [] */
    R = expression_to_constant_paths(current, rhs, incur);
  }
  else if( operator_expression_p(rhs, POINT_TO_OPERATOR_NAME) )
 {
    /* case -> operator */
   entity e = expression_to_entity(lhs);
   if( ! entity_undefined_p(e) && entity_variable_p(e)  ) {
        if(top_level_entity_p(e)|| formal_parameter_p(e) ) {
          reference nr = make_reference(e, NIL);
          cell nc = make_cell_reference(nr);
          if (!source_in_set_p(nc, incur)) {
            tmp = formal_points_to_parameter(nc);
            incur = set_union(incur, incur, tmp);
            set_clear(tmp);
            set_free(tmp);
          }
        }
    }
    R = expression_to_constant_paths(current, rhs, incur);
  }
  else if( expression_field_p(rhs) )
    {
      entity e = expression_to_entity(lhs);
      if( ! entity_undefined_p(e) && entity_variable_p(e)  ) {
        if(top_level_entity_p(e)|| formal_parameter_p(e) ) {
          reference nr = make_reference(e, NIL);
          cell nc = make_cell_reference(nr);
          if (!source_in_set_p(nc, incur)) {
            tmp = formal_points_to_parameter(nc);
            incur = set_union(incur, incur, tmp);
            set_clear(tmp);
            set_free(tmp);
          }
        }
    }
    R = expression_to_constant_paths(current, rhs,incur);
  }
  else if( operator_expression_p(rhs, DEREFERENCING_OPERATOR_NAME) )
 {
    R = expression_to_constant_paths(current, rhs,incur);
  }
  else if( operator_expression_p(rhs, C_AND_OPERATOR_NAME) )
    {
    /* case && operator */
  }
  else if( operator_expression_p(rhs, C_OR_OPERATOR_NAME) )
 {
    /* case || operator */
  }
  else if( operator_expression_p(rhs, CONDITIONAL_OPERATOR_NAME) )
 {
    /* case ? operator is similar to an if...else instruction */
    call c = expression_call(rhs);
    args = call_arguments(c);
    expression cond = EXPRESSION(CAR(args));
    expression arg1 = EXPRESSION(CAR(CDR(args)));
    expression arg2 = EXPRESSION(CAR(CDR(CDR(args))));
    incur = points_to_expression(cond, incur, true);
    out1 = points_to_assignment(current, lhs, arg1, incur);
    out2 = points_to_assignment(current,lhs, arg2, incur);
    return merge_points_to_set(out1, out2);
  }
  else if( expression_call_p(rhs) )
 {
    if(ENTITY_MALLOC_SYSTEM_P(expression_to_entity(rhs)) ||
       ENTITY_CALLOC_SYSTEM_P(expression_to_entity(rhs))){
      expression sizeof_exp = EXPRESSION (CAR(call_arguments(expression_call(rhs))));
      type t = expression_to_type(lhs);
      reference nr = original_malloc_to_abstract_location(lhs,
                                                          t,
                                                          type_undefined,
                                                          sizeof_exp,
                                                          get_current_module_entity(),
                                                          current);
      cell nc = make_cell_reference(nr);
      R = CONS(CELL, nc, NIL);
      address_of_p = true;
    }
   else if( user_function_call_p(rhs) )
     {
       type t = entity_type(call_function(expression_call(rhs)));
       entity ne = entity_undefined;
       if(type_sensitive_p)
	 ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
       else
	 ne = entity_all_xxx_locations(ANYWHERE_LOCATION);
       
       reference nr = make_reference(ne, NIL);
       cell nc = make_cell_reference(nr);
       R = CONS(CELL, nc, NIL);
       address_of_p = true;
     }
   else
     {
      type t = entity_type(call_function(expression_call(rhs)));
      entity ne = entity_undefined;
      if(type_sensitive_p)
         ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
      else
         ne = entity_all_xxx_locations(ANYWHERE_LOCATION);
      reference nr = make_reference(ne, NIL);
      cell nc = make_cell_reference(nr);
      R = CONS(CELL, nc, NIL);
      address_of_p = true;
    }
  }
  else{
    type t = expression_to_type(rhs);
    type lt = expression_to_type(lhs);
    /* Handle strut assignment m = n */
    type ct = compute_basic_concrete_type(t);
    type lct = compute_basic_concrete_type(lt);
    if(type_struct_p(ct) && type_struct(lct)) {
        list l1 = type_struct(ct);
        list l2 = type_struct(lct);

        FOREACH(ENTITY, i, l1) {
        if( expression_pointer_p(entity_to_expression(i)) )
          {
            entity ent2 = ENTITY (CAR(l2));
            expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
                                            lhs,
                                            entity_to_expression(i));
            expression ex2 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
                                            rhs,
                                            entity_to_expression(ent2));
            expression_consistent_p(ex1);
            expression_consistent_p(ex1);
            return points_to_assignment(current, ex1, ex2, incur);
          }
          l2 = CDR(l2);
      }      
        }
    else  {
    entity ne = entity_undefined;
    if( type_sensitive_p)
      ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
    else
      ne = entity_all_xxx_locations(ANYWHERE_LOCATION);
    reference nr = make_reference(ne, NIL);
    cell nc = make_cell_reference(nr);
    R = CONS(CELL, nc, NIL);
    address_of_p = true;
  }
  }
 
 /* Extract MAY/MUST points to relations from the input set "incur" */
  in_may = points_to_may_filter(incur);
  in_must = points_to_must_filter(incur);
  kill_may = kill_may_set(L, in_may);
  kill_must = kill_must_set(L, incur);
  gen_may = gen_may_set(L, R, in_may, &address_of_p);
  gen_must = gen_must_set(L, R, in_must, &address_of_p);
  set_union(kill, kill_may, kill_must);
  set_union(gen, gen_may, gen_must);
  if( set_empty_p(gen) ) 
    {
      if( type_sensitive_p )
        gen = points_to_anywhere_typed(L, incur);
    else
      gen = points_to_anywhere(L, incur); 
  }
  set_difference(incur, incur, kill);
  set_union(out, incur, gen);

  set_free(in_may);
  set_free(in_must);
  set_free(kill_may);
  set_free(kill_must);
  set_free(gen_may);
  set_free(gen_must);
  set_free(gen);
  set_free(kill);
  set_free(out1);
  set_free(out2);
  set_free(cur);
  set_clear(incur); // FI: why not free?
  return out;
}
