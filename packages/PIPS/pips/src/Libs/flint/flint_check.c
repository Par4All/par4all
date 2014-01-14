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
/*
 * flint_check.c
 *
 * Fabien Coelho & Laurent Aniort May 1992
 *
 * Modification : 92 09 Author       : Arnauld Leservot
 *
 */

#include "local.h"

#define FATAL(msg,value) {(void) fprintf(stderr,msg,value); exit(1); }

/* some defines usefull to clean the code */

#define BOOL_UNDEF -1
#define param_ref_p(P) \
  (mode_reference_p(parameter_mode(P)))

#define arg_const_p(arg)                           \
  (syntax_call_p(expression_syntax(arg)) &&      \
     value_constant_p(entity_initial(call_function( \
	   syntax_call(expression_syntax(arg))))))

#define call_constant_p(C) \
  (value_constant_p(entity_initial(call_function(C))))

/* The following define may be replaced by #define ... (true) */
#define effect_may_or_must_p(my_effect) \
  ((approximation_may_p(effect_approximation(my_effect))) || \
   (approximation_exact_p(effect_approximation(my_effect))) )

#define effect_to_name(the_effect)\
  entity_name(reference_variable(effect_any_reference(the_effect)))

#define entity_is_a_common_p(Ent)               \
  (type_area_p(entity_type(Ent)))

#define special_common(Ent)                                     \
  ((!strcmp(entity_local_name(Ent),DYNAMIC_AREA_LOCAL_NAME)) || \
   (!strcmp(entity_local_name(Ent),STATIC_AREA_LOCAL_NAME))   )


/***************************************************************************/

/* extern char    *current_module_name; unused and modified */

/*
 * check call sites
 *
 */


/*
 * check_procedure
 *
 * this function verify that a statement_call is a subroutine call. if not, a
 * message is broadcast. intrinsics are not checked. Calls to things that are
 * not functions are not checked, (for instance, there are calls to labels
 * which describe the format of a write or a read; that's quite strange!)
 */
bool check_procedure(c)
    call            c;
{
    entity          the_fnct;
    type            the_tp;
    functional      ft;
    type            result;
    bool            tmpbool= BOOL_UNDEF;

    if (call_intrinsic_p(c))
	tmpbool = BOOL_UNDEF;
    else {
	the_fnct = call_function(c);
	the_tp = entity_type(the_fnct);
	if (!type_functional_p(the_tp))
	    tmpbool = BOOL_UNDEF;
	else {
	    ft = type_functional(the_tp);
	    result = functional_result(ft);
	    if (!type_void_p(result)) {
		flint_message("check procedure",
			      "warning, function used as a procedure : %s\n",
			      entity_name(the_fnct));
		tmpbool = false;
	    }
	}
    }
    return (tmpbool);
}

/***************************************************************************/

/*
 * check_call
 *
 * check other calls : number of arguments, the basic of these args, and if
 * possible the dimensions. There is also a warning if a reference to a
 * constant may be modified. Calls to intrinsics are checked separately.
 * Calls to things that are not functions are not checked, (for instance,
 * there are calls to labels which describe the format of a write or a read;
 * that's quite strange!)
 */
bool check_the_call(c)
    call            c;
{
    list            la = call_arguments(c);
    entity          the_fnct = call_function(c);
    type            the_tp = entity_type(the_fnct);
    functional      ft;
    list            lt;

    if (!type_functional_p(the_tp))
	return (BOOL_UNDEF);

    ft = type_functional(the_tp);
    lt = functional_parameters(ft);


    if (!check_call_args_number(la, lt, c))
	return (false);

    /* else */
    if (call_intrinsic_p(c))
	return (check_call_intrinsic(la, lt, c));

    if ((int) gen_length(la) == 0)
	return (true);

    /* else */
    if (!check_call_types_compatibility(la, lt, c))
	return (false);

    /* else */
    if (call_constant_p(c))
	return (true);

    /* else */
    /* Errors in parameter modes are found out by effect computation.
     * A second check, later, is meaningless.
     */
    /* return (check_call_mode_consistency(la, lt, the_fnct)); */

    return true;
}


/*
 * check_call_intrinsic
 *
 * This function is dedicated to the check of calls to intrinsics. Only the
 * assignment is checked: Same basic and dimension for both arguments.
 * problem :there is no casting, so there may be messages despite the call
 * should be considered as ok. overloaded basics are not checked, and no
 * message is broadcast.
 */
bool
check_call_intrinsic(list la,
		     list __attribute__ ((unused)) lt,
		     call c)
{
    entity          the_fnct = call_function(c);
    bool            ok1, ok2;
    basic           ba1, ba2;
    list            da1, da2;

    /* check a call to the assign operator */
    if (ENTITY_ASSIGN_P(the_fnct)) {
	ok1 = find_bd_expression(EXPRESSION(CAR(la)), &ba1, &da1);
	ok2 = find_bd_expression(EXPRESSION(CAR(CDR(la))), &ba2, &da2);

	if (!(ok1 && ok2))
	    return (BOOL_UNDEF);

	if (basic_overloaded_p(ba1) || basic_overloaded_p(ba2))
	    return (BOOL_UNDEF);

	if (!check_call_basic(ba2, ba1, c, 0))
	    return (false);

	return (check_call_dim(da1, da2, c, 0) && check_call_dim(da2, da1, c, 0));
    }
    /* other intrinsics */
    return (BOOL_UNDEF);
}



/*
 * check_call_args_number
 *
 * This function check that the number of arguments of a call is valid.
 * intrinsics without parameter are not checked. (they are supposed to be
 * varryings)
 */
bool check_call_args_number(
    list            la, /* list of actual arguments */
    list            lt, /* list of parameters */
    call            c)
{
    int             na = gen_length(la);
    int             nt = gen_length(lt);

    if (na == nt ||
	(nt<=na && type_varargs_p(parameter_type(PARAMETER(CAR(gen_last(lt)))))))
	return (true);

    if (call_intrinsic_p(c) && (nt == 0)) {	/* sometimes out... */
	return (BOOL_UNDEF);
    }
    flint_message("check call",
		  "too %s arguments (%d) in call to %s (%d)\n",
		  ((na > nt) ? "many" : "few"),
		  na,
		  entity_name(call_function(c)),
		  nt);
    return (false);
}


/*
 * check_call_types_compatibility
 *
 * This function checks that the list of parameters and the list of arguments
 * are compatible.
 */
bool
check_call_types_compatibility(la, lt, c)
    list            la, lt;
    call            c;
{
    expression      exp;
    parameter       param;
    bool            ok = true;
    int             i, len = gen_length(lt);
    list            ca = la, ct = lt;

    for (i = 1; i <= len; i++) {
	bool            temp;
	exp = EXPRESSION(CAR(ca));
	POP(ca);
	param = PARAMETER(CAR(ct));
	POP(ct);
	temp = check_call_one_type(exp, param, c, i);
	ok = (ok && temp);
    }

    return (ok);
}

/*-----------------------------------------------------------------------*/

/*
 * check_call_one_type
 *
 * this function checks that an argument and a parameter are compatible. It is
 * not very interesting a function, but it may have other calls to
 * check-functions later
 *
 */
bool check_call_one_type(exp, param, c, i)
    expression      exp;
    parameter       param;
    call            c;
    int             i;
{
    return (check_call_basic_and_dim(exp, param, c, i));
}


/*-----------------------------------------------------------------------*/

/*
 * check_call_basic
 *
 * This function checks that two basics are compatible (ie the same) if not, a
 * message is broadcast
 */
bool check_call_basic(be, bp, c, i)
    basic           be, bp;
    call            c;
    int             i;
{
    if (basic_tag(be) == basic_tag(bp))
	return (true);

    if (basic_overloaded_p(be))
	flint_message("check_call: WARNING",
		      "may be incompatible basic type, %dth arg in call to %s\n",
		      i, entity_name(call_function(c)));
    else
	flint_message("check_call",
		      "incompatible basic type, %dth arg in call to %s, %s>%s\n",
		      i, entity_name(call_function(c)),
		      basic_to_string(be), basic_to_string(bp));
    return (false);
}


/*-----------------------------------------------------------------------*/
/* loose version */

/*
 * check_call_dim
 *
 * This function checks that the dimensions of two arrays are compatible. if
 * not, a message... (dimension means here the number of elements of the
 * array)
 */
bool check_call_dim(list de, list dp, call c, int i)
{
  intptr_t n_de, n_dp;
    bool
	ok_de = number_of_elements(de, &n_de),
	ok_dp = number_of_elements(dp, &n_dp);

    if (!(ok_de && ok_dp))
	return (BOOL_UNDEF);

    /* else */
    if (n_de >= n_dp)
	return (true);

    /* else */
    flint_message("check_call",
		  "incompatible dim, %dth arg in call to %s, (%d<%d)\n",
		  i, entity_name(call_function(c)),
		  n_de, n_dp);
    return (false);
}


/*-----------------------------------------------------------------------*/

/*
 * check_call_basic_and_dim
 *
 * This function checks that the list of parameters and the list of arguments
 * are compatible. ie same basics and compatible dimensions.
 */
bool check_call_basic_and_dim(exp, param, c, i)
    expression      exp;
    parameter       param;
    call            c;
    int             i;
{
    basic           bexp, bpar;
    list            dexp, dpar;
    bool            okexp = find_bd_expression(exp, &bexp, &dexp), okpar = find_bd_parameter(param, &bpar, &dpar);

    if (!(okexp && okpar))
	return (BOOL_UNDEF);

    /* else */
    if (!check_call_basic(bexp, bpar, c, i))
	return (false);

    /* else */
    return (check_call_dim(dexp, dpar, c, i));
}

/***************************************************************************/

/*
 * check_reference
 *
 * this function checks that the indexes of an array are all mere integers.
 * maybe it could accept floats with a cast, if the ANSI says so.
 */
void check_the_reference(ref)
    reference       ref;
{
    list            the_indices = reference_indices(ref);
    int             i, len = gen_length(the_indices);
    basic           base;
    list            dims;
    entity          var = reference_variable(ref);
    type            tp = entity_type(var);
    int             len_ind = gen_length(the_indices), len_dim;
    bool            ok;

    ok = find_bd_type_variable(tp, &base, &dims);
    if (!ok)
	return;

    len_dim = gen_length(dims);

    if (len_dim < len_ind) {
	flint_message("reference",
		      "too many indices (%d>%d) in reference to %s\n",
		      len_ind, len_dim, entity_local_name(var));
	return;
    }

    if (len_dim > len_ind) {
	flint_message("reference",
		      "too few indices (%d<%d) in reference to %s\n",
		      len_ind, len_dim, entity_local_name(var));
	return;
    }

    if (basic_overloaded_p(base))
	return;

    for (i = 1; i <= len; i++) {
	if (control_type_in_expression(is_basic_overloaded, 1,
				       EXPRESSION(CAR(the_indices))))
	    flint_message("check reference: WARNING",
			  "the %dth index may not be a mere integer\n", i);
	else
	    if (!control_type_in_expression(is_basic_int, 1,
					    EXPRESSION(CAR(the_indices))))
		flint_message("check reference",
			  "the %dth index is not a mere integer\n", i);

	the_indices = CDR(the_indices);
    }
}

/***************************************************************************/

/*
 * check_call_mode_consistency
 *
 * this function checks all the arguments of a call, looking for so called "mode
 * inconsistency".
 */
bool check_call_mode_consistency(la, lt, the_fnct)
    list            la, lt;
    entity          the_fnct;
{
	const char*module_name;
    list
	sefs_list = list_undefined;
    expression
	exp;
    parameter
	param;
    bool
	ok = true,
	temp = -1;
    int
	i, len = gen_length(lt);

    module_name = module_local_name(the_fnct);

    /* FI: the last argument should be pure=TRUE
     * since summary effects should not be touched
     */
    sefs_list = effects_to_list( (effects)
	db_get_memory_resource(DBR_SUMMARY_EFFECTS, module_name, true));

    pips_debug(7, "summary effects list for %s (%p)\n",
	       module_name, sefs_list);

    for (i = 1; i <= len; i++) {
	exp = EXPRESSION(CAR(la));
	la = CDR(la);
	param = PARAMETER(CAR(lt));
	lt = CDR(lt);
	temp = check_call_one_mode(exp, param, the_fnct, sefs_list, i);
	ok = (ok && temp);
    }

    return (ok);
}

/***************************************************************************/

/* This function checks that a reference to a constant in a call may not be
 * modified, if it could happen, a message is broadcast.
 */
bool check_call_one_mode(expression exp,
			 parameter param,
			 entity the_fnct,
			 list sefs_list,
			 int i)
{
  list            sefl = sefs_list;	/* locally */
  bool            encountered = false;
  effect          the_effect;
  entity          the_ent;

  if (!(param_ref_p(param) && arg_const_p(exp)))
    return (true);

  /* else : control */

  the_ent = find_ith_formal_parameter(the_fnct, i);

  while ((sefl != NULL) && (!encountered))
    {
      the_effect = EFFECT(CAR(sefl));
      sefl = CDR(sefl);
      encountered = (effect_write_p(the_effect) &&
		     effect_may_or_must_p(the_effect) &&
		     (!strcmp(entity_name(the_ent), effect_to_name(the_effect))));
    }

  if (encountered)
    flint_message("check call mode",
		  "modified reference to a constant, %dth argument in call to %s\n",
		  i, entity_name(the_fnct));

  return (!encountered);
}

/*-----------------------------------------------------------------------*/

/*
 * look_at_the_commons
 *
 * this function looks at the common declaration, just to see what it looks
 * like, in order to decide what we can do about it. it should not be
 * necessary if good and up to date documentations were provided.
 */
bool look_at_the_commons(entity module)
{
    list            ldecl = code_declarations(value_code(entity_initial(module)));
    entity          local;

    while (ldecl != NULL) {
	local = ENTITY(CAR(ldecl));
	ldecl = CDR(ldecl);

	if (type_area_p(entity_type(local))) {
	    fprintf(stdout,
		    "common : %s, in module %s\n",
		    entity_name(local),
		    entity_name(module));
	    if (!special_common(local)) {
		list            llayout = area_layout(type_area(entity_type(local)));
		while (llayout != NULL) {

		    fprintf(stdout, "variable %s at offset %td\n",
			    entity_name(ENTITY(CAR(llayout))),
			    ram_offset(storage_ram(entity_storage(ENTITY(CAR(llayout))))));
		    llayout = CDR(llayout);
		}
	    }
	}
    }
    return (true);
}

/*-----------------------------------------------------------------------*/

/*
 * position_in_the_area
 *
 * this function gives back, if possible, the starting and ending offsets of the
 * variable in the area
 */
bool position_in_the_area(entity the_var, intptr_t *inf, intptr_t *sup)
{
    basic           base;
    list            dims;
    intptr_t             len_unit = 0;
    intptr_t             nb_of_elements = 0;

    if (!find_bd_type_variable(entity_type(the_var), &base, &dims))
	return (false);
    if (!number_of_elements(dims, &nb_of_elements))
	return (false);
    if (!(basic_int_p(base) || basic_float_p(base) ||
	  basic_logical_p(base) || basic_complex_p(base)))
	return (false);

    switch (basic_tag(base)) {
    case is_basic_int:{
	    len_unit = basic_int(base);
	    break;
	}
    case is_basic_float:{
	    len_unit = basic_float(base);
	    break;
	}
    case is_basic_logical:{
	    len_unit = basic_logical(base);
	    break;
	}
    case is_basic_complex:{
	    len_unit = basic_complex(base);
	    break;
	}
    default:
	pips_internal_error("unknown basic tag %d", basic_tag(base));
    }

    *inf = ram_offset(storage_ram(entity_storage(the_var)));
    *sup = (*inf) + (nb_of_elements * len_unit) - 1;

    return (true);
}

/*-----------------------------------------------------------------------*/

/*
 * check_commons
 *
 * this function checks the commons of the module, looking for all variables of
 * the module in commons, that could overlap with other variables and may be
 * incompatible. the bool given back is set to true whatever happens.
 * Special Commons used by PIPS, (dynamic and static) are not checked.
 */
bool check_commons(entity module)
{
    list
	ldecl = code_declarations(value_code(entity_initial(module)));
    entity
	local;

    while (ldecl != NULL)
    {
	local = ENTITY(CAR(ldecl));
	ldecl = CDR(ldecl);

	if (entity_is_a_common_p(local) && (!special_common(local)))
	    check_one_common(local, module);
    }

    return (true);
}

/*
 * check_one_common
 *
 * This function checks one given common in one given module. for each argument
 * belonging to the common [while llayout], every other arguments belonging
 * to another [while lothers] declaration of the common is checked to find if
 * there is a possible overlap [call to check_overlap_in_common]. possible
 * synonymous variable within the same module and common are also checked.
 */
void check_one_common(entity local, entity module)
{
    list
	llayout_init = area_layout(type_area(entity_type(local))),
	llayout = llayout_init;

    pips_debug(4, "checking common %s of module %s\n",
	       entity_local_name(local),
	       module_local_name(module));

    while (!ENDP(llayout))
    {
	intptr_t
	    common_beginning_offset = -1,
	    common_ending_offset = -1,
	    other_beginning_offset = -1,
	    other_ending_offset = -1;
	entity
	    current_variable = ENTITY(CAR(llayout));
	list
	  lothers = llayout_init;

	llayout = CDR(llayout);

	if (strcmp(entity_module_name(current_variable),
		   module_local_name(module)))
	    continue;

	if (!position_in_the_area(current_variable,
				  &common_beginning_offset,
				  &common_ending_offset))
	    continue;

	while (lothers != NULL)
	{
	    entity          head = ENTITY(CAR(lothers));
	    int             synonymous = 0;
	    lothers = CDR(lothers);

	    if (!strcmp(entity_name(current_variable), entity_name(head)))
	    {
		if (synonymous)
		    flint_message("check common",
				  "%s used twice in common %s\n",
				  entity_name(current_variable),
				  module_local_name(local));
		synonymous++;
		continue;
	    }
	    if (!position_in_the_area(head,
				      &other_beginning_offset,
				      &other_ending_offset))
		continue;

	    check_overlap_in_common(local,
				    current_variable,
				    common_beginning_offset,
				    common_ending_offset,
				    head,
				    other_beginning_offset,
				    other_ending_offset);
	}
    }
}

/*-----------------------------------------------------------------------*/

/*
 * check_overlap_in_common
 *
 * This function is used to detect possible overlap between two variables of the
 * same common in two declarations, with incompatible basics (ANSI says that
 * a compiler must allows that one complex is also two floats) if so, a
 * message is broadcast.
 */
bool check_overlap_in_common(the_common, e1, inf1, sup1, e2, inf2, sup2)
    entity          the_common;
    entity          e1;
    int             inf1, sup1;
    entity          e2;
    int             inf2, sup2;
{
    basic           b1, b2;
    bool            ok1, ok2;
    list            l1, l2;	/* unused */

    /* testing overlap */
    if ((sup1 < inf2) || (sup2 < inf1))
	return (true);

    /* else I must check basic compatibility */
    ok1 = find_bd_type_variable(entity_type(e1), &b1, &l1);
    ok2 = find_bd_type_variable(entity_type(e2), &b2, &l2);

    if (!(ok1 && ok2))
	return (true);		/* benefice du doute! */

    if (basic_tag(b1) == basic_tag(b2))
	return (true);

    if ((basic_float_p(b1) && (basic_complex_p(b2))) ||
	(basic_float_p(b2) && (basic_complex_p(b1))))
	return (true);

    flint_message_2("check common",
		"overlap of incompatible variables (%s, %s) in common %s\n",
		    entity_name(e1),
		    entity_name(e2),
		    module_local_name(the_common));
    return (false);
}


