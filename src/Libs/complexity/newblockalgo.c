complexity new_block_to_complexity(block, precond, effects_list)
list block;
transformer precond;
list effects_list;
{
    complexity comp = make_zero_complexity();
    /* initilize the complexity as zero complexity */

    Pbase final_var_comp = NIL;
    /* variables in final complexity result */

    list block_reverse = gen_nreverse(block);
    /* get reverse order, because we want to go over the complexity
       bottom-up */

    Pbase block_var_must = var_in_must_be_written(effects_list);
    /* Get the must_be_written variables from the block effects,
       which means that these variables should NOT be contained in
       the block complexity results */

    trace_on("block"); /* begining of trace */

    MAPL (pa, {
	statement stat = STATEMENT(CAR(pa)); /* find each stmt of the block */

	complexity ctemp = statement_to_complexity(stat, precond, effects_list);
	/* Get the un-evaluated complexity for the stmt */

	transformer prec = load_statement_precondition(stat);
	/* Get the precondition for the stmt */

	list cumu_list = load_statement_cumulated_effects(stat);
	/* Get the cumulated effects of the stmt */

	Pbase var_comp = var_in_comp(ctemp);
	/* Obtain all the eventual variables from the complexity result */

	Pbase var_must = var_in_must_be_written(cumu_list);
	/* Find from effects that which variable is must be written */

	for ( var in (var_comp and var_must) ) {
	    /* if var is in var_comp and is must_be_written variable */

	    complexity csubst = evaluate_var_to_complexity(var, cumu_list, prec);
	    /* Evaluate that variable to complexity */

	    if ( csubst is not unknown )
		complexity_var_subst(ctmp, var, csubst);
	    else
		complexity_var_subst(ctmp, var, UU_);
	}

	complexity_add(&comp, ctemp);
	/* add the complexity togather */
    }, block_reverse);

    complexity_check_and_warn("block_to_complexity", comp);    
    /* verification of the final complexity */

    trace_off(); /* end of trace */
    return(comp);
}



