statement_mapping contexts_mapping_of_nest(stat, ce_map)
statement stat;
statement_mapping ce_map;
{
    pips_assert(contexts_mapping_of_nest, statement_loop_p(stat));

    ifdebug(5)  {
	STATEMENT_MAPPING_MAP(st, context, {
	    statement stp = (statement) st;

	    if (statement_call_p(stp)) {
		fprintf(stderr, "Execution context of statement %d :\n", 
			statement_number(stp));
		sc_fprint(stderr, (Psysteme) context, entity_local_name);
	    }
	}, contexts_map);
    }

    return(contexts_map);
}
