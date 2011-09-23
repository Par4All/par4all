computation_intensity_param p;
init_computation_intensity_param(&p);
gen_context_recurse(get_current_module_statement(),&p,
    statement_domain,do_computation_intensity,gen_null);
