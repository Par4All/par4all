#define PIPS_PHASE_PRELUDE(mn, debug_env_var)                   \
  (statement) db_get_memory_resource(DBR_CODE, mn, TRUE);       \
              set_current_module_statement(s);                  \
              entity mod = module_name_to_entity(mn);           \
              set_current_module_entity(mod);                   \
              debug_on(debug_env_var);                          \
              pips_debug(1, "Entering...\n");                   \
              pips_assert("Statement should be OK before...",   \
                          statement_consistent_p(s));
