#define PIPS_PHASE_POSTLUDE(s)                      \
  pips_assert("Statement should be OK after...",    \
              statement_consistent_p(s));           \
  pips_debug(1, "done\n"); debug_off();             \
  DB_PUT_MEMORY_RESOURCE(DBR_CODE,                  \
                         get_current_module_name(), \
                         s);                        \
  reset_current_module_statement();                 \
  reset_current_module_entity();                    \
  return TRUE;
