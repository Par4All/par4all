#include "safescale.h"


/**
   Analyze a given module
 */
bool safescale_module_analysis(string module_name)
{
  statement module_statement;
  entity module;  

  /* Initialize the resources */
  module_statement = (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);
  // module = local_name_to_top_level_entity(module_name);
  module = module_name_to_entity(module_name);  

  set_current_module_statement(module_statement);
  set_current_module_entity(module);
  
  /* Get effects of the module */
  set_cumulated_rw_effects((statement_effects) db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));

  /* Build hash tables between variables and values and between values and names for the module */
  module_to_value_mappings(module);
  
  //debug_on("PHRASE_DISTRIBUTOR_DEBUG_LEVEL");

  /* Get regions of the module */
  set_rw_effects((statement_effects) db_get_memory_resource(DBR_REGIONS, module_name, TRUE));
  set_in_effects((statement_effects) db_get_memory_resource(DBR_IN_REGIONS, module_name, TRUE));
  set_out_effects((statement_effects) db_get_memory_resource(DBR_OUT_REGIONS, module_name, TRUE));

  /* Do the job */
  //pips_debug(2, "BEGIN of PHRASE_DISTRIBUTOR\n");

  // convertion_to_kaapi(....); ou identification des blocs délimités par des pragmas selon méthode retenue.

  //pips_debug(2, "END of PHRASE_DISTRIBUTOR\n");

  print_statement(module_statement);

  //pips_assert("Statement structure is consistent after PHRASE_DISTRIBUTOR", gen_consistent_p((gen_chunk*) statement));
  //pips_assert("Statement is consistent after PHRASE_DISTRIBUTOR", statement_consistent_p(statement));
  
  /* Update/release resources */
  reset_current_module_statement();
  reset_current_module_entity();
  //dynamic_area = entity_undefined;
  reset_cumulated_rw_effects();
  reset_rw_effects();
  reset_in_effects();
  reset_out_effects();
  free_value_mappings();
  
  //debug_off();
  
  return TRUE;
}
