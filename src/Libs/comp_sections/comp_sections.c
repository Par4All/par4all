/*{{{  includes*/
#include "all.h"
/*}}}*/

bool summary_complementary_sections(char *module_name)
{
    /*comp_global_regions(module_name);*/
    DB_PUT_MEMORY_RESOURCE(DBR_SUMMARY_COMPSEC,
                        strdup(module_name),
                      (char*) make_comp_desc_set(NIL));
    return(TRUE);
}

bool complementary_sections(char *module_name)
{
    comp_regions(module_name);
    return(TRUE);
}

bool print_code_complementary_sections(char *module_name)
{
    print_code_comp_regions(module_name);
    return(TRUE);
}

