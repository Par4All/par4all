/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

