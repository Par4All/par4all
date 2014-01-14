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
/* static bool rule_multi_produced_consistent_p(rule mp_rule, makefile make_file)
 * input    : a rule that produces more than one resource and a the current makefile
 * output   : true if the others rules of the make file that produce at least
 *            one of the resource that mp_rule produces, produce exactly the
 *            same resources, or if no other rule produces the same resources. 
 *            false otherwise.
 * modifies : nothing
 * comment  :	
 */
static bool rule_multi_produced_consistent_p(mp_rule, make_file)
rule mp_rule;
makefile make_file;
{
    static bool rule_produced_consistent_p(rule rule_1, rule rule_2);
    list all_rules = makefile_rules(make_file);
    rule c_rule = rule_undefined;
    bool consistent = true;
    
    
    while (consistent && !ENDP(all_rules)) {
	c_rule = RULE(CAR(all_rules));

	if ( (c_rule != mp_rule)
	    && !rule_produced_consistent_p(mp_rule, c_rule))
	    consistent = false;
	all_rules = CDR(all_rules);
    }

    return(consistent);

}


/* static bool rule_produced_consistent_p(rule rule_1, rule_2)
 * input    : two rules
 * output   : true if they produce exactly the same resources, or if
 *            they produce no common resource. false otherwise.
 * modifies : nothing.
 * comment  :	
 */
static bool rule_produced_consistent_p(rule_1, rule_2)
rule rule_1, rule_2;
{
    list l_prod1 = rule_produced(rule_1);
    list l_prod2 = rule_produced(rule_2);
    list l2;
    bool consistent = true;
    bool first_common = true;
    bool first = true;
    bool same_length = (gen_length(l_prod1) == gen_length(l_prod2));
    string vr1, vr2;

    while (consistent && !ENDP(l_prod1)) {
	
	bool found = false;
	vr1 = virtual_resource_name(VIRTUAL_RESOURCE(CAR(l_prod1)));
	l2 = l_prod2;

	while (!found && !ENDP(l2)) {
	    vr2 = virtual_resource_name(VIRTUAL_RESOURCE(CAR(l2)));
	    if (same_string_p(vr1,vr2))
		found = true;
	    else
		l2 = CDR(l2);
	}

	
	if (first && !found )  first_common = false;
	if (first) first = false;
 
	consistent = (first_common && found && same_length) || (!first_common && !found);

	l_prod1 = CDR(l_prod1);
    }
    return(consistent);
}


