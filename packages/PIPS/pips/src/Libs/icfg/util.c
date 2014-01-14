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
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"
#include "properties.h"  /* get_bool_property */
#include "misc.h"
#include "database.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "resources.h"
#include "complexity_ri.h"
#include "pipsdbm.h"      /* DB_PUT_FILE_RESOURCE is defined there */
#include "text-util.h"
#include "icfg.h"

void safe_free_vertex(vertex ver, list l_of_vers)
{
    if (!gen_in_list_p(ver, l_of_vers))
        free_vertex(ver);
    return;
}

list safe_add_vertex_to_list(vertex ver, list l_of_vers)
{
    if (!gen_in_list_p(ver, l_of_vers))
        ADD_ELEMENT_TO_LIST(l_of_vers, VERTEX, ver);
    return l_of_vers;
}

list list_of_connected_nodes(vertex ver, list l_of_vers)
{
    MAP(SUCCESSOR, succ, {
        vertex ver_child = successor_vertex(succ);
	l_of_vers = safe_add_vertex_to_list(ver_child, l_of_vers);
	l_of_vers = list_of_connected_nodes(ver_child, l_of_vers);
    }, vertex_successors(ver));
    return l_of_vers;
}

static string convert_string_for_daVinci_graph (string s)
{
  string r;
  int l = strlen(s);
  if (l == 0) /* if empty, do nothing */
    return s;
  else if (*(s + l - 1) == '\n') {
    r = (string)malloc(l + 1);
    memset(r, 0, l + 1);
    if (strstr(s, "C               <") == s) {/* effects lines */
      r[0] = ' '; /* delete the comment of effects */
      strncpy(r + 1, s + 1, l - 2);
    } else {
      strncpy(r, s, l - 1);
    }
  } else {
    r = (string)malloc(l + 2);
    memset(r, 0, l + 2);
    if (strstr(s, "C               <") == s) {/* effects lines */
      r[0] = ' '; /* delete the comment of effects */
      strcpy(r + 1, s + 1);
    } else {
      strcpy(r, s);
    }
  }
  string ret =  concatenate (r, "\\n", NULL);
  free (r);
  return ret;
}

vertex get_vertex_by_string(const char* str_name, list l_of_vers)
{
    MAP(VERTEX, ver, {
        text ver_label = (text)vertex_vertex_label(ver);
	list sens = text_sentences(ver_label);
        sentence ver_first_sen = SENTENCE(CAR(sens));
	string ver_name = first_word_of_sentence(ver_first_sen);
	string ver_real_name = remove_newline_of_string(ver_name);
	if (same_string_p(ver_real_name, str_name) || same_string_p(ver_name, str_name)) {
	    free(ver_real_name);
	    return ver;
	} else
	    free(ver_real_name);
    }, l_of_vers);
    return vertex_undefined;
}


list safe_make_successor(vertex ver_parent, vertex ver_child, list l_of_vers)
{
    successor succ;
    sentence ver_child_first_sen = SENTENCE(CAR(text_sentences((text)vertex_vertex_label(ver_child))));
    string ver_child_name = first_word_of_sentence(ver_child_first_sen);

    /* search if it was made yet, do not make */
    MAP(SUCCESSOR, succ, {
        vertex ver = successor_vertex(succ);
	sentence ver_first_sen = SENTENCE(CAR(text_sentences((text)vertex_vertex_label(ver))));
	string ver_name = first_word_of_sentence(ver_first_sen);
	if (same_string_p(ver_name, ver_child_name)) return l_of_vers;
    }, vertex_successors(ver_parent));

    /* if it was not made, do it */
    succ = make_successor(NULL, ver_child);
    ADD_ELEMENT_TO_LIST(vertex_successors(ver_parent), SUCCESSOR, succ);
    /* add vertex parent to the list of vertex */
    l_of_vers = safe_add_vertex_to_list(ver_parent, l_of_vers);

    return l_of_vers;
}


void print_graph_of_text_to_daVinci(FILE * f_out, list l_of_vers)
{
    bool first_node_parent = true;
    fprintf(f_out, "[\n");

    FOREACH (VERTEX, ver_parent, l_of_vers) {
      bool first_node_child = true;
      text node_parent_text = (text)vertex_vertex_label(ver_parent);
      bool first_sentence = true;

      if (first_node_parent)
	first_node_parent = false;
      else
	fprintf(f_out, ",\n");

      FOREACH (SENTENCE, sen, text_sentences(node_parent_text)){
	string s = sentence_to_string(sen);
	if (first_sentence) {
	  string tmp = remove_newline_of_string (s);
	  fprintf(f_out, "l(\"%s\",n(\"\",[a(\"OBJECT\",\"", tmp);
	  first_sentence = false;
	  free (tmp);
	}
	if (strstr(s, CALL_MARK)) {
	  /*fprintf(f_out, convert_string_for_daVinci_graph(s + strlen(CALL_MARK)));*/
	} else {
	  fputs(convert_string_for_daVinci_graph(s), f_out);
	}
      }

      fprintf(f_out, "\")],[\n");

      FOREACH (SUCCESSOR, succ, vertex_successors(ver_parent)){
	vertex ver_child = successor_vertex(succ);
	text node_child_text = (text)vertex_vertex_label(ver_child);
	sentence node_child_sen = SENTENCE(CAR(text_sentences(node_child_text)));
	string node_name_child = remove_newline_of_string(first_word_of_sentence(node_child_sen));

	if (first_node_child)
	  first_node_child = false;
	else
	  fprintf(f_out, ",\n");
	fprintf(f_out, "  l(\"\",e(\"\",[],r(\"%s\")))", node_name_child);
	free(node_name_child);
      }

      fprintf(f_out, "]))");
    }

    fprintf(f_out, "\n]");

    return;
}

void print_graph_daVinci_from_starting_node(FILE * f_out, vertex start_ver)
{
    list l = NIL;
    ADD_ELEMENT_TO_LIST(l, VERTEX, start_ver);
    l = list_of_connected_nodes(start_ver, l);
    print_graph_of_text_to_daVinci(f_out, l);
    gen_free_list(l);
    return;
}

void print_marged_text_from_starting_node(FILE *fd, int margin, vertex start_ver, list l_of_vers)
{
    if (!vertex_undefined_p(start_ver)) {
        text txt = (text)vertex_vertex_label(start_ver);
	MAP(SENTENCE, sen, {
            string s = sentence_to_string(sen);
	    string call_mark = strstr(s, CALL_MARK);
	    if(call_mark) {
	        vertex ver_child = get_vertex_by_string(call_mark + strlen(CALL_MARK), l_of_vers);
		print_marged_text_from_starting_node(fd, margin + (call_mark - s), ver_child, l_of_vers);
	    } else if (strlen(s)) { /* if s in not empty, ok write out */
	      string tmp = remove_newline_of_string(s);
	      fprintf(fd, "%*s%s\n", margin, "", tmp);
	      free (tmp);
	    }
	}, text_sentences(txt));
    }
    return;
}

bool make_resource_from_starting_node
(const char* mod_name, string res_name, string file_ext, vertex start_ver, list l_of_vers, bool res_text_type)
{
    string filename, localfilename, dir;
    FILE *fd;
    
    localfilename = db_build_file_resource_name(res_name, mod_name, file_ext);
    dir = db_get_current_workspace_directory();
    filename = strdup(concatenate(dir, "/", localfilename, NULL));
    /*free(dir);*/ /* this line may cause a problem of bus errone*/

    fd = safe_fopen(filename, "w");
    if (!vertex_undefined_p(start_ver)) {
        if (res_text_type) {
	    print_marged_text_from_starting_node(fd, 0, start_ver, l_of_vers);
	    safe_fclose(fd, filename);
	    write_an_attachment_file(filename);
	} else {
	    print_graph_daVinci_from_starting_node(fd, start_ver);
	    safe_fclose(fd, filename);
	}
    }
    DB_PUT_FILE_RESOURCE(res_name, mod_name, localfilename);
    free(filename); /* this line must be after DB_PUT_FILE_RESOURCE for the reason of memory liberation */
    
    return true;
}

list /* of entity */ get_list_of_variable_to_filter() {
    list l = NIL;
    string variables_names = strdup(get_string_property("EFFECTS_FILTER_ON_VARIABLE"));
    string saved = variables_names, s;
    entity var = NULL;

    for (s = variables_names; *s; s++) {
        if (*s == ',') {
	  *s = '\0';
	  var = gen_find_tabulated(variables_names, entity_domain);
	  
	  if (!var || entity_undefined_p(var))
	      pips_user_warning("reference variable '%s' not found\n", variables_names);
	  else
	      ADD_ELEMENT_TO_LIST(l, ENTITY, var);
	  *s = ',';
	  variables_names = s+1;
	}
    }
	
    var = gen_find_tabulated(variables_names, entity_domain);
    
    if (!var || entity_undefined_p(var))
        pips_user_warning("reference variable '%s' not found\n", variables_names);
    else
        ADD_ELEMENT_TO_LIST(l, ENTITY, var);

    free(saved);
    saved = NULL;

    return l;
}

list /* of effect */ effects_filter(list l_effs, list l_ents)
{
    list l_flt = NIL;
    MAP(EFFECT, eff, {
        action ac = effect_action(eff);
	reference ref = effect_any_reference(eff);
	entity ent = reference_variable(ref);
	MAP(ENTITY, e_flt, {
	    if (entities_may_conflict_p(e_flt, ent)) {
	        bool found = false;
	        switch(get_int_property(RW_FILTERED_EFFECTS)) {
		case READ_ALL:
		case READ_END:
		    if (action_read_p(ac)) {
		        ADD_ELEMENT_TO_LIST(l_flt, EFFECT, eff);
		        found = true;
		    }
		    break;
		case WRITE_ALL:
		case WRITE_END:
		    if (!action_read_p(ac)) {
		        ADD_ELEMENT_TO_LIST(l_flt, EFFECT, eff);
			found = true;
		    }
		    break;
		case READWRITE_ALL:
		case READWRITE_END:
		default:
		    ADD_ELEMENT_TO_LIST(l_flt, EFFECT, eff);
		    found = true;
		    break;
		}
		if (found) break;
	    }
	}, l_ents);
    }, l_effs);
    return l_flt;
}









