#include "local.h"

#define ADD_ELEMENT_TO_LIST( _list, _type, _element) \
    (_list = gen_nconc( _list, CONS( _type, _element, NIL)))

vertex get_vertex_in_list(list in_l, string in_s);
void print_graph_text(FILE * out_f, graph in_gr);
void print_graph_daVinci(FILE * out_f, graph in_gr);
bool entity_in_list_pattern(list in_l, entity ent);

void
variable_trace(string mod_name)
{
  cons *pv1, *ps, *pc;
  statement mod_stat = get_current_module_statement();
  graph mod_graph = (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);
  string local_dg_name = db_build_file_resource_name(DBR_DG, mod_name, ".dg");
  string dg_name = strdup(concatenate(db_get_current_workspace_directory(), "/", local_dg_name, NULL));
  FILE * fd = safe_fopen(dg_name,"a");
  
  string local_graph_name = db_build_file_resource_name(DBR_DVCG_FILE, mod_name, TRACE_DV_SUFFIX);
  string graph_name = strdup(concatenate(db_get_current_workspace_directory(), "/", local_graph_name, NULL));
  FILE * gr = safe_fopen(graph_name, "w");
  list verlist = NIL;
  
  persistant_statement_to_int s_to_l = persistant_statement_to_int_undefined;
  int dl = -1;
  
  graph trace_graph;
  
  string var_names = NULL;
  list var_names_entities = NIL;

  /* USER REQUEST */
  string rep = user_request("Which variables do you want to trace?\n");
  if (rep[0] == '\0') {
    user_log("No variable to trace\n");
    return;
  }
  else var_names = rep;
  var_names_entities = string_to_entity_list(mod_name, var_names);
  
  debug_on("TRACING_VARIABLE_DEBUG_LEVEL");
  
  fprintf(fd,"-----Tracing variable-----\n");
  fprintf(fd,"Variables to trace: %s\n", var_names);
  fprintf(fd,"--------------------------\n");
  free(var_names);
    
  if (!statement_undefined_p(mod_stat)) {
    /* compute the line numbers for statements */
    s_to_l = statement_to_line_number(mod_stat);
    dl = module_to_declaration_length(get_current_module_entity());
  }

  for (pv1 = graph_vertices(mod_graph); !ENDP(pv1); pv1 = CDR(pv1)) {
    vertex v1 = VERTEX(CAR(pv1));
    statement s1 = vertex_to_statement(v1);
    
    for (ps = vertex_successors(v1); !ENDP(ps); ps = CDR(ps)) {
      successor su = SUCCESSOR(CAR(ps));
      vertex v2 = successor_vertex(su);
      statement s2 = vertex_to_statement(v2);
      dg_arc_label dal = (dg_arc_label) successor_arc_label(su);
      
      for(pc = dg_arc_label_conflicts(dal); !ENDP(pc); pc = CDR(pc)) {
	conflict c = CONFLICT(CAR(pc));
	if (!statement_undefined_p(mod_stat)) {
	  if (entity_in_list_pattern(var_names_entities,reference_variable(effect_reference(conflict_source(c))))) {
	    string succ_label = (string)malloc(sizeof(char *)*30);
	    int l1 = dl + apply_persistant_statement_to_int(s_to_l, s1);
	    int l2 = dl + apply_persistant_statement_to_int(s_to_l, s2);

	    vertex vertex_parent = NULL;
	    vertex vertex_child = NULL;
	    char statement_action_parent = action_read_p(effect_action(conflict_source(c))) ? 'R' : 'W';
	    char statement_action_child = action_read_p(effect_action(conflict_sink(c))) ? 'R' : 'W';
	    string variable_name_parent = strdup(words_to_string(effect_words_reference(effect_reference(conflict_source(c)))));
	    string variable_name_child = strdup(words_to_string(effect_words_reference(effect_reference(conflict_sink(c)))));
	    string node_name_parent = (string)malloc(sizeof(char *)*strlen(variable_name_parent) + 20);
	    string node_name_child = (string)malloc(sizeof(char *)*strlen(variable_name_child) + 20);
	    successor succ = NULL;
	    sprintf(node_name_parent, "%d-%s-%c", l1, variable_name_parent, statement_action_parent);
	    sprintf(node_name_child, "%d-%s-%c", l2, variable_name_child, statement_action_child);
	    free(variable_name_parent);
	    free(variable_name_child);

	    memset(succ_label, 0, strlen(succ_label));
	    if (conflict_cone(c) != cone_undefined) {
	      if (!statement_undefined_p(mod_stat)) {
		strcat(succ_label, "levels(");
		MAPL(pl, {
		  sprintf(succ_label + strlen(succ_label), pl == cone_levels(conflict_cone(c)) ? "%d" : ",%d", INT(CAR(pl)));
		}, cone_levels(conflict_cone(c)));
		strcat(succ_label, ")");
	      }
	    }
	    
	    vertex_parent = get_vertex_in_list(verlist, node_name_parent);
	    if (vertex_parent == vertex_undefined) {
	      vertex_parent = make_vertex((vertex_label)node_name_parent, NIL);
	      ADD_ELEMENT_TO_LIST(verlist, VERTEX, vertex_parent);
	    }
	    vertex_child = get_vertex_in_list(verlist, node_name_child);
	    if (vertex_child == vertex_undefined) {
	      vertex_child = make_vertex((vertex_label)node_name_child, NIL);
	      ADD_ELEMENT_TO_LIST(verlist, VERTEX, vertex_child);
	    }
	    succ = make_successor((dg_arc_label)succ_label, vertex_child);
	    ADD_ELEMENT_TO_LIST(vertex_successors(vertex_parent), SUCCESSOR, succ);
	  }
	}
      }
    }
  }
  gen_free_list(var_names_entities);

  trace_graph = make_graph(verlist);
  print_graph_text(fd, trace_graph);
  print_graph_daVinci(gr, trace_graph);
    
  if (!statement_undefined_p(mod_stat))
    free_persistant_statement_to_int(s_to_l);
  else
    fprintf(fd, "------------------End of tracing variable----------------\n");
  
  debug_off();
  
  safe_fclose(fd, dg_name);
  free(dg_name);
  safe_fclose(gr, graph_name);
  free(graph_name);
  
  return;
}

vertex get_vertex_in_list(list in_l, string in_s)
{
  vertex ver = NULL;
  for(; !ENDP(in_l); POP(in_l)) {
    string s;
    ver = VERTEX(CAR(in_l));
    s = (string)vertex_vertex_label(ver);
    if (same_string_p(in_s, s)) return ver;
  }
  return vertex_undefined;
}

void print_graph_text(FILE * out_f, graph in_gr)
{
  MAPL(ver_ptr, {
    vertex ver = VERTEX(CAR(ver_ptr));
    MAPL(succ_ptr, {
      successor succ = SUCCESSOR(CAR(succ_ptr));
      fprintf(out_f, (string)vertex_vertex_label(ver));
      fprintf(out_f, " ");
      fprintf(out_f, (string)vertex_vertex_label(successor_vertex(succ)));
      fprintf(out_f, " ");
      fprintf(out_f, (string)successor_arc_label(succ));
      fprintf(out_f, "\n");
    }, vertex_successors(ver));
  }, graph_vertices(in_gr));  
}

void print_graph_daVinci(FILE * out_f, graph in_gr)
{
  string gr_buffer = "";
  bool first_node_parent = TRUE;
  fprintf(out_f, "[\n");
  MAPL(ver_ptr, {
    vertex ver = VERTEX(CAR(ver_ptr));
    string node_name_parent = (string)vertex_vertex_label(ver);
    bool first_node_child = TRUE;
    if (first_node_parent)
      first_node_parent = FALSE;
    else
      fprintf(out_f, ",\n");
    fprintf(out_f,"l(\"%s\",n(\"\",[a(\"OBJECT\",\"%s\")],[\n", node_name_parent, node_name_parent); 
    MAPL(succ_ptr, {
      successor succ = SUCCESSOR(CAR(succ_ptr));
      string node_name_child = (string)vertex_vertex_label(successor_vertex(succ));
      if (first_node_child)
	first_node_child = FALSE;
      else
	fprintf(out_f, ",\n");
      if (strlen((string)successor_arc_label(succ)) == 0) {
	fprintf(out_f, "  l(\"\",e(\"\",[],r(\"%s\")))", node_name_child);
      } else {
	string temp_buffer = strdup(concatenate(gr_buffer, ",\nl(\"", node_name_parent, "-", node_name_child, "\",n(\"\",[a(\"OBJECT\",\"", (string)successor_arc_label(succ), "\"),a(\"_GO\",\"ellipse\")],[\n  l(\"\",e(\"\",[],r(\"", node_name_child, "\")))]))", NULL));
	free(gr_buffer);
	gr_buffer = temp_buffer;
	fprintf(out_f, "  l(\"\",e(\"\",[],r(\"%s-%s\")))", node_name_parent, node_name_child);
      } 
    }, vertex_successors(ver));
    fprintf(out_f, "]))");
  }, graph_vertices(in_gr));
  fprintf(out_f, gr_buffer);
  fprintf(out_f, "\n]");
  free(gr_buffer);
} 

bool entity_in_list_pattern(list in_l, entity ent)
{
  MAP(ENTITY, e, {
    if (same_string_p(entity_local_name(e), "*")) return TRUE;
    if (same_entity_p(e,ent)) return TRUE;
  }, in_l);
  return FALSE;
}


























































