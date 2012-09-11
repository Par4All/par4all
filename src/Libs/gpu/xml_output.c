/*

 $Id $

 Copyright 1989-2012 MINES ParisTech
 Copyright 2012 Silkan

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
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "control.h"
#include "pipsdbm.h"
#include "text-util.h"
#include "properties.h"

#define NL "\n"

#define XMLPRETTY    ".xml"

static FILE *out_fp = 0;

// Define output
static void xml_set_output(FILE *new_fp) {
  out_fp = new_fp;
}

/*
 * Print to the defined output the format string and an optional string argument
 */
#define xml_print(format, args...) \
{ \
  pips_assert("Output is set",out_fp); \
  fprintf(out_fp,format,##args); \
}

static void xml_output(string s) {
  xml_print("%s", s);
}

static void xml_print_entity_name(entity e) {
  xml_output(entity_name( e ));
}

static void xml_print_type(type t) {
  if(t == type_undefined) {
    xml_print("*undefined*");
  } else {
    switch(type_tag( t )) {
      case is_type_statement:
      xml_output("Statement (unit ?) ");
      break;
      case is_type_area:
      xml_output("Area unsupported");
      break;
      case is_type_variable:
      xml_output("Variable unsupported");
      //xml_print_variable(type_variable( t ));
      break;
      case is_type_functional:
      xml_output("Functional unsupported");
      //xml_print_functional(type_functional( t ));
      break;
      case is_type_varargs:
      xml_output("VarArgs unsupported");
      //xml_print_type(type_varargs( t ));
      break;
      case is_type_unknown:
      xml_output("Unknown");
      break;
      case is_type_void:
      xml_output("void");
      break;
      case is_type_struct: {
        FOREACH(entity, e, type_struct( t ) )
        {
          xml_print_entity_name(e);
        }
        break;
      }
      case is_type_union:
      xml_output("Union");
      FOREACH(entity, e, type_union( t ) )
      {
        xml_print_entity_name(e);
      }
      break;
      case is_type_enum:
      xml_output("Enum");
      FOREACH(entity, e, type_enum( t ) )
      {
        xml_print_entity_name(e);
      }
      break;
      default:
      break;
    }
  }
}

void xml_print_entity_full(entity e) {
  xml_output(entity_name( e ));
  xml_print_type(entity_type( e ));
  //xml_print_storage(entity_storage( e ));
  //xml_print_value(entity_initial( e ));
}

static void xml_print_parameter(entity p, bool is_a_dim) {
  xml_print("<Arg type=\"")
  type t = entity_type( p );

  // Pointeur or Array/scalar
  // BTW it would be too easy if "array" means Array type, in fact it means pointer...
  type ut = ultimate_type(t);
  if(basic_pointer_p(variable_basic(type_variable(ut)))) {
    xml_print("array");
  } else {
    xml_print("param");
  }

  // Name
  xml_print("\" name=\"%s\" ", entity_user_name(p));

  // Type
  xml_print(" typeData=\"%s\"",
            words_to_string(words_type(t, NIL, false)));

  if(is_a_dim) {
    xml_print(" value=\"0\"")
  }
  xml_print("/>" NL)
}


void gather_grid_dim(statement s, void *ctx) {
  // ctx is a pointer to a list of variable
  list *dims = (list *)ctx;

  string comment = statement_comments(s);
  pips_debug(3,"Statement %p, Compare comment '%s'\n",s,comment);
  if(!string_undefined_p(comment)) {
    size_t sentinel_len = strlen("// To be assigned to a call to P4A_vp_X: ");
    string sentinel;
    while((sentinel=strstr(comment,"// To be assigned to a call to P4A_vp_"))!=NULL) {
      string varname_ptr = sentinel+sentinel_len;
      string varname_end = strchr(varname_ptr,'\n');
      int varname_len = varname_end-varname_ptr;
      string varname = strndup(varname_ptr,varname_len); // to be freed
      pips_debug(2,"Catch dimension %s\n",varname);
      *dims = CONS(string,varname,*dims);
      comment = varname_end;
    }
  }
}

/** PIPSMAKE INTERFACE */
bool gpu_xml_dump(string mod_name) {

  statement module_statement = (statement) db_get_memory_resource(DBR_CODE,
                                                                  mod_name,
                                                                  true);

  set_current_module_statement(module_statement);

  /* Set the current module entity required to have many things
   working in PIPS: */
  set_current_module_entity(module_name_to_entity(mod_name));

  debug_on("GPU_XML_DUMP_DEBUG_LEVEL");
  pips_assert("Statement should be OK at entry...",
              statement_consistent_p(module_statement));

  // Prepare the output file
  string xml_file_name = db_build_file_resource_name("RI",
                                                     mod_name,
                                                     ".out.xml");
  string output_file = strdup(concatenate(db_get_current_workspace_directory(),
                                          "/",
                                          xml_file_name,
                                          NULL));
  pips_debug(2, "Output in %s\n", output_file);
  FILE *fp = safe_fopen(output_file, "w");
  xml_set_output(fp);

  /* First find grid dimension by looking for magic comments */
  list /* of const strings */ dims = NIL;
  gen_context_recurse(module_statement,&dims,statement_domain,gen_true,gather_grid_dim);


  /* Print current module */
  entity module = get_current_module_entity();
  type module_type = entity_type(module);
  pips_assert("Functional module", type_functional_p(module_type));

  // Remove wrapper prefix FIXME: bad hack...
  string original_name = mod_name + strlen("p4a_wrapper_");

  // Print Task
  xml_print("<Task name=\"%s\" kernel=\"%s\" nbParallelLoops=\"%zu\">" NL,
            original_name, mod_name, gen_length(dims));

  // Params
  int nparams = gen_length(functional_parameters(type_functional(module_type)));
  for(int i = 1; i <= nparams; i++) {
    entity param = find_ith_parameter(module, i);
    bool is_a_dim = false;
    FOREACH(string, dim, dims) {
      if(same_string_p(entity_user_name(param),dim)) {
        is_a_dim=true;
        break;
      }
    }
    xml_print_parameter(param, is_a_dim );
  }

  //xml_print_statement(module_statement);
  xml_print("</Task>" NL);

  // Reset output file
  xml_set_output(0);
  safe_fclose(fp, output_file);

  DB_PUT_FILE_RESOURCE( DBR_GPU_XML_FILE, strdup( mod_name ), xml_file_name );

  debug_off();

  reset_current_module_statement();
  reset_current_module_entity();

  return true;
}
