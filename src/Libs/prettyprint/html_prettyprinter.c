/*

 $Id: cprettyprinter.c 15792 2009-11-27 10:26:07Z amini $

 Copyright 1989-2009 MINES ParisTech

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
/*

 Try to prettyprint the RI in C.
 Very basic at the time.
 Functionnal.
 All arguments are assumed newly allocated.
 It might be really slow, but it should be safe.
 I should use some kind of string accumulator (array/list...)

 html_print_crough > MODULE.crough
 < PROGRAM.entities
 < MODULE.code

 html_print_c_code  > MODULE.c_printed_file
 < MODULE.crough
 */

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"
#include "properties.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "text-util.h"
#include "transformations.h"

#define NL "\n"
/**************************************************************** MISC UTILS */

#define current_module_is_a_function() \
(entity_function_p(get_current_module_entity()))

static void html_print_expression( expression e );
static void html_print_range( range r );
static void html_print_statement( statement r );
static void html_print_type( type t );

static void begin_block( char *block ) {
  printf( "<li class=\"%s\"><span>%s</span><ul>" NL, block, block );
}

static void end_block( char *block ) {
  printf( "</ul></li><!-- %s -->" NL, block );
}

void html_output( char *out ) {
  printf( "<li>%s</li>" NL, out );
}

static void html_print_entity( entity e ) {
  begin_block( "entity" );
  html_output( entity_name( e ) );
  end_block( "entity" );
}

static void html_print_ram( ram r ) {
  begin_block( "ram" );

  html_output( "Function" );
  html_print_entity( ram_function( r ) );

  html_output( "Section" );
  html_print_entity( ram_section( r ) );

  char str[100];
  snprintf( str, 100, "Offset : %d", (int) ram_offset( r ) );
  html_output( str );

  MAP(ENTITY, an_entity, {html_print_entity(an_entity);}, ram_shared( r ) );

  end_block( "ram" );
}

static void html_print_formal( formal f ) {
  begin_block( "formal" );

  html_output( "Function" );
  html_print_entity( formal_function( f ) );

  char str[100];
  snprintf( str, 100, "Offset : %d", (int) formal_offset( f ) );
  html_output( str );

  end_block( "formal" );
}

static void html_print_rom( unit r ) {
  begin_block( "rom" );

  html_output( "unit ?" );

  end_block( "rom" );
}

static void html_print_storage( storage s ) {
  begin_block( "storage" );
  switch ( storage_tag( s ) ) {
    case is_storage_return:
      html_output( "Return" );
      html_print_entity( storage_return( s ) );
      break;
    case is_storage_ram:
      html_print_ram( storage_ram( s ) );
      break;
    case is_storage_formal:
      html_print_formal( storage_formal( s ) );
      break;
    case is_storage_rom:
      html_print_rom( storage_rom( s) );
      break;
    default:
      html_output( "Unknown storage" );
      break;
  }
  end_block( "storage" );
}

static void html_print_basic( basic b ) {
  begin_block( "basic" );

  switch ( basic_tag( b ) ) {
    case is_basic_int:
      html_output( "int" );
      break;
    case is_basic_float:
      html_output( "float" );
      break;
    case is_basic_logical:
      html_output( "logical" );
      break;
    case is_basic_overloaded:
      html_output( "overloaded" );
      break;
    case is_basic_complex:
      html_output( "complex" );
      break;
    case is_basic_string:
      html_output( "string" );
      break;
    case is_basic_bit:
      html_output( "bit" );
      break;
    case is_basic_pointer:
      html_output( "pointer" );
      break;
    case is_basic_derived:
      html_output( "deriver" );
      break;
    case is_basic_typedef:
      html_output( "typedef" );
      break;
    default:
      html_output( "unknown" );
      break;
  }

  end_block( "basic" );
}

static void html_print_area( area a ) {
  begin_block( "area" );

  char str[100];
  snprintf( str, 100, "Size : %d", (int) area_size( a ) );
  html_output( str );

  html_output( "Layout" );
  MAP(ENTITY, an_entity, {html_print_entity(an_entity);}, area_layout( a ) );

  end_block( "area" );
}

static void html_print_qualifier( qualifier q ) {
  begin_block( "qualifier" );

  switch ( qualifier_tag( q ) ) {
    case is_qualifier_const:
      html_output( "const" );
      break;
    case is_qualifier_restrict:
      html_output( "restrict" );
      break;
    case is_qualifier_volatile:
      html_output( "volatile" );
      break;
    case is_qualifier_register:
      html_output( "register" );
      break;
    case is_qualifier_auto:
      html_output( "auto" );
      break;
    default:
      html_output( "unknown" );
      break;
  }

  end_block( "qualifier" );
}

static void html_print_dimension( dimension d ) {
  begin_block( "dimension" );

  html_print_expression( dimension_lower( d ) );
  html_print_expression( dimension_upper( d ) );

  end_block( "dimension" );
}

static void html_print_variable( variable v ) {
  begin_block( "variable" );

  html_print_basic( variable_basic( v ) );

  MAP(DIMENSION, a_dim, {html_print_dimension(a_dim);}, variable_dimensions( v ) );
  MAP(QUALIFIER, a_qual, {html_print_qualifier(a_qual);}, variable_qualifiers( v ) );

  end_block( "variable" );
}

static void html_print_mode( mode m ) {
  begin_block( "mode" );

  switch ( mode_tag( m ) ) {
    case is_mode_value:
      html_output( "value" );
      break;
    case is_mode_reference:
      html_output( "reference" );
      break;
    default:
      html_output( "unknown" );
      break;
  }

  end_block( "mode" );
}

static void html_print_parameter( parameter p ) {
  begin_block( "parameter" );

  html_print_type( parameter_type( p ) );
  html_print_mode( parameter_mode( p ) );
  html_output( "Dummy unimplemented" );

  end_block( "parameter" );
}

static void html_print_functional( functional f ) {
  begin_block( "functional" );

  MAP(PARAMETER, param, {html_print_parameter(param);}, functional_parameters( f ) );

  html_print_type( functional_result( f ) );

  end_block( "functional" );
}

static void html_print_type( type t ) {
  begin_block( "type" );
  switch ( type_tag( t ) ) {
    case is_type_statement:
      html_output( "Statement (unit ?) " );
      break;
    case is_type_area:
      html_print_area( type_area( t ) );
      break;
    case is_type_variable:
      html_print_variable( type_variable( t ) );
      break;
    case is_type_functional:
      html_print_functional( type_functional( t ) );
      break;
    case is_type_varargs:
      html_output( "VarArgs" );
      html_print_type( type_varargs( t ) );
      break;
    case is_type_unknown:
      html_output( "Unknown" );
      break;
    case is_type_void:
      html_output( "Void" );
      break;
    case is_type_struct:
      html_output( "Struct" );
      MAP(ENTITY, an_entity, {html_print_entity(an_entity);}, type_struct( t ) )
      ;
      break;
    case is_type_union:
      html_output( "Union" );
      MAP(ENTITY, an_entity, {html_print_entity(an_entity);}, type_union( t ) )
      ;
      break;
    case is_type_enum:
      html_output( "Enum" );
      MAP(ENTITY, an_entity, {html_print_entity(an_entity);}, type_enum( t ) )
      ;
      break;
    default:
      break;
  }
  end_block( "type" );
}

static void html_print_value( value v ) {
  begin_block( "value" );

  switch ( value_tag( v ) ) {
    case is_value_code:
      html_output( "code" );
      break;
    case is_value_symbolic:
      html_output( "symbolic" );
      break;
    case is_value_constant:
      html_output( "constant" );
      break;
    case is_value_intrinsic:
      html_output( "intrinsic" );
      break;
    case is_value_unknown:
      html_output( "unknown" );
      break;
    case is_value_expression:
      html_output( "expression" );
      html_print_expression( value_expression( v ) );
      break;
    default:
      html_output( "error" );
      break;
  }

  end_block( "value" );
}
void html_print_entity_full( entity e ) {
  begin_block( "entity" );
  html_output( entity_name( e ) );
  html_print_type( entity_type( e ) );
  html_print_storage( entity_storage( e ) );
  html_print_value( entity_initial( e ) );

  end_block( "entity" );
}

static void html_print_call( call c ) {
  begin_block( "call" );

  begin_block( "function" );
  html_print_entity( call_function(c) );
  end_block( "function" );

  if ( call_arguments( c ) ) {
    begin_block( "arguments" );
    MAP(EXPRESSION, e,
        {
          html_print_expression( e );
        }, call_arguments( c ) );
    end_block( "arguments" );
  }
  end_block( "call" );
}

static void html_print_unstructured( unstructured u ) {
  begin_block( "unstructured" );
  printf( "<li>Not implemented</li>" NL );
  end_block( "unstructured" );
}

static void html_print_reference( reference r ) {
  begin_block( "reference" );
  begin_block( "variable" );
  html_print_entity( reference_variable( r ) );
  end_block( "variable" );

  if ( reference_indices( r ) ) {
    begin_block( "indices" );
    MAP(EXPRESSION, e,
        {
          html_print_expression(e);
        }, reference_indices(r));
    end_block( "indices" );
  }

  end_block( "reference" );
}

static void html_print_expression( expression e ) {
  begin_block( "expression" );

  begin_block( "syntax" );
  syntax s = expression_syntax(e);
  switch ( syntax_tag(s) ) {
    case is_syntax_call:
      html_print_call( syntax_call( s ) );
      break;
    case is_syntax_range:
      html_print_range( syntax_range( s ) );
      break;
    case is_syntax_reference:
      html_print_reference( syntax_reference( s ) );
      break;
      /* add cast, sizeof here FIXME*/
    default:
      pips_internal_error("unexpected syntax tag");
  }
  end_block( "syntax" );

  end_block( "expression" );
}

static void html_print_range( range r ) {
  begin_block( "range" );
  begin_block( "lower" );
  html_print_expression( range_lower( r ) );
  end_block( "lower" );
  begin_block( "upper" );
  html_print_expression( range_upper( r ) );
  end_block( "upper" );
  begin_block( "increment" );
  html_print_expression( range_increment( r ) );
  end_block( "increment" );
  end_block( "range" );
}

static void html_print_loop( loop l ) {
  begin_block( "loop" );
  /* partial implementation ??? */

  begin_block( "index" );
  html_print_entity( loop_index( l ) );
  end_block( "index" );

  html_print_range( loop_range( l ) );

  begin_block( "body" );
  html_print_statement( loop_body( l ) );
  end_block( "body" );

  end_block( "loop" );
}

static void html_print_whileloop( whileloop w ) {
  /* partial implementation...  ?? */

  begin_block( "whileloop" );

  begin_block( "condition" );
  html_print_expression( whileloop_condition( w ) );
  end_block( "condition" );

  begin_block( "statement" );
  html_print_statement( whileloop_body( w ) );
  end_block( "statement" );

  begin_block( "evaluation" );
  evaluation eval = whileloop_evaluation(w);
  /*do while and while do loops */
  if ( evaluation_before_p(eval) )
    html_output( "while() {}" NL );
  else
    html_output( "do {} while(); " NL );
  end_block( "evaluation" );

  begin_block( "body" );
  html_print_statement( whileloop_body( w ) );
  end_block( "body" );

  end_block( "whileloop" );
}

static void html_print_forloop( forloop f ) {
  /* partial implementation... */

  begin_block( "forloop" );

  begin_block( "initialization" );
  html_print_expression( forloop_initialization( f ) );
  end_block( "initialization" );

  begin_block( "condition" );
  html_print_expression( forloop_condition( f ) );
  end_block( "condition" );

  begin_block( "increment" );
  html_print_expression( forloop_increment( f ) );
  end_block( "increment" );

  begin_block( "body" );
  html_print_statement( forloop_body( f ) );
  end_block( "body" );

  end_block( "forloop" );
}

static void html_print_sequence( sequence seq ) {
  begin_block( "sequence" );
  MAP(STATEMENT, s,
      {
        html_print_statement( s );
      },
      sequence_statements(seq));
  end_block( "sequence" );
}

static void html_print_test( test t ) {
  begin_block( "test" );

  begin_block( "cond" );
  html_print_expression( test_condition( t ) );
  end_block( "cond" );

  begin_block( "strue" );
  html_print_statement( test_true( t ) );
  end_block( "cond" );

  if ( !empty_statement_p( test_false( t ) ) ) {
    begin_block( "sfalse" );
    html_print_statement( test_false(t) );
    end_block( "sfalse" );
  }

  end_block( "test" );
}

static void html_print_statement( statement s ) {
  begin_block( "statement" );

  list l = statement_declarations(s);
  if ( l ) {
    begin_block( "declarations" );
    if ( !ENDP(l) ) {
      MAP(ENTITY, var,
          {
            html_print_entity( var );
          },l);
    }
    end_block( "declarations" );
  }

  begin_block( "instructions" );
  instruction i = statement_instruction(s);
  switch ( instruction_tag(i) ) {
    case is_instruction_test:
      html_print_test( instruction_test( i ) );
      break;
    case is_instruction_sequence:
      html_print_sequence( instruction_sequence( i ) );
      break;
    case is_instruction_loop:
      html_print_loop( instruction_loop( i ) );
      break;
    case is_instruction_whileloop:
      html_print_whileloop( instruction_whileloop( i ) );
      break;
    case is_instruction_forloop:
      html_print_forloop( instruction_forloop( i ) );
      break;
    case is_instruction_call:
      html_print_call( instruction_call( i ) );
      break;
    case is_instruction_unstructured:
      html_print_unstructured( instruction_unstructured( i ) );
      break;
    case is_instruction_expression:
      html_print_expression( instruction_expression( i ) );
      break;
    case is_instruction_goto:
      /*      statement g = instruction_goto(i);
       entity el = statement_label(g);
       string l = entity_local_name(el) + strlen(LABEL_PREFIX);
       printf("%s", strdup( concatenate( "goto ", l, SEMICOLON, NULL ) ) );*/
      break;
      /* add switch, forloop break, continue, return instructions here*/
    default:
      html_output( " Instruction not implemented" NL );
      break;
  }
  end_block( "instructions" );

  end_block( "statement" );
}

bool html_prettyprint( char *module_name ) {
  statement module_statement =
      PIPS_PHASE_PRELUDE(module_name,"PREPEND_COMMENT_DEBUG_LEVEL");

  /* Print current module */
  printf("<li><ul class=\"module\">");
  begin_block( module_name );
  html_print_statement( module_statement );
  end_block( module_name );
  printf("</li></ul>");

  /* Put back the new statement module */
  PIPS_PHASE_POSTLUDE(module_statement);

  return TRUE;
}

bool html_prettyprint_symbol_table( char /* unused */module ) {
  /* Print symbol table */
  begin_block( "Symbol table" );
  list entities = gen_filter_tabulated( gen_true, entity_domain );
  int i = 0;
  MAP(ENTITY,
      an_entity,
      {
        html_print_entity_full(an_entity);
      },
      entities);
  end_block( "Symbol table" );
}
