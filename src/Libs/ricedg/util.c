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
#include <stdlib.h>
/* For strdup: */
#define _GNU_SOURCE
#include <string.h>

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "local.h"
#include "rice.h"


/* Vertex_to_statement looks for the statement that is pointed to by
vertex v. This information is kept in a static hash_table named
OrderingToStatement. See ri-util/ordering.c for more information.
*/

statement 
vertex_to_statement(v)
vertex v;
{
    dg_vertex_label dvl = (dg_vertex_label) vertex_vertex_label(v);
    return(ordering_to_statement(dg_vertex_label_statement(dvl)));
}

int 
vertex_ordering(v)
vertex v;
{
    dg_vertex_label dvl = (dg_vertex_label) vertex_vertex_label(v);
    return dg_vertex_label_statement(dvl);
}

/* compare two vertices based on their ordering */
int compare_vertex(const void *v0, const void *v1) {
    const vertex V0 = *(const vertex *)v0,
                 V1 = *(const vertex *)v1;
    if(V0==V1) return 0;
    return vertex_ordering(V0) > vertex_ordering(V1);
}


/* Define a mapping from the statement ordering to the dependence
   graph vertices: */
hash_table
compute_ordering_to_dg_mapping(graph dependance_graph)
{
   hash_table ordering_to_dg_mapping = hash_table_make(hash_int, 0);
   
   MAP(VERTEX,
       a_vertex,
       {
          debug(7, "compute_ordering_to_dg_mapping",
                "\tSuccessor list: %p for statement ordering %p\n", 
                vertex_successors(a_vertex),
                dg_vertex_label_statement(vertex_vertex_label(a_vertex)));

          hash_put(ordering_to_dg_mapping,
                   (char *) dg_vertex_label_statement(vertex_vertex_label(a_vertex)),
                   (char *) a_vertex);
       },
       graph_vertices(dependance_graph));
   
   return ordering_to_dg_mapping;
}


static string dependence_graph_banner[8] = {
	"\n *********************** Use-Def Chains *********************\n",
	"\n **************** Effective Dependence Graph ****************\n",
	"\n ********* Dependence Graph (ill. option combination) *******\n",
	"\n ********* Dependence Graph (ill. option combination) *******\n",
	"\n ******** Whole Dependence Graph with Dependence Cones ******\n",
	"\n ********* Dependence Graph (ill. option combination) *******\n",
	"\n ********* Dependence Graph (ill. option combination) *******\n",
	"\n **** Loop Carried Dependence Graph with Dependence Cones ***\n"
    };


/**
 * @brief This is a callback for qsort function, it compares two vertex
 * @return 0 if these are equals, -1 if first is lower, 1 if second is lower
 */
int vertex_sort_callback( const vertex *v1, const vertex *v2 ) {
  statement s1 = vertex_to_statement( *v1 );
  statement s2 = vertex_to_statement( *v2 );
  return statement_number(s1) > statement_number(s2);
}


/**
 * @brief This is a callback for qsort function, it compares two successor
 * @return 0 if these are equals, -1 if first is lower, 1 if second is lower
 */
int successor_sort_callback( const successor *succ1, const successor *succ2 ) {
  vertex v1 = successor_vertex(*succ1);
  vertex v2 = successor_vertex(*succ2);
  return vertex_sort_callback( &v1, &v2 );
}

/**
 * @brief This is a callback for qsort function, it compares two conflicts
 * @return 0 if conflicts are equals, -1 if first is lower, 1 if second is lower
 */
int conflicts_sort_callback( conflict *c1, conflict *c2 ) {
  int r_value = 0;
  effect e1_sink = conflict_sink((*c1));
  effect e2_sink = conflict_sink((*c2));
  action a1_sink = effect_action(e1_sink);
  action a2_sink = effect_action(e2_sink);
  effect e1_source = conflict_source((*c1));
  effect e2_source = conflict_source((*c2));
  action a1_source = effect_action(e1_source);
  action a2_source = effect_action(e2_source);



  string s1 = strdup(concatenate(full_action_to_short_string(a1_source),
                                 full_action_to_short_string(a1_sink),
                                 NULL));
  string s2 = strdup(concatenate(full_action_to_short_string(a2_source),
                                 full_action_to_short_string(a2_sink),
                                 NULL));

  // Compare Action in lexical order
  r_value = strcmp( s1, s2 );
  free( s1 );
  free( s2 );

  if ( r_value == 0 ) {
    // Compare string representing source reference
    reference r1_source = effect_any_reference(e1_source);
    string s1 = words_to_string( effect_words_reference( r1_source ) );
    reference r2_source = effect_any_reference(e2_source);
    string s2 = words_to_string( effect_words_reference( r2_source ) );
    r_value = strcmp( s1, s2 );
    free( s1 );
    free( s2 );
  }
  if ( r_value == 0 ) {
    // Compare string representing sink reference
    reference r1_sink = effect_any_reference(e1_sink);
    string s1 = words_to_string( effect_words_reference( r1_sink ) );
    reference r2_sink = effect_any_reference(e2_sink);
    string s2 = words_to_string( effect_words_reference( r2_sink ) );
    r_value = strcmp( s1, s2 );
    free( s1 );
    free( s2 );
  }

  return r_value;
}


/* Print all edges and arcs */
void prettyprint_dependence_graph( FILE * fd,
                                   statement mod_stat,
                                   graph mod_graph ) {
  cons *pv1, *ps, *pc;
  Ptsg gs;
  int banner_number = 0;
  bool sru_format_p =
      get_bool_property( "PRINT_DEPENDENCE_GRAPH_USING_SRU_FORMAT" );
  persistant_statement_to_int s_to_l = persistant_statement_to_int_undefined;
  int dl = -1;
  debug_on("RICEDG_DEBUG_LEVEL");

  ifdebug(8) {
    /* There is no guarantee that the ordering_to_statement()
     * hash table is the proper one */
    print_ordering_to_statement( );
  }

  if ( sru_format_p && !statement_undefined_p(mod_stat) ) {
    /* compute line numbers for statements */
    s_to_l = statement_to_line_number( mod_stat );
    dl = module_to_declaration_length( get_current_module_entity( ) );
  } else {
    banner_number
        = get_bool_property( "PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS" )
          + 2 * get_bool_property( "PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS" )
          + 4 * get_bool_property( "PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES" );
    fprintf( fd, "%s\n", dependence_graph_banner[banner_number] );
  }

  gen_sort_list(graph_vertices(mod_graph), (gen_cmp_func_t)vertex_sort_callback);
  for ( pv1 = graph_vertices(mod_graph); !ENDP(pv1); pv1 = CDR(pv1) ) {
    vertex v1 = VERTEX(CAR(pv1));
    statement s1 = vertex_to_statement( v1 );
    gen_sort_list(vertex_successors(v1), (gen_cmp_func_t)successor_sort_callback);
    for ( ps = vertex_successors(v1); !ENDP(ps); ps = CDR(ps) ) {
      successor su = SUCCESSOR(CAR(ps));
      vertex v2 = successor_vertex(su);
      statement s2 = vertex_to_statement( v2 );
      dg_arc_label dal = (dg_arc_label) successor_arc_label(su);

      /* If we have zero printable conflicts, move on.
       *
       * If we have more than one conflict, let's sort them !
       *
       * FI: the number of conflicts should take into account the
       * filtering due to PRETTYPRINT_MEMORY_EFFECTS_ONLY
       */
      list cl = dg_arc_label_conflicts(dal);
      int nb_conflicts = gen_length(cl);
      if(get_bool_property("PRETTYPRINT_MEMORY_EFFECTS_ONLY")) {
	FOREACH(CONFLICT, c, cl) {
	  effect efsrc = conflict_source(c);
	  if(!store_effect_p(efsrc))
	    nb_conflicts--;
	}
      }

      if(nb_conflicts==0)
	continue;

      if ( !sru_format_p || statement_undefined_p(mod_stat) ) {
        /* Modification at revision 12484 because statement
         numbers were not initialized by C parser, with no
         validation of ricedg available at that time*/
        /* factorize line numbers */
        //fprintf(fd, "\t%s -->",
        // external_statement_identification(s1)
        // Revision 10893: %02d and statement_number (Pham Dat)
        fprintf( fd, "\t%02td -->", statement_number(s1) );
        //fprintf(fd, " %s with conflicts\n",
        // external_statement_identification(s2)
        fprintf( fd, " %02td with conflicts\n", statement_number(s2) );
      }

      if ( nb_conflicts > 1 ) {
        /*
         * Convert the list to an array for sorting
         * 20 is the initial size, should be enough for most cases
	 *
	 * FI: should generate a bug anytime when calls are
	 * involved. At least use dependent types... or pass gen_length()
         */
        gen_array_t conflicts_array = gen_array_make( 20 );
        list_to_array( dg_arc_label_conflicts(dal), conflicts_array );
        qsort( gen_array_pointer( conflicts_array ),
               gen_array_nitems( conflicts_array ),
               sizeof(void *),
               (gen_cmp_func_t) conflicts_sort_callback);
        list conflicts_list = NIL;
        GEN_ARRAY_FOREACH(conflict, s, conflicts_array)
          conflicts_list = CONS(conflict, s, conflicts_list);
        gen_array_free(conflicts_array);
        dg_arc_label_conflicts(dal)=conflicts_list;
      }

      /*
       * Loop over conflict and print them
       */
      for ( pc = dg_arc_label_conflicts(dal); !ENDP(pc); pc = CDR(pc) ) {
        conflict c = CONFLICT(CAR(pc));
	effect efsrc = conflict_source(c);

	/* FI: I should use another property, specific to the use-def
	   chains, but this is quite close */
	if(store_effect_p(efsrc)
	   || !get_bool_property("PRETTYPRINT_MEMORY_EFFECTS_ONLY")) {

        /* if (!entity_scalar_p(reference_variable
         (effect_any_reference(conflict_source(c))))) {
         */
        if ( sru_format_p && !statement_undefined_p(mod_stat) ) {
          int l1 = dl + apply_persistant_statement_to_int( s_to_l, s1 );
          int l2 = dl + apply_persistant_statement_to_int( s_to_l, s2 );

          fprintf( fd, "%d %d ", l1, l2 );
          /*
           fprintf(fd, "%d %d ",
           statement_number(s1), statement_number(s2));
           */
          fprintf( fd,
                   "%c %c ",
                   action_read_p(effect_action(conflict_source(c))) ? 'R' : 'W',
                   action_read_p(effect_action(conflict_sink(c))) ? 'R' : 'W' );
          fprintf( fd, "<" );
          print_words( fd,
                       effect_words_reference(
                           effect_any_reference(conflict_source(c)) ) );
          fprintf( fd, "> - <" );
          print_words( fd,
                       effect_words_reference(
                           effect_any_reference(conflict_sink(c)) ) );
          fprintf( fd, ">" );

          /* Additional information for EDF prettyprint.
           Instruction calls are given with  statement numbers
           */
          if ( get_bool_property( "PRETTYPRINT_WITH_COMMON_NAMES" ) ) {
            if ( instruction_call_p(statement_instruction(s1)) )
              fprintf( fd,
                       " %td-%s",
                       statement_number(s1),
                       entity_local_name(
                           call_function(
                               instruction_call(statement_instruction(s1))) ) );
            else
              fprintf( fd, " %td", statement_number(s1) );
            if ( instruction_call_p(statement_instruction(s2)) )
              fprintf( fd,
                       " %td-%s",
                       statement_number(s2),
                       entity_local_name(
                           call_function(
                               instruction_call(statement_instruction(s2))) ) );
            else
              fprintf( fd, " %td", statement_number(s2) );
          }

        } else {
          fprintf( fd, "\t\tfrom " );
          print_words( fd, words_effect( conflict_source(c) ) );

          fprintf( fd, " to " );
          print_words( fd, words_effect( conflict_sink(c) ) );
        }

        if ( conflict_cone(c) != cone_undefined ) {
          if ( sru_format_p && !statement_undefined_p(mod_stat) ) {
            fprintf( fd, " levels(" );
            MAPL(pl, {
                  fprintf(fd, pl==cone_levels(conflict_cone(c))? "%td" : ",%td",
                      INT(CAR(pl)));
                }, cone_levels(conflict_cone(c)));
            fprintf( fd, ") " );
          } else {
            fprintf( fd, " at levels " );
            MAPL(pl, {
                  fprintf(fd, " %td", INT(CAR(pl)));
                }, cone_levels(conflict_cone(c)));
            fprintf( fd, "\n" );
          }

          if( get_bool_property( "PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES" ) ) {
            gs = (Ptsg) cone_generating_system(conflict_cone(c));
            if( !SG_UNDEFINED_P( gs ) ) {
              if( sru_format_p && !statement_undefined_p(mod_stat) ) {
                if( sg_nbre_sommets( gs ) == 1 && sg_nbre_rayons( gs ) == 0
                    && sg_nbre_droites( gs ) == 0 ) {
                  /* uniform dependence */
                  fprintf( fd, "uniform" );
                  fprint_lsom_as_dense( fd, sg_sommets( gs ), gs->base );
                } else {
                  sg_fprint_as_ddv( fd, gs );
                }
              } else {
                /* sg_fprint(fd,gs,entity_local_name); */
                /* FI: almost print_dependence_cone:-( */
                sg_fprint_as_dense( fd, gs, gs->base );
                ifdebug(2) {
                  Psysteme sc1 = sc_new( );
                  sc1 = sg_to_sc_chernikova( gs );
                  (void) fprintf( fd,
                                  "syst. lin. correspondant au syst. gen.:\n" );
                  sc_fprint( fd, sc1, (get_variable_name_t) entity_local_name );
                }
              }
            }
          }
        } else {
          if ( sru_format_p && !statement_undefined_p(mod_stat) )
            fprintf( fd, " levels()" );
        }
        fprintf( fd, "\n" );
	}
      }
    }
  }

  if ( sru_format_p && !statement_undefined_p(mod_stat) ) {
    free_persistant_statement_to_int( s_to_l );
  } else {
    fprintf( fd,
             "\n****************** End of Dependence Graph ******************\n" );
  }

  debug_off();
}



/****************************************************************
 * FOLLOWING FUNCTIONS ARE INTENDED TO PRODUCE DEPENDENCE GRAPH
 * IN GRAPHIZ DOT FORMAT
 */

/* Context structure used by gen recurse */
struct prettyprint_dot_context {
  int previous_ordering;
  bool ordered;
  bool print_statement;
  FILE *fd;
  statement current;
};
typedef struct prettyprint_dot_context *dot_ctx;

/** \def dot_nodes_recurse( ctx, s )
  Recurse on statement s with context ctx
  Intended to be called while already on a gen_recurse recursion
 */
#define dot_nodes_recurse( ctx, s )   { \
  ctx->current = s; \
  gen_context_recurse( s, \
    ctx, \
    statement_domain, \
    prettyprint_dot_nodes, \
    gen_true ); \
}

/** \def dot_print_label_string( fd, str )
  print the string str in file descriptor fd, removing all \n
 */
#define dot_print_label_string( fd, str ) \
while ( *str ) { \
  char c = *str++; \
  if ( c == '"' ) { /* some char must be escaped */ \
    (void) putc( '\\', fd); \
  } \
  if ( c != '\n' ) { \
    (void) putc( c, fd); \
  } \
}

/** \fn static void prettyprint_dot_label( FILE *fd, statement s, bool print_statement )
 *  \brief Print the label for a statement. It will only output the first line
 *  after having removed comments.
 *  \param fd is the file descriptor
 *  \param s is the statement
 *  \param print_statement, if set to false then only print the ordering
 */
static void prettyprint_dot_label( FILE *fd, statement s, bool print_statement ) {

  if( ! print_statement ) {
    // Print only ordering
    long int o = statement_ordering(s);
    fprintf( fd, "(%ld,%ld)", ORDERING_NUMBER(o), ORDERING_STATEMENT(o));
  } else {
    // Print the code

    // saving comments
    string i_comments = statement_comments(s);

    // remove thems
    statement_comments(s) = string_undefined;

    // Get the text without comments
    list
        sentences =
      text_sentences(text_statement_enclosed(entity_undefined,0,s,false,true,NIL ) );

    // Restoring comments
    statement_comments(s) = i_comments;

    // Print the first sentence
    if(sentences) {
      sentence sent = SENTENCE(CAR( sentences ));
      if(sentence_formatted_p(sent)) {
        string str = sentence_formatted(sent);
        dot_print_label_string( fd, str );
      } else {
        unformatted u = sentence_unformatted(sent);
        cons *lw = unformatted_words(u);
        while(lw) {
          string str = STRING(CAR(lw));
          dot_print_label_string( fd, str )
          lw = CDR(lw);

        }
      }
    }
  }
}

/** \fn static bool prettyprint_dot_nodes( statement s, dot_ctx ctx )
 *  \brief Print nodes for statements, the recursion is quite complicated.
 *  for instance when we see a loop, we print the header, and we call a
 *  self-gen_recursion on the body to create separate node for each statement
 *  inside the loop.
 *  Called by gen_recurse_context
 *  \param s is the current statement
 *  \param ctx is the gen_recurse context
 *  \return true for blocks and simple statement, false for loop, test, ...
 */
static bool prettyprint_dot_nodes( statement s, dot_ctx ctx ) {
  bool gen_recurse = true;

  // We ignore the current statement (infinite recursion) and blocks
  if(ctx->current != s && ! statement_block_p( s ) ) {
    int ordering = statement_ordering(s);

    if ( ctx->ordered ) {
      // When we have produced a previous statement,
      // we chain it with current one with a very high weight
      if ( ctx->previous_ordering > 0 ) {
        fprintf( ctx->fd,
                 "    \"%d\" -> \"%d\";\n",
                 ctx->previous_ordering,
                 ordering ); // We really want ordering to be respected :-)
      }
      ctx->previous_ordering = ordering;
    }

    // Create the node
    fprintf( ctx->fd, "    %d  [label=\"", ordering);

    // Specials cases
    instruction i = statement_instruction(s);
    switch ( instruction_tag(i) ) {
      case is_instruction_test:
        // It's a test, we print the test itself first
        prettyprint_dot_label( ctx->fd, s, ctx->print_statement );
        fprintf( ctx->fd, "\"];\n" );
        // FIXME "else" won't appear in output...
        // but I've no "simple" solution for the moment :-(

        // Recurse on test bodies (true & false)
        dot_nodes_recurse( ctx, s );
        gen_recurse = false; // No more further recursion
        break;
      case is_instruction_sequence:
        break;
      case is_instruction_loop:
      case is_instruction_whileloop:
      case is_instruction_forloop:
        // We have a loop, first print the header
        prettyprint_dot_label( ctx->fd, s, ctx->print_statement );
        fprintf( ctx->fd, "\"];\n" );
        // Recurse on loop body now
        dot_nodes_recurse( ctx, s )
        ;
        gen_recurse = false; // No more further recursion
        break;
      case is_instruction_goto:
      case is_instruction_unstructured:// FIXME ???
      case is_instruction_call:
      case is_instruction_expression:
      default:
        // Standard output, print the statement
        prettyprint_dot_label( ctx->fd, s, ctx->print_statement );
        fprintf( ctx->fd, "\"];\n\n" );
        break;
    }
  }
  return gen_recurse;
}



/** \fn void prettyprint_dot_dependence_graph( FILE * fd,
 *                                             statement mod_stat,
 *                                             graph mod_graph )
 *  \brief Output dependence graph in a file in graphviz dot format
 *  \param fd is the file descriptor where to output
 *  \param mod_stat is the module statement (not necessary module, it can be
 *  a block statement for instance
 *  \param graph is the dependence graph to print
 */
void prettyprint_dot_dependence_graph( FILE * fd,
                                       statement mod_stat,
                                       graph mod_graph ) {
  debug_on( "RICEDG_DEBUG_LEVEL" );

  ifdebug(8) {
    /* There is no guarantee that the ordering_to_statement()
     * hash table is the proper one */
    print_ordering_to_statement( );
  }
  // Begin graph
  fprintf( fd, "digraph {\n" );


  bool centered = get_bool_property( "PRINT_DOTDG_CENTERED" );
  const char* title = get_string_property( "PRINT_DOTDG_TITLE" );
  const char* title_position = get_string_property( "PRINT_DOTDG_TITLE_POSITION" );
  const char* background = get_string_property( "PRINT_DOTDG_BACKGROUND" );
  const char* nodeshape= get_string_property( "PRINT_DOTDG_NODE_SHAPE" );
  const char* nodeshapecolor = get_string_property( "PRINT_DOTDG_NODE_SHAPE_COLOR" );
  const char* nodefillcolor = get_string_property( "PRINT_DOTDG_NODE_FILL_COLOR" );
  const char* nodefontcolor = get_string_property( "PRINT_DOTDG_NODE_FONT_COLOR" );
  const char* nodefontsize = get_string_property( "PRINT_DOTDG_NODE_FONT_SIZE" );
  const char* nodefontface = get_string_property( "PRINT_DOTDG_NODE_FONT_FACE" );


  /* graph style */
  fprintf( fd,
           "\n"
           "  /* graph style */\n");
  // Print title if not empty
  if( !same_string_p( title, "" ) ) {
    fprintf( fd, "  label=\"%s\";\n", title);
  }
  // Print title location if not empty
  if( !same_string_p( title_position, "" ) ) {
    fprintf( fd, "  labelloc=\"%s\";\n", title_position);
  }
  // Print background color if not empty
  if( !same_string_p( background, "" ) ) {
    fprintf( fd, "  bgcolor=\"%s\";\n", background);
  }
  if( centered ) {
    fprintf( fd, "  center=\"true\";\n");
  }
  fprintf( fd, "\n\n");



  /* Nodes style */
  fprintf( fd,
           "\n"
             "  /* Nodes style */\n"
             "  node [shape=\"%s\",color=\"%s\",fillcolor=\"%s\","
             "fontcolor=\"%s\",fontsize=\"%s\",fontname=\"%s\"];\n\n",
           nodeshape,
           nodeshapecolor,
           nodefillcolor,
           nodefontcolor,
           nodefontsize,
           nodefontface );




  // Should we print the statement or only its ordering ?
  bool print_statement = get_bool_property( "PRINT_DOTDG_STATEMENT" );
  // Should node be ordered top down according to the statement ordering ?
  bool ordered = get_bool_property( "PRINT_DOTDG_TOP_DOWN_ORDERED" );

  if( ordered || print_statement ) {
  fprintf( fd,
               "\n  {\n    /* Print nodes for statements %s order them */\n\n",
             ordered ? "and" : "but don't" );

    if ( ordered ) {
      fprintf( fd,
               "    /* ordering edges must be invisible, so set background color */\n"
               "    edge  [weight=100,color=%s];\n\n",
               background );
    }

    // Generate nodes
    struct prettyprint_dot_context ctx;
    ctx.fd = fd;
    ctx.current = NULL;
    ctx.ordered = ordered;
    ctx.print_statement = print_statement;
    ctx.previous_ordering = 0;

    gen_context_recurse( mod_stat,
        &ctx,
        statement_domain,
        prettyprint_dot_nodes,
        gen_true );
    fprintf( fd, "  }\n\n" );
  }


  fprintf( fd, "  /* Print arcs between statements */\n\n" );
  fprintf( fd, "  /* Dependence arcs won't constrain node positions */\nedge [constraint=false];\n\n" );



  const char* flowdep_color = get_string_property( "PRINT_DOTDG_FLOW_DEP_COLOR" );
  const char* flowdep_style = get_string_property( "PRINT_DOTDG_FLOW_DEP_STYLE" );
  const char* antidep_color = get_string_property( "PRINT_DOTDG_ANTI_DEP_COLOR" );
  const char* antidep_style = get_string_property( "PRINT_DOTDG_ANTI_DEP_STYLE" );
  const char* outputdep_color = get_string_property( "PRINT_DOTDG_OUTPUT_DEP_COLOR" );
  const char* outputdep_style = get_string_property( "PRINT_DOTDG_OUTPUT_DEP_STYLE" );
  const char* inputdep_color = get_string_property( "PRINT_DOTDG_INPUT_DEP_COLOR" );
  const char* inputdep_style = get_string_property( "PRINT_DOTDG_INPUT_DEP_STYLE" );

  bool mask_loop_carried = get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS");
  set_enclosing_loops_map( loops_mapping_of_statement(mod_stat) );

  // Loop over the graph and print all dependences
  FOREACH( vertex, v1 , graph_vertices( mod_graph ) ) {
    statement s1 = vertex_to_statement( v1 );
    list loops1 = load_statement_enclosing_loops(s1);
    FOREACH( successor, su, vertex_successors(v1) )
    {
      vertex v2 = successor_vertex( su );
      statement s2 = vertex_to_statement( v2 );
      list loops2 = load_statement_enclosing_loops(s2);
      dg_arc_label dal = (dg_arc_label) successor_arc_label( su );
      int nbrcomloops = FindMaximumCommonLevel(loops1, loops2);
      FOREACH( conflict, c, dg_arc_label_conflicts( dal ) )
      {
        action source_act = effect_action( conflict_source( c ) );
        action sink_act = effect_action( conflict_sink( c ) );
        reference sink_ref = effect_any_reference( conflict_sink( c ) );
        reference source_ref = effect_any_reference( conflict_source( c ) );
        const char* color = inputdep_color;
        const char* style = inputdep_style;

        /*
         * Here we check if this arc is only loop carried or not and we might
         * skip it if required by property
         */
        bool keep_this_conflict = true;
        if(mask_loop_carried && conflict_cone(c) != cone_undefined) {
          list lls = cone_levels(conflict_cone(c));
          keep_this_conflict = false;
          FOREACH(int, level, lls) {
            if(level > nbrcomloops) {
              //
              keep_this_conflict=true;
              break;
            }
          }
        }
        if(!keep_this_conflict) {
          continue;
        }


        // Ok let's display it now.

        if( action_read_p( source_act ) && action_write_p( sink_act ) ) {
          color = antidep_color;
          style = antidep_style;
        } else if( action_write_p( source_act ) && action_write_p( sink_act ) ) {
          color = outputdep_color;
          style = outputdep_style;
        } else if( action_write_p( source_act ) && action_read_p( sink_act ) ) {
          color = flowdep_color;
          style = flowdep_style;
        }
        fprintf( fd,
                 "%d -> %d [color=%s,style=%s,label=\"",
                 (int) statement_ordering(s1),
                 (int) statement_ordering(s2),
                 color,
                 style );
        fprintf( fd,
                 "%c <",
                 action_read_p( source_act ) ? 'R'
                                             : 'W' );
        print_words( fd,
                     effect_words_reference( source_ref ) );
        fprintf( fd, ">\\n" );
        fprintf( fd,
                 "%c <",
                 action_read_p( sink_act ) ? 'R'
                                           : 'W' );
        print_words( fd,
                     effect_words_reference( sink_ref ) );
        fprintf( fd, ">\\n" );


        // Print the levels and the cone
        if ( conflict_cone( c ) != cone_undefined ) {
          fprintf( fd, " levels(" );
          MAPL(pl, {
                fprintf(fd, pl==cone_levels(conflict_cone(c))? "%td" : ",%td",
                    INT(CAR(pl)));
              }, cone_levels(conflict_cone(c)));
          fprintf( fd, ") " );

          if ( get_bool_property( "PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES" ) ) {
            Ptsg gs = (Ptsg) cone_generating_system( conflict_cone( c ) );
            if ( !SG_UNDEFINED_P( gs ) ) {
              if ( sg_nbre_sommets( gs ) == 1 && sg_nbre_rayons( gs ) == 0
                  && sg_nbre_droites( gs ) == 0 ) {
                /* uniform dependence */
                fprintf( fd, "uniform" );
                fprint_lsom_as_dense( fd, sg_sommets( gs ), gs->base );
              } else {
                sg_fprint_as_ddv( fd, gs );
              }
            }
          }
        } else {
          fprintf( fd, " levels()" );
        }
        fprintf( fd, "\"];\n" );
      }
    }
  }

  fprintf( fd, "\n}\n" );

  reset_enclosing_loops_map();

  debug_off( );
}

/*
 * END OF OUTPUT IN GRAPHIZ DOT FORMAT
 ****************************************************************/



/* Do not print vertices and arcs ignored by the parallelization algorithms.
 * At least, hopefully...
 */
void 
prettyprint_dependence_graph_view(FILE * fd,
				  statement mod_stat,
				  graph mod_graph)
{
    cons *pv1, *ps, *pc;
    Ptsg gs;
    int banner_number = 0;

    banner_number =
	get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS") +
	    2*get_bool_property
		("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS") +
		    4*get_bool_property
			("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES");

    fprintf(fd, "%s\n", dependence_graph_banner[banner_number]);

    set_enclosing_loops_map( loops_mapping_of_statement(mod_stat) );

    debug_on("RICEDG_DEBUG_LEVEL");

    for (pv1 = graph_vertices(mod_graph); !ENDP(pv1); pv1 = CDR(pv1)) {
	vertex v1 = VERTEX(CAR(pv1));
	statement s1 = vertex_to_statement(v1);
	list loops1 = load_statement_enclosing_loops(s1);

	for (ps = vertex_successors(v1); !ENDP(ps); ps = CDR(ps)) {
	    successor su = SUCCESSOR(CAR(ps));
	    vertex v2 = successor_vertex(su);
	    statement s2 = vertex_to_statement(v2);
	    list loops2 = load_statement_enclosing_loops(s2);
	    dg_arc_label dal = (dg_arc_label) successor_arc_label(su);

      /*
       * If we have more than one conflict, let's sort them !
       */
      int nb_conflicts = gen_length(dg_arc_label_conflicts(dal));
      if ( nb_conflicts > 1 ) {
        /*
         * Convert the list to an array for sorting
         * 20 is the initial size, should be enough for most of the case
         */
        gen_array_t conflicts_array = gen_array_make( 20 );
        list_to_array( dg_arc_label_conflicts(dal), conflicts_array );
        qsort( gen_array_pointer( conflicts_array ),
               gen_array_nitems( conflicts_array ),
               sizeof(void *),
               (gen_cmp_func_t) conflicts_sort_callback);
        list conflicts_list = NIL;
        GEN_ARRAY_FOREACH(conflict, s, conflicts_array)
          conflicts_list = CONS(conflict, s, conflicts_list);
        gen_array_free(conflicts_array);
        dg_arc_label_conflicts(dal)=conflicts_list;
      }
	    int nbrcomloops = FindMaximumCommonLevel(loops1, loops2);
	    for (pc = dg_arc_label_conflicts(dal); !ENDP(pc); pc = CDR(pc)) {
		conflict c = CONFLICT(CAR(pc));

		if(conflict_cone(c) == cone_undefined) continue;
		else {
		    cons * lls = cone_levels(conflict_cone(c));
		    cons *llsred =NIL;
		    if (get_bool_property("PRINT_DEPENDENCE_GRAPH_WITHOUT_PRIVATIZED_DEPS")){
			MAPL(pl,{
			    _int level = INT(CAR(pl));
			    if (level <= nbrcomloops) {
				if (! ignore_this_conflict(v1,v2,c,level)) {
				    llsred = gen_nconc(llsred, CONS(INT, level, NIL));
				}
			    }
			    else {
				if (get_bool_property
				    ("PRINT_DEPENDENCE_GRAPH_WITHOUT_NOLOOPCARRIED_DEPS")) {
				    continue;
				}
				else llsred = gen_nconc(llsred, CONS(INT, level, NIL));
			    }
			}, lls);
		    }
		    if (llsred == NIL) continue;
		    else {
			/*if (!	entity_scalar_p(reference_variable
			  (effect_any_reference(conflict_source(c))))) { */

			fprintf(fd, "\t%02td --> %02td with conflicts\n", 
				statement_number(s1), statement_number(s2));
			
			fprintf(fd, "\t\tfrom ");
			print_words(fd, words_effect(conflict_source(c)));

			fprintf(fd, " to ");
			print_words(fd, words_effect(conflict_sink(c)));
			
			fprintf(fd, " at levels ");
			MAPL(pl, {
			    fprintf(fd, " %td", INT(CAR(pl)));
			}, llsred);

			fprintf(fd, "\n");
			if(get_bool_property
			   ("PRINT_DEPENDENCE_GRAPH_WITH_DEPENDENCE_CONES")) {
			    gs = (Ptsg)cone_generating_system(conflict_cone(c));
			    if (!SG_UNDEFINED_P(gs)) {
				/* sg_fprint(fd,gs,entity_local_name); */
				sg_fprint_as_dense(fd, gs, gs->base);
				ifdebug(2) {
				    Psysteme sc1 = sc_new();
				    sc1 = sg_to_sc_chernikova(gs);
				    (void) fprintf(fd,"syst. lin. correspondant au  syst. gen.:\n");
				    sc_fprint(fd, sc1, (get_variable_name_t) entity_local_name);
				}
			    }
			    fprintf(fd, "\n");
			}
		    }
		}
	    }
	}
    }
    clean_enclosing_loops();
    debug_off();
    fprintf(fd, "\n****************** End of Dependence Graph "
	    "******************\n");
}

void 
print_vect_in_vertice_val(fd,v,b)
FILE *fd;
Pvecteur v;
Pbase b;
{
    fprintf(fd,"(");
    for(; b!=NULL; b=b->succ) {
	fprint_string_Value(fd, " ",vect_coeff(b->var,v));
	if(b->succ !=NULL)
	    fprintf(fd,",");
    }
    fprintf(fd," )");
}

void 
print_dependence_cone(fd,dc,basis)
FILE *fd;
Ptsg dc;
Pbase basis;
{
    Psommet v;
    Pray_dte r;
    fprintf(fd,"\nDependence cone :\n");
    if (SG_UNDEFINED_P(dc)) fprintf(fd, "NULL \n");
    else {
	fprintf(fd,"basis :");
	base_fprint(fd, basis, (get_variable_name_t) safe_entity_name);
	fprintf(fd,"%d vertice(s) :",sg_nbre_sommets(dc));
	v = sg_sommets(dc);
	for(; v!=NULL; v= v->succ) {
	    fprint_string_Value(fd, " \n\t denominator = ", v->denominateur);
	    fprintf(fd, "\t");
	    print_vect_in_vertice_val(fd,v->vecteur,basis);	
	}

	fprintf(fd,"\n%d ray(s) : ",sg_nbre_rayons(dc));
   
	for(r = sg_rayons(dc); r!=NULL; r=r->succ) 
	    print_vect_in_vertice_val(fd,r->vecteur,basis);
    
	fprintf(fd,"\n%d line(s) : ",sg_nbre_droites(dc));
   
	for(r = sg_droites(dc); r!=NULL; r=r->succ) 
	    print_vect_in_vertice_val(fd,r->vecteur,basis);
	fprintf(fd,"\n");
    }
}


/* for an improved dependence test (Beatrice Creusillet)
 *
 * The routine name says it all. Only constraints transitively connected
 * to a constraint referencing a variable of interest with a non-zero 
 * coefficient are copied from sc to sc_res.
 *
 * Input:
 *  sc: unchanged
 *  variables: list of variables of interest (e.g. phi variables of regions)
 * Output:
 *  sc_res: a newly allocated restricted version of sc
 * Temporary:
 *  sc: the pointer is modified to make debugging more interesting:-(
 *      (no impact on the value pointed to by sc on procedure entry)
 *
 * FI: I'm sceptical... OK for speed, quid of accuracy? Nonfeasibility
 * due to existencial variables is lost if these variables are not
 * transitively related to the so-called variables of interest, isn'it?
 * Well, I do not manage to build a counter example because existential
 * problems are caught by the precondition normalization. Although it is
 * not as strong as one could wish, it gets lots of stuff...
 */

Psysteme 
sc_restricted_to_variables_transitive_closure(sc, variables)
Psysteme sc;
Pbase variables;
{
    Psysteme sc_res;
    
    Pcontrainte c, c_pred, c_suiv;
    Pvecteur v;    
    
    /* cas particuliers */ 
    if(sc_rn_p(sc)){ 
	/* we return a sc_rn predicate,on the space of the input variables */ 
	sc_res = sc_rn(variables);
	return sc_res;
    }
    if (sc_empty_p(sc)) {
	/* we return an empty predicate,on the space of the input variables */ 
	sc_res = sc_empty(variables);
	return sc_res;
    } 

    /* sc has no particularity. We just scan its equalities and inequalities 
     * to find which variables are related to the PHI variables */ 
    sc = sc_dup(sc);
    base_rm(sc->base);
    sc->base = BASE_NULLE;
    sc_creer_base(sc);

    sc_res = sc_new();
    sc_res->base = variables;
    sc_res->dimension = vect_size(variables);

    while (vect_common_variables_p(sc->base, sc_res->base)) {
	
	/* equalities first */ 
	c = sc_egalites(sc);
	c_pred = (Pcontrainte) NULL; 
	while (c != (Pcontrainte) NULL) { 
	    c_suiv = c->succ; 

	    /* if a constraint is found in sc, that contains a variable 
	     * that already belongs to the base of sc_res, then this 
	     * constraint is removed from sc, added to sc_res  and its 
	     * variables are added to the base of sc_res */ 
	    if (vect_common_variables_p(c->vecteur,sc_res->base)){ 

		/* the constraint is removed from sc */ 		
		if (c_pred != (Pcontrainte) NULL) 
		    c_pred->succ = c_suiv; 
		else 
		    sc_egalites(sc) = c_suiv; 

		/* and added to sc_res */
		c->succ = (Pcontrainte) NULL;
		sc_add_egalite(sc_res, c);

		/* sc_res base is updated with the variables occuring in c */
		for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ) { 
		    if (vecteur_var(v) != TCST) 
			sc_base_add_variable(sc_res, vecteur_var(v));
		} 
	    } 
	    else 
		c_pred = c; 
	    c = c_suiv;
	} 
	
	/* inequalities then */
	c = sc_inegalites(sc);
	c_pred = (Pcontrainte) NULL; 
	while (c != (Pcontrainte) NULL) { 
	    c_suiv = c->succ; 

	    /* if a constraint is found in sc, that contains a variable 
	     * that already belongs to the base of sc_res, then this 
	     * constraint is removed from sc, added to sc_res  and its 
	     * variables are added to the base of sc_res */ 

	    if (vect_common_variables_p(c->vecteur,sc_res->base)){
 
		/* the constraint is removed from sc */ 		
		if (c_pred != (Pcontrainte) NULL) 
		    c_pred->succ = c_suiv; 
		else 
		    sc_inegalites(sc) = c_suiv; 

		/* and added to sc_res */
		c->succ = (Pcontrainte) NULL;
		sc_add_inegalite(sc_res, c);

		/* sc_res base is updated with the variables occuring in c */
		for(v = c->vecteur; !VECTEUR_NUL_P(v); v = v->succ) { 
		    if (vecteur_var(v) != TCST) 
			sc_base_add_variable(sc_res, vecteur_var(v));
		} 
	    } 
	    else 
		c_pred = c; 
	    c = c_suiv;
	} 
		
	/* update sc base */
	base_rm(sc->base);
	sc_base(sc) = (Pbase) NULL; 
	(void) sc_creer_base(sc); 
    } /* while */ 
    
    
    sc_rm(sc);
    sc = NULL;
        

    return(sc_res);
}
/** 
 * creates a hash_table containing statements from @a statements as keys and their respective succesors according to @a dg as values
 * 
 * @param statements input statements
 * @param dg dependecy graph
 * 
 * @return allocated hash_table with (statement,successors pairs)
 */
hash_table statements_to_successors(list statements, graph dg)
{
    hash_table successors = hash_table_make(hash_pointer, HASH_DEFAULT_SIZE);
    FOREACH(VERTEX,v,graph_vertices(dg))
    {
        statement s = vertex_to_statement(v);
        if( !statement_undefined_p(gen_find_eq(s,statements)))
        {
            list succ = vertex_successors(v);
            hash_put(successors,s,succ);
        }
    }
    return successors;
}


/* That's all */
