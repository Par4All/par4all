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
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "resources.h"
#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
/* #include "statistics.h" */

static int *nbr_no_perf_nest_loop_of_depth, *nbr_perf_nest_loop_of_depth,
            nbr_loop, nbr_perfectly_nested_loop, nbr_interested_loop,
            nbr_no_perfectly_nested_loop,nbr_nested_loop,
            max_depth_perfect, max_depth_no_perfect,cpt ;
static bool first_turn ;


typedef enum { is_unknown, is_a_perf_nes_loop, is_a_no_perf_nes_loop, 
	       is_a_no_perf_nes_loop_t, is_a_call, is_a_continue,
                is_a_while, is_a_test, } contenu_t;
typedef struct {
   hash_table contenu;
   hash_table depth;
   stack statement_stack;
} * context_p, context_t;

static bool loop_flt(loop l, context_p context )
{ 
  cpt++;  
  if( ! first_turn ) first_turn = true; 
  return true;
} 
static void loop_rwt(loop l, context_p context  ) 
{  
  contenu_t contenu;
  intptr_t depth;
  if (first_turn) {
    depth =1;
    hash_put(context->depth,stack_head(context->statement_stack), 
	     ( void *) depth);  
    contenu =is_a_perf_nes_loop;
    hash_put(context->contenu,stack_head(context->statement_stack), 
	   (void *) contenu );
    first_turn= false;
  }
  else {  
    statement s = loop_body(l);
    contenu = (contenu_t) hash_get(context->contenu, s);
    if (contenu==is_a_no_perf_nes_loop_t)
        contenu=is_a_no_perf_nes_loop;
    depth= (intptr_t) hash_get(context->depth, s); 
    depth++;
    hash_put(context->depth,stack_head(context->statement_stack), 
             ( void *) depth); 
    hash_put(context->contenu,stack_head(context->statement_stack), 
	     (void *) contenu  );
  };
  nbr_loop++;
  cpt--;
  if ( cpt ==0) {
     statement s= loop_body(l);
     contenu = (contenu_t) hash_get(context->contenu, s);
     if ( contenu == is_a_no_perf_nes_loop_t){
      nbr_interested_loop++;
     };
  };
      
}
static bool stmt_flt(statement s,context_p context )
{  
 
   stack_push(s, context->statement_stack); 
   return true;
}


static void stmt_rwt( statement s, context_p context)
{
  /*  pips_assert("information associated to statement",
       hash_defined_p(context->contenu, s)         );  

     pips_assert("information associated to statement",
     hash_defined_p(context->depth, s)         );  */
       stack_pop(context->statement_stack);    
	 
} 
static bool seq_flt( sequence sq, context_p context  )
{
  return true;
}
static void seq_rwt(sequence sq, context_p context)
{
  
  contenu_t contenu= is_unknown, contenu_loop=is_unknown ;
  intptr_t depth=0;
  intptr_t max=0;
  intptr_t nbr=0;
  bool found=false;
  list l= sequence_statements(sq);
   MAP(STATEMENT, s,
  {
     contenu = (contenu_t) hash_get(context->contenu, s);
     if ( (nbr !=0)&&((contenu==  is_a_perf_nes_loop)||
                     ( contenu==  is_a_no_perf_nes_loop)|| 
                     ( contenu==   is_a_no_perf_nes_loop_t) ) )
         found=true;
     if ((contenu != is_a_perf_nes_loop)&& 
         ( contenu !=  is_a_no_perf_nes_loop)&& 
         ( contenu!=  is_a_no_perf_nes_loop_t) && 
         ( contenu != is_a_continue) )
         found=true; 
     switch ( contenu ){
        case is_a_perf_nes_loop : {nbr++; contenu_loop=contenu;
          depth = (intptr_t ) hash_get(context->depth, s);
          (*(nbr_perf_nest_loop_of_depth+depth))++;
          if( depth > max ) max =depth;
          nbr_perfectly_nested_loop++;
          nbr_nested_loop++; 
          if (max > max_depth_perfect) max_depth_perfect=max;
          break;
        }
    
        case is_a_no_perf_nes_loop: {nbr++; contenu_loop=contenu;
           depth = (intptr_t ) hash_get(context->depth, s);
           (*(nbr_no_perf_nest_loop_of_depth+depth))++;
           nbr_no_perfectly_nested_loop++;
           nbr_nested_loop++;
           if( depth > max ) max =depth;
           if (max > max_depth_no_perfect) max_depth_no_perfect=max;
           break ;
        }
        case is_a_no_perf_nes_loop_t: { nbr++; contenu_loop=contenu;
           depth = (intptr_t ) hash_get(context->depth, s);
           if( depth > max ) max =depth; break ;}   
        default: 
        break;
     
     };
  
  };, l);
  if (max > 0) {
       if (found){ 
          contenu =is_a_no_perf_nes_loop_t;
          hash_put(context->contenu,
	  stack_head(context->statement_stack), 
	  (void *)contenu); 
          hash_put(context->depth,
	  stack_head(context->statement_stack), 
	  (void * ) max );
       }
       else{
          hash_put(context->contenu,
	      stack_head(context->statement_stack), 
	      (void *) contenu_loop); 
          hash_put(context->depth,
	      stack_head(context->statement_stack), 
	      (void * ) max);
          switch ( contenu_loop ){  
             case is_a_perf_nes_loop: {(*(nbr_perf_nest_loop_of_depth+depth))--;
	       nbr_perfectly_nested_loop--; nbr_nested_loop--;   
               break; 
             }
             case is_a_no_perf_nes_loop :{
               (*(nbr_no_perf_nest_loop_of_depth+depth))--;
	       nbr_no_perfectly_nested_loop--;nbr_nested_loop--;
               break;
             }
	  default:break;                               
          };
       }

  }
  else {   
      contenu =is_unknown;
      hash_put(context->contenu,
	       stack_head(context->statement_stack), 
	       (void *) contenu);
      depth=0;
      hash_put(context->depth,
	      stack_head(context->statement_stack), 
	      (void *) depth);
  };

} 
static bool uns_flt(unstructured u, context_p context   )
{
    return true;
}
static void uns_rwt(unstructured u, context_p context)
{ 
  list blocks = NIL;
  statement s;
  contenu_t contenu;
  control c_in=unstructured_control(u);
  intptr_t max =0;
  intptr_t depth;
  CONTROL_MAP(c, 
     {
        s=control_statement(c);
        contenu = (contenu_t) hash_get(context->contenu, s);
        switch ( contenu ){
           case is_a_perf_nes_loop : {      
              depth = (intptr_t ) hash_get(context->depth, s);
              (*(nbr_perf_nest_loop_of_depth+depth))++;
               if( depth > max ) max =depth;
               nbr_perfectly_nested_loop++;
               nbr_nested_loop++; 
               if (max > max_depth_perfect) max_depth_perfect=max;
               break;
           }
           case is_a_no_perf_nes_loop: {        
              depth = (intptr_t ) hash_get(context->depth, s);
              (*(nbr_no_perf_nest_loop_of_depth+depth))++;
              nbr_no_perfectly_nested_loop++;
              nbr_nested_loop++;
              if( depth > max ) max =depth;
              if (max > max_depth_no_perfect) max_depth_no_perfect=max;
              break ;
           }
           case is_a_no_perf_nes_loop_t: {     
              depth = (intptr_t ) hash_get(context->depth, s);
              if( depth > max ) max =depth; break ;}   
              default: 
          break;
        };
    }, 
    c_in, blocks);
    gen_free_list(blocks);
    if (max > 0) {
        contenu =is_a_no_perf_nes_loop_t;
        hash_put(context->contenu,
	stack_head(context->statement_stack), 
	(void *)contenu); 
        hash_put(context->depth,
	stack_head(context->statement_stack), 
	 (void * ) max );
    }
    else
    { contenu =is_unknown;            ;
      hash_put(context->contenu,
      stack_head(context->statement_stack), 
      (void *)contenu);
    }
} 
static bool test_flt(test t, context_p context)
{
  return true;
}
static void test_rwt(test t, context_p context)
{
  statement s1=test_true(t);
  statement s2=test_false(t);  
  intptr_t max=0;
  intptr_t depth;
  contenu_t contenu1=is_unknown, contenu2=is_unknown,contenu =is_unknown;
  contenu1 = (contenu_t) hash_get(context->contenu, s1);
  switch ( contenu1 ){
     case is_a_perf_nes_loop: {
       depth= (intptr_t) hash_get(context->depth, s1);
       (*(nbr_perf_nest_loop_of_depth+depth))++;
       nbr_perfectly_nested_loop++;
       nbr_nested_loop++;
       max=depth;
       contenu= is_a_no_perf_nes_loop_t;
       hash_put(context->contenu,
       stack_head(context->statement_stack), 
       (void *) contenu );
       if (max > max_depth_perfect) max_depth_perfect=max;
       break;
     };
     case is_a_no_perf_nes_loop :{
        depth= (intptr_t) hash_get(context->depth, s1);
        (*(nbr_no_perf_nest_loop_of_depth+depth))++;
	nbr_no_perfectly_nested_loop++;
        nbr_nested_loop++;
        max=depth;
        contenu= is_a_no_perf_nes_loop_t;
        hash_put(context->contenu,
	stack_head(context->statement_stack), 
	(void *) contenu  );
        if (max > max_depth_no_perfect) max_depth_no_perfect=max;
         break;
      };
      case is_a_no_perf_nes_loop_t: {
        depth= (intptr_t) hash_get(context->depth, s1);
        max=depth;
        contenu= is_a_no_perf_nes_loop_t;
        hash_put(context->contenu,
	stack_head(context->statement_stack), 
	(void *) contenu  );
        break;
     };
      default:break; 
  };
  contenu2 = (contenu_t) hash_get(context->contenu, s2);
      switch ( contenu2 ){
         case is_a_perf_nes_loop : {
           depth= (intptr_t) hash_get(context->depth, s2);
           (*(nbr_perf_nest_loop_of_depth+depth))++;
           nbr_perfectly_nested_loop++;
           nbr_nested_loop++;
           if (depth > max ) max =depth;
	   contenu= is_a_no_perf_nes_loop_t;
           hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	   (void *) contenu);
           if (max > max_depth_perfect) max_depth_perfect=max;
           break;
         };

         case is_a_no_perf_nes_loop: {
           depth= (intptr_t) hash_get(context->depth, s2);
           (*(nbr_no_perf_nest_loop_of_depth+depth))++;
           nbr_no_perfectly_nested_loop++;
           nbr_nested_loop++;
           if (depth > max ) max =depth;
	   contenu= is_a_no_perf_nes_loop_t;
           hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	   (void *) contenu  );
           if (max > max_depth_no_perfect) max_depth_no_perfect=max;
           break;
         };
         case is_a_no_perf_nes_loop_t: {
           depth= (intptr_t) hash_get(context->depth, s2);
           if ( depth > max)  max =depth;
	   contenu= is_a_no_perf_nes_loop_t;
           hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	   (void *) contenu );
           break;
         };
         default: break;
      }; 
      hash_put(context->depth,
      stack_head(context->statement_stack), 
      (void *) max   );
      if (max==0) {
	  contenu=is_a_test;
          hash_put(context->contenu,
	  stack_head(context->statement_stack), 
	  (void *) contenu  ); 
      };
} 
static bool call_flt( call  ca, context_p context)
{
 
return true ;
}
static void call_rwt(call  ca, context_p context)
{
  contenu_t contenu;
  if (  strcmp(  entity_name(  call_function(ca)),"TOP-LEVEL:CONTINUE" )==0){
    contenu=is_a_continue; 
    hash_put(context->contenu,
    stack_head(context->statement_stack), 
    (void *) contenu);
  }
  else{
    contenu=is_a_call; 
    hash_put(context->contenu,
    stack_head(context->statement_stack), 
    (void *)contenu );
  };
} 
static void wl_rwt(whileloop w, context_p context)
{  
  contenu_t contenu=is_a_while;
   hash_put(context->contenu,
   stack_head(context->statement_stack), 
   (void *) contenu );
} 
static void initialize ()
{
    int i;
    nbr_no_perf_nest_loop_of_depth =(int  *) malloc(100 * sizeof (int ));
    nbr_perf_nest_loop_of_depth =(int  *) malloc(100* sizeof (int )) ;
    for (i=0; i<=99;i++)
       *( nbr_no_perf_nest_loop_of_depth+i) =0;
    for (i=0; i<=99;i++)
       *( nbr_perf_nest_loop_of_depth+i) =0;
    cpt=0; 
    nbr_perfectly_nested_loop=0; nbr_no_perfectly_nested_loop=0;
    nbr_interested_loop=0;
    nbr_nested_loop=0 ;  max_depth_perfect=0  ;
    nbr_loop=0;max_depth_no_perfect=0;first_turn=true;
    first_turn = true;
} 
static void put_result(string filename)
{
  int i;
  FILE * file = safe_fopen(filename, "w");  
  fprintf(file,
          "loops: %d\n"
	  "nested loops: %d\n"
	  "perfectly nested loops: %d\n"
	  "non perfectly nested loops: %d\n"
	  "non perfectly nested loops which we can treat : %d\n",
	  nbr_loop, nbr_nested_loop, nbr_perfectly_nested_loop,
          nbr_no_perfectly_nested_loop, nbr_interested_loop);


  for (i=1; i<= max_depth_perfect; i++)
  if (*(nbr_perf_nest_loop_of_depth+i) !=0)
  fprintf(file,"perfectly  nested loops of depth %d: %d \n", i, 
  nbr_perf_nest_loop_of_depth[i]);
  for (i=1; i<= max_depth_no_perfect; i++)
  if (*(nbr_no_perf_nest_loop_of_depth+i) !=0)
  fprintf(file,"non perfectly  nested loops of depth %d: %d \n", i, 
  *(nbr_no_perf_nest_loop_of_depth+i));
 
  safe_fclose(file, filename);
}


int loop_statistics(string name)
{
  statement stat;
  string filename, localfilename; 
  context_t context;
                                  
  debug_on("STATISTICS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", name);
  set_current_module_entity(local_name_to_top_level_entity(name));
 
  initialize() ;
  
  stat = (statement) db_get_memory_resource(DBR_CODE, name, true); 

  context.contenu = hash_table_make(hash_pointer, 0);  
  context.depth   = hash_table_make(hash_pointer, 0);
  context.statement_stack = stack_make(statement_domain, 0, 0); 
   
  
  gen_context_multi_recurse
    (stat, & context, 
     statement_domain, stmt_flt, stmt_rwt,
     sequence_domain, seq_flt, seq_rwt,
     test_domain, test_flt, test_rwt,
     loop_domain, loop_flt, loop_rwt,
     call_domain, call_flt, call_rwt,
     whileloop_domain, gen_true, wl_rwt,
     unstructured_domain, uns_flt, uns_rwt,
     expression_domain, gen_false, gen_null,
     NULL); 


  hash_table_free(context.contenu);


  hash_table_free(context.depth);
  stack_free(&context.statement_stack); 

  localfilename = db_build_file_resource_name(DBR_STATS_FILE,
					      name, ".loop_stats");

  filename = strdup(concatenate(db_get_current_workspace_directory(), 
				"/", localfilename, NULL));

   put_result(filename);  
   free(nbr_no_perf_nest_loop_of_depth);
   free(nbr_perf_nest_loop_of_depth);
  free(filename);
  DB_PUT_FILE_RESOURCE(DBR_STATS_FILE, name, localfilename);
  reset_current_module_entity();
  
pips_debug(1, "done.\n");
  debug_off();
   
  return true;
}






















