 
#include <stdio.h>

#include "genC.h"
#include "linear.h"

#include "ri.h"
#include "database.h"

#include "resources.h"
#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "statistics.h"     

static int depth, nbr_perfectly_nested_loop, nbr_no_perfectly_nested_loop, 
nbr_interested_loop,cpt, nbr_nested_loop, nbr_perf_nest_loop_of_depth[100],
  nbr_no_perf_nest_loop_of_depth[100], max_depth_perfect, nbr_loop,som,ind,
max_depth_no_perfect;
static bool perfectly_nested ,first_turn ;
 
static bool loop_flt( loop l)
{
    int k=0,i=0,position=0,pos=0;
    bool local=FALSE;
    bool loop_rwt();
    instruction lbi = statement_instruction(loop_body(l));
    if ((cpt==0) && instruction_sequence_p(lbi)) {
        bool cond1=TRUE,  found =FALSE;
        sequence seq = instruction_sequence(lbi);
        list l =sequence_statements(seq); 
        MAP(STATEMENT,s,{
          i++; if( instruction_loop_p(statement_instruction(s))) pos=i;
        };, l);
        if ( pos !=0){
	     i=0;
	     MAP(STATEMENT,s,{
	     i++; 
	     if (i!=pos){  
		 if (instruction_loop_p(statement_instruction(s)))
		   cond1 =FALSE; 
	     } 
	     },l);
	     i=0;
	     MAP(STATEMENT,s,{ 
             i++; 
             if (i!=pos){
	           if (!instruction_call_p(statement_instruction(s))) 
                      found =TRUE; 
	           else {
	               call ca = instruction_call(statement_instruction(s));
	               if(  strcmp(  entity_name(  call_function(ca))
                       ,"TOP-LEVEL:CONTINUE" )!=0) found =TRUE ;
	           }; 
	 
	     } 
             };,l);
        };
        if (( cond1) && (found))  nbr_interested_loop++ ;
    }; 
    cpt++;
    depth++;
    if ( instruction_loop_p(lbi)) return TRUE;  
    else {
         if(instruction_sequence_p(lbi)) { 
              sequence seq = instruction_sequence(lbi);
              list l =sequence_statements(seq); 
              MAP(STATEMENT,s, {k++;if
              ( instruction_loop_p(statement_instruction(s)))
              position=k ;    ; };, l); 
	      if (position!=0 ){ 
                   k=0; 
	           MAP(STATEMENT,s, 
                   {k++; if (k!=position){  
		              if(!instruction_call_p(statement_instruction(s)))
                                   local =TRUE; 
                              else{ 
                                    call ca= instruction_call
                                    (statement_instruction(s)) ;
				    if(strcmp(  entity_name(call_function(ca))
                                    ,"TOP-LEVEL:CONTINUE" )!=0)
                                    local =TRUE ;
                              }; 
                         }  
                   };, l);
                   if ( local){
                       perfectly_nested=FALSE; if (cpt==1) {
                                                      cpt++; loop_rwt();
                                               }; 
                       return FALSE; 
                   } 
                   else return TRUE;
	      }
              else { k=0; 
                   MAP(STATEMENT,s, {if
                   ( instruction_test_p(statement_instruction(s)))
                    { };    ; };, l); 
              };
                    
	      if (cpt ==1){
                   cpt++; loop_rwt();
                };  
	      return FALSE;
           
	 };     
         if (cpt ==1) { 
             cpt++; loop_rwt();
         }; 
         return FALSE;
    }; 
} 
bool loop_rwt( ) 
{  
  cpt--;
  if (cpt==1){
      nbr_nested_loop++;
      if (perfectly_nested){ 
          nbr_perfectly_nested_loop++;
          nbr_perf_nest_loop_of_depth[depth]++;
          if (depth >max_depth_perfect) max_depth_perfect =depth;
      }
      else{
	   perfectly_nested=TRUE  ; 
           nbr_no_perfectly_nested_loop++;
      };
      depth=0;
      cpt--;
    
  };
  return TRUE;
}
 
static bool loop_flt1( loop l)
{ 
  int  k=0, position=0;
  instruction lbi = statement_instruction(loop_body(l));
  first_turn =TRUE;
  nbr_loop++;
  cpt++;
  if(instruction_sequence_p(lbi)) { 
              sequence seq = instruction_sequence(lbi);
              list l =sequence_statements(seq); 
              MAP(STATEMENT,s, {k++;if
              ( instruction_loop_p(statement_instruction(s)))
              position=k ; };, l); 
              if (position!=0 ) {  
                   k=0; 
	            MAP(STATEMENT,s, 
                   {k++; if (k!=position){  
		              if(!instruction_call_p(statement_instruction(s)))
                                   perfectly_nested =FALSE; 
                              else{ 
                                    call ca= instruction_call
                                    (statement_instruction(s)) ;
				    if(strcmp(  entity_name(call_function(ca))
                                    ,"TOP-LEVEL:CONTINUE" )!=0)
                                    perfectly_nested=FALSE ;
                              }; 
		   };  
                    };, l);
	       };
                  
  }
   return  TRUE;
}  
  

   
bool loop_rwt1( )
{  
  if (first_turn) { 
 
     if   (cpt > depth)  depth = cpt; 
     first_turn= FALSE;
  }; 
   cpt--;
   if (cpt==0) {
     if ( ! perfectly_nested ) {
        nbr_no_perf_nest_loop_of_depth[depth]++;
        if (depth > max_depth_no_perfect) max_depth_no_perfect=depth;
     };
     perfectly_nested=TRUE;
     depth=0;
    
   };     
  return TRUE ;
}
static void initialize ()
{
  int i;
  for(i=0; i<=1000;i++)
   nbr_perf_nest_loop_of_depth[i]=0;
   depth=0;
   nbr_perfectly_nested_loop=0; nbr_no_perfectly_nested_loop=0;
   nbr_interested_loop=0;
   cpt=0; nbr_nested_loop=0 ; perfectly_nested =TRUE, max_depth_perfect=0  ;
   nbr_loop=0;max_depth_no_perfect=0;first_turn=TRUE,som=0;
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
  if (nbr_perf_nest_loop_of_depth[i] !=0)
  fprintf(file,"perfectly  nested loops of depth %d: %d \n", i, 
  nbr_perf_nest_loop_of_depth[i]);
  for (i=1; i<= max_depth_no_perfect; i++)
  if (nbr_no_perf_nest_loop_of_depth[i] !=0)
  fprintf(file,"non perfectly  nested loops of depth %d: %d \n", i, 
  nbr_no_perf_nest_loop_of_depth[i]);
 
  safe_fclose(file, filename);
}

typedef enum { is_unknown, is_a_loop, is_a_call } contenu_t;

typedef struct {
  hash_table contenu;
  stack statement_stack;
  // ...
} * context_p, context_t;


static bool stmt_flt(statement s, context_p context)
{
  stack_push(context->statement_stack, s);
  return TRUE;
}


static void stmt_rwt(statement s, context_p context)
{
  stack_pop(context->statement_stack);
}

/*
static void seq_rwt(sequence sq, context_p context)
{
  contenu_t contenu = is_unknown;

  MAP(STATEMENT, s,
  {
    contenu = ...;
  },
      sequence_statements(sq));

  hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	   contenu);
} 

static void uns_rwt(unstructured u, context_p context)
{
  hash_put(context->contenu,
	   stack_head(context->statement_stack), 
	   is_bad... );
} */

int loop_statistics(string name)
{
  statement stat;
  string filename, localfilename; 
  context_t context;

  debug_on("STATISTICS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", name);
  set_current_module_entity(local_name_to_top_level_entity(name));
  initialize() ;
  stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE); /* ??? */

  context.contenu = hash_table_make(hash_pointer, 0); /* statement -> etat */
  context.statement_stack = stack_make(stack_pointer, 0, 0);

   gen_context_multi_recurse
     (stat, &context,loop_domain, loop_flt, loop_rwt, NULL);
    
    gen_context_multi_recurse
    (stat, & context, 
     statement_domain, stmt_flt, stmt_rwt,
  
     NULL)
 
  gen_context_multi_recurse
    (stat, & context, 
     statement_domain, stmt_flt, stmt_rwt,
     sequence_domain, gen_true, seq_rwt,
     test_domain, gen_true, test_rwt,
     loop_domain, loop_flt, loop_rwt,
     whileloop_domain, gen_true, wl_rwt,
     unstructured_domain, gen_true, uns_rwt,
     call_domain, gen_true, call_rwt,
     expression_domain, gen_false, gen_null,
     NULL); 
  
  hash_table_free(context.contenu);
  stack_free(context.statement_stack);

  localfilename = db_build_file_resource_name(DBR_STATS_FILE,
					      name, ".loop_stats");
  filename = strdup(concatenate(db_get_current_workspace_directory(), 
				"/", localfilename, NULL));
 
  put_result(filename);
  free(filename);
  DB_PUT_FILE_RESOURCE(DBR_STATS_FILE, name, localfilename);
  reset_current_module_entity();
  


pips_debug(1, "done.\n");
  debug_off();
   
  return TRUE;
}






















