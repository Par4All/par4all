/* 
 * $Id$
 */

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

static int depth[1000], nbr_nested_loop, nbr_no_nested_loop,
nbr_interested_loop,cpt, all ;
static bool nested  ;
 
static bool loop_flt( loop l)
{
    int k=0,i=0,position=0,pos=0;
    bool local=FALSE;
    bool loop_rwt();
    instruction lbi = statement_instruction(loop_body(l));
    if ((cpt==0) && instruction_sequence_p(lbi)) 
     {
        bool cond1=TRUE,  found =FALSE;
        sequence seq = instruction_sequence(lbi);
        list l =sequence_statements(seq); 
        MAP(STATEMENT,s, {i++;if( instruction_loop_p(statement_instruction(s)))
              pos=i ;     };, l);
        if ( pos !=0) 
        {
	   i=0;
	   MAP(STATEMENT,s, 
	   {
	     i++; 
	     if (i!=pos) 
	       {  
		 if (instruction_loop_p(statement_instruction(s)) )
		   cond1 =FALSE; 
	       } 
	   },
	       l);
	   i=0;
	   MAP(STATEMENT, s, {  i++; if (i!=pos) 
                                     {
	                                   if (!instruction_call_p(statement_instruction(s))) found =TRUE; 
	                                   else 
                                           {
	                                      call ca = instruction_call(statement_instruction(s));
	                                      if(  strcmp(  entity_name(  call_function(ca)),"TOP-LEVEL:CONTINUE" )!=0) found =TRUE ;
	                                    }; 
	 
	                               } 
                              };, l);
        };
       if (( cond1) && (found))  nbr_interested_loop++ ;
     }; 
     cpt++;
     depth[nbr_nested_loop]++;
     if ( instruction_loop_p(lbi))
	return TRUE;  
     else 
     {
         if(instruction_sequence_p(lbi)) 
          { 
              sequence seq = instruction_sequence(lbi);
              list l =sequence_statements(seq); 
              MAP(STATEMENT,s, {k++;if( instruction_loop_p(statement_instruction(s)))
              position=k ; };, l); 
	      if (position!=0 )
              { 
                   k=0; 
	           MAP(STATEMENT,s, {k++; if    (k!=position) 
                                          {  
		                              if   (!instruction_call_p(statement_instruction(s)) )  local =TRUE; 
                                              else 
                                              { 
                                                call ca= instruction_call(statement_instruction(s)   ) ;
					       	                                if(  strcmp(  entity_name(  call_function(ca)),"TOP-LEVEL:CONTINUE" )!=0)
                                                     local =TRUE ;
                                              }; 
                                          }  
                                     };, l);

		  if ( local) 
                  {
                     nested=FALSE; if (cpt==1)
                                    {
                                       cpt++; loop_rwt();
                                    }; 
                     return FALSE; 
                  } 
                  else return TRUE;
	     };
	     if (cpt ==1)
             {
                cpt++; loop_rwt();
             }; 
	     return FALSE;
           
	};     
        if (cpt ==1) 
        {
            cpt++; loop_rwt();
        }; 
          return FALSE;
     }; 
   
    
  
} 
bool loop_rwt( ) 
{  
  cpt--;
  if (cpt==1)
    {
      all++;
      if (nested) 
        nbr_nested_loop++;
       else
	 {
	   nested=TRUE  ; 
           depth[nbr_nested_loop]=0;
           nbr_no_nested_loop++;
         };
      cpt--;
    };
  return TRUE;
}
static void initialize ()
{
  int i;
  for(i=0; i<=1000;i++)
      depth[i]=0;
   nbr_nested_loop=0; nbr_no_nested_loop=0;nbr_interested_loop=0;
   cpt=0; all=0 ; nested =TRUE  ;
  
} 
static void put_result(string filename)
{
  int i;
  FILE * file = safe_fopen(filename, "w");  
  
  fprintf(file,
	  "nested loops: %d\n"
	  "perfectly nested loops: %d\n"
	  "non perfectly nested loops: %d\n"
	  "depth 2 non perfectly nested loops: %d\n",
	  all, nbr_interested_loop, nbr_no_nested_loop, nbr_interested_loop);

  for (i=0; i<=nbr_nested_loop-1; i++)
    fprintf(file," depth of loop %d: %d\n", i+1, depth[i]);

  safe_fclose(file, filename);
}

int loop_statistics(string name)
{
  statement stat;
  string filename, localfilename; 
  debug_on("STATISTICS_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", name);

  set_current_module_entity(local_name_to_top_level_entity(name));
 
  
  
  initialize() ;
  stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE); /* ??? */

  gen_context_multi_recurse
    (stat, NULL,
     loop_domain, loop_flt, loop_rwt,
     NULL);

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











