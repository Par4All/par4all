/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include "defines-local.h"

static boolean reduction_op_p(string op)
{
  if (!strcasecmp(op,"+")) return TRUE;
  if (!strcasecmp(op,"*")) return TRUE;
  if (!strcasecmp(op,"-")) return TRUE;
  if (!strcasecmp(op,".and.")) return TRUE;
  if (!strcasecmp(op,".or.")) return TRUE;
  if (!strcasecmp(op,".eqv.")) return TRUE;
  if (!strcasecmp(op,".neqv.")) return TRUE;
  if (!strcasecmp(op,"max")) return TRUE;
  if (!strcasecmp(op,"min")) return TRUE;
  if (!strcasecmp(op,"iand")) return TRUE;
  if (!strcasecmp(op,"ior")) return TRUE;
  if (!strcasecmp(op,"ieor")) return TRUE;
  return FALSE;
}

clause step_check_reduction(entity module, string directive_txt)
{
  string reduction_,reduction_clause,reduction_clause_dup,remaining;
  string operator_,operator;
  string list_,item,comma;
  int nb_item, nb_comma;
  boolean erreur=false;
  map_entity_string reductions=make_map_entity_string();
  string msg_err;

  remaining=strdup(directive_txt);
  while (remaining && !erreur)
    {
      reduction_=strcasestr(remaining,"reduction");
      if (!reduction_) break;
      reduction_clause=strchr(reduction_,'(');
      if (!reduction_clause)
	{
	  msg_err="missing clause detail";
	  erreur=TRUE;
	  break;
	}
      reduction_clause=strtok (reduction_clause,"()");
      reduction_clause_dup=strdup(reduction_clause);
      remaining=strtok (NULL,"");
      
      operator_=strtok (reduction_clause_dup,":");
      list_=strtok (NULL,"");
      if (!list_)
	{
	  msg_err="missing variable list";
	  erreur=TRUE;
	  break;
	}
      
      //check operator
      operator=strtok(operator_," ");
      if (!reduction_op_p(operator)||strtok(NULL," "))
	{
	  msg_err="unknow operator";
	  erreur=TRUE;
	  break;
	}

      //check item
      nb_item=0;
      nb_comma=0;
      comma=strchr(list_,',');
      while (comma!=NULL)
	{
	  nb_comma++;
	  comma=strchr(comma+1,',');
	}
      item=strtok(list_," ,");
      while (item != NULL && nb_item<=nb_comma)
	{
	  entity e=global_name_to_entity(entity_local_name(module), step_upperise(item));
	  if (entity_undefined_p (e)) 
	    {
	      msg_err="unknow variable";
	      erreur=TRUE;
	      break;
	    }
	  /* verifier la non existance d'une réduction précédente pour la variable*/
	  if (bound_map_entity_string_p(reductions,e))
	    {
	      msg_err="several reduction for the same variable.";
	      erreur=TRUE;
	      break;
	    }
	  extend_map_entity_string(reductions,e,strdup(operator));
	  pips_debug(2,"reduction %s %s\n",entity_name(e),operator);
	  nb_item++;
	  item = strtok (NULL, ", ");
	}
      if (item&&!erreur)
	{
	  msg_err="too many variable";
	  erreur=TRUE;
	}

      free(reduction_clause_dup);
    }

  if (erreur)
    {
      pips_user_error("\n\nSTEP : erreur in clause : reduction (%s)\n%s\n\n",reduction_clause,msg_err);
      free_map_entity_string(reductions);
      return clause_undefined;
    }

  return make_clause_reduction(reductions);
}


string clause_reduction_to_string(map_entity_string reductions)
{
  string s = string_undefined;
  string_buffer sb = string_buffer_make(FALSE);
  list entity_reduction=NIL;

  MAP_ENTITY_STRING_MAP(e, __attribute__ ((unused))op,{
      entity_reduction=CONS(ENTITY,e,entity_reduction);
    },reductions);
  sort_list_of_entities(entity_reduction);

  FOREACH(ENTITY,variable,entity_reduction)
    {
      string operator=apply_map_entity_string(reductions,variable);
      s=strdup(concatenate(" reduction(",operator," : ",entity_local_name(variable),")",NULL));
      string_buffer_append(sb, s);
    }
  gen_free_list(entity_reduction);

  s = string_buffer_to_string(sb);
  string_buffer_free_all(&sb);
  return s;
}
