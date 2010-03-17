/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

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

step_reduction step_check_reduction(entity module, string directive_txt)
{
  string reduction_,reduction_clause,reduction_clause_dup,remaining;
  string operator_,operator;
  string list_,item,comma;
  int nb_item, nb_comma;
  boolean erreur=false;
  step_reduction reductions=make_step_reduction();
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
	  if (bound_step_reduction_p(reductions,e))
	    {
	      msg_err="several reduction for the same variable.";
	      erreur=TRUE;
	      break;
	    }
	  extend_step_reduction(reductions,e,strdup(operator));
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
      free_step_reduction(reductions);
      return step_reduction_undefined;
    }

  return reductions;
}

clause clause_reductions(entity module, string directive_txt)
{
  return make_clause_step_reduction(step_check_reduction(module,directive_txt));
}

string clause_reduction_to_string(step_reduction reductions)
{
  string s = string_undefined;
  string_buffer sb = string_buffer_make(FALSE);

  STEP_REDUCTION_MAP(variable,operator,
     {
       s=strdup(concatenate(" reduction(",operator," : ",entity_local_name(variable),")",NULL));
       string_buffer_append(sb, s);
     },reductions);
  
  s = string_buffer_to_string(sb);
  string_buffer_free_all(&sb);
  return s;
}
