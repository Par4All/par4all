/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#include "defines-local.h"

step_private step_check_private(entity module, string directive_txt)
{
  string private_,private_clause,remaining;
  string list_,item,comma;
  int nb_item, nb_comma;
  boolean erreur=false;
  list privates=NIL;
  string msg_err;

  remaining=strdup(directive_txt);
  while (remaining && !erreur)
    {
      private_=strcasestr(remaining,"private");
      if (!private_) break;
      private_clause=strchr(private_,'(');
      if (!private_clause)
	{
	  msg_err="missing clause detail";
	  erreur=TRUE;
	  break;
	}
      private_clause=strtok (private_clause,"()");
      list_=strdup(private_clause);
      remaining=strtok (NULL,"");
            
      if (!list_)
	{
	  msg_err="missing variable list";
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
	      pips_user_warning("unknow variable %s in %s\n",item,entity_local_name(module));
	    }
	  else
	    {
	      nb_item++;
	      privates=CONS(ENTITY,e,privates);
	    }
	  item = strtok (NULL, ", ");
	}
      if (item&&!erreur)
	{
	  msg_err="too many variable";
	  erreur=TRUE;
	  break;
	}
    }

  if (erreur)
    {
      pips_user_error("\n\nSTEP : erreur in clause : private (%s)\n%s\n\n",private_clause,msg_err);
      gen_free_list(privates);
      return step_private_undefined;
    }

  return make_step_private(privates);
}

clause clause_private(entity module, string directive_txt)
{
  return make_clause_step_private(step_check_private(module,directive_txt));
}


string clause_private_to_string(step_private privates)
{
  string s = string_undefined;
  string_buffer sb = string_buffer_make(FALSE);

  FOREACH(ENTITY,variable,step_private_entity(privates))
    {
      if(s == string_undefined)
	s=strdup(concatenate(" private(",entity_local_name(variable),NULL));
      else
	s=strdup(concatenate(", ",entity_local_name(variable),NULL));
      string_buffer_append(sb, s);
    }

  if (s != string_undefined)
    {
      string_buffer_append(sb, strdup(")"));
      s = string_buffer_to_string(sb);
      string_buffer_free_all(&sb);
    }
  return s;
}
