/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
#include "defines-local.h"

clause directive_transformation(string directive_txt)
{
  if (strcasestr(directive_txt,STEP_CLAUSE_NOMPI_TXT)!=NULL)
    return make_clause_transformation(STEP_TRANSFORMATION_OMP);
  else if (strcasestr(directive_txt,STEP_CLAUSE_HYBRID_TXT)!=NULL)
    return make_clause_transformation(STEP_TRANSFORMATION_HYBRID);
  else if (strcasestr(directive_txt,STEP_CLAUSE_IGNORE_TXT)!=NULL)
    return make_clause_transformation(STEP_TRANSFORMATION_SEQ);
  else if (strcasestr(directive_txt,STEP_CLAUSE_MPI_TXT)!=NULL)
    return make_clause_transformation(STEP_TRANSFORMATION_MPI);
  else
    {
      string tranformation=get_string_property("STEP_DEFAULT_TRANSFORMATION");
      if (strncasecmp(tranformation,STEP_DEFAULT_TRANSFORMATION_MPI_TXT,strlen(STEP_DEFAULT_TRANSFORMATION_MPI_TXT))==0)
	  return make_clause_transformation(STEP_TRANSFORMATION_MPI);
      else if (strncasecmp(tranformation,STEP_DEFAULT_TRANSFORMATION_HYBRID_TXT,strlen(STEP_DEFAULT_TRANSFORMATION_HYBRID_TXT))==0)
	  return make_clause_transformation(STEP_TRANSFORMATION_HYBRID);
      else if (strncasecmp(tranformation,STEP_DEFAULT_TRANSFORMATION_OMP_TXT,strlen(STEP_DEFAULT_TRANSFORMATION_OMP_TXT))==0)
	return make_clause_transformation(STEP_TRANSFORMATION_OMP);
      else
	{
	  pips_user_error("Undefined transformation :\ndirective : %s\ndefault transformation : %s", directive_txt,tranformation);
	  return clause_undefined;
	}
    }
}

bool directive_transformation_p(directive d,int *transformation)
{
  bool found=false;
  FOREACH(CLAUSE,c,directive_clauses(d))
    {
      if (!found && clause_transformation_p(c))
	{
	  found=true;
	  *transformation=clause_transformation(c);
	}
    }
  return found;
}
