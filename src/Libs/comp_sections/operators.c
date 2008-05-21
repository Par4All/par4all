#include "all.h"

/* Only the Hull needs to be initialsed */
comp_desc InitCompDesc(reference ref, tag ReadWrite)
{
  /* precondition : this function should be called
     for array varaibles only
  */

  /*{{{  code*/
  /*{{{  inits*/
  comp_desc Descriptor;
  simple_section Hull;
  reference DupRef;
  entity e;
  action DupAct;
  comp_sec DupSec;
  /*}}}*/
  
  pips_debug(3, "begin\n");
  DupRef = reference_dup(ref);

  e = reference_variable(DupRef);
  
  /*{{{  code*/
  if (entity_scalar_p(e)) {
    pips_debug(1, "InitCompDesc : scalar variable encountered \n ");
    return NULL;
  }
  
  pips_debug(3, "InitCompDesc : Array entity name %s \n ",func_entity_name(e) );
	   
  Hull = AllocateSimpleSection(DupRef);
  DupAct = make_action(ReadWrite, UU);
  DupSec = make_comp_sec(Hull, NIL);
  Descriptor = make_comp_desc (DupRef, DupAct, DupSec) ;              
  /*}}}*/
  pips_debug(3, "end\n");
  
  return(Descriptor);
  /*}}}*/
}



/* The multidimensional union of two complementary sections
 * performed on a 2-d basis
 */
comp_sec
CompUnion(comp_sec __attribute__ ((unused)) cs1,
	  comp_sec __attribute__ ((unused)) cs2)
{
  comp_sec result;
  unsigned int i;
  unsigned int Rank = 0;
  unsigned int NoOfImages = (Rank*Rank)/2;
  
  for (i = 0; i < NoOfImages; i++)  {
  }
  /* insert a debug */
 
  return(result);
}

bool
CompIntersection(comp_sec __attribute__ ((unused)) cs1,
		 comp_sec __attribute__ ((unused)) cs2)
{
  return FALSE;
}

