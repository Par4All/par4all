bool prepend_comment(string mn) {

  // Use this module name to get the resources we need
  statement s = PIPS_PHASE_PRELUDE(mn,	
				   "PREPEND_COMMENT_DEBUG_LEVEL" );
  
  // Get the value of the property containing the comment to be prepended
  string c = ... ;

  // Add comment c to the module statement s
  s = ... ;
  
  // Put back the new statement module

  PIPS_PHASE_POSTLUDE(s);
}
