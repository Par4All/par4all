// To check the conversion of if in to while

// See whileif01 for the counterpart

// Support for the proof of improvement

// Transformer lists do not impact the result for the time being
// (Nov. 20, 2011)

// whilewhile01 reduced to its second loop

// It is not clear if the transformer for if(1) is correct or not. As
// the if is never returned, any overapproximation of the empty set is
// correct... The preconditions seem to be all correct, although the
// precondition for if(1) could be more precise. Since the if is never
// exited, it should be i==-1instead of -1<=i<=10

// Refine_transformers does not improve the result, neither does
// transformer_list. Maybe transformer list should be improved to
// study the behavior for the first iteration because the precondition
// is exactly known and then the behavior for other iterations.

void whilewhile03()
{
  int i = -1;

  while(i<=10) {
    if(1) {
      while(i<=10 && i>=0)
	i++;
      while(i<=10 && i<0)
	i--;
      // Never reached
      ;
    }
  }

  return;
}
