// To check the conversion of if in to while

// See whileif01 for the counterpart

// Support for the proof of improvement

// Transformer lists do not impact the result for the time being
// (Nov. 20, 2011)

void whilewhile01()
{
  int i = 0;

  // The second internal while is not entered
  while(i<=10) {
    while(i<=10 && i>=0)
      i++;
    while(i<=10 && i<0)
      i--;
  }

  // The non-termination is not detected, both loops are potentially entered
  i = -1;
  while(i<=10) {
    if(1) {
      while(i<=10 && i>=0)
	i++;
      while(i<=10 && i<0)
	i--;
    }
  }

  // This control point is not reachable, but the empty set is over
  // approximated by i==11

  // The non-termination is detected, the second inner loop is not
  // entered
  // Note that the inner loops have been exchanged
  i = -1;
  while(i<=10) {
    if(1) {
      while(i<=10 && i<0)
	i--;
      while(i<=10 && i>=0)
	i++;
    }
  }

  return;
}
