// To check the conversion of if in to while

// See whileif01 for the counterpart

// Support for the proof of improvement

// Transformer lists do not impact the result for the time being
// (Nov. 20, 2011)

// whilewhile01 reduced to its first loop

void whilewhile02()
{
  int i = 0;

  // The second internal while is not entered
  while(i<=10) {
    while(i<=10 && i>=0)
      i++;
    while(i<=10 && i<0)
      i--;
  }

  return;
}
