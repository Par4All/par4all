// Check the conversion of for loops into do loops

// Here the sign ot T is known: the for loop must be transformed
// into a do loop (difference with increasing_loop01: the
// incrementation formulae)

void increasing_loop02(unsigned int T, int NJ) {
  int tj;
  for(tj=T; tj<NJ-T; tj=tj+T) 
    ;

}



