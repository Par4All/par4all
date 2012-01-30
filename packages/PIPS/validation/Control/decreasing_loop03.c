// Check the conversion of for loops into do loops

// Here the sign ot T is known: the for loop must be transformed
// into a do loop

void decreasing_loop03(unsigned int T, int NJ) {
  int tj;
  for(tj=NJ-1; tj>=T-1; tj=tj-T) 
    ;

}



