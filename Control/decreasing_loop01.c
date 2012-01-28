// Check the conversion of for loops into do loops

// Here the sign ot T is unknown: the for loop must be transformed
// into a while loop or preserved as a for loop

void decreasing_loop01(int T, int NJ) {
  int tj;
  for(tj=NJ-1; tj>=T-1; tj-=T) 
    ;

}



