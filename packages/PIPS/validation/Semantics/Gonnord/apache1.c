/*Source Code From Laure Gonnord*/
//from apache1.fst (Fast defintion)

int random();

int apache1(){
  
  int c,tk_sz;

  assume(tk_sz>0);
  c=0;

  while(1){
    if (random()&&c<tk_sz-1) {
      c=c+1;
    }
  }
  return 0;
}

  //bad region : c>tk_sz in the loop
  //loop invariant : inside the loop {c<tk_sz, c>=0}
  // found by Aspic
