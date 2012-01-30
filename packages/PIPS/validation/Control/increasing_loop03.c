// Check the conversion of for loops into do loops

// Here the sign ot T is known or not depending on the loops

void increasing_loop03(unsigned int T, int NJ) {
  int tj;

  /* do loop */
  for(tj=T; tj<NJ-T; tj+=T+1)
    ;
  /* do loop */
  for(tj=T; tj<NJ-T; tj+=2*T)
    ;
  /* do loop */
  for(tj=T; tj<NJ-T; tj+=2*T+1)
    ;
  /* while loop */
  for(tj=T; tj<NJ-T; tj+=T-1)
    ;
  /* while loop */
  for(tj=T; tj<NJ-T; tj+=2*(T-1))
    ;
  /* while loop */
  for(tj=T; tj<NJ-T; tj+=2*T-1)
    ;
}



