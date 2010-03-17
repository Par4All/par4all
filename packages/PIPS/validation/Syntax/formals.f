C     Bug with formal parameter lists with ENTRY (CA)

      SUBROUTINE FPTRAK(NNEW,ISTAT)                                     
                                                                  
      CALL EXTEND(JSTAT)                                                
         
      END                                                               
                                                                        
      SUBROUTINE EXTEND(ISTAT)                                          
                                                                      
      ISTAT=0                                                           
                                                                   
      ENTRY EXTINT(GATE1,GATE2,NEQQ,ALVNEW,IPRT)                        
      GATEF=GATE1                                                       
      GATVEL=GATE2                                                      
      NEQUIV=NEQQ                                                       
      ALVERT=ALVNEW                                                     
      IF(IPRT.LT.1) THEN                                                
      RETURN                                                            
      ENDIF                                                             

      END                                                               
