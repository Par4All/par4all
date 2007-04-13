      PROGRAM GUERIN01
      INTEGER A, B, C
      INTEGER TATA, D, E,TOTO
      INTEGER FSM1
      FSM1 = 1
      DO WHILE (FSM1.NEQV.0)
         IF (FSM1.EQV.1) THEN
            FSM1 = 2
         ELSEIF (FSM1.EQV.2) THEN
            B = 5                                                       
            FSM1 = 3                                              
         ELSEIF (FSM1.EQV.3) THEN
            A = 2                                                       
            FSM1 = 4                                              
         ELSEIF (FSM1.EQV.4) THEN
            C = 3                                                       
            FSM1 = 5                                              
         ELSEIF (FSM1.EQV.5) THEN
            D = 1                                               
            E = 10                                                
            TOTO = 1                                           
            FSM1 = 6                                              
         ELSEIF (FSM1.EQV.6) THEN
            IF (TATA.LE.E) THEN                              
               FSM1 = 7                                           
            ELSE
               FSM1 = 12                                          
            ENDIF
         ENDIF
      ENDDO

      STOP                                                        
      END
