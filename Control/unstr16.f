      SUBROUTINE MSTADB
      IF(KT-1) goto 105
      goto 106
  104 FORMAT(1H1,4X,17HEQUIV POT TEMP = ,E14.6,
     C23H OUT OF RANGE IN MSTADB)                                                                
105   CONTINUE                                                          
      RETURN                                                            
  106 CONTINUE                                                          
      END                                                               
