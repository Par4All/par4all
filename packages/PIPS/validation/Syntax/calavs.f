C     Bug: DATA is found inconsistent "too few initializers"

C================= CALAVSK.N.S. 3/22 84 === REV.001K.N.S. 4/26 84 ======
C
      SUBROUTINE CALAVS(NAVRS)
      IMPLICIT DOUBLE PRECISION (A-H,O-Z)
      CHARACTER*4 J,K,L,M,N
      COMMON /AVERIZ/ V(13),VV(13)
      DIMENSION AV(13),AVV(13)
      DIMENSION J(13),K(13),L(13),M(13),N(13)
      SAVE J, K, L, M, N
      DATA J(1),K(1),L(1),M(1),N(1)/'WW  ','POT.','EN. ','(KJ/','MOL)'/
     X J(2),K(2),L(2),M(2),N(2)/'WI  ','POT.','EN. ','(KJ/','MOL)'/
     X J(3),K(3),L(3),M(3),N(3)/'WGA ','POT.','EN. ','(KJ/','MOL)'/
     X J(4),K(4),L(4),M(4),N(4)/'II  ','POT.','EN. ','(KJ/','MOL)'/
     X J(5),K(5),L(5),M(5),N(5)/'IGA ','POT.','EN. ','(KJ/','MOL)'/
     X J(6),K(6),L(6),M(6),N(6)/'TOT.','POT.','EN. ','(KJ/','MOL)'/
     X J(7),K(7),L(7),M(7),N(7)/'H2O ','TR. ',' KE ','(KJ/','MOL)'/
     X J(8),K(8),L(8),M(8),N(8)/'H2O ','ROT.',' KE ','(KJ/','MOL)'/
     X J(9),K(9),L(9),M(9),N(9)/'ION ','TR. ',' KE ','(KJ/','MOL)'/
     XJ(10),K(10),L(10),M(10),N(10)/'TOT-','ENER','GY  ','(KJ/','MOL)'/
     XJ(11),K(11),L(11),M(11),N(11)/'H2O ','TR. ','TEMP','  ( ',' K )'/
     XJ(12),K(12),L(12),M(12),N(12)/'H2O ','ROT ','TEMP','  ( ',' K )'/
     XJ(13),K(13),L(13),M(13),N(13)/'ION ','TR. ','TEMP','  ( ',' K )'/
      FNAVS=DBLE(NAVRS)
      DO 100 I=1,13
      AV(I)=V(I)/FNAVS
      AVV(I)=SQRT(ABS(VV(I)/FNAVS-AV(I)*AV(I)))
  100 CONTINUE
      WRITE(66,150) NAVRS
  150 FORMAT(//1X,'*** AVERAGE VALUES AFTER ',I8,' STEPS:'//)
      DO 200 I=1,10
  200 WRITE(66,250) J(I),K(I),L(I),M(I),N(I),AV(I),AVV(I)
  250 FORMAT(/1X,5A4,4X,-3PF12.4,' +/-  ',-3PF12.4)
      WRITE(1,2)
      WRITE(1,5) AV(10)
      IF( AV(10) .LT. -7495.5   .AND.   AV(10) .LT. -7495.3 ) THEN
        WRITE(1,3)
      ELSE
        WRITE(1,4)
      ENDIF
C
      DO 300 I=11,13
  300 WRITE(66,350) J(I),K(I),L(I),M(I),N(I),AV(I),AVV(I)
  350 FORMAT(/1X,5A4,4X, 0PF12.2,' +/-  ', 0PF12.2)
2     FORMAT(1X,'VALIDATION PARAMETERS:'/)
3     FORMAT(//1X,'RESULTS FOR THIS RUN ARE:   VALID')
4     FORMAT(//1X,'RESULTS FOR THIS RUN ARE: INVALID')
5     FORMAT(1X,-3PF12.4)
      RETURN
      END
