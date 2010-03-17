C     excerpt from SPEC/CFP95/applu after array bound checking instrumentation

C     Check for array bound check: information collected in the first
C     loop can be propagated because you know all loops are entered
C     (interprocedural information introduced here thru an artificial
C     test)

      SUBROUTINE SETBV

      IMPLICIT REAL*8 (A-H,O-Z)
c
      PARAMETER ( C1 = 1.40D+00, C2 = 0.40D+00,
     $            C3 = 1.00D-01, C4 = 1.00D+00,
     $            C5 = 1.40D+00 )
c
c***grid
c
      COMMON/CGCON/ NX, NY, NZ,
     $              II1, II2, JI1, JI2, KI1, KI2, ITWJ,
     $              DXI, DETA, DZETA,
     $              TX1, TX2, TX3,
     $              TY1, TY2, TY3,
     $              TZ1, TZ2, TZ3
c
c***dissipation
c
      COMMON/DISP/ DX1,DX2,DX3,DX4,DX5,
     $             DY1,DY2,DY3,DY4,DY5,
     $             DZ1,DZ2,DZ3,DZ4,DZ5,
     $             DSSP
c
c***field variables and residuals
c
      COMMON/CVAR/ U(5,33,33,33),
     $             RSD(5,33,33,33),
     $             FRCT(5,33,33,33)
c
c***output control parameters
c
      COMMON/CPRCON/ IPR, IOUT, INORM
c
c***newton-raphson iteration control parameters
c
      COMMON/CTSCON/ ITMAX, INVERT,
     $               DT, OMEGA, TOLRSD(5),
     $               RSDNM(5), ERRNM(5), FRC, TTOTAL
c
      COMMON/CJAC/ A(5,5,33,33,33),
     $             B(5,5,33,33,33),
     $             C(5,5,33,33,33),
     $             D(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      COMMON/CEXACT/ CE(5,13)
c

      if(nx.lt.5.or.ny.lt.5.or.nz.lt.5) stop

c
c***set the dependent variable values along the top and bottom faces
c
      DO J = 1, NY
         DO I = 1, NX
            IF (J.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 3rd dimension"
            IF (I.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 2nd dimension"
c
            CALL EXACT(I, J, 1, U(1,I,J,1))
            IF (NZ.LT.1.OR.NZ.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 4th dimension"
            CALL EXACT(I, J, NZ, U(1,I,J,NZ))
c
         ENDDO
      ENDDO
c
c***set the dependent variable values along north and south faces
c
      DO K = 1, NZ
         DO I = 1, NX
            IF (K.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 4th dimension"
            IF (I.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 2nd dimension"
c
            CALL EXACT(I, 1, K, U(1,I,1,K))
            IF (NY.LT.1.OR.NY.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 3rd dimension"
            CALL EXACT(I, NY, K, U(1,I,NY,K))
c
         ENDDO
         print *, nx, ny, nz
      ENDDO
      print *, nx, ny, nz
c
c***set the dependent variable values along east and west faces
c
      DO K = 1, NZ
         DO J = 1, NY
            IF (K.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 4th dimension"
            IF (J.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 3rd dimension"
c
            CALL EXACT(1, J, K, U(1,1,J,K))
            IF (NX.LT.1.OR.NX.GT.33) STOP 
     &      "Bound violation:array SETBV:U, 2nd dimension"
            CALL EXACT(NX, J, K, U(1,NX,J,K))
c
         ENDDO
         print *, nx, ny, nz
      ENDDO

      print *, nx, ny, nz
c
      END
      SUBROUTINE EXACT ( I, J, K, U000IJK )

      IMPLICIT REAL*8 (A-H,O-Z)
c
      PARAMETER ( C1 = 1.40D+00, C2 = 0.40D+00,
     $            C3 = 1.00D-01, C4 = 1.00D+00,
     $            C5 = 1.40D+00 )
c
c***grid
c
      COMMON/CGCON/ NX, NY, NZ,
     $              II1, II2, JI1, JI2, KI1, KI2, ITWJ,
     $              DXI, DETA, DZETA,
     $              TX1, TX2, TX3,
     $              TY1, TY2, TY3,
     $              TZ1, TZ2, TZ3
c
c***dissipation
c
      COMMON/DISP/ DX1,DX2,DX3,DX4,DX5,
     $             DY1,DY2,DY3,DY4,DY5,
     $             DZ1,DZ2,DZ3,DZ4,DZ5,
     $             DSSP
c
c***field variables and residuals
c
      COMMON/CVAR/ U(5,33,33,33),
     $             RSD(5,33,33,33),
     $             FRCT(5,33,33,33)
c
c***output control parameters
c
      COMMON/CPRCON/ IPR, IOUT, INORM
c
c***newton-raphson iteration control parameters
c
      COMMON/CTSCON/ ITMAX, INVERT,
     $               DT, OMEGA, TOLRSD(5),
     $               RSDNM(5), ERRNM(5), FRC, TTOTAL
c
      COMMON/CJAC/ A(5,5,33,33,33),
     $             B(5,5,33,33,33),
     $             C(5,5,33,33,33),
     $             D(5,5,33,33,33)
c
c***coefficients of the exact solution
c
      COMMON/CEXACT/ CE(5,13)
c



c
      DIMENSION U000IJK(*)
c
      XI = DFLOAT(I-1)/(NX-1)
      ETA = DFLOAT(J-1)/(NY-1)
      ZETA = DFLOAT(K-1)/(NZ-1)
c
      DO M = 1, 5
c
         U000IJK(M) = CE(M,1)+CE(M,2)*XI+CE(M,3)*ETA+CE(M,4)*ZETA+CE(
     &   M,5)*XI*XI+CE(M,6)*ETA*ETA+CE(M,7)*ZETA*ZETA+CE(M,8)*XI*XI*
     &   XI+CE(M,9)*ETA*ETA*ETA+CE(M,10)*ZETA*ZETA*ZETA+CE(M,11)*XI*
     &   XI*XI*XI+CE(M,12)*ETA*ETA*ETA*ETA+CE(M,13)*ZETA*ZETA*ZETA*
     &   ZETA
c
      ENDDO
c
      END
