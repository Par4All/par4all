c SOR 2x2 parameter and flip-flop and real*8
      PROGRAM HPFTEST54
c
c version avec 4 processeurs
c

C     Modification: doall replaced by do so as not to have implictly 
C     real indexes for all loops (FI)

      INTEGER N, TIMES
      PARAMETER (N=500)
      PARAMETER (TIMES=100)
      REAL*8 TEMP(N,N,2), NORTH(N), X
      EXTERNAL REDMAX1
      REAL*8 REDMAX1
      EXTERNAL TIME
      INTEGER TIME
      INTEGER OLD, NEW, T1, T2, T3, T4, T5
      INTEGER I, J, K
      LOGICAL T(N,N)
      LOGICAL P(2,2)
      PRINT *,'HPFTEST54 RUNNING, THERMO'                               0006
c
c initialization
c
      PRINT *,'INITIALIZING'                                            0007
      T1 = TIME()                                                       0008
c
      DO I = 1, N                                                       0011
         NORTH(I) = 100.0                                               0012
      ENDDO
      DO K = 1, 2                                                       0014
         DO I = 1, N                                                    0017
            TEMP(1,I,K) = NORTH(I)                                      0018
         ENDDO
         DO J = 1, N                                                    0021
            DO I = 2, N                                                 0023
               TEMP(I,J,K) = 10.0                                       0024
            ENDDO
         ENDDO
      ENDDO
c
      PRINT *,'RUNNING'                                                 0025
      T2 = TIME()                                                       0026
c
      NEW = 2                                                           0027
      DO K = 1, TIMES                                                   0029
         OLD = NEW                                                      0030
         NEW = 3-NEW                                                    0031
         DO J = 2, N-1                                                  0034
            DO I = 2, N-1                                               0036
               TEMP(I,J,NEW) = 0.25*(TEMP(I-1,J,OLD)+TEMP(I+1,J,OLD)+   0037
     &         TEMP(I,J-1,OLD)+TEMP(I,J+1,OLD))                         0037
            ENDDO
         ENDDO
      ENDDO
c
c print results
c
      PRINT *,'REDUCTION'                                               0038
      T3 = TIME()                                                       0039
c
      DO I = 1, N                                                       0042
         NORTH(I) = TEMP(2,I,OLD)                                       0043
      ENDDO
      X = REDMAX1(NORTH(1), 1, N)                                       0044
c
      PRINT *,'RESULTS:'                                                0045
      T4 = TIME()                                                       0046
c
      PRINT *,'MAX is ',X                                               0047
10    FORMAT(F8.2,F8.2,F8.2,F8.2,F8.2)                                  0048
      DO I = 2, 10, 2                                                   0050
         WRITE (FMT=10,UNIT=6) TEMP(I,12,OLD),TEMP(I,24,OLD),TEMP(I,    0051
     &   36,OLD),TEMP(I,48,OLD),TEMP(I,60,OLD)                          0051
      ENDDO
      PRINT *,'HPFTEST52 ENDED'                                         0052
      T5 = TIME()                                                       0053
      PRINT *,'Timing: init ',T2-T1,' run ',T3-T2,' red ',T4-T3,        0054
     &' IO ',T5-T4,' total ',T5-T1                                      0054
      END
c
c reduction
c
      real*8 function REDMAX1(a,l,u)
      integer l, u, i
      real*8 a(l:u), amax
      amax = a(l)
      do i=l+1, u
         if (a(i).GT.amax) amax = a(i)
      enddo
      redmax1 = amax
      return
      end
      integer function time()
      time=-1
      end
