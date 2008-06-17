! README file:
! 		  SX-3/44 Supercomputer at Dorval 
! 
! 		       Sample Performance Run 
! 
!      The following fortran program is a "dusty deck" used by myself to
! make quick comparisons  of  the various computers offered for sale  to
! this  site. The program is known as a shallow water model, a type   of
! test  harness used by  meteorologists  to   experiment  with   various
! integration schemes. By changing  the resolution of  the model it   is
! possible to scale this code to a  wide range of computational engines.
! I  have  run  this  on everything from  an  IBM  PC  to  all  of   the
! supercomputers installed at this site.
! 
!      The code is a  "dusty  deck" in  the sense that it  is  run using
! compiler options at   the   command  line  only. The code   is     not
! particularly   written for any   type  of  computer, however, in   the
! implementation below, would qualify for something in  the  long vector
! domain. The program consists of 3 modules, the main program "SHALLOW",
! and 2 subroutines "PLOT"  and "RANGE". The latter 2 routines are  used
! to get output for display on a graphics  device, in this  case, an IBM
! PC  with a  CGA  graphics  screen. The statistics  on  the run are  as
! follows...
! 
!      Vectorization  Ratio in  SHALLOW 99.75% Average Vector Length  in
! SHALLOW 213 Computation rate    in   SHALLOW 1.436 GFLOPS Number    of
! processors 1
! 
!      This is certainly not a  high profile benchmark code, but is  has
! shown up the differences amongst various computers for application  to
! meteorological applications at  this  site. I call   this  my 1 minute
! benchmark because it usually requires less than 1 minute to move it to
! a new machine, compile it and get it running.
! 
!      If anyone runs this on other machines, I would  like  to get back
! the numbers.
! 
! 
! Iain B. Findleton 
! 
!
! port to HPF/HPFC (FC, 12/06/96):
! - the code is simple because no border conditions
! - should be flip-flopped... (but arg passing in trouble -> copy)
! - wrapped around... should do sg about corners...
!
! Log:
! - sizes moved as parameters m & n [MAIN,RANGE,PLOT]
! - explicit types
! - DO/ENDDO
! - loop distribution [60,70,75,100,110,115,200,210,215,320,325]
! - loop interchange [801,901]
! - loop structure changed [802-803(ijk->ij),901(ifs->min/max)]
! - INDEPENDENT, NEW & REDUCTION directives 
! - (commented) HOST/END HOST directives for I/Os
! - static mapping directives [MAIN,RANGE,PLOT]
!
      PROGRAM SHALLOW

      integer m,n
      parameter(m=320,n=200)

C ***
C *** BENCHMARK WEATHER PREDICTION PROGRAM FOR COMPARING THE PERFORMANCE
C *** OF CURRENT SUPERCOMPUTERS. THE MODEL IS BASED ON THE PAPER, "THE
C *** DYNAMICS OF FINITE-DIFFERENCE MODELS OF THE SHALLOW-WATER
C *** EQUATIONS," BY R. SADOURNY, J. ATM. SCI., VOL.32, NO.4, APRIL 1975
C *** CODE BY PAUL N. SWARZTRAUBER, NATIONAL CENTER FOR ATMOSPHERIC
C *** RESEARCH, BOULDER, COLORADO   OCTOBER 1984.
C ***

      REAL U(m,n),V(m,n),P(m,n),UNEW(m,n),PNEW(m,n),UOLD(m,n),
     $     VOLD(m,n),POLD(m,n),CU(m,n),CV(m,n),Z(m,n),H(m,n),PSI(m,n) 
      REAL VNEW(m,n)
      CHARACTER*16 LABEL(5)

      real A,DT,TDT,TIME,DX,DY,ALPHA,PI,TPI,DI,DJ
      integer ITMAX,MPRINT,MM1,NM1,I,J

!hpf$ processors procs(2,2)
!hpf$ template domain(m,n)
!hpf$ align with domain:: U,V,P,UNEW,PNEW,UOLD,VOLD,POLD,
!hpf$x      CU,CV,Z,H,PSI,VNEW
!hpf$ distribute (block,block) onto procs:: domain

      OPEN(UNIT=2,FORM='UNFORMATTED')

      LABEL(1)='PRESSURE FIELD  '
      LABEL(2)='EAST/WEST WIND  '
      LABEL(3)='NORTH/SOUTH WIND'
      LABEL(4)='GEOPOTENTIAL    '
      LABEL(5)='HEIGHT FIELD    '

      A = 1.E6
      DT = 90.
      TDT = 90.
      TIME = 0.
      DX = 1.E5
      DY = 1.E5
      ALPHA = .001
      ITMAX = 2400
      MPRINT = 100
      MM1 = m-1
      NM1 = n-1
      PI = 4.*ATAN(1.)
      TPI = PI+PI
      DI = TPI/FLOAT(M)
      DJ = TPI/FLOAT(N)

C *** INITIAL VALUES OF THE STREAM FUNCTION [50]

!hpf$ independent(J,I)
      DO J=1,n
         DO I=1,m
            PSI(I,J) = A*SIN((FLOAT(I)-.5)*DI)*SIN((FLOAT(J)-.5)*DJ)
         ENDDO
      ENDDO

C *** INITIALIZE VELOCITIES [60]

!hpf$ independent(J,I)
      DO J=1,NM1
         DO I=2,m
            U(I,J) = -(PSI(I,J+1)-PSI(I,J))/DY
         ENDDO
      ENDDO
!hpf$ independent(J,I)
      DO J=2,n
         DO I=1,MM1
            V(I,J) = (PSI(I+1,J)-PSI(I,J))/DX
         ENDDO
      ENDDO

C *** PERIODIC CONTINUATION [70,75,86]

!hpf$ independent
      DO J=1,NM1
         U(1,J) = U(m,J)
      ENDDO
!hpf$ independent
      DO J=1,NM1
         V(m,J+1) = V(1,J+1)
      ENDDO

!hpf$ independent
      DO I=2,m
         U(I,n) = U(I,1)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         V(I,1) = V(I,n)
      ENDDO

! corners...
      U(1,n) = U(m,1)
      V(m,1) = V(1,n)

!hpf$ independent(J,I)
      DO J=1,n
         DO I=1,m
            UOLD(I,J) = U(I,J)
            VOLD(I,J) = V(I,J)
            POLD(I,J) = 50000.
            P(I,J) = 50000.
         ENDDO
      ENDDO

C *** PRINT INITIAL VALUES

C!fcd$ host
C     WRITE(*,390) N,M,DX,DY,DT,ALPHA
C 390 FORMAT('1NUMBER OF POINTS IN THE X DIRECTION',I8/
C    1     ' NUMBER OF POINTS IN THE Y DIRECTION',I8/
C    2     ' GRID SPACING IN THE X DIRECTION    ',F8.0/
C    3     ' GRID SPACING IN THE Y DIRECTION    ',F8.0/
C    4     ' TIME STEP                          ',F8.0/
C    5     ' TIME FILTER PARAMETER              ',F8.3)
C     MNMIN = MIN0(M,N)
C     WRITE(*,391) (POLD(I,I),I=1,MNMIN)
C 391 FORMAT(/' INITIAL DIAGONAL ELEMENTS OF P ' //,(8E16.6))
C     WRITE(*,392) (UOLD(I,I),I=1,MNMIN)
C 392 FORMAT(/' INITIAL DIAGONAL ELEMENTS OF U ' //,(8E16.6))
C     WRITE(*,393) (VOLD(I,I),I=1,MNMIN)
C 393 FORMAT(/' INITIAL DIAGONAL ELEMENTS OF V ' //,(8E16.6))
C!fcd$ end host
      NCYCLE = 0

!
! INFINITE LOOP with a STOP
!
   90 NCYCLE = NCYCLE + 1

C *** COMPUTE CAPITAL U, CAPITAL V, Z, AND H [100]

      FSDX = 4./DX
      FSDY = 4./DY

!hpf$ independent(J,I)
      DO J=1,NM1
         DO I=2,m
            CU(I,J) = .5*(P(I,J)+P(I-1,J))*U(I,J)
         ENDDO
      ENDDO
!hpf$ independent(J,I)
      DO J=2,n
         DO I=1,MM1
            CV(I,J) = .5*(P(I,J)+P(I,J-1))*V(I,J)
         ENDDO
      ENDDO
!hpf$ independent(J,I)
      DO J=2,n
         DO I=2,m
            Z(I,J) = (FSDX*(V(I,J)-V(I-1,J))-FSDY*(U(I,J)
     1           -U(I,J-1)))/(P(I-1,J-1)+P(I,J-1)+P(I,J)+P(I-1,J))
         ENDDO
      ENDDO
!hpf$ independent(J,I)
      DO J=1,NM1
         DO I=1,MM1
            H(I,J) = P(I,J)+.25*(U(I+1,J)*U(I+1,J)+U(I,J)*U(I,J)
     1           +V(I,J+1)*V(I,J+1)+V(I,J)*V(I,J))
         ENDDO
      ENDDO

C *** PERIODIC CONTINUATION [110,115]

!hpf$ independent
      DO J=1,NM1
         CU(1,J) = CU(m,J)
      ENDDO
!hpf$ independent
      DO J=2,n
         CV(m,J) = CV(1,J)
      ENDDO
!hpf$ independent
      DO J=2,n
         Z(1,J) = Z(m,J)
      ENDDO
!hpf$ independent
      DO J=1,NM1
         H(m,J) = H(1,J)
      ENDDO

!hpf$ independent
      DO I=2,m
         CU(I,n) = CU(I,1)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         CV(I,1) = CV(I,n)
      ENDDO
!hpf$ independent
      DO I=2,m
         Z(I,1) = Z(I,n)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         H(I,n) = H(I,1)
      ENDDO

! corners exchange, could be a generalized shift iff 1d distribution
      CU(1,n) = CU(m,1)
      CV(m,1) = CV(1,n)
      Z(1,1) = Z(m,n)
      H(m,n) = H(1,1)

C *** COMPUTE NEW VALUES U, V, AND P [200]

      TDTS8 = TDT/8.
      TDTSDX = TDT/DX
      TDTSDY = TDT/DY

!hpf$ independent(J,I)
      DO J=1,NM1
         DO I=2,m
            UNEW(I,J) = UOLD(I,J)+
     1           TDTS8*(Z(I,J+1)+Z(I,J))*(CV(I,J+1)+CV(I-1,J+1)
     2           +CV(I-1,J)+CV(I,J))-TDTSDX*(H(I,J)-H(I-1,J))
         ENDDO
      ENDDO
!hpf$ independent(J,I)
      DO J=2,n
         DO I=1,MM1
            VNEW(I,J) = VOLD(I,J)-TDTS8*(Z(I+1,J)+Z(I,J))
     1           *(CU(I+1,J)+CU(I,J)+CU(I,J-1)+CU(I+1,J-1))
     2           -TDTSDY*(H(I,J)-H(I,J-1))
         ENDDO
      ENDDO
!hpf$ independent(J,I)
      DO J=1,NM1
         DO I=1,MM1
            PNEW(I,J) = POLD(I,J)-TDTSDX*(CU(I+1,J)-CU(I,J))
     1           -TDTSDY*(CV(I,J+1)-CV(I,J))
         ENDDO
      ENDDO

C *** PERIODIC CONTINUATION [210,215]

!hpf$ independent
      DO J=1,NM1
         UNEW(1,J) = UNEW(m,J)
      ENDDO
!hpf$ independent
      DO J=2,n
         VNEW(m,J) = VNEW(1,J)
      ENDDO
!hpf$ independent
      DO J=1,NM1
         PNEW(m,J) = PNEW(1,J)
      ENDDO

!hpf$ independent
      DO I=2,m
         UNEW(I,n) = UNEW(I,1)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         VNEW(I,1) = VNEW(I,n)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         PNEW(I,n) = PNEW(I,1)
      ENDDO

! corners again...
      UNEW(1,N+1) = UNEW(M+1,1)
      VNEW(M+1,1) = VNEW(1,N+1)
      PNEW(M+1,N+1) = PNEW(1,1)

C     IF(NCYCLE .GT. ITMAX) WRITE (*,220)
C 220 FORMAT('0   *****  END OF PROGRAM SHALLOW  *****')

      IF (NCYCLE.GT.ITMAX) THEN
         CLOSE(2)
! was STOP => HPFC coredump...
         GOTO 1000
      ENDIF

      TIME = TIME + DT

      IF(MOD(NCYCLE,MPRINT) .NE. 0) GO TO 370

      PTIME = TIME/3600.
C
C     WRITE(*,350) NCYCLE,PTIME
C 350 FORMAT(//,' CYCLE NUMBER',I5,' MODEL TIME IN  HOURS',F6.2)
C     WRITE(*,355) (PNEW(I,I),I=1,MNMIN)
C 355 FORMAT(/,' DIAGONAL ELEMENTS OF P ' //,(8E16.6))
C     WRITE(*,360) (UNEW(I,I),I=1,MNMIN)
C 360 FORMAT(/,' DIAGONAL ELEMENTS OF U ' //,(8E16.6))
C     WRITE(*,365) (VNEW(I,I),I=1,MNMIN)
C 365 FORMAT(/,' DIAGONAL ELEMENTS OF V ' //,(8E16.6))
C
      CALL PLOT (P,NCYCLE,LABEL(1))
      CALL PLOT (Z,NCYCLE,LABEL(4))
      CALL PLOT (H,NCYCLE,LABEL(5))
      CALL PLOT (U,NCYCLE,LABEL(2))
      CALL PLOT (V,NCYCLE,LABEL(3))
C
  370 IF(NCYCLE .LE. 1) GO TO 310

!loop 300
!hpf$ independent(J,I)
      DO J=1,NM1
         DO I=1,MM1
            UOLD(I,J) = U(I,J)+ALPHA*(UNEW(I,J)-2.*U(I,J)+UOLD(I,J))
            VOLD(I,J) = V(I,J)+ALPHA*(VNEW(I,J)-2.*V(I,J)+VOLD(I,J))
            POLD(I,J) = P(I,J)+ALPHA*(PNEW(I,J)-2.*P(I,J)+POLD(I,J))
            U(I,J) = UNEW(I,J)
            V(I,J) = VNEW(I,J)
            P(I,J) = PNEW(I,J)
         ENDDO
      ENDDO

C *** PERIODIC CONTINUATION

!loop 320
!hpf$ independent
      DO J=1,NM1
         UOLD(m,J) = UOLD(1,J)
      ENDDO
!hpf$ independent
      DO J=1,NM1
         VOLD(m,J) = VOLD(1,J)
      ENDDO
!hpf$ independent
      DO J=1,NM1
         POLD(m,J) = POLD(1,J)
      ENDDO
!hpf$ independent
      DO J=1,NM1
         U(m,J) = U(1,J)
      ENDDO
!hpf$ independent
      DO J=1,NM1
         V(m,J) = V(1,J)
      ENDDO
!hpf$ independent
      DO J=1,NM1
         P(m,J) = P(1,J)
      ENDDO

!loop 325
!hpf$ independent
      DO I=1,MM1
         UOLD(I,n) = UOLD(I,1)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         VOLD(I,n) = VOLD(I,1)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         POLD(I,n) = POLD(I,1)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         U(I,n) = U(I,1)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         V(I,n) = V(I,1)
      ENDDO
!hpf$ independent
      DO I=1,MM1
         P(I,n) = P(I,1)
      ENDDO

! corners...
      UOLD(m,n) = UOLD(1,1)
      VOLD(m,n) = VOLD(1,1)
      POLD(m,n) = POLD(1,1)
      U(m,n) = U(1,1)
      V(m,n) = V(1,1)
      P(m,n) = P(1,1)

!loop
      GO TO 90

  310 TDT = TDT+TDT

! loop 400 (should be a size-3 flip-flop!)
!hpf$ independent(J,I)
      DO J=1,n
         DO I=1,m
            UOLD(I,J) = U(I,J)
            VOLD(I,J) = V(I,J)
            POLD(I,J) = P(I,J)
            U(I,J) = UNEW(I,J)
            V(I,J) = VNEW(I,J)
            P(I,J) = PNEW(I,J)
         ENDDO
      ENDDO

      GO TO 90

 1000 CONTINUE

      END
C
      SUBROUTINE PLOT (FIELD,NCYCLE,FTITLE)
C
C *** SUBROUTINE TO MAKE TWO-DIMENSIONAL PLOTS OF FIELDS
C
      integer m,n
      parameter(m=320,n=200)
      REAL FIELD (m,n)

!hpf$ processors procs(2,2)
!hpf$ distribute (block,block) onto procs:: FIELD

      integer I,J,ICOLOR
      real top,bottom,scale

      CHARACTER*16 FTITLE
      CHARACTER*1 IMAGE(m,n),LOOKUP(16)
      DATA LOOKUP/'A','B','C','D','E','F','G','H',
     1            'I','J','K','L','M','N','O','P'/

!hpf$ align with FIELD:: IMAGE

      WRITE(2) FTITLE,NCYCLE

      CALL RANGE(FIELD,TOP,BOTTOM)
      SCALE = TOP-BOTTOM

! loop 801
!hpf$ independent(J,I), new(ICOLOR)
      DO J=1,n
         DO I=1,m
            ICOLOR = ((FIELD(I,J)-BOTTOM)/SCALE)*16.0
            IMAGE(I,J) = LOOKUP(ICOLOR + 1)
         ENDDO
      ENDDO
C
C * INSERT A SCALE OF COLORS ALONG THE SIDE OF THE IMAGE
C
! loop 802-803
!hpf$ independent(I,J), new(ICOLOR)
      DO I=20,169
         ICOLOR = (I-10)/10
         DO J=1,20
            IMAGE(J,I)=LOOKUP(ICOLOR)
         ENDDO
      ENDDO

C
C * WRITE THE IMAGE TO THE OUTPUT FILE [804]
C
      DO J=1,n
         WRITE(2) (IMAGE(I,J),I=1,m)
      ENDDO
C
      RETURN
C
      END
C
      SUBROUTINE RANGE(FIELD,TOP,BOTTOM)

      integer n,m
      parameter (m=320,n=200)

      REAL FIELD(m,n), TOP, BOTTOM

!hpf$ processors procs(2,2)
!hpf$ distribute (block,block) onto procs:: FIELD

      TOP = FIELD(1,1)
      BOTTOM = FIELD(1,1)
! loop 901
!hpf$ independent(J,I), reduction(BOTTOM,TOP)
      DO J=1,n
         DO I=1,m
            BOTTOM=MIN(FIELD(I,J),BOTTOM)
            TOP=MAX(FIELD(I,J),TOP)
         ENDDO
      ENDDO
C
      RETURN
C
      END

