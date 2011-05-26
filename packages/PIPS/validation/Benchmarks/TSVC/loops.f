c***********************************************************************
c                TEST SUITE FOR VECTORIZING COMPILERS                  *
c                        (File 2 of 2)                                 *
c                                                                      *
c  Version:   2.0                                                      *
c  Date:      3/14/88                                                  *
c  Authors:   Original loops from a variety of                         *
c             sources. Collection and synthesis by                     *
c                                                                      *
c             David Callahan  -  Tera Computer                         *
c             Jack Dongarra   -  University of Tennessee               *
c             David Levine    -  Argonne National Laboratory           *
c***********************************************************************
c  Version:   3.0                                                      *
c  Date:      1/4/91                                                   *
c  Authors:   David Levine    -  Executable version                    *
c***********************************************************************
c                         ---DESCRIPTION---                            *
c                                                                      *
c  This test consists of a variety of  loops that represent different  *
c  constructs  intended   to  test  the   analysis  capabilities of a  *
c  vectorizing  compiler.  Each loop is  executed with vector lengths  *
c  of 10, 100, and 1000.   Also included  are several simple  control  *
c  loops  intended  to  provide  a  baseline  measure  for  comparing  *
c  compiler performance on the more complicated loops.                 *
c                                                                      *
c  The  output from a  run  of the test  consists of seven columns of  *
c  data:                                                               *
c     Loop:        The name of the loop.                               *
c     VL:          The vector length the loop was run at.              *
c     Seconds:     The time in seconds to run the loop.                *
c     Checksum:    The checksum calculated when running the test.      *
c     PreComputed: The precomputed checksum (64-bit IEEE arithmetic).  *
c     Residual:    A measure of the accuracy of the calculated         *
c                  checksum versus the precomputed checksum.           *
c     No.:         The number of the loop in the test suite.           *
c                                                                      *
c  The  residual  calculation  is  intended  as  a  check  that   the  *
c  computation  was  done  correctly  and  that 64-bit arithmetic was  *
c  used.   Small   residuals    from    non-IEEE    arithmetic    and  *
c  nonassociativity  of   some calculations  are   acceptable.  Large  *
c  residuals  from   incorrect  computations or  the  use   of 32-bit  *
c  arithmetic are not acceptable.                                      *
c                                                                      *
c  The test  output  itself  does not report   any  results;  it just  *
c  contains data.  Absolute  measures  such as Mflops and  total time  *
c  used  are  not   appropriate    metrics  for  this  test.   Proper  *
c  interpretation of the results involves correlating the output from  *
c  scalar and vector runs  and the  loops which  have been vectorized  *
c  with the speedup achieved at different vector lengths.              *
c                                                                      *
c  These loops  are intended only  as  a partial test of the analysis  *
c  capabilities of a vectorizing compiler (and, by necessity,  a test  *
c  of the speed and  features  of the underlying   vector  hardware).  *
c  These loops  are  by no means  a  complete  test  of a vectorizing  *
c  compiler and should not be interpreted as such.                     *
c                                                                      *
c***********************************************************************
c                           ---DIRECTIONS---                           *
c                                                                      *
c  To  run this  test,  you will  need  to  supply  a  function named  *
c  second() that returns user CPU time.                                *
c                                                                      *
c  This test is distributed as two separate files, one containing the  *
c  driver  and  one containing the loops.   These  two files MUST  be  *
c  compiled separately.                                                *
c                                                                      *
c  Results must  be supplied from  both scalar and vector  runs using  *
c  the following rules for compilation:                                *
c                                                                      *
c     Compilation   of the  driver  file must  not  use any  compiler  *
c     optimizations (e.g., vectorization, function  inlining,  global  *
c     optimizations,...).   This   file   also must  not  be analyzed  *
c     interprocedurally to  gather information useful  in  optimizing  *
c     the test loops.                                                  *
c                                                                      *
c     The file containing the  loops must be compiled twice--once for  *
c     a scalar run and once for a vector run.                          *
c                                                                      *
c        For the scalar  run, global (scalar) optimizations should be  *
c        used.                                                         *
c                                                                      *
c        For  the  vector run,  in  addition   to  the  same   global  *
c        optimizations specified  in the scalar   run,  vectorization  *
c        and--if available--automatic  call generation to   optimized  *
c        library  routines,  function inlining,  and  interprocedural  *
c        analysis should be  used.  Note again that function inlining  *
c        and interprocedural  analysis  must  not be  used to  gather  *
c        information  about any of the  program  units  in the driver  *
c        program.                                                      *
c                                                                      *
c     No changes  may  be made  to   the   source code.   No compiler  *
c     directives may be used, nor may  a file  be split into separate  *
c     program units.  (The exception is  filling  in  the information  *
c     requested in subroutine "info" as described below.)              *
c                                                                      *
c     All files must be compiled to use 64-bit arithmetic.             *
c                                                                      *
c     The  outer  timing  loop  is  included  only   to increase  the  *
c     granularity of the calculation.  It should not be vectorizable.  *
c     If it is found to be so, please notify the authors.              *
c                                                                      *
c  All runs  must be  made  on a standalone  system  to minimize  any  *
c  external effects.                                                   *
c                                                                      *
c  On virtual memory computers,  runs should be  made with a physical  *
c  memory and working-set  size  large enough  that  any  performance  *
c  degradation from page  faults is negligible.   Also,  the  timings  *
c  should be repeatable  and you  must  ensure  that timing anomalies  *
c  resulting from paging effects are not present.                      *
c                                                                      *
c  You should edit subroutine "info"   (the  last subroutine  in  the  *
c  driver program) with  information specific to your  runs, so  that  *
c  the test output will be annotated automatically.                    *
c                                                                      *
c  Please return the following three files in an electronic format:    *
c                                                                      *
c  1. Test output from a scalar run.                                   *
c  2. Test output from a vector run.                                   *
c  3. Compiler output listing (source echo, diagnostics, and messages) *
c     showing which loops have been vectorized.                        *
c                                                                      *
c  The preferred media  for receipt, in order  of preference, are (1)  *
c  electronic mail, (2) 9-track  magnetic or  cartridge tape in  Unix  *
c  tar  format, (3) 5" IBM PC/DOS  floppy   diskette, or  (4) 9-track  *
c  magnetic  tape in  ascii  format,   80 characters per card,  fixed  *
c  records, 40 records per block, 1600bpi.  Please return to           *
c                                                                      *
c  David Levine       		                                       *
c  Mathematics and Computer Science Division                           *
c  Argonne National Laboratory                                         *
c  Argonne, Illinois 60439                                             *
c  levine@mcs.anl.gov                                                  *
c***********************************************************************
c%1.1
      subroutine s111 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     linear dependence testing
c     no dependence - vectorizable
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s111 ')
      t1 = second()
      do 1 nl = 1,2*ntimes
      do 10 i = 2,n,2
         a(i) = a(i-1) + b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(2*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,2*ntimes*(n/2),n,t2,'s111 ')
      return
      end
c%1.1
      subroutine s112 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     linear dependence testing
c     loop reversal
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s112 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = n-1,1,-1
         a(i+1) = a(i) + b(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s112 ')
      return
      end
c%1.1
      subroutine s113 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     linear dependence testing
c     a(i)=a(1) but no actual dependence cycle
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s113 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n
         a(i) = a(1) + b(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s113 ')
      return
      end
c%1.1
      subroutine s114 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     linear dependence testing
c     transpose vectorization
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s114 ')
      t1 = second()
      do 1 nl = 1,2*(ntimes/n)
      do 10 j = 1,n
         do 20 i = 1,j-1
            aa(i,j) = aa(j,i) + bb(i,j)
   20    continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(2*(ntimes/n)) )
      chksum = cs2d(n,aa)
      call check (chksum,2*(ntimes/n)*((n*n-n)/2),n,t2,'s114 ')
      return
      end
c%1.1
      subroutine s115 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     linear dependence testing
c     triangular saxpy loop
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s115 ')
      t1 = second()
      do 1 nl = 1,2*(ntimes/n)
      do 10 j = 1,n
         do 20 i = j+1, n
            a(i) = a(i) - aa(i,j) * a(j)
  20     continue
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(2*(ntimes/n)) )
      chksum = cs1d(n,a)
      call check (chksum,2*(ntimes/n)*((n*n-n)/2),n,t2,'s115 ')
      return
      end
c%1.1
      subroutine s116 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     linear dependence testing
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s116 ')
      t1 = second()
      do 1 nl = 1,5*ntimes
      do 10 i = 1,n-5,5
         a(i)   = a(i+1) * a(i)  
         a(i+1) = a(i+2) * a(i+1)
         a(i+2) = a(i+3) * a(i+2)
         a(i+3) = a(i+4) * a(i+3)
         a(i+4) = a(i+5) * a(i+4)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(5*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,5*ntimes*(n/5),n,t2,'s116 ')
      return
      end
c%1.1
      subroutine s118 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     linear dependence testing
c     potential dot product recursion
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s118 ')
      t1 = second()
      do 1 nl = 1,2*(ntimes/n)
      do 10 i = 2,n
         do 20 j = 1,i-1
               a(i) = a(i) + bb(i,j) * a(i-j)
  20     continue
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(2*(ntimes/n)) )
      chksum = cs1d(n,a)
      call check (chksum,2*(ntimes/n)*((n*n-n)/2),n,t2,'s118 ')
      return
      end
c%1.1
      subroutine s119 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     linear dependence testing
c     no dependence - vectorizable
c     
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s119 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 j = 2,n
         do 20 i = 2,n
            aa(i,j) = aa(i-1,j-1) + bb(i,j)
  20     continue
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa)
      call check (chksum,(ntimes/n)*(n-1)*(n-1),n,t2,'s119 ')
      return
      end
c%1.2
      subroutine s121 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     induction variable recognition
c     loop with possible ambiguity because of scalar store
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s121 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1, n-1
         j = i+1
         a(i) = a(j) + b(i)
10    continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
1     continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s121 ')
      return
      end
c%1.2
      subroutine s122 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,n1,n3)
c
c     induction variable recognition
c     variable lower and upper bound, and stride
c
      integer ntimes, ld, n, i, nl, j, k, n1, n3
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s122 ')
      t1 = second()
      do 1 nl = 1,ntimes
      j = 1
      k = 0
      do 10 i=n1,n,n3
         k = k + j
         a(i) = a(i) + b(n-k+1)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s122 ')
      return
      end
c%1.2
      subroutine s123 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     induction variable recognition
c     induction variable under an if
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s123 ')
      t1 = second()
      do 1 nl = 1,2*ntimes
      j = 0
      do 10 i = 1,n/2
         j = j + 1
         a(j) = b(i) + d(i) * e(i)
         if(c(i) .gt. 0.) then
            j = j + 1
            a(j) = c(i)+ d(i) * e(i)
         endif
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(2*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,2*ntimes*(n/2),n,t2,'s123 ')
      return
      end
c%1.2
      subroutine s124 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     induction variable recognition
c     induction variable under both sides of if (same value)
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s124 ')
      t1 = second()
      do 1 nl = 1,2*ntimes
      j = 0
      do 10 i = 1,n/2
         if(b(i) .gt. 0.) then
            j = j + 1
            a(j) = b(i) + d(i) * e(i)
            else
            j = j + 1
            a(j) = c(i) + d(i) * e(i)
         endif
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(2*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,2*ntimes*(n/2),n,t2,'s124 ')
      return
      end
c%1.2
      subroutine s125 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     induction variable recognition
c     induction variable in two loops; collapsing possible
c

      integer ntimes, ld, n, i, nl, j, k, nn
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, array
      parameter(nn=1000)
      common /cdata/ array(nn*nn)

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s125 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      k = 0
      do 10 j = 1,n
         do 20 i = 1,n
            k = k + 1
            array(k) = aa(i,j) + bb(i,j) * cc(i,j)
   20    continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs1d(n*n,array)
      call check (chksum,(ntimes/n)*n*n,n,t2,'s125 ')
      return
      end
c%1.2
      subroutine s126 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     induction variable recognition
c     induction variable in two loops; recurrence in inner loop
c
      integer ntimes, ld, n, i, nl, j, k, nn
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d, array
      parameter(nn=1000)
      common /cdata/ array(nn*nn)

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s126 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      k = 1
      do 10 i = 1,n
         do 20 j = 2,n
            bb(i,j) = bb(i,j-1) + array(k) * cc(i,j)
            k = k + 1
   20    continue
         k = k + 1
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,bb)
      call check (chksum,(ntimes/n)*n*(n-1),n,t2,'s126 ')
      return
      end
c%1.2
      subroutine s127 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     induction variable recognition
c     induction variable with multiple increments
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s127 ')
      t1 = second()
      do 1 nl = 1,2*ntimes
      j = 0
      do 10 i = 1,n/2
         j = j + 1
         a(j) = b(i) + c(i) * d(i)
         j = j + 1
         a(j) = b(i) + d(i) * e(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(2*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,2*ntimes*(n/2),n,t2,'s127 ')
      return
      end
c%1.2
      subroutine s128 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     induction variables
c     coupled induction variables
c
      integer ntimes, ld, n, i, nl, j, k
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s128 ')
      t1 = second()
      do 1 nl = 1,2*ntimes
      j = 0
      do 10 i = 1, n/2
         k = j + 1
         a(i) = b(k) - d(i)
         j = k + 1
         b(k) = a(i) + c(k)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(2*ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,2*ntimes*(n/2),n,t2,'s128 ')
      return
      end
c%1.3
      subroutine s131 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     global data flow analysis
c     forward substitution
c
      integer ntimes, ld, n, i, nl, m
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      m = 1
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s131 ')
      if(a(1).gt.0)then
         a(1) = b(1)
      endif
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
         a(i) = a(i+m) + b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s131 ')
      return
      end
c%1.3
      subroutine s132 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     global data flow analysis
c     loop with multiple dimension ambiguous subscripts
c
      integer ntimes, ld, n, i, nl, j, k, m
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      m = 1
      j = m
      k = m+1
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s132 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i=2,n
         aa(i,j) = aa(i-1,k) + b(i) * c(k)
10    continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
1     continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs2d(n,aa)
      call check (chksum,ntimes*n-1,n,t2,'s132 ')
      return
      end
c%1.4
      subroutine s141 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     nonlinear dependence testing
c     walk a row in a symmetric packed array
c     element a(i,j) for (j>i) stored in location j*(j-1)/2+i
c
      integer ntimes, ld, n, i, nl, j, k, nn
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, array
      parameter(nn=1000)
      common /cdata/ array(nn*nn)

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s141 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 i = 1,n
         k = i*(i-1)/2+i
         do 20 j = i,n
	    array(k) = array(k) + bb(i,j)
	    k = k + j		
  20     continue
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs1d(n*n,array)
      call check (chksum,(ntimes/n)*n*n,n,t2,'s141 ')
      return
      end
c%1.5
      subroutine s151 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     interprocedural data flow analysis
c     passing parameter information into a subroutine
c
      integer ntimes, ld, n, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s151 ')
      t1 = second()
      do 1 nl = 1,ntimes
      call s151s(a,b,n,1)
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s151 ')
      return
      end
      subroutine s151s(a,b,n,m)
      integer i, n, m
      real a(n), b(n)
      do 10 i = 1,n-1
         a(i) = a(i+m) + b(i)
  10  continue
      return
      end
c%1.5
      subroutine s152 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     interprocedural data flow analysis
c     collecting information from a subroutine
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s152 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         b(i) = d(i) * e(i)
         call s152s(a,b,c,i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s152 ')
      return
      end
      subroutine s152s(a,b,c,i)
      integer i
      real a(*), b(*), c(*)
      a(i) = a(i) + b(i) * c(i)
      return
      end
c%1.6
      subroutine s161 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     tests for recognition of loop independent dependences
c     between statements in mutually exclusive regions.
c 
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s161 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
         if (b(i) .lt. 0.) go to 20
         a(i)   = c(i) + d(i) * e(i)
         go to 10
   20    c(i+1) = a(i) + d(i) * d(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,c)
      call check (chksum,ntimes*(n-1),n,t2,'s161 ')
      return
      end
c%1.6
      subroutine s162 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,k)
c
c     control flow
c     deriving assertions
c
      integer ntimes, ld, n, i, nl, k
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s162 ')
      t1 = second()
      do 1 nl = 1,ntimes
      if ( k .gt. 0 ) then
         do 10 i = 1,n-1
            a(i) = a(i+k) + b(i) * c(i)
10       continue
      endif
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s162 ')
      return
      end
c%1.7
      subroutine s171 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,inc)
c
c     symbolics
c     symbolic dependence tests
c
      integer ntimes, ld, n, i, nl, inc
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s171 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i*inc) = a(i*inc) + b(i)
 10   continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
 1    continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s171 ')
      return
      end
c%1.7
      subroutine s172 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,n1,n3)
c
c     symbolics
c     vectorizable if n3 .ne. 0
c
      integer ntimes, ld, n, i, nl, n1, n3
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s172 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = n1,n,n3
         a(i) = a(i) + b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s172 ')
      return
      end
c%1.7
      subroutine s173 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     symbolics
c     expression in loop bounds and subscripts
c
      integer ntimes, ld, n, i, nl, k
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      k = n/2
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s173 ')
      t1 = second()
      do 1 nl = 1,2*ntimes
      do 10 i = 1,n/2
            a(i+k) = a(i) +  b(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(2*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,2*ntimes*(n/2),n,t2,'s173 ')
      return
      end
c%1.7
      subroutine s174 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     symbolics
c     loop with subscript that may seem ambiguous
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s174 ')
      t1 = second()
      do 1 nl = 1,2*ntimes
      do 10 i= 1, n/2
         a(i) = a(i+n/2) + b(i)
10    continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
1     continue
      t2 = second() - t1 - ctime - ( dtime * float(2*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,2*ntimes*(n/2),n,t2,'s174 ')
      return
      end
c%1.7
      subroutine s175 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,inc)
c
c     symbolics
c     symbolic dependence tests
c
      integer ntimes, ld, n, i, nl, inc
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s175 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1,inc
         a(i) = a(i+inc) + b(i)
 10   continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
 1    continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s175 ')
      return
      end
c%1.7
      subroutine s176 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     symbolics
c     convolution
c
      integer ntimes, ld, n, i, nl, j, m
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      m = n/2
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s176 ')
      t1 = second()
      do 1 nl = 1,4*(ntimes/n)
      do 10 j = 1,n/2
        do 20 i = 1,m
           a(i) = a(i) + b(i+m-j) * c(j)
   20   continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(4*(ntimes/n)) )
      chksum = cs1d(n,a)
      call check (chksum,4*(ntimes/n)*(n/2)*(n/2),n,t2,'s176 ')
      return
      end
c
c**********************************************************
c                                                         *
c                      VECTORIZATION                      *
c                                                         *
c**********************************************************
c%2.1
      subroutine s211 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     statement reordering
c     statement reordering allows vectorization
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s211 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n-1
         a(i) = b(i-1) + c(i) * d(i)
         b(i) = b(i+1) - e(i) * d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-2),n,t2,'s211 ')
      return
      end
c%2.1
      subroutine s212 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     statement reordering
c     dependency needing temporary
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s212 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i=1,n-1
         a(i) = a(i) * c(i)
         b(i) = b(i) + a(i+1) * d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s212 ')
      return
      end
c%2.2
      subroutine s221 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop distribution
c     loop that is partially recursive
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s221 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n
         a(i) = a(i)   + c(i) * d(i)
         b(i) = b(i-1) + a(i) + d(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s221 ')
      return
      end
c%2.2
      subroutine s222 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop distribution
c     partial loop vectorization, recurrence in middle
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s222 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n
         a(i) = a(i)   + b(i)   * c(i)
         b(i) = b(i-1) * b(i-1) * a(i)
         a(i) = a(i)   - b(i)   * c(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s222 ')
      return
      end
c%2.3
      subroutine s231 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop interchange
c     loop with multiple dimension recursion
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s231 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 i=1,n
         do 20 j=2,n
            aa(i,j) = aa(i,j-1) + bb(i,j)
   20    continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa)
      call check (chksum,(ntimes/n)*n*(n-1),n,t2,'s231 ')
      return
      end
c%2.3
      subroutine s232 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop interchange
c     interchanging of triangular loops
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s232 ')
      t1 = second()
      do 1 nl = 1,2*(ntimes/n)
      do 10 j = 2,n
         do 20 i = 2,j
            aa(i,j) = aa(i-1,j)*aa(i-1,j)+bb(i,j)
  20     continue
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(2*(ntimes/n)) )
      chksum = cs2d(n,aa)
      call check (chksum,2*(ntimes/n)*((n*n-n)/2),n,t2,'s232 ')
      return
      end
c%2.3
      subroutine s233 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop interchange
c     interchanging with one of two inner loops
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s233 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 i = 2,n
         do 20 j = 2,n
            aa(i,j) = aa(i,j-1) + cc(i,j)
  20     continue
         do 30 j = 2,n
            bb(i,j) = bb(i-1,j) + cc(i,j)
  30     continue
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa) + cs2d(n,bb)
      call check (chksum,(ntimes/n)*(n-1)*(2*n-2),n,t2,'s233 ')
      return
      end
c%2.3
      subroutine s234 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop interchange
c     if loop to do loop, interchanging with if loop necessary
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s234 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      i = 1
   11 if(i.gt.n) goto 10
         j = 2
   21    if(j.gt.n) goto 20
            aa(i,j) = aa(i,j-1) + bb(i,j-1) * cc(i,j-1)
            j = j + 1
         goto 21
   20 i = i + 1
      goto 11
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa)
      call check (chksum,(ntimes/n)*n*(n-1),n,t2,'s234 ')
      return
      end
c%2.3
      subroutine s235 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop interchanging
c     imperfectly nested loops
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s235 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 i = 1,n
         a(i) =  a(i) + b(i) * c(i)
         do 20 j = 2,n
            aa(i,j) = aa(i,j-1) +  bb(i,j) * a(i)
  20     continue
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa) + cs1d(n,a)
      call check (chksum,(ntimes/n)*n*(n-1),n,t2,'s235 ')
      return
      end
c%2.4
      subroutine s241 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     node splitting
c     preloading necessary to allow vectorization
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s241 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
         a(i) = b(i) * c(i)   * d(i)
         b(i) = a(i) * a(i+1) * d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s241 ')
      return
      end
c%2.4
      subroutine s242 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,s1,s2)
c
c     node splitting
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, s1, s2

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s242 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n
         a(i) = a(i-1) + s1 + s2 + b(i) + c(i) + d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s242 ')
      return
      end
c%2.4
      subroutine s243 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     node splitting
c     false dependence cycle breaking
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s243 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
         a(i) = b(i) + c(i)   * d(i)
         b(i) = a(i) + d(i)   * e(i)
         a(i) = b(i) + a(i+1) * d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s243 ')
      return
      end
c%2.4
      subroutine s244 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     node splitting
c     false dependence cycle breaking
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s244 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
         a(i)   = b(i) + c(i)   * d(i)
         b(i)   = c(i) + b(i)
         a(i+1) = b(i) + a(i+1) * d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s244 ')
      return
      end
c%2.5
      subroutine s251 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar and array expansion
c     scalar expansion
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, s

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s251 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         s    = b(i) + c(i) * d(i)
         a(i) = s * s
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s251 ')
      return
      end
c%2.5
      subroutine s252 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar and array expansion
c     loop with ambiguous scalar temporary
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, s, t

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s252 ')
      t1 = second()
      do 1 nl = 1,ntimes
      t = 0.
      do 10 i=1,n
         s    = b(i) * c(i)
         a(i) = s + t
         t    = s
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s252 ')
      return
      end
c%2.5
      subroutine s253 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar and array expansion
c     scalar expansion, assigned under if
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, s

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s253 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         if(a(i) .gt. b(i))then
            s    = a(i) - b(i) * d(i)
            c(i) = c(i) + s
            a(i) = s
         endif
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,c)
      call check (chksum,ntimes*n,n,t2,'s253 ')
      return
      end
c%2.5
      subroutine s254 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar and array expansion
c     carry around variable
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, x

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s254 ')
      t1 = second()
      do 1 nl = 1,ntimes
      x = b(n)
      do 10 i = 1,n
         a(i) = (b(i) + x) * .5
         x    =  b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s254 ')
      return
      end
c%2.5
      subroutine s255 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar and array expansion
c     carry around variables, 2 levels
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, x, y

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s255 ')
      t1 = second()
      do 1 nl = 1,ntimes
      x = b(n)
      y = b(n-1)
      do 10 i = 1,n
         a(i) = (b(i) + x + y) * .333
         y    =  x
         x    =  b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s255 ')
      return
      end
c%2.5
      subroutine s256 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar and array expansion
c     array expansion
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s256 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 i = 1,n
         do 20 j = 2,n
            a(j)    = aa(i,j) - a(j-1)
            aa(i,j) = a(j) + bb(i,j)
   20    continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs1d(n,a) + cs2d(n,aa)
      call check (chksum,(ntimes/n)*n*(n-1),n,t2,'s256 ')
      return
      end
c%2.5
      subroutine s257 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar and array expansion
c     array expansion
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s257 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 i = 2,n
         do 20 j = 1,n
            a(i)    = aa(i,j) - a(i-1)
            aa(i,j) = a(i) + bb(i,j)
   20    continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs1d(n,a) + cs2d(n,aa)
      call check (chksum,(ntimes/n)*(n-1)*n,n,t2,'s257 ')
      return
      end
c%2.5
      subroutine s258 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar and array expansion
c     wrap-around scalar under an if
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, s

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s258 ')
      t1 = second()
      do 1 nl = 1,ntimes
      s = 0.
      do 10 i = 1,n
         if (a(i) .gt. 0.) then
            s = d(i) * d(i)
         endif
         b(i) = s * c(i) + d(i)
         e(i) = (s + 1.) * aa(i,1)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,b) + cs1d(n,e)
      call check (chksum,ntimes*n,n,t2,'s258 ')
      return
      end
c%2.6
      subroutine s261 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     scalar renaming
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, t

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s261 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n
         t    = a(i) + b(i) 
         a(i) = t    + c(i-1) 
         t    = c(i) * d(i)
         c(i) = t 
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,c)
      call check (chksum,ntimes*(n-1),n,t2,'s261 ')
      return
      end
c%2.7
      subroutine s271 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     loop with singularity handling
c    
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s271 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i=1,n
         if (b(i) .gt. 0.) a(i) = a(i) + b(i) * c(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s271 ')
      return
      end
c%2.7
      subroutine s272 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,t)
c
c     control flow
c     loop with independent conditional
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, t

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s272 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1, n
         if (e(i) .ge. t) then
            a(i) = a(i) + c(i) * d(i)
            b(i) = b(i) + c(i) * c(i)
         endif
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*n,n,t2,'s272 ')
      return
      end
c%2.7
      subroutine s273 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     simple loop with dependent conditional
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s273 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1 , n
         a(i) = a(i) + d(i) * e(i)
         if (a(i) .lt. 0.) b(i) = b(i) + d(i) * e(i)
         c(i) = c(i) + a(i) * d(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b) + cs1d(n,c)
      call check (chksum,ntimes*n,n,t2,'s273 ')
      return
      end
c%2.7
      subroutine s274 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     complex loop with dependent conditional
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s274 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1 , n
         a(i) = c(i) + e(i) * d(i)
         if (a(i) .gt. 0.) then
            b(i) = a(i) + b(i)
         else
            a(i) = d(i) * e(i)
         endif
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*n,n,t2,'s274 ')
      return
      end
c%2.7
      subroutine s275 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     if around inner loop, interchanging needed
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s275 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 i = 2,n
         if(aa(i,1) .gt. 0.)then
            do 20 j = 2,n
              aa(i,j) = aa(i,j-1) + bb(i,j) * cc(i,j)
  20        continue
         endif
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa)
      call check (chksum,(ntimes/n)*(n-1)*(n-1),n,t2,'s275 ')
      return
      end
c%2.7
      subroutine s276(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     if test using loop index
c
      integer ntimes, ld, n, i, nl, mid
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s276 ')
      t1 = second()
      mid = n/2
      do 1 nl = 1,ntimes
      do 10 i = 1, n
        if ( i .lt. mid ) then
           a(i) = a(i) + b(i) * c(i)
        else
           a(i) = a(i) + b(i) * d(i)
        endif
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s276 ')
      return
      end
c%2.7
      subroutine s277 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     test for dependences arising from guard variable computation.
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s277 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
        if (a(i) .ge. 0.) go to 20
        if (b(i) .ge. 0.) go to 30
           a(i)   = a(i) + c(i) * d(i)
   30   continue
           b(i+1) = c(i) + d(i) * e(i)
   20 continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s277 ')
      return
      end
c%2.7
      subroutine s278 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     if/goto to block if-then-else
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s278 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         if (a(i) .gt. 0.) goto 20
            b(i) = -b(i) + d(i) * e(i)
         goto 30
  20     continue
            c(i) = -c(i) + d(i) * e(i)
  30     continue
            a(i) =  b(i) + c(i) * d(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b) + cs1d(n,c)
      call check (chksum,ntimes*n,n,t2,'s278 ')
      return
      end
c%2.7
      subroutine s279 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     vector if/gotos
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s279 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         if (a(i) .gt. 0.) goto 20
            b(i) = -b(i) + d(i) * d(i)
         if (b(i) .le. a(i)) goto 30
            c(i) =  c(i) + d(i) * e(i)
         goto 30
  20     continue
            c(i) = -c(i) + e(i) * e(i)
  30     continue
            a(i) =  b(i) + c(i) * d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b) + cs1d(n,c)
      call check (chksum,ntimes*n,n,t2,'s279 ')
      return
      end
c%2.7
      subroutine s2710(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,x)
c
c     control flow
c     scalar and vector ifs
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, x

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s2710')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         if(a(i) .gt. b(i))then
            a(i) = a(i) + b(i) * d(i)
            if(n .gt. 10)then
               c(i) =  c(i) + d(i) * d(i)
            else
               c(i) =  1.0  + d(i) * e(i)
            endif
         else
            b(i) = a(i) + e(i) * e(i)
            if(x .gt. 0.)then
               c(i) =  a(i) + d(i) * d(i)
            else
               c(i) =  c(i) + e(i) * e(i)
            endif
         endif
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b) + cs1d(n,c)
      call check (chksum,ntimes*n,n,t2,'s2710')
      return
      end
c%2.7
      subroutine s2711(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     semantic if removal
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s2711')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         if(b(i) .ne. 0.) a(i) = a(i) + b(i) * c(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s2711')
      return
      end
c%2.7
      subroutine s2712(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control flow
c     if to elemental min
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s2712')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         if(a(i) .gt. b(i)) a(i) = a(i) + b(i) * c(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s2712')
      return
      end
c%2.8
      subroutine s281 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     crossing thresholds
c     index set splitting
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, x

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s281 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         x    = a(n-i+1) + b(i) * c(i)
         a(i) = x - 1.0
         b(i) = x
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*n,n,t2,'s281 ')
      return
      end
c%2.9
      subroutine s291 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop peeling
c     wrap around variable, 1 level
c
      integer ntimes, ld, n, i, nl, im1
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s291 ')
      t1 = second()
      do 1 nl = 1,ntimes
      im1 = n
      do 10 i = 1,n
         a(i) = (b(i) + b(im1)) * .5
         im1  = i
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s291 ')
      return
      end
c%2.9
      subroutine s292 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop peeling
c     wrap around variable, 2 levels
c
      integer ntimes, ld, n, i, nl, im1, im2
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s292 ')
      t1 = second()
      do 1 nl = 1,ntimes
      im1 = n
      im2 = n-1
      do 10 i = 1,n
         a(i) = (b(i) + b(im1) + b(im2)) * .333
         im2 = im1
         im1 = i
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s292 ')
      return
      end
c%2.9
      subroutine s293 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop peeling
c     a(i)=a(1) with actual dependence cycle, loop is vectorizable
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s293 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(1)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s293 ')
      return
      end
c%2.10
      subroutine s2101(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     diagonals
c     main diagonal calculation
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s2101')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         aa(i,i) = aa(i,i) + bb(i,i) * cc(i,i) 
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs2d(n,aa)
      call check (chksum,ntimes*n,n,t2,'s2101')
      return
      end
c%2.12
      subroutine s2102(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     diagonals
c     identity matrix, best results vectorize both inner and outer loops
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s2102')
      t1 = second()
      do 1 nl = 1,ntimes/n
      do 10 i = 1, n
         do 20 j = 1,n
            aa(i,j) = 0.
   20    continue
         aa(i,i) = 1.
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa)
      call check (chksum,(ntimes/n)*n*n,n,t2,'s2102')
      return
      end
c%2.11
      subroutine s2111 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     wavefronts
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s2111')
      t1 = second()
      do 1 nl = 1,(ntimes/n)
      do 10 j = 2,n
         do 20 i = 2,n
            aa(i,j) = aa(i-1,j) + aa(i,j-1)
   20    continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa)
      if (chksum .eq. 0.)chksum = 3.0
      call check (chksum,(ntimes/n)*(n-1)*(n-1),n,t2,'s2111')
      return
      end
c
c**********************************************************
c                                                         *
c                   IDIOM RECOGNITION                     *
c                                                         *
c**********************************************************
c%3.1
      subroutine s311 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     sum reduction
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, sum

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s311 ')
      t1 = second()
      do 1 nl = 1,ntimes
      sum = 0.
      do 10 i = 1,n
         sum = sum + a(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,sum)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = sum
      call check (chksum,ntimes*n,n,t2,'s311 ')
      return
      end
c%3.1
      subroutine s312 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     product reduction
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, prod

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s312 ')
      t1 = second()
      do 1 nl = 1,ntimes
      prod = 1.
      do 10 i = 1,n
         prod = prod * a(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,prod)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = prod
      call check (chksum,ntimes*n,n,t2,'s312 ')
      return
      end
c%3.1
      subroutine s313 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     dot product
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, dot

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s313 ')
      t1 = second()
      do 1 nl = 1,ntimes
      dot = 0.
      do 10 i = 1,n
         dot = dot + a(i) * b(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,dot)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = dot
      call check (chksum,ntimes*n,n,t2,'s313 ')
      return
      end
c%3.1
      subroutine s314 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     if to max reduction
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, x

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s314 ')
      t1 = second()
      do 1 nl = 1,ntimes
      x = a(1)
      do 10 i = 2,n
         if(a(i) .gt. x) x = a(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,x)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = x
      call check (chksum,ntimes*n,n,t2,'s314 ')
      return
      end
c%3.1
      subroutine s315 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     if to max with index reduction, 1 dimension
c
      integer ntimes, ld, n, i, nl, index
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, x

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s315 ')
      t1 = second()
      do 1 nl = 1,ntimes
      x     = a(1)
      index = 1
      do 10 i = 2,n
         if(a(i) .gt. x)then
            x     = a(i)
            index = i
         endif
   10 continue
      chksum = x+float(index)
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,chksum)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = x + float(index)
      call check (chksum,ntimes*n,n,t2,'s315 ')
      return
      end
c%3.1
      subroutine s316 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     if to min reduction
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, x

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s316 ')
      t1 = second()
      do 1 nl = 1,ntimes
      x = a(1)
      do 10 i = 2,n
         if (a(i) .lt. x) x = a(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,x)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = x
      call check (chksum,ntimes*n,n,t2,'s316 ')
      return
      end
c%3.1
      subroutine s317 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     product reduction, vectorize with
c     1. scalar expansion of factor, and product reduction
c     2. closed form solution: q = factor**n
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, q, factor
      parameter(factor=.99999)

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s317 ')
      t1 = second()
      do 1 nl = 1,ntimes
      q = 1.
      do 10 i = 1,n
         q = factor*q
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,q)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = q
      call check (chksum,ntimes*n,n,t2,'s317 ')
      return
      end
c%3.1
      subroutine s318 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,inc)
c
c     reductions
c     isamax, max absolute value, increments not equal to 1
c
c
      integer ntimes, ld, n, i, nl, inc, k, index
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, max

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s318 ')
      t1 = second()
      do 1 nl = 1,ntimes
      k     = 1
      index = 1
      max   = abs(a(1))
      k     = k + inc
      do 10 i = 2,n
         if(abs(a(k)) .le. max) go to 5
         index = i
         max   = abs(a(k))
    5    k     = k + inc
   10 continue
      chksum = max + float(index)
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,chksum)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = max + float(index)
      call check (chksum,ntimes*(n-1),n,t2,'s318 ')
      return
      end
c%3.1
      subroutine s319 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     coupled reductions
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, sum

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s319 ')
      t1 = second()
      do 1 nl = 1,ntimes
      sum = 0.
      do 10 i = 1,n
         a(i) = c(i) + d(i)
         sum  = sum + a(i)
         b(i) = c(i) + e(i)
         sum  = sum + b(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,sum)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = sum
      call check (chksum,ntimes*n,n,t2,'s319 ')
      return
      end
c%3.1
      subroutine s3110(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     if to max with index reduction, 2 dimensions
c
      integer ntimes, ld, n, i, nl, j, xindex, yindex
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, max

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s3110')
      t1 = second()
      do 1 nl = 1,ntimes/n
      max    = aa(1,1)
      xindex = 1
      yindex = 1
      do 10 j = 1,n
         do 20 i = 1,n
            if ( aa(i,j) .gt. max ) then
               max    = aa(i,j)
               xindex = i
               yindex = j
            endif
  20     continue
  10  continue
      chksum = max + float(xindex) + float(yindex)
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,chksum)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = max + float(xindex) + float(yindex)
      call check (chksum,(ntimes/n)*n*n,n,t2,'s3110')
      return
      end
c%3.1
      subroutine s3111(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     conditional sum reduction
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, sum

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s3111')
      t1 = second()
      do 1 nl = 1,ntimes
      sum = 0.
      do 10 i = 1,n
         if ( a(i) .gt. 0. ) sum = sum + a(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,sum)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = sum
      call check (chksum,ntimes*n,n,t2,'s3111')
      return
      end
c%3.1
      subroutine s3112(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     sum reduction saving running sums
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, sum

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s3112')
      t1 = second()
      do 1 nl = 1,ntimes
      sum = 0.
      do 10 i = 1,n
         sum  = sum + a(i)
         b(i) = sum
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,sum)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,b) + sum
      call check (chksum,ntimes*n,n,t2,'s3112')
      return
      end
c%3.1
      subroutine s3113(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     reductions
c     maximum of absolute value
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, max

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s3113')
      t1 = second()
      do 1 nl = 1,ntimes
      max = abs(a(1))
      do 10 i = 2,n
         if(abs(a(i)) .gt. max) max = abs(a(i))
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,max)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = max
      call check (chksum,ntimes*(n-1),n,t2,'s3113')
      return
      end
c%3.2
      subroutine s321 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     recurrences
c     first order linear recurrence
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s321 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n
         a(i) = a(i) + a(i-1) * b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s321 ')
      return
      end
c%3.2
      subroutine s322 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     recurrences
c     second order linear recurrence
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s322 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 3,n
         a(i) = a(i) + a(i-1) * b(i) + a(i-2) * c(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-2),n,t2,'s322 ')
      return
      end
c%3.2
      subroutine s323 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     recurrences
c     coupled recurrence
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s323 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n
         a(i) = b(i-1) + c(i) * d(i)
         b(i) = a(i)   + c(i) * e(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s323 ')
      return
      end
c%3.3
      subroutine s331 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     search loops
c     if to last-1
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s331 ')
      t1 = second()
      do 1 nl = 1,ntimes
      j  = -1
      do 10 i = 1,n
         if(a(i) .lt. 0) j = i
  10  continue
      chksum = float(j)
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,chksum)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = float(j)
      call check (chksum,ntimes*n,n,t2,'s331 ')
      return
      end
c%3.3
      subroutine s332 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,t)
c
c     search loops
c     first value greater than threshold
c
      integer ntimes, ld, n, i, nl, index
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, t, value

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s332 ')
      t1 = second()
      do 1 nl = 1,ntimes
      index = -1
      value = -1.
      do 10 i = 1,n
         if ( a(i) .gt. t ) then
            index = i
            value = a(i)
            goto 20
         endif
   10 continue
   20 continue
      chksum = value + float(index)
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,chksum)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = value + float(index)
      call check (chksum,ntimes*n,n,t2,'s332 ')
      return
      end
c%3.4
      subroutine s341 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     packing
c     pack positive values
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s341 ')
      t1 = second()
      do 1 nl = 1,ntimes
      j = 0
      do 10 i = 1,n
         if(b(i) .gt. 0.)then
            j    = j + 1
            a(j) = b(i)
         endif
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s341 ')
      return
      end
c%3.4
      subroutine s342 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     packing
c     unpacking
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s342 ')
      t1 = second()
      do 1 nl = 1,ntimes
      j = 0
      do 10 i = 1,n
         if(a(i) .gt. 0.)then
            j    = j + 1
            a(i) = b(j)
         endif
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s342 ')
      return
      end
c%3.4
      subroutine s343 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     packing
c     pack 2-d array into one dimension
c
      integer ntimes, ld, n, i, nl, j, k, nn
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, array
      parameter(nn=1000)
      common /cdata/ array(nn*nn)

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s343 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      k = 0
      do 10 i = 1,n
         do 20 j= 1,n
            if (bb(i,j) .gt. 0) then
               k = k + 1
               array(k) = aa(i,j)
            endif
   20    continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs1d(n*n,array)
      call check (chksum,(ntimes/n)*n*n,n,t2,'s343 ')
      return
      end
c%3.5
      subroutine s351 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop rerolling
c     unrolled saxpy
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, alpha, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s351 ')
      t1 = second()
      alpha = c(1)
      do 1 nl = 1,5*ntimes
      do 10 i = 1,n,5
        a(i)   = a(i)   + alpha * b(i)
        a(i+1) = a(i+1) + alpha * b(i+1)
        a(i+2) = a(i+2) + alpha * b(i+2)
        a(i+3) = a(i+3) + alpha * b(i+3)
        a(i+4) = a(i+4) + alpha * b(i+4)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(5*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,5*ntimes*(n/5),n,t2,'s351 ')
      return
      end
c%3.5
      subroutine s352 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop rerolling
c     unrolled dot product
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, dot

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s352 ')
      t1 = second()
      do 1 nl = 1,5*ntimes
      dot = 0.
      do 10 i = 1, n, 5
         dot = dot +   a(i)*b(i)   + a(i+1)*b(i+1) + a(i+2)*b(i+2)
     +             + a(i+3)*b(i+3) + a(i+4)*b(i+4)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,dot)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(5*ntimes) )
      chksum = dot
      call check (chksum,5*ntimes*(n/5),n,t2,'s352 ')
      return
      end
c%3.5
      subroutine s353 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip)
c
c     loop rerolling
c     unrolled sparse saxpy
c
      integer ntimes, ld, n, i, nl, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, alpha, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s353 ')
      t1 = second()
      alpha = c(1)
      do 1 nl = 1,5*ntimes
      do 10 i = 1,n,5
        a(i)   = a(i)   + alpha * b(ip(i))
        a(i+1) = a(i+1) + alpha * b(ip(i+1))
        a(i+2) = a(i+2) + alpha * b(ip(i+2))
        a(i+3) = a(i+3) + alpha * b(ip(i+3))
        a(i+4) = a(i+4) + alpha * b(ip(i+4))
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(5*ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,5*ntimes*(n/5),n,t2,'s353 ')
      return
      end
c
c**********************************************************
c                                                         *
c                 LANGUAGE COMPLETENESS                   *
c                                                         *
c**********************************************************
c%4.1
      subroutine s411 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop recognition
c     if loop to do loop, zero trip
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s411 ')
      t1 = second()
      do 1 nl = 1,ntimes
      i = 0
  10  continue
      i = i + 1
      if (i.gt.n) goto 20
      a(i) = a(i) + b(i) * c(i)
      goto 10
  20  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s411 ')
      return
      end
c%4.1
      subroutine s412 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,inc)
c
c     loop recognition
c     if loop with variable increment
c
      integer ntimes, ld, n, i, nl, inc
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s412 ')
      t1 = second()
      do 1 nl = 1,ntimes
      i = 0
  10  continue
      i = i + inc
      if(i .gt. n)goto 20
         a(i) = a(i) + b(i) * c(i)
      goto 10
  20  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s412 ')
      return
      end
c%4.1
      subroutine s413 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop recognition
c     if loop to do loop, code on both sides of increment
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s413 ')
      t1 = second()
      do 1 nl = 1,ntimes
      i = 1
  10  continue
      if(i .ge. n)goto 20
         b(i) = b(i) + d(i) * e(i)
         i    = i + 1
         a(i) = c(i) + d(i) * e(i)
      goto 10
  20  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a) + cs1d(n,b)
      call check (chksum,ntimes*(n-1),n,t2,'s413 ')
      return
      end
c%4.1
      subroutine s414 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop recognition
c     if loop to do loop, interchanging with do necessary
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s414 ')
      t1 = second()
      do 1 nl = 1,ntimes/n
      i = 1
  10  if(i .gt. n) goto 20
      do 30 j = 2,n
         aa(i,j) = aa(i,j-1) + bb(i,j-1) * cc(i,j-1)
   30 continue
      i = i + 1
      goto 10
  20  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
      chksum = cs2d(n,aa) + cs2d(n,bb) + cs2d(n,cc)
      call check (chksum,(ntimes/n)*n*(n-2),n,t2,'s414 ')
      return
      end
c%4.1
      subroutine s415 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop recognition
c     while loop
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s415 ')
      t1 = second()
      do 1 nl = 1,ntimes
      i = 0
10    continue
      i = i + 1
      if ( a(i) .lt. 0. ) goto 20
         a(i) = a(i) + b(i) * c(i)
      goto 10
20    continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s415 ')
      return
      end
c%4.2
      subroutine s421 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     storage classes and equivalencing
c     equivalence- no overlap
c
      integer ntimes, ld, n, i, nl, nn
      parameter(nn=1000)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d
      real x(nn),y(nn)
      equivalence (x(1),y(1))

      call set1d(n,x,0.0,1)
      call set1d(n,y,1.0,1)
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s421 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
         x(i) = y(i+1) + a(i)
  10  continue
      call dummy(ld,n,x,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,x)
      call check (chksum,ntimes*(n-1),n,t2,'s421 ')
      return
      end
c%4.2
      subroutine s422 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     storage classes and equivalencing
c     common and equivalence statement
c     anti-dependence, threshold of 4
c
      integer ntimes, ld, n, i, nl, nn, vl
      parameter(nn=1000,vl=64)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      common /cdata/ array(1000000)
      real x(nn), array
      equivalence (x(1),array(5))
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call set1d(n,x,0.0,1)
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s422 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         x(i) = array(i+8) + a(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,x)
      call check (chksum,ntimes*(n-8),n,t2,'s422 ')
      return
      end
c%4.2
      subroutine s423 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     storage classes and equivalencing
c     common and equivalenced variables - with anti-dependence
c
      integer ntimes, ld, n, i, nl, nn, vl
      parameter(nn=1000,vl=64)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real x(nn), array
      common /cdata/ array(1000000)
      equivalence (array(vl),x(1))
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call set1d(n,x,1.0,1)
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s423 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1, n-1
         array(i+1) = x(i) + a(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,array)
      call check (chksum,ntimes*(n-1),n,t2,'s423 ')
      return
      end
c%4.2
      subroutine s424 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     storage classes and equivalencing
c     common and equivalenced variables - overlap
c     vectorizeable in strips of 64 or less
c
      integer ntimes, ld, n, i, nl, nn, vl
      parameter(nn=1000,vl=64)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real x(nn), array
      real t1, t2, second, chksum, ctime, dtime, cs1d
      common /cdata/ array(1000000)
      equivalence (array(vl),x(1))

      call set1d(n,x,0.0,1)
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s424 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n-1
         x(i+1) = array(i) + a(i)
   10 continue
      call dummy(ld,n,x,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,x)
      call check (chksum,ntimes*(n-1),n,t2,'s424 ')
      return
      end
c%4.3
      subroutine s431 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     parameters
c     parameter statement
c
      integer ntimes, ld, n, i, nl, k, k1, k2
      parameter(k1=1, k2=2, k=2*k1-k2)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s431 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i+k) + b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s431 ')
      return
      end
c%4.3
      subroutine s432 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     parameters
c     data statement
c
      integer ntimes, ld, n, i, nl, k, k1, k2
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d
      data k1,k2 /1,2/

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s432 ')
      k=2*k1-k2
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i+k) + b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s432 ')
      return
      end
c%4.4
      subroutine s441 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     non-logical if's
c     arithmetic if
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s441 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1, n
         if (d(i)) 20,30,40
   20    a(i) = a(i) + b(i) * c(i)
         goto 50
   30    a(i) = a(i) + b(i) * b(i)
         goto 50
   40    a(i) = a(i) + c(i) * c(i)
   50 continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s441 ')
      return
      end
c%4.4
      subroutine s442 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,indx)
c
c     non-logical if's
c     computed goto
c
      integer ntimes, ld, n, i, nl, indx(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s442 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1, n
         goto (15,20,30,40) indx(i)
   15    a(i) = a(i) + b(i) * b(i)
         goto 50
   20    a(i) = a(i) + c(i) * c(i)
         goto 50
   30    a(i) = a(i) + d(i) * d(i)
         goto 50
   40    a(i) = a(i) + e(i) * e(i)
   50 continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s442 ')
      return
      end
c%4.4
      subroutine s443 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     non-logical if's
c     arithmetic if
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s443 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1, n
         if (d(i)) 20,20,30
   20    a(i) = a(i) + b(i) * c(i)
         goto 50
   30    a(i) = a(i) + b(i) * b(i)
   50 continue
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s443 ')
      return
      end
c%4.5
      subroutine s451 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     intrinsic functions
c     intrinsics
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s451 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = sin(b(i)) + cos(c(i))
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s451 ')
      return
      end
c%4.5
      subroutine s452 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     intrinsic functions
c     seq function
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s452 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1, n
         a(i) = b(i) + c(i) * float(i)
 10   continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
 1    continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s452 ')
      return
      end
c%4.5
      subroutine s453 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     intrinsic functions
c     seq function
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, s

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s453 ')
      t1 = second()
      do 1 nl = 1,ntimes
      s = 0.
      do 10 i = 1, n
         s    = s + 2.
         a(i) = s * b(i)
 10   continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
 1    continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s453 ')
      return
      end
c%4.7
      subroutine s471 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,c471s)
c
c     call statements
c
      integer ntimes, ld, n, i, nl, nn, m
      parameter(nn=1000)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, c471s, x(nn)

      m = n
      call set1d(n,x,0.0,1)
      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s471 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,m
         x(i) = b(i) + d(i) * d(i)
         call s471s
         b(i) = c(i) + d(i) * e(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) ) -
     +     ( (n*ntimes) * c471s )
      chksum = cs1d(n,x) + cs1d(n,b)
      call check (chksum,ntimes*n,n,t2,'s471 ')
      return
      end
c%4.8
      subroutine s481 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     non-local goto's
c     stop statement
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s481 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1, n
         if (d(i) .lt. 0.) stop 'stop 1'
         a(i) = a(i) + b(i) * c(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s481 ')
      return
      end
c%4.8
      subroutine s482 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     non-local goto's
c     other loop exit with code before exit
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s482 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i) + b(i) * c(i)
         if(c(i) .gt. b(i))goto 20
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s482 ')
  20  continue
      return
      end
c%4.9
      subroutine s491(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip)
c
c     vector semantics
c     indirect addressing on lhs, store in sequence
c
      integer ntimes, ld, n, i, nl, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s491 ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(ip(i)) = b(i) + c(i) * d(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s491 ')
      return
      end
c%4.11
      subroutine s4112(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip,s)
c
c     indirect addressing
c     sparse saxpy
c
      integer ntimes, ld, n, i, nl, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, s

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s4112')
      t1 = second()
      do 1 nl = 1,ntimes 
      do 10 i = 1, n
          a(i) = a(i) + b(ip(i)) * s
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s4112')
      return
      end
c%4.11
      subroutine s4113(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip)
c
c     indirect addressing
c     indirect addressing on rhs and lhs
c
      integer ntimes, ld, n, i, nl, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s4113')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(ip(i)) = b(ip(i)) + c(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s4113')
      return
      end
c%4.11
      subroutine s4114(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip,n1)
c
c     indirect addressing
c     mix indirect addressing with variable lower and upper bounds
c
      integer ntimes, ld, n, i, nl, k, n1, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s4114')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i=n1, n
         k = ip(i)
         a(i) = b(i) + c(n-k+1) * d(i)
         k = k + 5
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s4114')
      return
      end
c%4.11
      subroutine s4115(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip)
c
c     indirect addressing
c     sparse dot product
c
      integer ntimes, ld, n, i, nl, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, sum

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s4115')
      t1 = second()
      do 1 nl = 1,ntimes
      sum = 0.
      do 10 i = 1, n
         sum = sum + a(i) * b(ip(i))
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = sum
      call check (chksum,ntimes*n,n,t2,'s4115')
      return
      end
c%4.11
      subroutine s4116(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip,
     +                 j,inc)
c
c     indirect addressing
c     more complicated sparse sdot
c
      integer ntimes, ld, n, i, nl, j, off, inc, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, sum

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s4116')
      t1 = second()
      do 1 nl = 1,ntimes
      sum = 0.
      do 10 i = 1, n-1
         off = inc + i
         sum = sum + a(off) * aa(ip(i),j)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = sum
      call check (chksum,ntimes*(n-1),n,t2,'s4116')
      return
      end
c%4.11
      subroutine s4117(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     indirect addressing
c     seq function
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s4117')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 2,n
         a(i) = b(i) + c(i/2) * d(i)
 10   continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
 1    continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*(n-1),n,t2,'s4117')
      return
      end
c%4.12
      subroutine s4121 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     statement functions
c     elementwise multiplication
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, f, x, y
      f(x,y) = x*y

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s4121')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i) + f(b(i),c(i))
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'s4121')
      return
      end
c%5.1
      subroutine va(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector assignment
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'va   ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'va   ')
      return
      end
c%5.1
      subroutine vag(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip)
c
c     control loops
c     vector assignment, gather
c
      integer ntimes, ld, n, i, nl, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vag  ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = b(ip(i))
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vag  ')
      return
      end
c%5.1
      subroutine vas(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,ip)
c
c     control loops
c     vector assignment, scatter
c
      integer ntimes, ld, n, i, nl, ip(n)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vas  ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(ip(i)) = b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vas  ')
      return
      end
c%5.1
      subroutine vif(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector if 
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vif  ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         if (b(i) .gt. 0.) then
            a(i) = b(i)
         endif
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vif  ')
      return
      end

c%5.1
      subroutine vpv(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector plus vector
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vpv  ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i) + b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vpv  ')
      return
      end
c%5.1
      subroutine vtv(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector times vector
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vtv  ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i) * b(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vtv  ')
      return
      end
c%5.1
      subroutine vpvtv(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector plus vector times vector
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vpvtv')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i) + b(i) * c(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vpvtv')
      return
      end
c%5.1
      subroutine vpvts(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc,s)
c
c     control loops
c     vector plus vector times scalar
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d, s

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vpvts')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i) + b(i) * s
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vpvts')
      return
      end
c%5.1
      subroutine vpvpv(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector plus vector plus vector
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vpvpv')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i) + b(i) + c(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vpvpv')
      return
      end
c%5.1
      subroutine vtvtv(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector times vector times vector
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vtvtv')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a(i) = a(i) * b(i) * c(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,a)
      call check (chksum,ntimes*n,n,t2,'vtvtv')
      return
      end
c%5.1
      subroutine vsumr(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector sum reduction
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, sum

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vsumr')
      t1 = second()
      do 1 nl = 1,ntimes
      sum = 0.
      do 10 i = 1,n
         sum = sum + a(i)
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,sum)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = sum
      call check (chksum,ntimes*n,n,t2,'vsumr')
      return
      end
c%5.1
      subroutine vdotr(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     vector dot product reduction
c
      integer ntimes, ld, n, i, nl
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, dot

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vdotr')
      t1 = second()
      do 1 nl = 1,ntimes
      dot = 0.
      do 10 i = 1,n
         dot = dot + a(i) * b(i)
   10 continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,dot)
   1  continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = dot
      call check (chksum,ntimes*n,n,t2,'vdotr')
      return
      end
c%5.1
      subroutine vbor(ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     control loops
c     basic operations rates, isolate arithmetic from memory traffic
c     all combinations of three, 59 flops for 6 loads and 1 store. 
c
      integer ntimes, ld, n, i, nl, nn
      parameter(nn=1000)
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d
      real a1, b1, c1, d1, e1, f1, s(nn)

      call init(ld,n,a,b,c,d,e,aa,bb,cc,'vbor ')
      t1 = second()
      do 1 nl = 1,ntimes
      do 10 i = 1,n
         a1 = a(i)
         b1 = b(i)
         c1 = c(i)
         d1 = d(i)
         e1 = e(i)
         f1 = aa(i,1)
         a1   = a1*b1*c1 + a1*b1*d1 + a1*b1*e1 + a1*b1*f1
     +                   + a1*c1*d1 + a1*c1*e1 + a1*c1*f1
     +                              + a1*d1*e1 + a1*d1*f1
     +                                         + a1*e1*f1
         b1   = b1*c1*d1 + b1*c1*e1 + b1*c1*f1
     +                   + b1*d1*e1 + b1*d1*f1
     +                              + b1*e1*f1
         c1   = c1*d1*e1 + c1*d1*f1
     +                   + c1*e1*f1
         d1   = d1*e1*f1
         s(i) = a1 * b1 * c1 * d1
  10  continue
      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
  1   continue
      t2 = second() - t1 - ctime - ( dtime * float(ntimes) )
      chksum = cs1d(n,s)
      call check (chksum,ntimes*n,n,t2,'vbor ')
      return
      end
