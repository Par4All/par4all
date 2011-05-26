!     Induction variable j not found because of non-affine upper boud of
!     loop 10

c%1.2
      subroutine s124 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     induction variable recognition
c     induction variable under both sides of if (same value)
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs1d

!      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s124 ')
!      t1 = second()
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
!      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
!      t2 = second() - t1 - ctime - ( dtime * float(2*ntimes) )
!      chksum = cs1d(n,a)
!      call check (chksum,2*ntimes*(n/2),n,t2,'s124 ')
      return
      end
