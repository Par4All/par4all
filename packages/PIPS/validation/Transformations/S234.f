!     Loop recovery fails. Might be due to semantics not providing the
!     necessary preconditions

c%2.3
      subroutine s234 (ntimes,ld,n,ctime,dtime,a,b,c,d,e,aa,bb,cc)
c
c     loop interchange
c     if loop to do loop, interchanging with if loop necessary
c
      integer ntimes, ld, n, i, nl, j
      real a(n), b(n), c(n), d(n), e(n), aa(ld,n), bb(ld,n), cc(ld,n)
      real t1, t2, second, chksum, ctime, dtime, cs2d

!      call init(ld,n,a,b,c,d,e,aa,bb,cc,'s234 ')
!      t1 = second()
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
!      call dummy(ld,n,a,b,c,d,e,aa,bb,cc,1.)
   1  continue
!      t2 = second() - t1 - ctime - ( dtime * float(ntimes/n) )
!      chksum = cs2d(n,aa)
!      call check (chksum,(ntimes/n)*n*(n-1),n,t2,'s234 ')
      return
      end
