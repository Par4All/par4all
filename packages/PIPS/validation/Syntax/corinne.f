      program corinne

c     Example used at SC'95

      real a(10,10), b(10,10)

      do i = 1, 10
         b(i,i) = 10.
      enddo

cfirst      do i = 1. 10
csecond      do i = 1, 10
cthird      do i = 1, 10
         do j = i+1, 10
            b(i,j) = 20.
            a(i,j) = b(i-1,j)
         enddo
      enddo

      if(i.gt.n) then
csecond         t(i-1,1) = 0.
cthird        a(i-1,1) = 0.
      endif
      endif

      call doendif(a,10)

      call ifendo

      end

      subroutine doendif(a,n)
      real a(n,n)

      do i = 1, n
         a(i,i) = 0.
      endif

      end

      subroutine ifenddo(a,n)
      real a(n,n)

      if(a(n,n).gt.0.) then
         a(n,n) = 0.
      enddo

      end
