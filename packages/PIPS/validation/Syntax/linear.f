C     a functional parameter is used. See the calls to relax in the main.
c------------------------------------------------------------------------------
      subroutine init
      parameter (l=2, m=3, n=4)
c
      real*8 a(l,m), b(m,n)
c
      common /d/ a, b
c
      real*8 drand
c
      do i = 1, l
         do j = 1, m
            a(i,j) = srand(0)
         enddo
      enddo
c
      do i = 1, m
         do j = 1, n
            b(i,j) = drand(0)
         enddo
      enddo
c
      return
      end
c------------------------------------------------------------------------------
      program linear
c
      parameter (l=2, m=3, n=4)
      parameter (niter=100)
c
      real*8 a(l,m), b(m,n)
      real*8 c(l,n)
c
      common /d/ a, b
      common /r/ c
c
      real*8 mean1, mean2
      external mean1, mean2
c
      call init
c
      call prmat(a, l, m)
      call prmat(b, m, n)
c
      call matmul(a, b, c, l, m, n)
c
      call prmat(c, l, n)
c
      call relax(c, l, n, niter, mean1)
      call relax(c, l, n, niter, mean2)
c
      call prmat(c, l, n)
c
      stop 'end of program'
      end
c------------------------------------------------------------------------------
      subroutine matmul(a, b, c, l, m, n)
c
      integer l, m, n
      real*8 a(l,m), b(m,n)
      real*8 c(l,n)
      real*8 t
c
      do i = 1, l
         do j = 1, n
            t = 0.0
            do k = 1, m
               t = t + a(i,k)*b(k,j)
            enddo
            c(i,j) = t
         enddo
      enddo
c
      return
      end

c------------------------------------------------------------------------------
      real*8 function mean1(q1, q2)
c
      real*8 q1, q2
c
      mean1 = 0.5*(q1+q2)
c
      return
      end

c------------------------------------------------------------------------------
      real*8 function mean2(q1, q2)
c
      real*8 q1, q2
c
      mean2 = 0.75*q1 + .25*q2
c
      return
      end

c------------------------------------------------------------------------------
      subroutine prmat(mat, d1, d2)
c
      real*8 mat(d1, d2)
      integer d1, d2
c
      do i = 1, d1
         do j = 1, d2
            write (6, 1000) i, j, mat(i,j)
         enddo
      enddo
c
 1000 format('i = ', i3, '   j = ', i3, '   mat(i,j) = ', e18.7)
c
      return
      end

c------------------------------------------------------------------------------
      subroutine relax(mat, d1, d2, n, f)
c
      integer d1, d2
      real*8 mat(d1, d2)
      real*8 f
c
      if (n .le. 0) stop 'nombre d iterations incorrect'
c
      do iter = 1, n
         do i = 2, d1-1
            do j = 2, d2-1
               q = (mat(i-1,j)+mat(i+1,j)+mat(i,j-1)+mat(i,j+1))/4.0
               mat(i,j) = f(mat(i,j), q)
            enddo
         enddo
      enddo
c
      return
      end
