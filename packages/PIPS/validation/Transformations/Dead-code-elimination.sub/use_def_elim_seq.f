      program use_def_elim_seq

c     Goal: test the consistency of the
c     ordering_to_statement tables.

      parameter (n=10)
      parameter (lx=32, mx=92, nx=42)

      real a(n,n), b(n,n), c(n,n,2)

      real d(lx, mx, nx)

      read *, a

      call matmul(b, a, a, n, n, n)

      call smatmul(b, a, a, n, n, n)

      call transpose(b, n)

      call flipflop(c, n, 0.25)

      call init
      call testinit
      call digitalize(d, lx, mx, nx)

      end

      subroutine matmul(x, y, z, l, m, n)
c     compute x := y * z
      real x(l,n), y(l,m), z(m,n)

      do i = 1, l
         do j = 1, n
            x(i,j) = 0.
            do k = 1, m
               x(i,j) = x(i,j) + y(i,k)*z(k,j)
            enddo
         enddo
      enddo

      end

      subroutine smatmul(x, y, z, l, m, n)
c     compute x := y * z
      real x(l,n), y(l,m), z(m,n)

      do i = 1, l
         do j = 1, n
            s = 0.
            do k = 1, m
               s = s + y(i,k)*z(k,j)
            enddo
            x(i,j) = s
         enddo
      enddo

      end

      subroutine transpose(x,n)
      real x(n,n)

      do i = 1, n-1
         do j = i+1, n
            t = x(i,j)
            x(i,j) = x(j,i)
            x(j,i) = t
         enddo
      enddo

      end

      subroutine flipflop(x,n,c)
      integer n
      real x(n,n,2), c

      integer i, j, new, old, t

      new=1

 10   continue
        old=new
        new=3-old

        do j=2, n-1
           do i=2, n-1
              x(i,j,new) = c*x(i,j,old) + (1-c)*
     $         (x(i+1,j,old)+x(i-1,j,old)+x(i,j-1,old)+x(i,j+1,old))
           enddo
        enddo

      if (abs(x(5,5,new)-x(5,5,old)).gt.0.01) goto 10

      end

      subroutine digitalize(x, lx, mx, nx)
      real x(mx,nx,lx)

      common /c2/ m,mm1,mp1

      ms=mm1/2
      mm=mp1/2

      do k=2,n-1
         do j=2,mm
            jj=j+ms
            x(j,k,1)  = x(jj,k,2)
            x(jj,k,1) = x(j,k,2)
         enddo
      enddo

      end

      subroutine testinit

      parameter (lx=32, mx=92, nx=42)

      common /c1/ px,dfi

      common /c2/ m,mm1,mp1
c
c input data consistency checking
c
      m=nint(px/dfi+1.)
      mm1=m-1
      mp1=m+1
      if(.not.( m .gt. 1 .and. m .le. mx) ) then
         write(6,*) l,m,n,lcry
         stop 3
      endif
      end

      subroutine init

      common /c1/ px,dfi

      read(5,*) px,dfi

      end
