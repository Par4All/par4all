      program seq

c     Goal: show different dependence tests and privatization

      parameter (n=10)
      parameter (lx=32, mx=92, nx=42)

      real a(n,n), b(n,n), c(n,n,2)

      real d(lx, mx, nx)

      call init
      call testinit
      call digitalize(d, lx, mx, nx)

      end

      subroutine digitalize(x, lx, mx, nx)
      real x(mx,nx,lx)

      common /c2/ m,mm1,mp1

!!     i = m

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
