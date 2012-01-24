      subroutine gewpp(a,n)
      real a(n,n)
      
C     Gaussian elimination with partial pivoting (courtesy of Rob Schreiber)
      
      do k = 1, n-1
c     find pivot
         imax = k
         do i = k+1, n
            if (abs(a(i,k)) .gt. abs(a(imax,k))) then
               imax = i
            endif
         enddo
c     exchange elements and compute multipliers
         t = a(imax,k)
         a(imax, k) = a(k,k)
         a(k,k) = t
         do i = k+1, n
            a(i,k) = a(i,k) / a(k,k)
         enddo
c     a(i,j)  -= a(i,k) * a(k,j) for i, j > k
         do j = k+1, n
            t = a(imax, j)
            a(imax, j) = a(k,j)
            a(k,j) = t
            do i = k+1, n
               a(i,j) = a(i,j) - a(i,k) * a(k,j)
            enddo
         enddo
      enddo
      
      return
      end
