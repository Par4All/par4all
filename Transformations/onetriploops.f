      program onetriploops

      integer i, n
      read *, n

      do i = 1, 1
         print *, 'i = ', i
      enddo

      do i = 2, 2
         print *, 'i = ', i
      enddo

      do i = -10, -10
         print *, 'i = ', i
      enddo

      do i = n, n
         print *, 'i = ', i
      enddo

      do i = 1, 1, 1
         print *, 'i = ', i
      enddo

      do i = 1, 1, -1
         print *, 'i = ', i
      enddo

      do i = 2*n+3, 3+n*2
         print *, 'i = ', i
      enddo

      do i = 100*cos(float(n)), 100*cos(float(n))
         print *, 'i = ', i
      enddo
      
! loupe...
      do i = 100*cos(float(n)), cos(float(n))*100
         print *, 'i = ', i
      enddo
      
      m = n

      do i = n, m
         print *, 'i = ', i
      enddo

      m = n + 2

      do i = n, m, 3
         print *, 'i = ', i
      enddo

! pas simplifiable
      do i = n, m+1, 3
         print *, 'i = ', i
      enddo

! pas simplifiable
      do i = n, m, 2
         print *, 'i = ', i
      enddo

      do i = m, n, -3
         print *, 'i = ', i
      enddo

! pas simplifiable
      do i = m, n, -2
         print *, 'i = ', i
      enddo

! pas simplifiable (signe de n...)
      do i = n, 2*n-1, n
         print *, 'i = ', i
      enddo

      if (n.gt.0) then
! pas simplifiable
         do i = n, 2*n+1, n
            print *, 'i = ', i
         enddo
         do i = n, 2*n-1, n
            print *, 'i = ', i
         enddo
      endif

      if (n.lt.0) then
! pas simplifiable
         do i = n, 2*n-1, n
            print *, 'i = ', i
         enddo
         do i = n, 2*n+1, n
            print *, 'i = ', i
         enddo
      endif

      end

