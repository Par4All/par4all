      program onetriploop02

      read *, n

      
      m = n

      m = n + 2

      if (n.gt.0) then
         do i = n, 2*n-1, n
            print *, 'i = ', i
         enddo
      endif

      if (n.lt.0) then
         do i = n, 2*n+1, n
            print *, 'i = ', i
         enddo
      endif

      end

