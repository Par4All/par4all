      program zerotrip

c     The loop body should not disturb the postcondition

      real t(10)

      if(n.ge.1) then

         k = 3
         do i = 1, n
            k = 2*k
            t(i) = 0
         enddo
         print *, k, i

      endif

      if(n.ge.0) then

         k = 3
         do i = 1, n
            k = 2*k
            t(i) = 0
         enddo
         print *, k, i

      endif 

      if(n.lt.0) then

         k = 3
         do i = 1, n
            k = 2*k
            t(i) = 0
         enddo
         print *, k, i

      endif 

      end
