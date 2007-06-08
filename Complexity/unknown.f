      program unknown

c     See how unknown variable names are generated
c     for DO/ENDDO loops without a label:more than
c     one unknown upper bound should generated!

      real t(100), u(100,100,100)

      do i = 1, l
         t(i) = 0.
      enddo

      do j = 1, m
         t(j) = 0.
      enddo

      do k = n, 10
         t(k) = 0.
      enddo

      do l = n, n+10
         t(l) = 0.
      enddo

      do 100 i = 1, n
         do 100 j = 1, m
            do 100 k = 1, l
               u(i,j,k) = 0.
 100        continue

      end
