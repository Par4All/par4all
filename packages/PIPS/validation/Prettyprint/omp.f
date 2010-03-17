      program p
      real a(10,10,10,10,10)
      do m=2,5
         do l=5,7
            do k=1, 2
               do j=1, 2
                  do i=1, 10
                     x = 2.3
                     y=5.3 + x
                     z=2.32 + y
                     w=1.2 + z
                     a(i,j,k,l,m) = x*x+x+z
                  enddo
               enddo
            enddo
         enddo
      enddo
      end
