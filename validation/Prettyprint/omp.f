      program p
      real a(10,10,10,10,10)

      i = 0

      if(i.gt.0) then
         print *, 'RETURN 1 in FOO'
         i = i + 1
      endif

      i = i + 1

      if(i.gt.0) then
         i = i + 1
      endif

      do m=2,5
         do l=5,7
            do k=1, 2                      
               do j=1, 2
                  do i=1, 10
                     x = 2.3
C dsgsgsgs
C frfrfr
                     y=                 5.3 + x
C frfrfr
C frfrfr
C frfrrf
                     z=2.32 + y

                     w=1.2 + z
                     a(i,j,k,l,m) = x*x+x+z
                  enddo
               enddo
            enddo
         enddo
      enddo

      if(w.gt.0) then
         print *, 'RETURN 1 in FOO'
      endif

C troto   
      i= i + 1

      end

      subroutine foo()
C      print *, "foo is entered with ", i
C      if(i.gt.0) then
C         print *, 'RETURN 1 in FOO'
C      endif
      i = i + 1
      end
