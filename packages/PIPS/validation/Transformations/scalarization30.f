C     Scalar replacement in the presence of conditional control flow
C
C     Steve Carr and Ken Kennedy
C
C     Software - Practive and Experience, Vol. 24, No. 1, pp. 51-77, 1994
C
C     First example, which turns out to be wrong if the program
C     behaviors must be respected

      subroutine scalarization30(a,b,m,n)
      real a(n), b(m)

C     a(i) is not scalarized because m<1 is a guard for it. Adding s =
C     a(i) before do j = ... may generate out of bounds accesses that do
C     not exist in the initial code
C
C     Except if declarations can be trusted
      do i = 1,n
         do j = 1,m
            a(i) = a(i) + b(j)
         enddo
      enddo

C     Let's make sure that the inner loop is executed
      if(m.lt.1) stop

C     The result is copied out for the next loop nest
      do i = 1,n
         do j = 1,m
            a(i) = a(i) + b(j)
         enddo
      enddo

C     The inner loop is certainly executed when the outer loop is entered
      do i = 1,n
         do j = 1,n
            a(i) = a(i) + b(j)
         enddo
C     No copy out because of the interprocedural analysis: no caller here
      enddo

      end

