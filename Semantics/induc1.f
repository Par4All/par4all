      subroutine induc1(n)
      
      j = 0
      
      do 100 i = 1, n
         j = j + 2
 100  continue

c     The precondition here about i and j, j==2i-2, is wrong because i
c     was incremented one more time
c
c     No, it is wrong because you cannot prove that Loop 100
c     is executed at least once! And anyway, i was incremented
c     exactly as it should have! (FI, 04 June 1997)
c
c     We should end up with j >= 0 and i >= n + 1 using the convex
c     hull of 0 trip and n trips.
c
c     No! If the loop is not entered, you end up with i == 1 and j == 0
c     If the loop is entered, you end up with j==2i-2... which also
c     stand for the previous case! For the same kind of reason, index I
c     must end up greater than N+1.

c     Expected precondition:
C     P(I,J) {J==2I-2, N+1<=I}

      print *, i, j
      
      end
