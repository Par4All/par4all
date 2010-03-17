      program test1

      integer i,j,NC,JC
C      parameter (NC=100)
      real  ZG,V(100),W(100),G(100,100),GAMAI(100)

      NC=50
      JC = 30
      do 320 i = 1,NC
        V(i) = 0.
        W(i) = 0.
        if (I.GT.10) then
        else
           do j = 1,jc
              W(i) = W(i) / V(i)
              GAMAI(i) = W(i) +G(i,j)
           enddo
        endif
 320  continue
      
      end
      
