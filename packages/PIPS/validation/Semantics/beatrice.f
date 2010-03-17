      subroutine beatrice(i)

C     Bug found in paper about interprocedural analyses for Zapata

      if(i.le.10) then
         i = 10
      endif

      end

      subroutine beatrice2(i,x)

C     Bug found in paper about interprocedural analyses for Zapata
C     First variation

      if(x.gt.0.) then
         i = i+1
      endif

      end

      subroutine beatrice3(i, j, x)

C     Bug found in paper about interprocedural analyses for Zapata
C     Second variation

      if(x.gt.0.) then
         i = i + 1
      else
         j = j + 1
      endif

      end
