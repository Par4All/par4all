      subroutine andne(kpress, lmapp, leig, lpress)

      kpress = 1
      lmapp = 1

      if (kpress.gt.1.and.lmapp.ne.1) then        
         leig = lpress-nweig                      
      endif

      print *, kpress, lmapp, leig, lpress

      end
