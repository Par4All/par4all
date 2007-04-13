C     how is STOP handled? The proper transformer increments call

      subroutine testerror
      common /error/ nerror
      save ncall

      if(nerror.gt.0) then
         print *, nerror
         stop
      else
         ncall = ncall + 1
      endif

      end
