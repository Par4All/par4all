C     To debug sc_bounded_normalization

      program bounded06
      real a, b

      if (a.gt.0.0.and.b.gt.0.0) then
         if (a+b.gt.0.0) then
            print *, 'test redondant a+b>0'
         endif
      endif

      end
