      subroutine cst_test2

c     Example taken from ocean (Perfect Club)

      leig = 132097
      if (.false.) then
         leig = lpress-nweig
      endif
      le = 1

      print *, leig

c     First synthetic example

      if(.true.) then
         i = 1
      else
         i = 2
      endif

      if(.false.) then
         i = i + 2
      else
         i = i + 1
      endif

c     Should be I == 2

      print *, i

c     Second synthetic example

      if(.true..and.i.eq.1) then
         i = 1
      else
         i = 2
      endif

      if(.false..or.i.eq.2) then
         i = i + 2
      else
         i = i + 1
      endif

c     Should be I == 4

      print *, i

      end
