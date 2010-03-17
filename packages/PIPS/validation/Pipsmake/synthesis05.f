      program synthesis05

C     Bug: wrong typing for a parameter. PIPS sees lload as an OVERLOADED...

      integer lload
      parameter  (lload  = 1 )
 
      call  hidenh ( lhiden, lload, 9999)

      end
