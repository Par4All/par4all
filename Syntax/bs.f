      program bs

C     Standard (?) extension found in Zebulon
C     SUn f77 and GNU g77 do not produce code with same behavior
C     Non-ANSI extension only recognized by xlf, which produces
C     code with same behavior as SUN f77

c      print *, 'Hello!'
      print *, '---\'\a\\\n''---'
      end
