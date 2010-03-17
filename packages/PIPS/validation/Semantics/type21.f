      program type21

C     Variable lengths must be taken into account

      character*3 s1
      character*5 s2

C     Cannot be expressed as a transformer because a subtring operator
C     is implictly applied
      s1 = s2

C     Can be expressed as a transformer because a padding SPACEs
C     are implictly added and are not considered in string comparisons
      s2 = s1

      end
