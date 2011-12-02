from pyps import workspace

with workspace("duplicate.c","duplicate/duplicate.c", preprocessor_file_name_conflict_handling=True) as w:
    print ":".join([f.name for f in w.fun])

