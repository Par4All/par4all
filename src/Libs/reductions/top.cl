;;;; Reduction detection

(in-package :user)

(require :init)
(require :ri)
(require :reduction)

;;; FILES-DIRECTORY has to be defined by the user

(require :util 
	 (pathname (concatenate 'string files-directory "/util")))
(require :simplify
	 (pathname (concatenate 'string files-directory "/simplify")))
(require :eval
	 (pathname (concatenate 'string files-directory "/eval")))
(require :match
	 (pathname (concatenate 'string files-directory "/match")))
(require :reduc
	 (pathname (concatenate 'string files-directory "/reduc")))

(use-package '(:newgen :ri))

(defvar *code*)
(defvar *effects-mapping*)

;;; Test 

(defun entities-file (program)
    (concatenate 'string 
		 program ".database/"
		 program ".ENTITIES"))

(defun db-file (program entity which)
    (concatenate 'string 
		 program ".database/"
		 (entity-local-name entity) which))    
    
(defun code-file (program entity)
    (db-file program entity ".CODE"))

(defun effects-file (program entity)
    (db-file program entity ".CUMULATED_EFFECTS"))

(defun dbget-code (program entity)
    (with-open-file (c (code-file program entity))
	(format *debug-io* "~%Reading ~A's code" 
		(entity-local-name entity))
	(prog1 (gen-read c)
	    (format *debug-io* "~%Reading completed"))))

(defun dbget-effects-mapping (program entity)
    (with-open-file (c (effects-file program entity))
	(format *debug-io* "~%Reading ~A's effects" 
		(entity-local-name entity))
	(let* ((size (read c))
	       (map (make-hash-table :size size)))
	    (dotimes (i size
			(progn (format *debug-io* "~%Reading completed")
			       map))
		     (setf (gethash (read c) map)
			   (gen-read c))))))



(defun reductions (&optional (load-entities-p nil)
			     (program "foo")
			     (module "FOO"))
    (when load-entities-p
	(with-open-file (e (entities-file program))
	    (format *debug-io* "~%Reading ~A's entities" program)
	    (gen-read-tabulated e entity-domain)
	    (format *debug-io* "~%Read done")
	    (load (pathname (concatenate 'string files-directory "/patterns"))
		  :verbose nil)))
    (gen-mapc-tabulated
     #'(lambda (entity)
	   (when (and (string= module (entity-local-name entity))
		      (type-functional-p (entity-type entity))
		      (value-code-p (entity-initial entity)))
	       (setf *code* (dbget-code program entity))
	       (setf *effects-mapping*
		     (dbget-effects-mapping program entity))
	       (detect-reduction *code*)
	       (format *debug-io* "~%Writing new code")
	       (with-open-file (c (code-file program entity)
				  :direction :output
				  :if-exists :supersede)
		   (gen-write c *code*))
	       (format *debug-io* "~%Updating entities' file")
	       (with-open-file (c (entities-file program)
				  :direction :output
				  :if-exists :supersede)
		   (gen-write-tabulated c entity-domain))))
     entity-domain))


