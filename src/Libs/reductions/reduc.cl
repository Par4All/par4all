;;;; Detection of reductions
;;;;
;;;; Cf. A Unified Semantic Approach for the Vectorization and Parallelization
;;;; of Generalized Reductions} (with B. Dehbonei), 1989 ACM SIGARCH Int. 
;;;; Conf. on Supercomputing, Crete, Jun.89

(require :init)
(require :ri)
(require :reduction)
(require :util)
(require :eval)
(require :match)

(use-package '(:newgen :ri :reduction))

;;; Detection of reductions

;; Watch out for embedded loops and side-effects !!!
;;
(defun detect-reduction (statement)
  (gen-recurse
   (statement-instruction statement)
   ((instruction tag) 
    (unless (instruction-block-p instruction)
      (funcall tag instruction))
    :instruction)
   ((loop body)
     #'(lambda (instruction)
	 (when (simple-loop-p loop)
	   (let ((reds (detect-reduction-loop loop)))
	     (unless (endp reds)
	       (prune (loop-body loop) (mapcar #'car reds))
	       (setf (instruction-tag instruction) 
		 is-instruction-block)
	       (setf (instruction-block instruction) 
		 `(,(make-statement-loop loop)
		   ,@(mapcar #'cdr reds))))))
	 :loop))
   ((call) #'identity)
   ((test true false) #'identity)
   ((unstructured)
    #'(lambda (ignore)
	(let ((blocs '()))
	  (control-map #'(lambda (c)
			   (detect-reduction (control-statement c)))
		       (unstructured-control unstructured)
		       blocs)
	  :unstructured)))
   ((control statement) :statement)
   ((statement instruction) instruction)))

;;; Loop management

(defun detect-reduction-loop (loop)
    (let* ((body (loop-body loop))
	   (vars (remove-if-not #'scalar-entity-p
				(statement-written-entities body)))
	   (s (eval-statement body (init-store vars)))                   
	   (v-reds 
	    (mapcan
	     #'(lambda (var)
		   (let ((inst (create-instance var (get-store s var))))
		       (if inst
			       (let ((template (match-pattern
						inst
						(loop-index loop))))
				   (if (and template
					    (not (reduction-used-p var body)))
					   `((,var .
					      ,(reduction-code var 
							       template
							       loop)))
				       '()))
			   '())))
	     vars)))
	(unless (endp v-reds)
	    (format *debug-io* "~%Reductions detected")
	    (print-store s (mapcar #'car v-reds)))
	v-reds))	

(defun simple-loop-p (loop)
    (let ((range (loop-range loop)))
	(every #'pure-expression-p
	       `(,(range-lower range)
		 ,(range-upper range)
		 ,(range-increment range)))))

(defun reduction-used-p (var body) 
    (flet ((check-rw 
	    (read written)
	    (and (member var read :test #'entity-equal-p)
		 (not (member var written :test #'entity-equal-p)))))
	(gen-recurse
	 body
	 ((instruction tag)
	  #'(lambda (statement)
		(if (instruction-block-p instruction)
			(some #'identity tag)
		    (funcall tag statement))))
	 ((statement instruction) (funcall instruction statement))
	 ((call) 
	  #'(lambda (statement)
		(check-rw (statement-read-entities statement)
			  (statement-written-entities statement))))
	 ((test true false) 
	  #'(lambda (statement)
		(or true false)))
	 ((loop body) 
	  #'(lambda (statement)
		(let ((range (loop-range loop)))
		    (or (check-rw
			 `(,@(free-vars (range-lower range))
			     ,@(free-vars (range-upper range))
			     ,@(free-vars (range-increment range)))
			 '())
			body)))))))

;;; Pruning of statements that refer to the reduction variables

(defun prune (statement vars)
    (flet ((recurse (stat)
		    (prune-statement stat vars)))
	(funcall
	 (gen-recurse
	  (statement-instruction statement)
	  ((instruction tag)
	   #'(lambda (statement)
		 (unless (instruction-block-p instruction)
		     (funcall tag statement))))
	  ((statement instruction) 
	   (funcall instruction statement))
	  ((call) #'recurse)
	  ((test) #'recurse)
	  ((loop) #'recurse))
	 statement)))
	    
(defun prune-statement (statement vars)
    (let ((instruction (statement-instruction statement)))
	(when (some 
	       #'(lambda (f)
		     (some
		      #'(lambda (var)
			    (entity-equal-p var
					    (effect-reference-variable f)))
		      vars))
	       (statement-cumulated-effects statement))
	    (format *debug-io* "~%Deleting statement ~D"
		    (statement-number statement))
	    (setf (instruction-tag instruction) is-instruction-block)
	    (setf (instruction-block instruction) '()))))
		  
(defun reduction-code (var temp loop)
    (let ((pattern (template-pattern temp)))
	(funcall (template-generator temp)
		 var 
		 (pattern-condition pattern)
		 (template-value temp)
		 (pattern-operator pattern)
		 (pattern-indices pattern)
		 loop)))

;;;

(provide :reduc)
