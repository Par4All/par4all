;;
;; $Id$
;;
;; Copyright 1989-2014 MINES ParisTech
;;
;; This file is part of PIPS.
;;
;; PIPS is free software: you can redistribute it and/or modify it
;; under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; any later version.
;;
;; PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
;; WARRANTY; without even the implied warranty of MERCHANTABILITY or
;; FITNESS FOR A PARTICULAR PURPOSE.
;;
;; See the GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with PIPS.  If not, see <http://www.gnu.org/licenses/>.
;;

;;;; epips.el : Emacs mode for (W)PIPS
;;;; Version 1.0
;;;;
;;;; Ronan.Keryell@cri.ensmp.fr

;;; Bug with 19.31:
;;  (make-variable-buffer-local 'x-sensitive-text-pointer-shape)
;;  (setq x-sensitive-text-pointer-shape x-pointer-hand2)
;; Then make-frame -> error in process filter: can't set cursor shape: BadValue (integer parameter out of range for operation)
;;
;; When on sensitive text, if we mask some text with
;; buffer-invisibility-spec and the cursor is no longer on a sensitive
;; text, the cursor shape is still x-sensitive-text-pointer-shape

;; Try to be XEmacs compatible:
(defvar epips-running-xemacs-p (string-match "XEmacs\\|Lucid" emacs-version))

;; To store the module name in a buffer:
(make-variable-buffer-local 'epips-local-module-name)

(if (x-display-color-p)
    (progn
     ; We have a color screen:
     (copy-face 'bold-italic 'epips-face-prompt-user)
     (set-face-foreground 'epips-face-prompt-user "grey")

     (copy-face 'bold-italic 'epips-face-user-error)
     (set-face-foreground 'epips-face-user-error "red")

     (copy-face 'default 'epips-face-user-log)
     (set-face-foreground 'epips-face-user-log "DodgerBlue1")

     (copy-face 'bold-italic 'epips-face-user-warning)
     (set-face-foreground 'epips-face-user-warning "orange")

     (copy-face 'bold 'epips-face-module-head)
     (set-face-foreground 'epips-face-module-head "red")
     (set-face-background 'epips-face-module-head "lightblue")

     (copy-face 'default 'epips-face-parallel-loop)
     (set-face-background 'epips-face-parallel-loop "lemonchiffon")

     (copy-face 'default 'epips-face-declaration)
     (set-face-foreground 'epips-face-declaration "magenta")

     (make-face 'epips-face-reference)
     (set-face-underline-p 'epips-face-reference t)
     (copy-face 'highlight 'epips-mouse-face-reference)

     (copy-face 'bold-italic 'epips-face-call)
     (set-face-underline-p 'epips-face-call t)
     (copy-face 'highlight 'epips-mouse-face-call)

     (copy-face 'default 'epips-face-preconditions)
     (set-face-foreground 'epips-face-preconditions "black")
     (set-face-background 'epips-face-preconditions "lightblue")

     (copy-face 'default 'epips-face-transformers)
     (set-face-foreground 'epips-face-transformers "black")
     (set-face-background 'epips-face-transformers "lightgreen")

     (copy-face 'default 'epips-face-cumulated-effect)
     (set-face-foreground 'epips-face-cumulated-effect "black")
     (set-face-background 'epips-face-cumulated-effect "dodgerblue1")
     )
  (progn
     ; No...
     (copy-face 'bold-italic 'epips-face-prompt-user)
     (set-face-underline-p 'epips-face-prompt-user t)

     (copy-face 'bold 'epips-face-user-error)
     (invert-face 'epips-face-user-error)

     (copy-face 'bold 'epips-face-user-log)
      
     (copy-face 'bold-italic 'epips-face-user-warning)
   )
  )

;;; The position of the EPips frames:


(let
    (
     (epips-frame-height (if epips-running-xemacs-p
			     400 ;; Why not ?
			   ;; Else compute half the height of the screen:
			   (- (/ (- (/ (x-display-pixel-height)
				       2) ; 2 stacked frames
				    26) ; estimated window manager decor
					; per frame
				 (frame-char-height))
			      1 ;; Keep room for the menu-bar
			      )			  
			   )
			 )
     (epips-log-frame-height (if epips-running-xemacs-p
				 12
				 (/ 250
				    (frame-char-height))
				 )
			     )
     )
  (add-to-list
   'special-display-regexps `("Pips-Log" (top - 0) (left . 0) (height . ,epips-log-frame-height)))
  (add-to-list
   'special-display-regexps
   `("EPips-[048]" (top . 0) (left - 0) (height . ,epips-frame-height)))
  (add-to-list
   'special-display-regexps
   `("EPips-[159]" (top - 0) (left - 0) (height . ,epips-frame-height)))
  (add-to-list
   'special-display-regexps
   `("EPips-[26]" (top . 0) (left . 0) (height . ,epips-frame-height)))
  (add-to-list
   'special-display-regexps
   `("EPips-[37]" (top - 0) (left . 0) (height . ,epips-frame-height)))
  )


;; The token to mark the begin and end of command. Use some strings to
;; be usually never used in ISO-8859-1:
(setq
 epips-begin-of-command-token "\200"
 epips-end-of-command-token "\201"
 epips-send-begin-of-command-token "\202"
 epips-send-end-of-command-token "\203"
 epips-command-separator ":"
 )

;; All the command string that are used to name commands between Pips
;; and emacs:
(setq
 epips-array-data-flow-graph-view-command-name "Array Data Flow Graph"
 epips-available-modules-command-name "AVAILABLE_MODULES"
 epips-bank-view-command-name "BANK"
 epips-call-graph-view-command-name "Callgraph View"
 epips-daVinci-view-command-name "VIEW_DAVINCI_GRAPH"
 epips-dependance-graph-view-command-name "Dependence Graph View"
 epips-distributed-view-command-name "Distributed View"
 epips-edit-view-command-name "Edit"
 epips-emacs-sequential-view-command-name "Emacs Sequential View"
 epips-flint-view-command-name "Flint View"
 epips-hpfc-file-view-command-name "HPFC File"
 epips-ICFG-view-command-name "ICFG View"
 epips-module-command-name "MODULE_NAME"
 epips-new-daVinci-context-command-name "NEW_DAVINCI_CONTEXT"
 epips-parallel-view-command-name "Parallel View"
 epips-placement-view-command-name "Placement"
 epips-prompt-user-command-name "PROMPT_USER"
 epips-scheduling-view-command-name "Time stamps"
 epips-sequential-view-command-name "Sequential View"
 epips-user-error-command-name "USER_ERROR"
 epips-user-log-command-name "USER_LOG"
 epips-user-view-command-name "User View"
 epips-user-warning-command-name "USER_WARNING"
 epips-window-number-command-name "WINDOW_NUMBER"
 )


;;; Some utilities:

(defun epips-debug (something)
  "To print some messages during the epips debugging"
  (print something (get-buffer "*Messages*"))
  )


(defun epips-dispatch-regexp (a-string default-function &rest a-map-list)
  "Try do find a regexp match in a-map-list (e1 f1 e2 f2 ...)
 for a-string and if any, call the function dual. 
If not, call default-function"
  (catch 'match-has-been-found
    (let (
	  (length-of-list (length a-map-list))
	  (i 0)
	  )
      (while (< i length-of-list)
	;; Try to match a daVinci syntax:
	(if (string-match (elt a-map-list i) a-string)
	    (progn
	      ;; Call the according function with the parameter if any:
	      (funcall (elt a-map-list (1+ i))
		       (match-string 1)) ; The parameter
	      ;; Finished
	      (throw 'match-has-been-found nil)
	      )
	  )
	(setq i (+ i 2))
	)
      ;; Well, nothing has matched, use the default-function:
      (funcall default-function a-string)
      )
    )
  )
    

(defun epips-clean-up-fortran-expression (a-string)
  "Remove surrounding blanks, double blanks near to ',', LF, 
continuation mark, etc."
  ;; Remove the leading and trailing blanks:
  (while (string-match "\\`[ \n\t]+\\|[ \n\t]+\\'" a-string)
    (setq a-string (replace-match "" nil nil a-string)))
  ;; Remove continuation marks:
  (while (string-match "\n     &" a-string)
    (setq a-string (replace-match "" nil nil a-string)))
  a-string
  )


(defun epips-list-to-completion-alist (a-list)
  "Transform a list in an alist suitable for completion stuff"
  (let ((i 1))
    (mapcar '(lambda (x) (prog1
			     (list x i)
			   (setq i (1+ i))))
	    a-list)
    )
  )

  

(defun epips-select-and-display-a-buffer (name)
  "The function used to select and display a buffer."
  (let (
        ; Create a new frame if necessary to display the buffer:
	;(pop-up-frames t)
        )
    (pop-to-buffer name)
    )
  )


(defun epips-find-the-next-buffer-name-to-use ()
  "Return the buffer name of the next Pips buffer to use.
If no buffer can be found, just return nil."
  (let (
	(epips-new-current-window-number
	 (% (1+ epips-current-window-number) epips-window-number)
	 )
	(do-not-exit-loop t)
	)
    (catch 'a-usable-pips-buffer-has-been-found
					; While we do not have tested
					; all the Pips buffers:
      (while do-not-exit-loop
	(let* (
	       (a-buffer (get-buffer-create
			  (aref epips-buffers
				epips-new-current-window-number))
			 )
	       ;; Get the local variable buffer-read-only in the
	       ;; buffer:
	       (buffer-is-read-only-p
		(save-excursion
		  (set-buffer a-buffer)
		  buffer-read-only)
		)
	       )
	  (if (not (or (buffer-modified-p a-buffer)
		       buffer-is-read-only-p
		       )
		   )
	      ;; Well, return this buffer:
	      (progn
		(setq epips-current-window-number epips-new-current-window-number)
		(throw 'a-usable-pips-buffer-has-been-found
		       (buffer-name a-buffer))
		)
	    )
	  ;; Try the next buffer:
	  (setq
	   do-not-exit-loop (/= epips-new-current-window-number epips-current-window-number)
	   epips-new-current-window-number (% (1+ epips-new-current-window-number) epips-window-number)
	   )
	  )
	)
      ;; Else the while return nil.
      )
    )
  )


(defun epips-raw-insert (a-string
			 begin-position
			 end-position
			 a-process)
  "A function that insert the characters begin-position to end-position from a-string."
  (epips-select-and-display-a-buffer (process-buffer a-process))
  (epips-debug 'epips-raw-insert)
  (epips-debug begin-position)
  (epips-debug end-position)
  (epips-debug (substring a-string
			  begin-position
			  end-position))
  (insert (substring a-string
		     begin-position
		     end-position))
  )


; A function that insert some text with a property list :
(defun epips-insert-log-with-properties (some-text property-list)
  (epips-select-and-display-a-buffer epips-process-buffer)
  (let ((old-point (point)))
    (insert some-text)
    (add-text-properties old-point
			 (point)
			 property-list))
  )


;;;
;;; The daVinci stuff to display graphs:
;;;

(defvar epips-current-daVinci-context -1
  "Store the current daVinci context used to display a graph")

(defvar epips-daVinci-process nil
  "Store the current daVinci process used to display a graph")

(defvar epips-daVinci-selected-nodes nil
  "Store the current selected nodes")

(defvar epips-daVinci-selected-edge nil
  "Store the current selected edge")


(defun epips-daVinci-clear-state ()
  "Just put the daVinci variable in a clean state"
  (setq epips-current-daVinci-context -1
	epips-daVinci-selected-nodes nil
	epips-daVinci-selected-edge nil)
  )


(defun epips-daVinci-sentinel (process event)
  "Handler for daVinci process state change"
  (epips-debug (format "Process %s had event '%s'." process event))
  (epips-debug "Just consider the daVinci process is no longer usable...")
  (setq epips-daVinci-process nil)
  )


(defun epips-daVinci-ignore (&rest nothing-at-all)
  "Do nothing"
  )


(defun epips-daVinci-close-context ()
  "Close the current context from daVinci"
  (setq epips-daVinci-curent-context nil)
  )


(defun epips-daVinci-set-curent-context (context)
  "Set the current context from daVinci"
  (epips-debug (format "Current context is \"%s\"" context))
  (setq epips-daVinci-curent-context context)
  )


(defun epips-daVinci-nodes-selection (nodes)
  "Select the user pointed nodes"
  (epips-debug (format "Selection of nodes \"%s\"" nodes))
  (setq epips-daVinci-selected-nodes nodes)
  )


(defun epips-daVinci-node-double-click ()
  "Action to node double click"
  (epips-user-warning-command "Node double click\n")
  )


(defun epips-daVinci-edge-selection (edge)
  "Select the user pointed edge"
  (epips-debug (format "Selection of edge \"%s\"" edge))
  (setq epips-daVinci-selected-edge edge)
  )


(defun epips-daVinci-node-selection ()
  "Action to edge double click"
  (epips-user-warning-command "Edge double click\n")
  )


(defun epips-daVinci-output-filter (process output)
  "Accept the daVinci output and displatch functions"
  (epips-debug (format "Process %s had output '%s'." process output))
  ;; The daVinci syntax with respective action:
  (epips-dispatch-regexp output 'epips-user-warning-command
			 "^ok$" 'epips-daVinci-ignore
			 "^context(\\(\".*\"\\))$" 'epips-daVinci-set-curent-context
			 "^close$" 'epips-daVinci-close-context
			 "^node_selections_labels([\\(.*\\)])$" 'epips-daVinci-nodes-selection
			 "^node_double_click$" 'epips-daVinci-node-double-click
			 "^edge_selection_labels([\\(.*\\)])$" 'epips-daVinci-edge-selection
			 "^edge_double_click$" 'epips-daVinci-edge-double-click
			 "^quit$" 'epips-daVinci-ignore
			 )
  )


(defun epips-send-command-to-daVinci (some-text)
  "Send a daVinci command"
  (let (
	;; Commands end with a '\n':
	(the-command (format "%s\n" some-text))
	)
    (epips-debug (format "epips-send-command-to-daVinci: %s" the-command))
    (process-send-string epips-daVinci-process the-command)
    )
  )


(defun epips-new-daVinci-context-user-command (epips-command-content)
  "Launch a new daVinci window (context)"
  ;; If there is no running daVinci process yet, launch one:
  (if (not epips-daVinci-process)
      (let ((process-connection-type nil))
	(epips-daVinci-clear-state)
	(setq epips-daVinci-process
	      (start-process "DaVinci" "DaVinci" "daVinci" "-pipe"))	
	(set-process-sentinel epips-daVinci-process 'epips-daVinci-sentinel)
	(set-process-filter epips-daVinci-process 'epips-daVinci-output-filter)
	)
    )
  ;; Open a new context:
  (setq epips-current-daVinci-context (1+ epips-current-daVinci-context))
  (epips-send-command-to-daVinci (format "multi(open_context(\"Context_%d\"))"
				      epips-current-daVinci-context))
  )


(defun epips-daVinci-view (epips-command-name epips-command-content)
  "Ask daVinci for displaying a graph in the current daVinci context"
  (epips-debug (format "epips-daVinci-view: %s" epips-command-content))
  (let (
	(graph-file-name (if (string-match "-graph$" epips-command-content)
			     ;; Load the "-daVinci" file instead of
			     ;; the "-graph" one if any:
			     (replace-match "-daVinci" t t epips-command-content)
			   epips-command-content))
	)
    (epips-send-command-to-daVinci (format "menu(file(open_graph(\"%s\")))"
					   graph-file-name))
    )
  )


;;;
;;; Here are defined the various function to deal with the Pips actions:
;;;

;; Executed from send_module_name_to_emacs() in emacs.c:
(defun epips-module-name-command (epips-command-content)
  (epips-debug 'epips-module-name-command)
  (setq epips-current-module-name epips-command-content)
  )  


; Executed from send_prompt_user_to_emacs() in emacs.c:
(defun epips-prompt-user-command (epips-command-content)
  (epips-debug 'epips-prompt-user-command)
  (epips-insert-log-with-properties epips-command-content
				    '(face epips-face-prompt-user))
  (epips-raw-insert "\n"
		    0
		    nil
		    a-process)
  )  


; Executed from send_user_error_to_emacs() in emacs.c:
(defun epips-user-error-command (epips-command-content)
  (epips-debug 'epips-user-error-command)
  (epips-insert-log-with-properties epips-command-content
				    '(face epips-face-user-error))
  )  


; Executed from send_user_log_to_emacs() in emacs.c:
(defun epips-user-log-command (epips-command-content)
  (epips-debug 'epips-user-log-command)
  (epips-insert-log-with-properties epips-command-content
				    '(face epips-face-user-log))
  )  


(defun epips-user-warning-command (epips-command-content)
  "Executed from send_user_warning_to_emacs() in emacs.c"
  (epips-debug 'epips-user-warning-command)
  (epips-insert-log-with-properties epips-command-content
				    '(face epips-face-user-warning))
  )  


(defun epips-fortran-mode-and-hilit ()
  "Go in Fortran mode"
  (fortran-mode)
  ;;(epips-add-keymaps-and-menu-in-the-current-buffer)
  )


; Initialize the associative list that record the output sent by each
; xtree process:
(setq epips-xtree-list-output '())

(defun epips-xtree-output-filter (a-process an-output-string)
  "Define a filter to interpret the standard output of xtree"
  (let*
      (
					; Allow interruption inside the filter
       (inhibit-quit nil)
					; Get the old output string of
					; this process:
       (epips-this-process-old-output
	(assoc a-process epips-xtree-list-output))
					; Concatenate the new output
					; to the old one
       (full-output-string (concat (cdr epips-this-process-old-output)
				   an-output-string))
       (epips-xtree-output-filter-newline nil)
       )
    					; Each new line is a module
					; name:
    (while (setq epips-xtree-output-filter-newline
		   (string-match "\n" full-output-string))
					; Ok, we received something up
					; a newline from xtree.
      (let* (
	     (one-line-from-xtree
	      (substring full-output-string
			 0
			 epips-xtree-output-filter-newline)
	      )
					; In fact, only the leaf name
					; is needed up to now:
	     (point-place (string-match "[^.]*$" one-line-from-xtree))
	     )
					; Apply a command with the
					; module name clicked with the
					; mouse:
	
					; For exemple, display the
					; code of the module:
      (epips-send-sequential-view-command
       (substring one-line-from-xtree point-place nil))
      )
       
					; Discard the part that is
					; already executed:
      (setq full-output-string
	    (substring full-output-string
		       (1+ epips-xtree-output-filter-newline)
		       nil))
      )
					; To remind the output of this
					; process:
    (setcdr epips-this-process-old-output full-output-string)
    )
  )

  
(defun epips-ICFG-or-graph-view-command (epips-command-name
					 epips-command-content)
  "Display a graph with xtree"
  (let*
      (
					; Do not use a pipe to
					; communicate since it looks
					; like a flush is lacking
					; somewhere and the output is
					; sent to emacs only when
					; xtree exits if so... No
					; (process-connection-type
					; nil)
       (epips-xtree-process
					; Do not use intermediate shell:
	(start-process "xtree"
		       "Pips-Xtree-Log"
		       "xtree"
		       "-name" epips-command-name
		       "-title" epips-current-module-name
		       "-bg" "LightSteelBlue1" "-fg" "purple4"
		       "-separator" "    "
		       "-oformat" "resource"
		       )
	)
					; That mean that we can not a
					; file through stdin and need
					; a temporary buffer:
       (epips-xtree-input-buffer (get-buffer-create "epips-xtree-input"))
       )
    (save-excursion
					; The process to understand
					; the click on nodes in xtree:
      (set-process-filter epips-xtree-process 'epips-xtree-output-filter)
      (set-buffer epips-xtree-input-buffer)
					; Read the ICFG output file:
      (insert-file-contents epips-command-content)
					; Remove the empty lines:
      (perform-replace "^\012" "" nil t nil)
					; Indent the comments lines as
					; the following one to have
					; nodes at the same depth:
      (goto-char (point-min))
      (perform-replace "^\\(C.*\\)\012\\( *\\)" "\\2\\1\012\\2" nil t nil)
      ;; Remove the new leading space from Fabien :
      (goto-char (point-min))
      (perform-replace "^ " "" nil t nil)
					; Send the file content to the
					; xtree process:
      (process-send-region epips-xtree-process (point-min) (point-max))
					; And close stdin to draw the
					; tree:
      (process-send-eof epips-xtree-process)
      (kill-buffer epips-xtree-input-buffer)
					; Reset the output string of
					; this xtree process:
      (setq epips-xtree-list-output
	    (append (` (( (, epips-xtree-process )  . "")) )
		    epips-xtree-list-output))
					; The end of an xtree process
					; would need to clean
					; epips-xtree-list-output...
      )
    )
  )



(defun epips-sequential-view-command (epips-command-name
				      epips-command-content)
  "Executed from send_view_to_emacs() in emacs.c.
epips-command-content contains the name of the file to display."
  (epips-debug 'epips-sequential-view-command)
  (message "Displaying \"%s\" for module \"%s\"..."
	   epips-command-name
	   epips-current-module-name)
  (save-excursion
					; Switch to a new buffer:
    (let (
	  (old-buffer-name (epips-find-the-next-buffer-name-to-use))
	  (decorated-file-name (concat epips-command-content
				       "-emacs"))
	  )
      (if old-buffer-name
	  ;; Well, there is a buffer available:
	  (progn
					; Switch to the new buffer,
					; called old compared with few
					; lines below...:
	   (epips-select-and-display-a-buffer old-buffer-name)
  
					; Erase an old content:
	   (delete-region (point-min) (point-max))
	   (setq
	    epips-local-module-name epips-current-module-name
	    )
	   (if (file-exists-p decorated-file-name)
	       ;; Execute the file generated by Pips decorated with
	       ;; Emacs properties if any:
	       (load-file decorated-file-name)
	     ;; Else, just insert its content that is plain text:
	     (insert-file-contents epips-command-content)
	     )
	   (goto-char (point-min))
	   (epips-fortran-mode-and-hilit)
	   ;; Add the file name, for a possible future user edit and
	   ;; save:
	   ;; Emacs 19.32: added t to avoid asking confirmation
	   ;; to the user if there is already a buffer displaying this
	   ;; file. It used to be trouble-shooting for unexperimented
	   ;; users... :-)
	   (set-visited-file-name epips-command-content t)
	   ;; Restore the original "Emacs-PIPS-<n>" name instead of
	   ;; the file name:
	   (rename-buffer old-buffer-name)
	   
	   (epips-initialize-current-buffer-stuff)
	   ;; No modification yet:
	   (set-buffer-modified-p nil)
	   ;; Change the window and icon headers:
	   (modify-frame-parameters
	    (selected-frame)
	    `((name . ,(format "%s - %s: %s"
			       (buffer-name)
			       epips-command-name
			       epips-current-module-name))
	      (icon-title-format . ,(format "%s - %s: %s"
					    (buffer-name)
					    epips-command-name
					    epips-current-module-name))
	      )
	    )
	   (message "Displaying \"%s\" for module \"%s\" done."
		    epips-command-name
		    epips-current-module-name)
	   )
	(epips-user-warning-command " No more buffer available... Unprotect or save some or increase the number of windows.\n")
	)
      )
    )
  )


(defun epips-view-command (epips-command-name epips-command-content)
  "Executed from send_view_to_emacs() in emacs.c"
  (epips-debug 'epips-view-command)
  (if (equal epips-command-name epips-array-data-flow-graph-view-command-name)
      (epips-sequential-view-command epips-command-name epips-command-content)
    (if (equal epips-command-name epips-bank-view-command-name)
	(epips-sequential-view-command epips-command-name epips-command-content)
      (if (equal epips-command-name epips-call-graph-view-command-name)
          (epips-ICFG-or-graph-view-command epips-command-name epips-command-content)
	(if (equal epips-command-name epips-daVinci-view-command-name)
	    (epips-daVinci-view epips-command-name epips-command-content)
	  (if (equal epips-command-name epips-dependance-graph-view-command-name)
	      (epips-sequential-view-command epips-command-name epips-command-content)
	    (if (equal epips-command-name epips-edit-view-command-name)
					; The .f edit is not something
					; special:
		(epips-sequential-view-command epips-command-name epips-command-content)
	      (if (equal epips-command-name epips-distributed-view-command-name)
		  (epips-sequential-view-command epips-command-name epips-command-content)
		(if (equal epips-command-name epips-emacs-sequential-view-command-name)
		    (epips-sequential-view-command epips-command-name epips-command-content)
		  (if (equal epips-command-name epips-flint-view-command-name)
		      (epips-sequential-view-command epips-command-name epips-command-content)
		    (if (equal epips-command-name epips-hpfc-file-view-command-name)
			(epips-sequential-view-command epips-command-name epips-command-content)
		      (if (equal epips-command-name epips-ICFG-view-command-name)
			  (epips-sequential-view-command epips-command-name epips-command-content)
			(if (equal epips-command-name epips-parallel-view-command-name)
			    (epips-sequential-view-command epips-command-name epips-command-content)
			  (if (equal epips-command-name epips-placement-view-command-name)
			      (epips-sequential-view-command epips-command-name epips-command-content)
			    (if (equal epips-command-name epips-scheduling-view-command-name)
				(epips-sequential-view-command epips-command-name epips-command-content)
			      (if (equal epips-command-name epips-sequential-view-command-name)
				  (epips-sequential-view-command epips-command-name epips-command-content)
				(if (equal epips-command-name epips-user-view-command-name)
				    (epips-sequential-view-command epips-command-name epips-command-content)
					; Else, command unknown:
				  (epips-user-error-command (concat "\nCommand name \""
								    epips-command-name
								    "\" with argument \""
								    epips-command-content
								    "\" not implemented !!!\n\n"))
				  )
				)
			      )
			    )
			  )
			)
		      )
		    )
		  )
		)
	      )
	    )
	  )
	)
      )
    )
  )


;; Executed from send_window_number_to_emacs() in emacs.c:
(defun epips-window-number-command (epips-command-content)
  (epips-debug 'epips-window-number-command)
  (setq epips-window-number (string-to-number epips-command-content))
  )  


; This function try to execute a command sent by Pips:
(defun epips-execute-output-command (command-string
				     command-begin-position
				     command-end-position
				     a-process)
  (epips-debug 'epips-execute-output-command)
					; Focus on the interesting
					; part of the string:
  (setq command-string (substring command-string
				  (+ command-begin-position
				     (length epips-begin-of-command-token))
				  command-end-position))
  (epips-debug command-string)
					;The command has the format
					;"command_name:command_content":
  (setq epips-command-separator-position
	(string-match epips-command-separator command-string))
  (setq epips-command-name (substring command-string
				      0
				      epips-command-separator-position))
  (epips-debug epips-command-name)
  (setq epips-command-content (substring command-string
					 (+ epips-command-separator-position
					    (length epips-command-separator))
					 nil))
  (epips-debug epips-command-content)
  (if (equal epips-command-name epips-available-modules-command-name)
      (epips-set-available-module-list-command epips-command-content)
    (if (equal epips-command-name epips-module-command-name)
	(epips-module-name-command epips-command-content)
      (if (equal epips-command-name  epips-new-daVinci-context-command-name)
	  (epips-new-daVinci-context-user-command epips-command-content)
	(if (equal epips-command-name epips-prompt-user-command-name)
	    (epips-prompt-user-command epips-command-content)
	  (if (equal epips-command-name epips-user-error-command-name)
	      (epips-user-error-command epips-command-content)
	    (if (equal epips-command-name epips-user-log-command-name)
		(epips-user-log-command epips-command-content)
	      (if (equal epips-command-name epips-user-warning-command-name)
		  (epips-user-warning-command epips-command-content)
		(if (equal epips-command-name epips-window-number-command-name)
		    (epips-window-number-command epips-command-content)
					; It may be a view command:
		  (epips-view-command epips-command-name epips-command-content)
		  )
		)
	      )
	    )
	  )
	)
      )
    )
  )

; Parse the output of wpips to see if there are some commands inside:
(defun epips-analyse-output (a-process an-output-string)
  (let ((epips-packet-begin-position nil)
	(epips-packet-end-position nil))
    (if
	(equal
	 epips-output-automaton-state
	 'epips-output-automaton-state-wait-for-begin
	 )
					; We are waiting for a begin
					; of packet:
	(progn
					; Search for a begin of packet:
	  (setq epips-packet-begin-position
		(string-match epips-begin-of-command-token an-output-string))
	  (epips-debug 'epips-packet-begin-position)
	  (epips-debug epips-packet-begin-position)
					; Display all the string up to
					; a potential
					; epips-begin-of-command-token:
	  (epips-raw-insert an-output-string
			    0
			    epips-packet-begin-position
			    a-process)
	  (if epips-packet-begin-position
					; If we have found a begin,
					; look for an end from the
					; begin position:
	      (progn
					; Raw output up to the command
					; begin :
		(setq epips-packet-end-position
		      (string-match epips-end-of-command-token an-output-string
				    (+ 1 epips-packet-begin-position)))
		(epips-debug 'epips-packet-end-position)
		(epips-debug epips-packet-end-position)
		(if epips-packet-end-position
					; Ok, we have the end of
					; packet in the same string
					; too
		    (progn
					; Execute the command
		      (epips-execute-output-command
		       an-output-string
		       epips-packet-begin-position
		       epips-packet-end-position
		       a-process)
		      (setq
					; We discard the command from
					; the an-output-string:
		       an-output-string
		       (substring an-output-string
				  (+
				   epips-packet-end-position
				   (length epips-end-of-command-token))
				  nil))
					; We stay in the
					; wait-for-begin state.
		      )
	  
					; Else we do not have the end
					; of packet yet so we store
					; the command...
		  (setq epips-output-command-string
			(substring
			 an-output-string
			 epips-packet-begin-position
			 nil)
					; We empty the output-string:
			an-output-string ""
					; And stay in the wait-for-end
					; state:
			epips-output-automaton-state
			'epips-output-automaton-state-wait-for-end
			)
		  )
		)
					; Else, no command found, we
					; stay in the wait-for-begin
					; state and empty the
					; output-string:
	    (setq an-output-string "")

	    )
	  )
					; Else:
					; We are waiting for an end of
					; packet:
      (progn
					; Search for an end of packet:
	(setq epips-packet-end-position
	      (string-match epips-end-of-command-token an-output-string))
	(if epips-packet-end-position
					; If we have found an end, we
					; can send the command:
	    (progn
	      (
	       epips-execute-output-command
	       (concat epips-output-command-string
		       (substring an-output-string
				  0
				  epips-packet-end-position))
	       0
	       nil
	       a-process)

	      (setq
					; We leave the wait-for-end
					; state:
	       epips-output-automaton-state
	       'epips-output-automaton-state-wait-for-begin
					; An leave the rest of the
					; line
	       an-output-string
	       (substring an-output-string
			  (+
			   epips-packet-end-position
			   (length epips-end-of-command-token))
			  nil)
	       )
	      )
	  (setq
					; Else it is a piece of
					; command, thus we just add
					; the string to the command
					; string
	   epips-output-command-string (concat
					epips-output-command-string
					an-output-string)
					; End empty the output string
	   an-output-string "")
	  )
	)
      )
    (epips-debug epips-packet-begin-position)
    (epips-debug epips-packet-end-position)
    (epips-debug epips-output-automaton-state)
					; Return the remaining
					; an-output-string:
    an-output-string
    )
  )


(defun epips-output-filter (a-process an-output-string)
  "Define a filter to interpret the standard output of wpips:
   The outline come from the E-Lisp manual about \"Process Filter Function\"."
  (let
      (
       (old-buffer (current-buffer))
       (old-window (selected-window))
       (inhibit-quit nil)		; Allow interruption inside the filter
       ;;(debug-on-error t)
       )
    (unwind-protect
	(let ((moving)
	      (old-point))
					; By default, go to the end of
					; the buffer controling the
					; process:
	  (set-buffer (process-buffer a-process))
	  ;; To be sure we always insert at the end of the buffer:
	  (set-marker (process-mark a-process) (point-max))
	  (setq old-point (point))
	  (setq moving (= (point) (point-max)))
	  (save-excursion
	    (goto-char (process-mark a-process))
					; Parse the output of wpips to
					; see if there are some
					; commands inside:
	    (while (progn
					; Loop on each semantical
					; piece
		     (setq an-output-string (epips-analyse-output
					     a-process
					     an-output-string))
		
		     (epips-debug an-output-string)
					; Until it returns an empty
					; string:
		     (not (equal an-output-string ""))))

	    (set-marker (process-mark a-process) (point-max))
	    )
	  (if moving
	      ;; Align the text on the window bottom:
	      (progn
		(goto-char (point-max))
		;; Recenter the Pips-Log window:
		(select-window (get-buffer-window (current-buffer) t))
		(recenter -1)
		)
	    ;; Else, go back where the user was before:
	    (goto-char old-point))
	  )
      ;; When the unwind-protect exits:
      (progn
	(set-buffer old-buffer)
	(select-window old-window)
	)
      )
    )
  )
  
; Here are the functions used to send a command to Pips:

(defun epips-send-a-command-to-pips (command-name &optional command-content)
  "Send a command with command-name and command-content to the Pips process"
  (if (not command-content)
      (setq command-content "")
    )
  (process-send-string epips-process
		       (concat epips-send-begin-of-command-token
			       command-name
			       epips-command-separator
			       command-content
			       epips-send-end-of-command-token))
  )


(defun epips-send-sequential-view-command (&optional module-name)
  "Send a command for displaying the sequential view of the current module"
  (epips-send-a-command-to-pips epips-sequential-view-command-name module-name)
  )


;;; Module selection:


(defun epips-send-module-select-command (module-name)
  "Send a command for choosing a module to the Pips process"
  (epips-send-a-command-to-pips epips-module-command-name module-name)
  )


(defun epips-mouse-module-select (event)
  "Select the module with the name under the mouse"
  (interactive "e")
  (epips-debug event)
  (mouse-set-point event)
					; Guess that the module name
					; is the word where the user
					; clicked:
  (let ((module-name (thing-at-point 'word)))
	(epips-send-sequential-view-command module-name)
	)
  )


(defvar epips-available-module-list '("A" "B" "TRUC")
  "Store the name of the modules available in the current workspace")


(defun epips-set-available-module-list-command (list-of-module)
  "Receive a string that is a list of the names of the available
modules in the current workspace and set the
variable epips-available-module-list accordingly"
  (setq epips-available-module-list
	(car (read-from-string list-of-module)))
  )


(defvar epips-select-module-history nil
  "Store the history of previously selected modules")


(defun epips-select-module (module-name)
  "Select a module from the module list"
  (interactive 
   (list ; "interactive" waits for a list of arguments...
    (completing-read
    "Enter the name of the module to select: "
    (epips-list-to-completion-alist epips-available-module-list)
    nil ; No predicate
    t ; Require the module choosen really exists
    nil ; No initial choice
    epips-select-module-history)))
  (epips-send-module-select-command module-name)
  )


;;;


(defun epips-save-to-seminal-file ()
  "Save the current file with a \".f\" file name.
  In this way, PIPS can reparse it and use the modifications done
  by the user or even by PIPS"
  (interactive)
  (write-region
   (point-min)
   (point-max)
   (concat
    (file-name-directory buffer-file-name)
    (downcase (file-name-sans-extension
	       (file-name-nondirectory buffer-file-name)))
    ".f")
   )
					; Mark the buffer as not modified:
  (set-buffer-modified-p nil)
  )


(defun epips-toggle-decoration-display ()
  "Enable or disable the display of the decorations added by PIPS,
such as the preconditions, the regions, etc."  
  (interactive)
  (setq buffer-invisibility-spec (not buffer-invisibility-spec))
  (recenter)
  )


; Various initialization things:

;(setq epips-menu-keymap (cons "Pips" (make-sparse-keymap "Pips")))
;(fset 'epips-menu epips-menu-keymap)

; All the keys and menus for the EPips mode: 
(defvar epips-keymap (make-sparse-keymap "The PIPS keymap")
  "Keymap for the Emacs PIPS mode.")
(defvar epips-menu-keymap (make-sparse-keymap "The PIPS menu")
  "Keymap for the menu of the Emacs PIPS mode.")
(define-key epips-keymap "\C-C\C-J" 'jpips)
;(define-key epips-keymap "\C-C\C-K" 'epips-kill-the-buffers)
(define-key epips-keymap "\C-C\C-L" 'epips-clear-log-buffer)
(define-key epips-keymap "\C-C\C-M" 'epips-select-module)
(define-key epips-keymap "\C-C\C-P" 'epips)
(define-key epips-keymap "\C-C\C-Q" 'epips-kill-the-buffers)
(define-key epips-keymap "\C-C\C-S" 'epips-save-to-seminal-file)
(define-key epips-keymap "\C-C\C-T" 'epips-toggle-decoration-display)
(define-key epips-keymap [print] 'epips-toggle-decoration-display)

;; The EPips main menu:
(fset 'epips-main-menu epips-menu-keymap)
;; Ne marche plus
;;(define-key epips-keymap [menu-bar epips-menu] '("EPips" . epips-main-menu))
(define-key epips-keymap [menu-bar epips-menu] (cons "Epips" (make-sparse-keymap "Main epips")))
(define-key epips-keymap [menu-bar epips-menu main] '("Main menu" . epips-main-menu))

(define-key epips-menu-keymap [epips-kill-the-buffers-menu-item]
  '("Quit and kill the Pips buffers" . epips-kill-the-buffers))
(define-key epips-menu-keymap [epips-another-jpips-process-menu-item]
  '("Launch another JPips process" . jpips))
(define-key epips-menu-keymap [epips-another-pips-process-menu-item]
  '("Launch another WPips process" . epips))
(define-key epips-menu-keymap [epips-clear-log-buffer-menu-item]
  '("Clear log buffer" . epips-clear-log-buffer))
(define-key epips-menu-keymap [epips-save-to-seminal-file-menu-item]
  '("Save the file after edit in the seminal .f" . epips-save-to-seminal-file))
(define-key epips-menu-keymap [epips-toggle-decoration-display-menu-item]
  '("Toggle decoration display" . epips-toggle-decoration-display))
(define-key epips-menu-keymap [epips-select-module-menu-item]
  '("Select a Fortran module" . epips-select-module))

;;(define-key epips-keymap [S-mouse-1]
;;  '("Avoid a bip" . ignore))
;;(define-key epips-keymap [S-down-mouse-3] 'epips-main-menu)
(define-key epips-keymap [(shift button3)] 'epips-main-menu)


;;(defun epips-add-keymaps-and-menu-in-the-current-buffer ()
;;  "This function add the menus and define some keyboard accelerators
;; to the current buffer"
;;  (use-local-map epips-keymap)
;;  )

(defvar epips-reference-keymap (make-sparse-keymap "The PIPS keymap for references")
  "Keymap active when the mouse is on a reference.")

(define-key epips-reference-keymap [down-mouse-1] 'epips-display-the-declaration-of-a-reference-variable)
(define-key epips-reference-keymap [double-mouse-1] 'epips-jump-to-a-declaration)
(define-key epips-keymap "\C-C\C-A" 'epips-show-property)


(defvar epips-call-keymap (make-sparse-keymap "The PIPS keymap for calls")
  "Keymap active when the mouse is on a call site.")

(define-key epips-call-keymap [double-mouse-1]
  '("Go to module" . epips-mouse-module-select))


(defun epips-build-menu-from-layout (layout keymap menu-name menu-vector)
  (setq keymap (make-sparse-keymap menu-name))
  (let (
	(length-of-list (length layout))
	(i 0)
	)
    (while (< i length-of-list)
      (define-key keymap (make-vector 1 (make-symbol (concat menu-name
							     "-"
							     i)))
	'epips-deal-with-menu)
      (setq i (1+ i))
      )
    )
  (define-key epips-keymap [menu-bar epips-menu view]
    (cons menu-name keymap))
  )

(defun epips-build-menu ()
  "Add the automatically generated menus"
  ;; Load the menu layout:
  (load-file (or (getenv "EPIPS_VIEW_MENU_LAYOUT")
		 (concat (getenv "PIPS_ROOT")
			 "/etc/epips_view_menu_layout.el")
		 )
	     )
  (epips-build-menu-from-layout epips-view-menu-layout
				'epips-view-keymap
				"epips view"
				[epips-view-menu-name])

  (load-file (or (getenv "EPIPS_TRANSFORM_MENU_LAYOUT")
		 (concat (getenv "PIPS_ROOT")
			 "/etc/epips_transform_menu_layout.el")
		 )
	     )
)


;;; The property stuff:


(defun epips-relative-insert-properties (offset some-properties)
  "Put some properties relative to offset. Properties are a list of
\"begin end (property-list)\" like with the #(\"...\" ...) format"
  (if some-properties
      (let ((begin (+ (nth 0 some-properties) offset))
	    (end (+ (nth 1 some-properties) offset))
	    (properties (nth 2 some-properties)))
	;; Insert the property list:
	(add-text-properties begin end properties)
	;; Deal with the next property set:
	(epips-relative-insert-properties offset (nthcdr 3 some-properties))
	)
    )
  )


(defun epips-insert-with-properties-read-syntax-like (a-string-with-some-properties)
  "Insert a string with some properties.
The properties can merge (that is not the case for read symtax '#(...)').
The format is
 (\"the string\" b1 e1 p1 b2 e2 b3 ...)"
  trucmuche
  (let ((old-point (point))
        (length-of-properties (length a-string-with-some-properties))
	(i 1))
    (insert (elt (a-string-with-some-properties) 0))
    (while (< i number-of-properties)
      (add-text-properties (elt a-string-with-some-properties
				i)
			   (elt a-string-with-some-properties
				(1+ i))
			   (elt a-string-with-some-properties
				(+ i 2)))
      (setq (+ i 3)))
    )
  )


(defun vieux-epips-insert-with-properties-read-syntax-like (a-string-with-some-properties)
  "Insert a string with some properties.
The properties can merge (that is not the case for read symtax '#(...)').
The format is
(\"the string\" b1 e1 p1 b2 e2 b3 ...)"
  (let ((old-point (point)))
    (insert (car a-string-with-some-properties))
    (epips-relative-insert-properties old-point
                                      (cdr a-string-with-some-properties))
    )
  )


(defun epips-show-property ()
  "Display the properties"
  (interactive)
  (print (text-properties-at (point)) (current-buffer))
  )


(defun epips-look-for-property-with-value (a-property a-value)
  "Find in the current buffer the first occurrence of a-property
with the value a-value. Return '(min-pos max-pos) if found, nil else"
  (catch 'the-success
    (save-excursion
      (goto-char (point-min))
      (while (not (eobp))
	(let ((next-change
	       (or (next-single-property-change (point) a-property)
		   (point-max))))
	  (epips-debug next-change)
	  (let ((value-here (get-text-property next-change
					       a-property)))
	    (epips-debug value-here)
	    (if value-here
		(if (string= value-here a-value)
		    ;; Give back the position of the correct
		    ;; property:
		    (throw 'the-success
			   (list next-change
				 (or (next-single-property-change
				      next-change
				      a-property)
				     (point-max))))))
	    )
	  (goto-char next-change)
	  )
	)
      ;; If we cannot find the property with the correct value, return
      ;; nil:
      nil
      )
    )
  )


(defun epips-get-the-declaration-of-the-reference-at-point ()
  "If there is a reference at point, give the position
of its declaration and the declaration string: (begin start string).
If not, barf and return nil."
  (let ((property-reference-variable
	 (get-text-property (point)
			    'epips-property-reference-variable)))
    (if property-reference-variable
	(let ((property-place (epips-look-for-property-with-value
			       'epips-property-declaration
			       property-reference-variable)))
	  (if property-place
	      ;; Give the result back:
	      (append property-place
		      (list (format "%s %s"
				    ;; Get the type of the variable:
				    (get-text-property (car property-place)
						       'epips-property-type)
				    ;; And its name:
				    (epips-clean-up-fortran-expression
				     (buffer-substring (car property-place)
						       (car (cdr property-place)))))))
	    (progn
	      (epips-user-warning "Cannot find the declaration here!")
	      nil)
	    )
	  )
      (progn
	(epips-user-warning "Cannot find a reference here!")
	nil)      
      )
    )
  )
  

(defun epips-display-the-declaration-of-a-reference-variable (event)
  "Display a message describing the declaration 
of the reference variable we are clicking on"
  (interactive "e")
  ;; Goto where we clicked on first:
  (goto-char (posn-point (event-start event)))
  (message "Variable declaration is \"%s\""
	   (elt (epips-get-the-declaration-of-the-reference-at-point) 2))
  )
  

(defun epips-jump-to-a-declaration (event)
  "Go to the declaration we are on"
  (interactive "e")
  ;; Goto where we clicked on first:
  (goto-char (posn-point (event-start event)))
  (let ((declaration (epips-get-the-declaration-of-the-reference-at-point)))
    (if declaration
	(progn
	  ;; Modify the mark to be able to come back later:
	  (push-mark (point) nil nil)
	  (message (substitute-command-keys
		    "Jumped to \"%s\". To go back to the reference, type C-u \\[set-mark-command]")
		   (elt declaration 2))          
          ;; Go to the variable declaration:
          (goto-char (car declaration)))
      (epips-user-warning "Cannot find the declaration here!")
      )
    )
  )


(defun epips-initialize-current-buffer-stuff ()
  "Initialize mode, local variables, etc."
  ;; Use the EPips minor mode:
  (epips-mode 1)
  ;; To store the module name in a buffer:
  (make-variable-buffer-local 'epips-local-module-name)
  ;; Do not hide any information:
  (setq buffer-invisibility-spec nil)
  ;; Set the active area mouse pointer:
;;;;  (make-variable-buffer-local 'x-sensitive-text-pointer-shape)
  ;;(setq x-sensitive-text-pointer-shape x-pointer-question-arrow)
  ;;(setq x-sensitive-text-pointer-shape x-pointer-target)
  ;;(setq x-sensitive-text-pointer-shape x-pointer-rtl-logo)
  ;;(setq x-sensitive-text-pointer-shape x-pointer-sizing)
  (setq x-sensitive-text-pointer-shape x-pointer-hand2)
  (modify-frame-parameters (selected-frame)
			   (list (assoc 'mouse-color (frame-parameters (selected-frame))))
			   )
  )


(defun epips-add-keymaps-and-menu ()
  "This function add the menus and define some keyboard accelerators"
					; Add the menus on all the
					; Pips windows and log window:
  (mapcar '(lambda (a-buffer)
	     (set-buffer a-buffer)
	     (epips-add-keymaps-and-menu-in-the-current-buffer)
	     )
	  (append epips-buffers (list epips-process-buffer))
	  )
)


(defun epips-clear-log-buffer ()
  "This function clean the log buffer"
  (interactive)
  (save-excursion
					; Switch to the log buffer:
    (set-buffer epips-process-buffer)
					; Erase an old content:
    (delete-region (point-min) (point-max))
    )
  )
  


(defun epips-create-the-buffers ()
  "The function to create all the EPips buffers"
  (setq epips-buffers (make-vector epips-buffer-number nil))
  (setq i 0)
  (while (< i epips-buffer-number)
					; Create each window:
    ;;(aset epips-buffers i (get-buffer-create
	;;		   (format "EPips-%d" i)))
    ;; Just create a buffer name. Buffer creation will be dynamic.
    (aset epips-buffers i (format "EPips-%d" i))
    (setq i (1+ i))
    )		     
  )


(defun epips-all-the-buffers ()
  "Return all the buffers used by EPips"
  (append
   (list epips-process-buffer)
   epips-buffers
   (if (boundp 'epips-xtree-input-buffer)
       epips-xtree-input-buffer)
   )
)


(defun epips-kill-the-log-buffer ()
  "The function to kill the EPips Log buffer"
  (interactive)
  (kill-buffer epips-process-buffer)
)


(defun epips-kill-the-buffers ()
  "The function to kill all the EPips buffers"
  (interactive)
  (mapcar '(lambda (a-buffer)
	     ;; Kill a buffer only if it exists:
	     (if (get-buffer a-buffer)
		 (kill-buffer a-buffer)
	       ))
	  (epips-all-the-buffers)
	  )
  )


(defun epips-kill-the-local-variables (a-buffer)
  "The function to kill the local variable in a buffer"
  (save-excursion
    (let (
	  (old-buffer (current-buffer))
	  )
      (set-buffer a-buffer)
					; Clean up the environment :
      (kill-all-local-variables)
      (set-buffer old-buffer)
      )
    )
)


(defun epips-kill-the-local-variables-in-the-buffers ()
  "The function to kill the local variable in all the EPips buffers"
  (mapcar 'epips-kill-the-local-variables (epips-all-the-buffers))
)


;; Initialize the buffers to display PIPS stuff:
(setq epips-buffer-number 9)

;; Launch the wpips process from Emacs:
      
(defun epips (&optional type)
  "The Emacs-PIPS mode.

The Fortran codes are dealt with the Fortran mode of Emacs.

Some hypertext is added to PIPS:
- by clicking with Shift-mouse-1 (Shift and the mouse left button)
on a function or procedure name, the module is selected and displayed.
The buffers used to display PIPS output are named from Emacs-Pips-0
to Emacs-Pips-9

You can edit the text in a PIPS buffer. After that, you can save the
modified file with \C-C \C-C in the \".f\" file so that PIPS will take
in account your modifications up to now.

The Call Graph and ICFG view are displayed with xtree.
By pressing an xtree node, it is possible to select a module
and display its code as with Shift-mouse-1 in an Emacs-PIPS window.

The log window is a buffer named Pips-Log.

In each PIPS buffer, a menu is available to empty the log buffer and 
even to kill the PIPS buffer when they are no longer useful after 
exiting the PIPS mode.

How does it work? It launches mainly a wpips with the -emacs option...

Killing the Pips-Log buffer kills also the WPips process.
It is useful to interrupt a core dump of 250 MBytes when it happens
for example... :-)

You can choose the wpips executable by setting the EPIPS_WPIPS 
variable to its path.

By the way, EPips is nicer when use with a package such as font-lock-mode...

If epips is launched with the string \"jpips\", the jpips backend (Java
interface) is used instead of the X11/XView backend. See also the jpips
command.

Special commands: 
\\{epips-keymap}
"
					; Just to have the function
					; for the user :
  (interactive)
  (message "Entering Emacs PIPS mode...")
					; Initialize the automaton
					; that analyses the pips
					; output:
  (setq epips-output-automaton-state
	'epips-output-automaton-state-wait-for-begin)
					; Initialize various variables:
  (setq
					; No module name, yet:
   epips-current-module-name nil
					; No window has been displayed yet:
   epips-current-window-number -1
   )
					; Create the display buffers:
  (epips-create-the-buffers)

  (epips-build-menu)

  (catch 'some-error
    (let
	(
	 (process-connection-type nil)	; Use a pipe to communicate
	 )
      (if type (if (equal type "jpips")
		   (setq epips-process (start-process "JPips"
						      "Pips-Log"
						      (or (getenv "EPIPS_JPIPS")
							  "jpips")
						      "-e"))
		 (progn
		   (epips-user-error-command
		    (format "Cannot launch epips with \"%s\" type." type))
		   (throw 'some-error))
		 )
	(
	 setq epips-process (start-process "WPips"
					   "Pips-Log"
					   (or (getenv "EPIPS_WPIPS")
					       "wpips")
					   "-emacs")
	      )
	)
      ;;(setq epips-process (start-process "WPips" "Pips-Log" "/projects/Pips/Development/Libs/effects/wpips" "-emacs"))
					;(goto-char (process-mark epips-process))
      (message (concat (process-name epips-process) " process launched..."))
      (setq epips-process-buffer (process-buffer epips-process))
      (set-process-filter epips-process 'epips-output-filter)
      (epips-select-and-display-a-buffer epips-process-buffer)
					; Clean up the environment :
      ;; (epips-kill-the-local-variables-in-the-buffers)
					;(switch-to-buffer
					; epips-process-buffer) Hum, I
					; do not know why I need to
					; initialize (process-mark
					; epips-process) if I do not
					; want a rude #<marker in no
					; buffer>. It used to work, but...
      (goto-char (point-max))
      (set-marker (process-mark epips-process) (point))
      ;;    (epips-add-keymaps-and-menu)
      ;; Enter EPips mode:
      (save-excursion
	(set-buffer epips-process-buffer)
	(epips-mode 1)
	)
      )
    )
  )

(defun jpips (&optional type)
  "The Emacs-PIPS mode with the jpips Java backend. See the documentation
of epips."

					; Just to have the function
					; for the user :
  (interactive)
  (epips "jpips")
  )


(defun epips-launch-alone ()
  "Act as epips but discard the current buffer after the wpips start-up.
Used by the epips shell script to run EPips as a stand alone Emacs."
  (let
      ((the-current-frame (window-frame (get-buffer-window (current-buffer)))))
    (epips)
    (delete-frame the-current-frame)
    )
  )


(defun jpips-launch-alone ()
  "Act as epips but discard the current buffer after the jpips start-up.
Used by the jepips shell script to run EPips as a stand alone Emacs."
  (let
      ((the-current-frame (window-frame (get-buffer-window (current-buffer)))))
    (jpips)
    (delete-frame the-current-frame)
    )
  )


(defun epips-mode (&optional enable)
  "For EPips acting as a minor mode, add this function mainly to enable
EPips keymap.

See the documentation about epips with
\C-h f epips

Special commands: 
\\{epips-keymap}"
  (interactive)
  ;; To store the EPips minor mode state in a buffer:
  (make-variable-buffer-local 'epips-mode)
  ;; Add the EPips minor mode in the minor mode list:
  (add-to-list 'minor-mode-alist '(epips-mode " EPips"))
  ;; Add the EPips keymap in the minor mode keymap list:
  (add-to-list 'minor-mode-map-alist (cons 'epips-mode epips-keymap))
  ;; Enter Epips minor mode if asked or toggle:
  (setq epips-mode (if (null enable)
		       (not epips-mode)
		     (> (prefix-numeric-value enable) 0)))
  )


(defun vieux-epips-output-filter (a-process an-output-string)
  (let
      (
       (old-buffer (current-buffer))
       (inhibit-quit nil)		; Allow interrupion inside the filter
       )
    (unwind-protect
	(let (moving)
	  (set-buffer (process-buffer a-process))
	  (setq moving (= (point) (process-mark a-process)))
	  (save-excursion
	    ;; Insert the text, moving the process-marker.
	    (goto-char (process-mark a-process))
	    (insert an-output-string)
	    (set-marker (process-mark a-process) (point)))
	  (if moving (goto-char (process-mark a-process)))
	  (set-buffer old-buffer)
	  )
      )
    )
  )
