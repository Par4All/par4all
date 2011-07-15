<!DOCTYPE html>

<%inherit file="frame.mako"/>

<%def name="head()">

	<title>PAWS</title>
        <link type="text/css" href="/jqueryui/css/cupertino/jquery-ui-1.8.10.custom.css" rel="stylesheet" />
        <link type="text/css" href="/jquery-linedtextarea/jquery-linedtextarea.css" rel="stylesheet" />
        <link type="text/css" href="/pygments/css/pygments.css" rel="stylesheet" />
        <link type="text/css" href="/jqzoom/css/jquery.jqzoom.css" rel="stylesheet" />
        <script type="text/javascript" src="/jqueryui/js/jquery-1.4.4.min.js"></script>
        <script type="text/javascript" src="/jqueryui/js/jquery-ui-1.8.10.custom.min.js"></script>
        <script type="text/javascript" src="/jquery-linedtextarea/jquery-linedtextarea.js"></script>
        <script type="text/javascript" src="/jqzoom/js/jquery.jqzoom-core.js"></script>
        <script type="text/javascript">

        var frameId = 'frame_ID';       // hidden frame id
        var jFrame = null;              // hidden frame object
        var formId = 'form_ID';         // hidden form id
        var jForm = null;               // hidden form object
        var uploadUrl = '/userfiles/upload';
	var files_no = 1;
	var marker = '\", \"';
	var multiple = false;
	var file_name;

        var result_id = 'result'; // linked to the div!
        var graph_id = 'graph';

	var directory;
	var performed = false;
	var created_graph = false;
	var functions = '';

	var options = {
		zoomHeight: 400,
		zoomWIDTH: 400
	};

/*********************************************************************
*								     *
* Variables and functions for popup descriptions.		     *
*								     *
*********************************************************************/

	var OffsetX = -10;
	var OffsetY = 0;
	var old, skn, iex = (document.all), yyy = -1000;
	var ns4 = document.layers;
	var ns6 = document.getElementById && !document.all;
	var ie4 = document.all;

	function popup(msg) {
        	var content = "<div class=boxpopup>" + msg + "</div>";
                yyy = OffsetY;
                if (ns4) {skn.document.write(content); skn.document.close(); skn.visibility="visible";}
                if (ns6) {document.getElementById('pdqbox').innerHTML = content; skn.display = ""; skn.visibility="visible";}
                if (ie4) {document.all("pdqbox").innerHTML = content; skn.display = "";}
	}

	function get_mouse(e) {
		var x = (ns4 || ns6) ? e.pageX : event.x + document.body.scrollLeft;
		skn.left = x + OffsetX;
                var y = (ns4 || ns6) ? e.pageY : event.y + document.body.scrollTop;
                skn.top = y + yyy;
	}

	function remove_popup() {
		yyy = - 1000;
                if (ns4) {skn.visibility = "hidden";}
                else if (ns6 || ie4)
                	skn.display = "none";
	}

/*********************************************************************
*								     *
* Functions for user files uploading.				     *
*								     *
*********************************************************************/

        function upload_prepare_forms() {
        	jForm = createForm(formId);
                jFrame = createUploadIframe(frameId);

                jForm.appendTo('body');
                jForm.attr('target', frameId);

                $("#inp").bind('change', startUpload);
                function startUpload(){
                	if(jForm!=null) {
                        	jForm.remove();
                                jForm = createForm(formId);
                                jForm.appendTo('body');
                                jForm.attr('target', frameId);
                        }

                        var newElement = $(this).clone(true);
                        newElement.bind('change',startUpload);
                        newElement.insertAfter($(this));
                        $(this).appendTo(jForm);

                        jForm.submit();
                        jFrame.unbind('load');
                        jFrame.load(function(){
                        	upload_load(upload_get_response($(this)));
                        });
		};
	}

        function upload_get_response(jframe) {
        	var myFrame = document.getElementById(jframe.attr('name'));
                var bod = $(myFrame.contentWindow.document.body);
                var response = $(myFrame.contentWindow.document.body).text().substring(3, bod.text().length - 3);
                setPseudoTextFileValue(document.getElementById('inp').value);
                return response;
	}

	function setPseudoTextFileValue(text) {
        	document.getElementById('pseudotextfile').value = text;
        }

	function createUploadIframe(id) {
        	return $('<iframe width="300" height="200" name="' + id + '" id="' + id + '"></iframe>')
                	.css({position: 'absolute', top: '270px', left: '450px', border:'1px solid #f00', display: 'none'})
                	.appendTo('body');
	};

        function createForm(formId) {
        	return $('<form method="post" action="' + uploadUrl + '" name="' + formId + '" id="' + formId +
                        '" enctype="multipart/form-data" style="position:absolute;width:10px;height:10px;left:45px;top:-45px;padding:5px;-moz-opacity:0;"></form>');
	};

	function split_response(response, sequence) {
        	return response.replace(/\\n/g,'\n').replace(/\\t/g, '\t').split(sequence);
	}

        function set_response_to_first_tab(splited_0) {
        	$("#source_tab_link1").text(splited_0[0]);
                document.source1.sourcecode1.value = splited_0[1];
		language_detection(splited_0[1], 1);
	}
               
        function upload_add(response) {
        	var splited = split_response(response, '\"], [\"');
                set_response_to_first_tab(splited[0].split(marker));
                change_tab_focus_src();
                files_no = splited.length;
                clear_multiple();
                if (files_no > 1) {
                	upload_add_multiple(splited);
                }
	};

	function upload_load(response){
		removeTabs();
		clear_result_tab();
		upload_add(response);
		performed = false;
		created_graph = false;
	}

	function upload_add_multiple(splited) {
		for (var i = 1; i < splited.length; i++) {
			var no = i + 1;
			df = $('<div id="tabs-' + no + '"><form name="language' + no + '"><label for="lang' + no + '">Language: </label><input name="lang' + no + '" value="not yet detected." readonly="readonly" /></form><table><tr><td><form name="source' + no + '"><textarea name="sourcecode' + no + '" id="sourcecode' + no + '" rows="27" cols="120" onkeydown="handle_key_down(this, event)">' + splited[i].split(marker)[1] + '</textarea></form></td></tr></table></div>');
			df.appendTo('#tabs');
			$('#sourcecode' + no).attr('spellcheck', false);
			$('#tabs').tabs('add', '#tabs-' + no, splited[i].split(marker)[0], i);
			language_detection(splited[i].split(marker)[1], no);
		}
		multiple = true;
	}

/*********************************************************************
*								     *
* Helper functions for performing operations.			     *
*								     *
*********************************************************************/

	function language_detection(source, number) {
        	var test;
                $.ajax({
                	type: "POST",
                        data: {code: source},
                        cache: false,
                        url: "/detector/detect_language",
                        async: false,
                        error: function() {
                        	$('#language_detection').html("Language not recognised!");
                                test = false;
			},
                        success: function(data) {
                        	if (data != "none") {
                                	document.forms['language' + number].elements['lang' + number].value = data;
                                        test = true;
				} else {
                                	document.language1.lang1.value = " not supported.";
                                        test = false;
                                }
                        }
		});
                return test;
	}

        function compile(source, number) {
        	var test;
                $.ajax({
                	type: "POST",
                        data: {
                        	code: source,
                                language: document.forms['language' + number].elements['lang' + number].value
                        },
                        cache: false,
                        url: "/detector/compile",
                        async: false,
                        error: function() {
                        	$('#language_detection').html("Cannot be compiled!");
                                test = false;
			},
                        success: function(data) {
                        	if (data != "0") {
                                	$('#resultcode').html('Error in tab number ' + number + ':<br/><br/>' + data);
                                        test = false;
                        	} else {
                                	test = true;
                               	}
			}
		});
                return test;
	}
	
	function preprocess_input(source, result_tab_index, tab_index, panel_id) {
		if (language_detection(source, tab_index)) {
			change_tab_focus_res(result_tab_index, panel_id);
			if (compile(source, tab_index)) {
				return true;
			}
		}
		return false;
	}
	
	function check_language(index) {
		var selected = $('#tabs').tabs('option', 'selected') + 1;
		var sourcee = document.forms['source' + selected].elements['sourcecode' + selected].value;
		language_detection(sourcee, selected);
	}

	function get_directory(panel_id) {
		$.ajax({
			type: 'GET',
			cache: false,
			url: '/operations/get_directory',
			async: false,
			error: function() {
				if (panel_id == result_id)
					$('#resultcode').html('Web error, try again!');
			},
			success: function(data) {
				directory = data;
				if (panel_id == result_id)
					document.getElementById('save_button_link').href = '/res/' + directory + '/' + directory;
			}
		});
	}
 	
	function get_functions() {
		var datas = {};
		datas['number'] = files_no;
		for(var u = 0; u < files_no; u++) {
			datas['code' + u] = document.forms['source' + (u + 1)].elements['sourcecode' + (u + 1)].value;
		}
		for(var t = 0; t < files_no; t++) {
			datas['lang' + t] = document.forms['language' + (t + 1)].elements['lang' + (t + 1)].value;
		}
		$.ajax({
                        type: "POST",
                        data: datas,
                        cache: false,
                        url: "/operations/get_functions",
                        async: false,
			error: function() {
			},
			success: function(data) {
				functions = data;
				$('#dialog-choose-function').html(data);
				$('#dialog-choose-function input:submit').button();
				$('#dialog-choose-function input:submit').click(function() {
					$.ajax({
						type: "POST",
						data: {
                                                       	functions: $(this).attr('value'),
							operation: ${self.operation_id()}
                                                },
                                                cache: false,
						url: "/operations/perform_multiple",
						error: function() {
                                                },
						success: function(data) {
                                                        $('#resultcode').html(data);
							activate_buttons();
						}
					});
					$('#dialog-choose-function').dialog('close');
				});
			}
		});
	}
	
	function check_sources(index, panel_id) {
		sources = new Array(files_no);
		languages = new Array(files_no);
		get_directory(panel_id);
		for (var i = 1; i <= files_no; i++) {
			sources[i] = document.forms['source' + i].elements['sourcecode' + i].value;
		}
		var u = 1;
		while ((u <= files_no) && preprocess_input(sources[u], index, u, panel_id)) {
			u = u + 1;
		}
		if (u == files_no + 1) {
			get_functions();
			return true;
		}
		return false;
	}

/*********************************************************************
*								     *
* Functions for creating graphics.				     *
*								     *
*********************************************************************/

	function create_graph(index, panel_id) {
		clear_graph_tab();
		if (check_sources(index, panel_id)) {
			if (multiple) {
				$.ajax({
                                        type: "GET",
                                        cache: false,
					url: "/graph/dependence_graph_multi",
					error: function() {
                                        },
					success: function(data) {
						enable_dependence_graph(data);
					}
				});
			} else {
				var lang = document.language1.lang1.value;
				var source_code = document.source1.sourcecode1.value;
				$.ajax({
					type: "POST",
					data: {
                                                code: source_code,
                                        	language: lang
                                        },
                                        cache: false,
                                        url: "/graph/dependence_graph",
					error: function() {
					},
					success: function(data) {
						enable_dependence_graph(data);
					}
				});
			}
		}
	}

	function enable_dependence_graph(data) {
		$('#dependence_graph').html(data);
		$('.ZOOM_IMAGE').jqzoom(options);
		activate_graph_buttons();
	}

/*********************************************************************
*								     *
* Helper functions for keeping consistence between input and output. *
*								     *
*********************************************************************/

	function change_tab_focus_src() {
		$('#tabs').tabs("select", 0);
	}

	function add_tab_title(text) {
		$("#result_tab_link").text(text);
	}

	function change_tab_focus_res(result_tab_index, panel_id) {
		$('#tabs').tabs("select", result_tab_index);
		add_waiting_notification(panel_id);
	}

	function add_waiting_notification(panel_id) {
		var tab = '#resultcode';
		if (panel_id == graph_id)
			tab = '#dependence_graph';
		$(tab).html("<p><b>Please wait while processing...<br/>It might take long time.</b><br/></p>");
	}

	function add_choose_function_notification() {
		$('#resultcode').html("<p><b>Choose function to display.</b><br/></p>");
	}

	function activate_buttons() {
		$("input:submit", ".save_results").button("option", "disabled", false);
		$("input:submit", ".print_results").button("option", "disabled", false);
	}

	function deactivate_buttons() {
		$("input:submit", ".save_results").button("option", "disabled", true);
		$("input:submit", ".print_results").button("option", "disabled", true);
	}

	function deactivate_graph_buttons() {
		//console.debug('DEACTIVATING');
	}

	function activate_graph_buttons() {
		//console.debug('ACTIVATING');
	}

	function clear_result_tab() {
		deactivate_buttons();
		$('#resultcode').html('');
		// all of other tabs are removed
		document.language1.lang1.value = 'not yet detected';
	}

	function clear_graph_tab() {
		deactivate_graph_buttons();
		$('dependence_graph').html('');
	}

	function clear_multiple() {
		multiple = false;
		$('#multiple-functions').html('');
	}

	function removeTabs() {
		for(var i = 1; i < files_no; i++) {
			$('#tabs').tabs('remove', 1);
		}
		files_no = 1;
		multiple = false;
	}

/*********************************************************************
*								     *
* Functions for loading examples.				     *
*								     *
*********************************************************************/

	function load_examples() {
		if (multiple)
                	removeTabs();
		$.ajax({
                        type: "POST",
                        data: {operation: ${self.operation_id()}},
                        cache: false,
                        url: "/examples/get_examples",
                        error: function() {
                        	$('#dialog-error-examples').dialog('open');
                        },
			success: function(data) {
                                var dialog_content = data;
                                $('#select-examples-buttons').html(dialog_content);
                                $('#select-examples-buttons input:submit').button();
                                $('#select-examples-buttons input:submit').click(function() {
                                        file_name = $(this).attr('value');
                                        $("#source_tab_link1").text(file_name);
                                        $.ajax({
                                                type: "POST",
                                                data: {
                                                        name: file_name,
                                                	operation: ${self.operation_id()}
                                                },
                                                cache: false,
                                                url: "/examples/get_file",
                                                error: function() {
                                                	document.source1.sourcecode1.value = "Web error, try again later.";
                                                },
                                        	success: function(data) {
                                                        document.source1.sourcecode1.value = data;
                                                        setPseudoTextFileValue("");
                                                        change_tab_focus_src();
                                                        deactivate_buttons();
                                                	load_example_success();
							language_detection(data, 1);
                                        	}
                                        });
                                	$('#dialog-load-examples').dialog("close");
                                });
                        	$('#dialog-load-examples').dialog('open');
                	}
        	});
	}

	function load_example_success() {
		performed = false;
		created_graph = false;
		clear_result_tab();
		clear_multiple();
	}
    
/*********************************************************************
*								     *
* Functions for handling special keys.				     *
*								     *
*********************************************************************/

	function handle_key_down(item, event) {
		c = event.keyCode;
		if (c == 9) {
			catch_tab(item);
		}
		else if (c == 13 || c == 86) {
			setTimeout("check_language()", 1000);
		}
		clear_result_tab();
		performed = false;
		created_graph = false;
	}

	function replaceSelection(input, replaceString) {
		if (input.setSelectionRange) {
			var selectionStart = input.selectionStart;
			var selectionEnd = input.selectionEnd;
			input.value = input.value.substring(0, selectionStart) + replaceString + input.value.substring(selectionEnd);
			if (selectionStart != selectionEnd) {
				setSelectionRange(input, selectionStart, selectionStart + replaceString.length);
			} else {
				setSelectionRange(input, selectionStart + replaceString.length, selectionStart + replaceString.length);
			}
		} else if (document.selection) {
			var range = document.selection.createRange();
			if (range.parentElement() == input) {
				var isCollapsed = range.text = '';
				range.text = replaceString;
				if (!isColllapsed) {
					range.moveStart('character', -replaceString.length);
					range.select();
				}
			}
		}
	}

	function setSelectionRange(input, selectionStart, selectionEnd) {
		if (input.setSelectionRange) {
			input.focus();
			input.setSelectionRange(selectionStart, selectionEnd);
		}
		else if (input.createTextRange) {
			var range = input.createTextRange();
			range.collapse(true);
			range.moveEnd('character', selectionEnd);
			range.moveStart('character', selectionStart);
			range.select();
		}
	}

	function catch_tab(item) {
		replaceSelection(item, String.fromCharCode(9));
		setTimeout("document.getElementById('" + item.id + "').focus();", 0);
		 return false;
	}

/*********************************************************************
*								     *
* Function for font resizing.					     *
*								     *
*********************************************************************/

	function resize(direction) {
		changeFontSize('body', direction);
		changeFontSize('textarea', direction);
		changeFontSize('.highlight', direction);
		changeFontSize('.lineno', direction);
		changeLinesHeight();
	}

	function changeFontSize(id, direction) {
		size = parseInt($(id).css('font-size'));
		if (direction == 1)
			size = size + 1;
		else
			size = size - 1;
		if (size > 1) {
			$(id).css('font-size', size);
		}
	}

	function changeLinesHeight() {
		size = parseInt($('.lines').css('height'));
		size = size * 1.04;
		$('.lines').css('height', size);
	}

/*********************************************************************
*								     *
* Function for components initialization.			     *
*								     *
*********************************************************************/

	function initialize() {

        	$('#tabs').tabs();

		$('#tabs').tabs({
			tabTemplate: "<li><a href='#{href}'>#{label}</a></li>",
			select: function(event, ui) {
				if ((performed == false) && ui.panel.id == result_id) {
					event.preventDefault();
					performed = true;
					perform_operation(ui.index, result_id);
				} else if ((created_graph == false) && ui.panel.id == graph_id) {
					event.preventDefault();
					created_graph = true;
					create_graph(ui.index, graph_id);
				}
			}
		});

                $('#sourcecode1').linedtextarea();
                $('#sourcecode1').attr('spellcheck', false);

                $('#dialog-error-examples').dialog({
                	autoOpen: false,
                        width: 400,
                        buttons: { "OK": function() { $(this).dialog("close");} }
		});

                $('#dialog-load-examples').dialog({
                	autoOpen: false,
                        width: 400
		});

		$('#dialog-choose-function').dialog({
			autoOpen: false,
			width: 400
		});

                add_tab_title(${self.operation_id()}.toUpperCase());

                $("input:submit", ".load_client_file_form").button();

                $("input:submit", ".save_results").button();
                $("input:submit", ".print_results").button();
                $("input:submit", ".print_results").click(function(event) {
                	var content = document.getElementById("resultcode");
                        var printFrame = document.getElementById("iframetoprint").contentWindow;
                        printFrame.document.open();
                        printFrame.document.write(content.innerHTML);
                        printFrame.document.close();
                        printFrame.focus();
                        printFrame.print();
		});

                deactivate_buttons();

                $("input:submit", ".operation").button();
                $("input:submit", ".operation").button("option", "label", "RUN");
		$("input:submit", ".operation").click(function(event) {
                        event.preventDefault();
			$('#tabs').tabs('select', 0);
			$('#tabs').tabs('select', '#' + result_id);
                });

                $("input:submit", ".load_examples").button();
		$("input:submit", ".load_examples").click(function(event) {     
                	event.preventDefault();
                	load_examples();
        	});

		$('#resizing_source input:submit').button();
	}
	
	</script>
	${self.level_head()}
</%def>
