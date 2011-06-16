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

                function preprocess_input(source, result_tab_index, tab_index, panel_id) {

                        if (language_detection(source, tab_index)) {
                                change_tab_focus_res(result_tab_index, panel_id);
                                if (compile(source, tab_index)) {
                                        return true;
                                }
                        }
                        return false;
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
                        console.debug('DEACTIVATING');
                }

                function activate_graph_buttons() {
                        console.debug('ACTIVATING');
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

                
                function handle_key_down(item, event) {
                        catch_tab(item, event);
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

                function catch_tab(item, e) {
                        c = e.keyCode;
                        if (c == 9) {
                                replaceSelection(item, String.fromCharCode(9));
                                setTimeout("document.getElementById('" + item.id + "').focus();", 0);
                                return false;
                        }
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

	function initialize() {

        	$('#tabs').tabs();

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
                	perform_operation();
                });

                $("input:submit", ".load_examples").button();
		$("input:submit", ".load_examples").click(function(event) {     
                	event.preventDefault();
                	load_examples();
        	});
	}

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
                                        	}
                                        });
                                	$('#dialog-load-examples').dialog("close");
                                });
                        	$('#dialog-load-examples').dialog('open');
                	}
        	});
	}

	</script>
	${self.level_head()}
</%def>
