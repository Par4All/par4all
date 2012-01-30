
var options = {
    zoomHeight: 400,
    zoomWidth: 400
};

function handle_key_down(item, event) {
    catch_tab(item, event);
    modified = true;
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

function initialize_slider(steps) {
    $('#demo_slider').slider('option', 'disabled', false);
    $('#demo_slider').slider('option', 'min', 0);
    $('#demo_slider').slider('option', 'max', parseInt(steps));
    $('#demo_slider').slider('option', 'animate', true);
    $('#demo_slider').slider('option', 'value', '0');
    $('#step').val('0');
    $('#all_steps').val(steps);		
}

function source_textarea(data) {
    return '<textarea name="sourcecode" id="sourcecode" class="output" rows="33" cols="85" onkeydown="handle_key_down(this, event)">' + data + '</textarea>'	
}

function source_resultarea(data) {
    return '<div id="sourcecode" name="sourcecode" class="output">' + data + '</div>'
}


$(function(){
    $('#demo_slider').slider({value:0});
    $('#demo_slider').slider('option', 'disabled', true);
    $('#demo_slider').slider('option', 'slide', function(event, ui) {
	$('#step').val(ui.value);
	if (modified) {
	    $.ajax({
		type: 'POST',
		data: { 
		    code: $('#sourcecode').val(),
		    name: ${self.file_name()}
		},
		cache: false,
		async: false,
		url: "/demo/change_source",
		error: function(){
		    $('#sourcetpips').html("Web error, can't perform demo.");
		},
		success: function(data){
		    modified = false;
		}
	    });
	}
	$.ajax({
	    type: 'POST',
	    data: {
		step: ui.value,
		name: ${self.file_name()}
	    },
	    cache: false,
	    url: "/demo/get_step_output",
	    error: function(){
		$('#sourcetpips').html("Web error, can't perform demo.");
	    },
	    success: function(data){
		if (ui.value != '0') {
		    $('#source').html(source_resultarea(data));
		    $('.ZOOM_IMAGE').jqzoom(options);
		} else {
		    $('#source').html(source_textarea(data));
		    $('#sourcecode').linedtextarea();
		}
	    }
	});
	$.ajax({
	    type: 'POST',
	    data: {step: ui.value},
	    cache: false,
	    url: "/demo/get_step_script",
	    error: function(){
		$('#sourcetpips').html("Web error, can't perform demo.");
	    },
	    success: function(data){
		$('#sourcetpips').html(data);
	    }
	});
    });

    $('#step').val($('#demo_slider').slider('value'));
    $('#all_steps').val('0');

    $('#sourcecode').linedtextarea();
    $('#sourcecode').attr('spellcheck', false);

    $('#dialog-error-examples').dialog({
	autoOpen: false,
	width: 400,
	buttons: {
	    "OK": function(){
		$(this).dialog("close");
	    }
	}
    });

    $('#dialog-load-examples').dialog({
	autoOpen: false,
	width: 400
    });

    initialize_demo();

    $("input:submit", ".load_examples").button();
    $("input:submit", ".load_examples").click(function(event){
				
	event.preventDefault();
	$.ajax({
	    type: "POST",
	    data: {operation: "demo"},
	    cache: false,
	    url: "/examples/get_examples",
	    error: function(){
		$('#dialog-error-examples').dialog('open');
	    },
	    success: function(data){
		var dialog_content = data;
		$('#select-examples-buttons').html(dialog_content);

		$('#select-examples-buttons input:submit').button();
		$('#select-examples-buttons input:submit').click(function(){

		    var filename = $(this).attr('value');
		    modified = false;
		    $.ajax({
			type: 'POST',
			data: { name: ${self.file_name()}},
			cache: false,
			async: false,
			url: '/demo/get_steps_number',
			error: function(){
			    $('#sourcetpips').html("Web error, demo can't be initialize!");
			},
			success: function(data){
			    initialize_slider(data);
			}
		    });
		    
		    $.ajax({
			type: 'POST',
			data: { name: ${self.file_name()}},
			cache: false,
			url: "/demo/load_demo_tpips",
			error: function(){
			    $('#sourcetpips').html("Web error, try again later.");
			},
			success: function(data){
			    $('#sourcetpips').html(data);
			}
		    });
		    $.ajax({
			type: 'POST',
			data: { name: ${self.file_name()}},
			cache: false,
			url: "/demo/load_demo_source",
			error: function(){
			    $('#sourcecode').html("Web error, try again later.");
			},
			success: function(data){
			    $('#source').html(source_textarea(data));
			    $('#sourcecode').linedtextarea();
			}
		    });
		    $('#dialog-load-examples').dialog('close');
		    $('#demo_slider').slider('option', 'value', '0');
		});
		
		$('#dialog-load-examples').dialog('open');
	    }
	});
    });
})
