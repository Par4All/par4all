!function(c){function b(e,d){d.find("> .active").removeClass("active").find("> .dropdown-menu > .active").removeClass("active");e.addClass("active");if(e.parent(".dropdown-menu")){e.closest("li.dropdown").addClass("active")}}function a(i){var h=c(this),f=h.closest("ul:not(.dropdown-menu)"),d=h.attr("href"),g,j;if(/^#\w+/.test(d)){i.preventDefault();if(h.parent("li").hasClass("active")){return}g=f.find(".active a").last()[0];j=c(d);b(h.parent("li"),f);b(j,j.parent());h.trigger({type:"change",relatedTarget:g})}}c.fn.tabs=c.fn.pills=function(d){return this.each(function(){c(this).delegate(d||".tabs li > a, .pills > li > a","click",a)})};c(document).ready(function(){c("body").tabs("ul[data-tabs] li > a, ul[data-pills] > li > a")})}(window.jQuery||window.ender);