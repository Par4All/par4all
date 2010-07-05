#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
#


'''
Par4All common option parsing and report email generation routines
XXX: kind of a p4a_util2 right now.
'''


import string, sys, optparse, smtplib, traceback, mimetypes, re

from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from p4a_util import *
from p4a_version import *


static_options = None
static_args = None
extra_report_files = []


def add_common_options(parser):

    group = optparse.OptionGroup(parser, "General Options")

    group.add_option("--verbose", "-v", action = "count", default = get_verbosity(),
        help = "Run in verbose mode: -v will display more information, -vv will display most information.")

    group.add_option("--log", action = "store_true", default = False,
        help = "Enable logging in current directory.")

    group.add_option("--report", metavar = "YOUR-EMAIL-ADDRESS", default = None,
        help = "Send a report email to the Par4All support email address in case of error. "
            + "This implies --log (it will log to a distinct file every time). "
            + "The report will contain the full log for the failed command, as well as "
            + "the runtime environment of the script like arguments and environment variables.")

    group.add_option("--report-files", action = "store_true", default = False,
        help = "If --report is specified, and if there were files specified as arguments to the script, they will be attached to the generated report email. "
            + "NB: This might be a privacy/legal concern for your organization, so please check twice you are allowed and willing to do so.")

    group.add_option("--report-dont-send", action = "store_true", default = False,
        help = "If --report is specified, generate an .eml file with the email which would have been send to the Par4All team, but do not actually send it.")

    group.add_option("--no-color", action = "store_true", default = False,
        help = "Disable coloring of terminal output.")

    group.add_option("-V", dest = "script_version", action = "store_true", default = False,
        help = "Display script version and exit.")

    parser.add_option_group(group)


def process_common_options(options, args):

    if options.no_color:
        p4a_term.disabled = True

    set_verbosity(options.verbose)

    if options.script_version:
        main_script = os.path.abspath(sys.argv[0])
        version = guess_file_revision(main_script)
        sys.__stdout__.write(version + "\n")
        return False

    if options.log and not options.report:
        setup_logging(suffix = "_" + utc_datetime(), remove = True)
    elif options.report:
        options.log = True
        setup_logging(suffix = "_report_" + utc_datetime(), remove = True)
        warn("--report enabled, I will send an email to the Par4All team in case of error")

    global static_options, static_args
    static_options = options
    static_args = args

    return True


def report_enabled():
    global static_options
    return static_options.report


def report_add_file(file):
    global extra_report_files
    extra_report_files.append(file)


def send_report_email(from_addr = "anonymous@par4all.org", recipient = "par4all@hpc-project.com"):

    global static_options, static_args, extra_report_files

    current_log_file = get_current_log_file()
    if not current_log_file or not os.path.exists(current_log_file):
        raise p4a_error("Current log file is invalid")
    
    eml_file = change_file_ext(current_log_file, ".eml")
    
    recipients = [ recipient, from_addr ]

    flush_log()

    #~ warn("Trying to send a report email to " + recipient + " (" + eml_file + ")", log = False)

    try:
        args = ""
        if static_args:
            for a in static_args:
                args += a + "\n"
        else:
            args = "[]\n"

        files = []
        abs_files = []
        files_desc = ""
        if static_options.report_files:
            for a in static_args + extra_report_files:
                if os.path.exists(a) and os.path.isfile(a):
                    abs = os.path.realpath(os.path.abspath(os.path.expanduser(a)))
                    if abs in abs_files:
                        continue
                    files.append(a)
                    abs_files.append(abs)
                    files_desc += a + " (" + abs + ")\n"
                else:
                    debug("Not a valid file: " + a, log = False)
        if files_desc == "":
            files_desc = "(none)\n"

        env = ""
        for e in os.environ:
            env += e + ": " + os.environ[e] + "\n"

        outer = MIMEMultipart()
        outer["Subject"] = "[" + get_program_name() + "] Par4All failure report"
        outer["From"] = from_addr
        outer["To"] = ", ".join(recipients)
        outer.preamble = "This message is in multipart MIME format.\n"

        outer.attach(MIMEText("This is an automated report email for the following command which failed:\n\n" + " ".join(sys.argv)
            + "\n\nTranslated options:\n\n" + repr(static_options)
            + "\n\nTranslated arguments:\n\n" + args
            + "\nAttached files:\n\n" + files_desc
            + "\nMachine: " + uname() + "\n"
            + "\nUsername: " + whoami() + "\n"
            + "\nEnvironment:\n\n" + env
            + "\nThe full log for this session follows:\n\n" + read_file(current_log_file)))

        for file in abs_files:
            # Guess the content type based on the file's extension.  Encoding
            # will be ignored, although we should check for simple things like
            # gzip'd or compressed files.
            ctype, encoding = mimetypes.guess_type(file)
            if ctype is None or encoding is not None:
                # No guess could be made, or the file is encoded (compressed), so
                # use a generic bag-of-bits type.
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split('/', 1)
            if maintype == "text":
                # Note: we should handle calculating the charset
                msg = MIMEText(read_file(file), _subtype = subtype)
            elif maintype == "image":
                msg = MIMEImage(read_file(file, test = False), _subtype = subtype)
            elif maintype == "audio":
                msg = MIMEAudio(read_file(file, test = False), _subtype = subtype)
            else:
                msg = MIMEBase(maintype, subtype)
                msg.set_payload(read_file(file, test = False))
                # Encode the payload using Base64
                encoders.encode_base64(msg)
            # Set the filename parameter
            msg.add_header("Content-Disposition", "attachment", filename = os.path.split(file)[1])
            outer.attach(msg)

        msg = outer.as_string()
        write_file(eml_file, msg)

    except:
        (t, e, tb) = sys.exc_info()
        warn("Generating the email failed: " + e.__class__.__name__ + ": " + str(e))
        debug("".join(traceback.format_exception(t, e, tb)), level = 3)
        if os.path.exists(current_log_file):
            suggest("Try sending " + current_log_file + " manually to " + recipient + ", thank you")
        return

    else:
        if static_options.report_dont_send:
            done("Report email generated as " + eml_file, log = False)

    if not static_options.report_dont_send:
        warn("Sending " + eml_file + " to " + recipient)
        try:
            server = None
            for serv in [ "mail.hpc-project.com", "mail-int.hpc-project.com" ]:
                if ping(serv):
                    server = serv
                    break
            if not server:
                raise p4a_error("No SMTP server is reachable")

            s = smtplib.SMTP(server)
            s.sendmail(from_addr, recipients, read_file(msg))
            s.quit()

        except:
            (t, e, tb) = sys.exc_info()
            warn("Sending the email failed: " + e.__class__.__name__ + ": " + str(e))
            debug("".join(traceback.format_exception(t, e, tb)), level = 3)
            if os.path.exists(eml_file):
                suggest("Try sending " + eml_file + " manually to " + recipient + ", thank you")

        else:
            done("Report email sent to " + recipient + ", thank you for your feedback", log = False)


def send_report_email_if_enabled():
    global static_options, static_args, extra_report_files
    if static_options.report:
        from_addr = static_options.report
        if not re.match(r"[\w\+\-\_\.]+\@[\w\-\.]+", from_addr):
            warn("Please pass a valid email address to --report if you want the Par4All team to help you with this error")
            from_addr = "anonymous@par4all.org"
        send_report_email(from_addr = from_addr)
    else:
        suggest("You may report this error to the Par4All team by running again using --report <your email address>")
    if not static_options.report_files and (static_args or extra_report_files):
        suggest("Pass --report-files to attach input/processed files to the report email")


def suggest_more_verbosity():
    global static_options
    if get_verbosity() < 2:
        v = "v" * (get_verbosity() + 1)
        suggest("To get a more verbose output, pass -" + v)
        if not static_options.log:
            suggest("Alternatively, you can pass --log to log -vv output to a file")
    current_log_file = get_current_log_file()
    if static_options.log and current_log_file and os.path.exists(current_log_file):
        warn("Log file was " + current_log_file)


def suggest_RC():
    global static_options
    if not static_options.rebuild or not static_options.clean:
        suggest("You may try running again with --rebuild --clean (-RC)")


if __name__ == "__main__":
    print(__doc__)
    print("This module is not directly executable")


# Some Emacs stuff:
### Local Variables:
### mode: python
### mode: flyspell
### ispell-local-dictionary: "american"
### tab-width: 4
### End:
