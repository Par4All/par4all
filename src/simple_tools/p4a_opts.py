#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Authors:
# - Grégoire Péan <gregoire.pean@hpc-project.com>
# - Ronan Keryell <Ronan Keryell@hpc-project.com>
#
import p4a_util
import p4a_version
import os
import sys
import optparse
import traceback
import re
import platform


'''
Par4All common option parsing and report email generation routines
'''





# A copy of the program options as parsed with optparse:
static_options = None

# A copy of the program arguments as parsed with optparse:
static_args = None

# The array containing additional files to add to a *potential* report
# if --report and --report-files are specified:
extra_report_files = []

# With Python 2.4 (previous versions untested/not allowed, see process_common_options),
# --report is unavailable because email modules are not present.
# You might as well set this to False to disable reporting once and for all.
report_available = (platform.python_version() >= "2.5")


def add_common_options(parser):
    '''Add common options (for setting verbosity, reporting errors etc.)
    to the optparse list of options.'''

    global report_available

    group = optparse.OptionGroup(parser, "General options")

    group.add_option("--verbose", "-v", action = "count", default = p4a_util.get_verbosity(),
        help = "Run in verbose mode: each -v increases verbosity mode and display more information, -vvv will display most information.")

    group.add_option("--log", action = "store_true", default = False,
        help = "Enable logging in current directory.")

    if report_available:
        group.add_option("--report", metavar = "YOUR-EMAIL-ADDRESS", default = None,
            help = "Send a report email to the Par4All support email address in case of error. "
                + "This implies --log (it will log to a distinct file every time). "
                + "The report will contain the full log for the failed command, as well as "
                + "the runtime environment of the script like arguments and environment variables.")

        group.add_option("--report-files", action = "store_true", default = False,
            help = "If --report is specified, and if there were files specified as arguments to the script, they will be attached to the generated report email. "
                + "WARNING: This might be a privacy/legal concern for your organization, so please check twice you are allowed and willing to do so. "
                + "The Par4All team cannot be held responsible for a misuse/unintended specification of the --report-files option.")

        group.add_option("--report-dont-send", action = "store_true", default = False,
            help = "If --report is specified, generate an .eml file with the email which would have been send to the Par4All team, but do not actually send it.")

    group.add_option("--plain", "--no-color", "--no-fancy", "-z", action = "store_true", default = False,
        help = "Disable coloring of terminal output and disable all fancy tickers and spinners and this kind of eye-candy things :-)")

    group.add_option("--no-spawn", action = "store_true", default = False,
		help = "Do not spawn a child process to run processing (this child process is normally used to post-process the PIPS output and reporting simpler error message for example).")

    # Interestingly, if I use --exec instead of --execute, later I get a
    # syntax error on options.exec... Strange.
    group.add_option("--execute", metavar = "PYTHON-CODE", default = None,
            help = """Execute the given Python code in order to change
            the behaviour of this script.
            It is useful to extend dynamically Par4All.
            The execution is done at the end of the common option processing""")

    group.add_option("--script-version", "--version", "-V", action = "store_true", default = False,
        help = "Display script version and exit.")

    parser.add_option_group(group)


def process_common_options(options, args):
    '''Process options and do some initialization depending on common options
    described above.'''

    global static_options, static_args, report_available

    p4a_util.set_verbosity(options.verbose)

    # Disable coloring and any terminal decorations if requested:
    p4a_util.set_fancy(not options.plain)

    # Only Python >= 2.4 supported.
    if platform.python_version() < "2.4":
        p4a_util.die("You must at least use Python 2.4 to run Par4All scripts")

    # We were asked what is the script version:
    # try to determine the git revision or look at the
    # p4a_version file somewhere:
    if options.script_version:
        (revision, versiond) = p4a_version.make_full_revision(os.environ.get("P4A_ROOT"))
        sys.stdout.write(revision + "\n")
        return False

    if report_available:
        if options.log and not options.report:
            p4a_util.setup_logging(suffix = "_" + p4a_util.utc_datetime(), remove = True)
        elif options.report:
            options.log = True
            p4a_util.setup_logging(suffix = "_report_" + p4a_util.utc_datetime(), remove = True)
            p4a_util.warn("--report enabled, I will send an email to the Par4All team in case of error")
            # Warn a lot if --report-files is specified...
            if options.report_files:
                p4a_util.warn("--report-files specified: This might be a privacy/legal/licensing concern for your organization/company")
                p4a_util.warn("--report-files specified: Please check twice you are allowed and willing to do so, or hit CTRL+C NOW!!")
                p4a_util.warn("--report-files specified: The Par4All team cannot be held responsible for a misuse/unintended specification of the --report-files option,")
                p4a_util.warn("--report-files specified: nor does the --report-files obliges the Par4All team to purchase any licensing for your software,")
                p4a_util.warn("--report-files specified: nor does the --report-files transfers the Par4All team any copyright/licensing rights on your software.")

    static_options = options
    static_args = args

    return True


def report_enabled():
    global static_options
    return static_options.report


def report_add_file(file):
    global extra_report_files
    extra_report_files.append(file)


def send_report_email(from_addr = "anonymous@par4all.org", recipient = "support@par4all.org"):

    import smtplib, mimetypes

    # Only for Python >= 2.5:
    from email import encoders
    from email.message import Message
    from email.mime.audio import MIMEAudio
    from email.mime.base import MIMEBase
    from email.mime.image import MIMEImage
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    global static_options, static_args, extra_report_files

    current_log_file = p4a_util.get_current_log_file()
    if not current_log_file or not os.path.exists(current_log_file):
        raise p4a_util.p4a_error("Current log file is invalid")

    eml_file = p4a_util.change_file_ext(current_log_file, ".eml")

    recipients = [ recipient, from_addr ]

    p4a_util.flush_log()

    #~ p4a_util.warn("Trying to send a report email to " + recipient + " (" + eml_file + ")", log = False)

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
                    p4a_util.debug("Not a valid file: " + a, log = False)
        if files_desc == "":
            files_desc = "(none)\n"

        env = ""
        for e in os.environ:
            env += e + ": " + os.environ[e] + "\n"

        outer = MIMEMultipart()
        outer["Subject"] = "[" + p4a_util.get_program_name() + "] Par4All failure report"
        outer["From"] = from_addr
        outer["To"] = ", ".join(recipients)
        outer.preamble = "This message is in multipart MIME format.\n"

        outer.attach(MIMEText("This is an automated report email for the following command which failed:\n\n" + " ".join(sys.argv)
            + "\n\nTranslated options:\n\n" + repr(static_options)
            + "\n\nTranslated arguments:\n\n" + args
            + "\nAttached files:\n\n" + files_desc
            + "\nMachine: " + repr(platform.uname()) + "\n"
            + "\nLinux distribution: " + repr(platform.linux_distribution()) + "\n"
            + "\nlibc version: " + repr(platform.libc_ver()) + "\n"
            + "\nUsername: " + p4a_util.whoami() + "\n"
            + "\nEnvironment:\n\n" + env
            + "\nThe full log for this session follows:\n\n" + p4a_util.read_file(current_log_file)))

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
                msg = MIMEText(p4a_util.read_file(file), _subtype = subtype)
            elif maintype == "image":
                msg = MIMEImage(p4a_util.read_file(file, text = False), _subtype = subtype)
            elif maintype == "audio":
                msg = MIMEAudio(p4a_util.read_file(file, text = False), _subtype = subtype)
            else:
                msg = MIMEBase(maintype, subtype)
                msg.set_payload(p4a_util.read_file(file, text = False))
                # Encode the payload using Base64
                encoders.encode_base64(msg)
            # Set the filename parameter
            msg.add_header("Content-Disposition", "attachment", filename = os.path.split(file)[1])
            outer.attach(msg)

        msg = outer.as_string()
        p4a_util.write_file(eml_file, msg)

    except:
        (t, e, tb) = sys.exc_info()
        p4a_util.warn("Generating the email failed: " + e.__class__.__name__ + ": " + str(e))
        p4a_util.debug("".join(traceback.format_exception(t, e, tb)), level = 3)
        if os.path.exists(current_log_file):
            p4a_util.suggest("Try sending " + current_log_file + " manually to " + recipient + ", thank you")
        return

    else:
        if static_options.report_dont_send:
            p4a_util.done("Report email generated as " + eml_file, log = False)

    if not static_options.report_dont_send:
        p4a_util.warn("Sending " + eml_file + " to " + recipient)
        try:
            server = None
            for serv in [ "mail.hpc-project.com", "mail-int.hpc-project.com" ]:
                if p4a_util.ping(serv):
                    server = serv
                    break
            if not server:
                raise p4a_util.p4a_error("No SMTP server is reachable")

            s = smtplib.SMTP(server)
            s.sendmail(from_addr, recipients, p4a_util.read_file(eml_file))
            s.quit()

        except:
            (t, e, tb) = sys.exc_info()
            p4a_util.warn("Sending the email failed: " + e.__class__.__name__ + ": " + str(e))
            p4a_util.debug("".join(traceback.format_exception(t, e, tb)), level = 3)
            if os.path.exists(eml_file):
                p4a_util.suggest("Try sending " + eml_file + " manually to " + recipient + ", thank you")

        else:
            p4a_util.done("Report email sent to " + recipient + ", thank you for your feedback", log = False)


def send_report_email_if_enabled():
    global static_options, static_args, extra_report_files, report_available
    if not report_available or not static_options:
        return
    if static_options.report:
        from_addr = static_options.report
        if not re.match(r"[\w\+\-\_\.]+\@[\w\-\.]+", from_addr):
            p4a_util.warn("Please pass a valid email address to --report if you want the Par4All team to help you with this error")
            from_addr = "anonymous@par4all.org"
        send_report_email(from_addr = from_addr)
    else:
        p4a_util.suggest("You may report this error to the Par4All team by running again using --report <your email address>")
    if not static_options.report_files and (static_args or extra_report_files):
        p4a_util.suggest("Pass --report-files to attach input/processed files to the report email (CAREFUL: this assumes you are allowed and willing to send us potentially confidential files!)")


def suggest_more_verbosity():
    global static_options
    if p4a_util.get_verbosity() < 2:
        v = "v" * (p4a_util.get_verbosity() + 1)
        p4a_util.suggest("To get a more verbose output, pass -" + v)
        if static_options and not static_options.log:
            p4a_util.suggest("Alternatively, you can pass --log to log -vv output to a file")
    current_log_file = p4a_util.get_current_log_file()
    if static_options and static_options.log and current_log_file and os.path.exists(current_log_file):
        p4a_util.warn("Log file was " + current_log_file)


def suggest_rebuild():
    global static_options
    if static_options and (not static_options.rebuild or not static_options.clean):
        p4a_util.suggest("You may try running again with --clean --rebuild --reconf all")


def exec_and_deal_with_errors(fun):
    """Execute a function and deal with some errors that may happen by
    enclosing in try/catch"""
    try:
        # Executing a function is always a lot of fun :-)
        fun()
    except p4a_util.p4a_error:
        (t, e, tb) = sys.exc_info()
        p4a_util.error(str(e))
        p4a_util.debug("".join(traceback.format_exception(t, e, tb)))
        if e.code == -2:
            p4a_util.error("Interrupted")
        else:
            suggest_more_verbosity()
            send_report_email_if_enabled()
        sys.exit(e.code)
    except EnvironmentError: # IOError, OSError.
        (t, e, tb) = sys.exc_info()
        p4a_util.error(str(e))
        p4a_util.debug("".join(traceback.format_exception(t, e, tb)))
        suggest_more_verbosity()
        send_report_email_if_enabled()
        sys.exit(e.errno)
    except KeyboardInterrupt:
        p4a_util.error("Interrupted")
        sys.exit(-2)
    except SystemExit:
        raise
    except: # All other exceptions.
        (t, e, tb) = sys.exc_info()
        p4a_util.error("Unhandled " + e.__class__.__name__ + ": " + str(e))
        p4a_util.error("".join(traceback.format_exception(t, e, tb)))
        suggest_more_verbosity()
        send_report_email_if_enabled()
        sys.exit(255)



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
