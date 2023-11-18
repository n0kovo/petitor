#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import re
import os
from time import localtime, gmtime, strftime, sleep, time
from functools import reduce
from operator import mul, itemgetter
from select import select
from itertools import islice, cycle
import string
import random
from decimal import Decimal
from base64 import b64encode
from datetime import timedelta, datetime
import subprocess
import hashlib
import multiprocessing
import signal
import glob
from xml.sax.saxutils import escape as xmlescape, quoteattr as xmlquoteattr
from binascii import hexlify, unhexlify
from queue import Empty, Full
from urllib.parse import quote, urlencode, urlparse, urlunparse, quote_plus, unquote
from io import StringIO
from sys import maxsize as maxint
from IPy import IP
from multiprocessing.managers import SyncManager
import html
import logging
from optparse import OptionParser
from optparse import OptionGroup
from optparse import IndentedHelpFormatter
import pycurl
from http.server import BaseHTTPRequestHandler
from multiprocessing import set_start_method


__author__  = 'Sebastien Macke'
__email__   = 'petitor@hsc.fr'
__url__     = 'http://www.hsc.fr/ressources/outils/petitor/'
__git__     = 'https://github.com/lanjelot/petitor'
__twitter__ = 'https://twitter.com/lanjelot'
__version__ = '1.0'
__license__ = 'GPLv2'
__pyver__   = '%d.%d.%d' % sys.version_info[0:3]
__banner__  = 'Petitor %s (%s) with python-%s' % (__version__, __git__, __pyver__)


# logging {{{
class Logger:
  def __init__(self, queue):
    self.queue = queue
    self.name = multiprocessing.current_process().name

  def send(self, action, *args):
    self.queue.put((self.name, action, args))

  def quit(self):
    self.send('quit')

  def headers(self):
    self.send('headers')

  def result(self, *args):
    self.send('result', *args)

  def save_response(self, *args):
    self.send('save_response', *args)

  def save_hit(self, *args):
    self.send('save_hit', *args)

  def setLevel(self, level):
    self.send('setLevel', level)

  def warn(self, msg):
    self.send('warning', msg)

  def info(self, msg):
    self.send('info', msg)

  def debug(self, msg):
    self.send('debug', msg)


class TXTFormatter(logging.Formatter):
  def __init__(self, indicatorsfmt):
    self.resultfmt = '%(asctime)s %(name)-7s %(levelname)7s - ' + ' '.join('%%(%s)%ss' % (k, v) for k, v in indicatorsfmt) + ' | %(candidate)-34s | %(num)5s | %(mesg)s'

    super(TXTFormatter, self).__init__(datefmt='%H:%M:%S')

  def format(self, record):
    if not record.msg or record.msg == 'headers':
      fmt = self.resultfmt

    else:
      if record.levelno == logging.DEBUG:
        fmt = '%(asctime)s %(name)-7s %(levelname)7s [%(pname)s] %(message)s'
      else:
        fmt = '%(asctime)s %(name)-7s %(levelname)7s - %(message)s'


    self._style._fmt = fmt


    pp = {}
    for k, v in record.__dict__.items():
      if k in ['candidate', 'mesg']:
        pp[k] = repr23(v)
      else:
        pp[k] = v

    return super(TXTFormatter, self).format(logging.makeLogRecord(pp))

class CSVFormatter(logging.Formatter):
  def __init__(self, indicatorsfmt):
    fmt = '%(asctime)s,%(levelname)s,'+','.join('%%(%s)s' % name for name, _ in indicatorsfmt)+',%(candidate)s,%(num)s,%(mesg)s'

    super(CSVFormatter, self).__init__(fmt=fmt, datefmt='%H:%M:%S')

  def format(self, record):
    pp = {}
    for k, v in record.__dict__.items():
      if k in ['candidate', 'mesg']:
        pp[k] = '"%s"' % v.replace('"', '""')
      else:
        pp[k] = v

    return super(CSVFormatter, self).format(logging.makeLogRecord(pp))

class XMLFormatter(logging.Formatter):
  def __init__(self, indicatorsfmt):
    fmt = '''<result time="%(asctime)s" level="%(levelname)s">
''' + '\n'.join('  <{0}>%({1})s</{0}>'.format(name.replace(':', '_'), name) for name, _ in indicatorsfmt) + '''
  <candidate>%(candidate)s</candidate>
  <num>%(num)s</num>
  <mesg>%(mesg)s</mesg>
  <target %(target)s/>
</result>'''

    super(XMLFormatter, self).__init__(fmt=fmt, datefmt='%H:%M:%S')

  def format(self, record):
    pp = {}
    for k, v in record.__dict__.items():
      if isinstance(v, str):
        pp[k] = xmlescape(v)
      else:
        pp[k] = v

    return super(XMLFormatter, self).format(logging.makeLogRecord(pp))

class MsgFilter(logging.Filter):

  def filter(self, record):
    if record.msg:
      return 0
    else:
      return 1

def process_logs(queue, indicatorsfmt, argv, log_dir, runtime_file, csv_file, xml_file, hits_file):

  ignore_ctrlc()

  logging._levelToName[logging.ERROR] = 'FAIL'
  encoding = 'latin1'


  handler_out = logging.StreamHandler()
  handler_out.setFormatter(TXTFormatter(indicatorsfmt))

  logger = logging.getLogger('petitor')
  logger.setLevel(logging.DEBUG)
  logger.addHandler(handler_out)

  names = [name for name, _ in indicatorsfmt] + ['candidate', 'num', 'mesg']

  if runtime_file or log_dir:
    runtime_log = os.path.join(log_dir or '', runtime_file or 'RUNTIME.log')

    with open(runtime_log, 'a') as f:
      f.write('$ %s\n' % ' '.join(argv))

    handler_log = logging.FileHandler(runtime_log, encoding=encoding)
    handler_log.setFormatter(TXTFormatter(indicatorsfmt))

    logger.addHandler(handler_log)

  if csv_file or log_dir:
    results_csv = os.path.join(log_dir or '', csv_file or 'RESULTS.csv')

    if not os.path.exists(results_csv):
      with open(results_csv, 'w') as f:
        f.write('time,level,%s\n' % ','.join(names))

    handler_csv = logging.FileHandler(results_csv, encoding=encoding)
    handler_csv.addFilter(MsgFilter())
    handler_csv.setFormatter(CSVFormatter(indicatorsfmt))

    logger.addHandler(handler_csv)

  if xml_file or log_dir:
    results_xml = os.path.join(log_dir or '', xml_file or 'RESULTS.xml')

    if not os.path.exists(results_xml):
      with open(results_xml, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<root>\n')
        f.write('<start utc=%s local=%s/>\n' % (xmlquoteattr(strfutctime()), xmlquoteattr(strflocaltime())))
        f.write('<cmdline>%s</cmdline>\n' % xmlescape(' '.join(argv)))
        f.write('<module>%s</module>\n' % xmlescape(argv[0]))
        f.write('<options>\n')

        i = 0
        del argv[0]
        while i < len(argv):
          arg = argv[i]
          if arg[0] == '-':
            if arg in ('-d', '--debug', '--allow-ignore-failures', '-y'):
              f.write('  <option type="global" name=%s/>\n' % xmlquoteattr(arg))
            else:
              if not arg.startswith('--') and len(arg) > 2:
                name, value = arg[:2], arg[2:]
              elif '=' in arg:
                name, value = arg.split('=', 1)
              else:
                name, value = arg, argv[i+1]
                i += 1
              f.write('  <option type="global" name=%s>%s</option>\n' % (xmlquoteattr(name), xmlescape(value)))
          else:
            name, value = arg.split('=', 1)
            f.write('  <option type="module" name=%s>%s</option>\n' % (xmlquoteattr(name), xmlescape(value)))
          i += 1
        f.write('</options>\n')
        f.write('<results>\n')

    else:  # remove "</results>...</root>"
      with open(results_xml, 'r+b') as f:
        offset = f.read().find(b'</results>')
        if offset != -1:
          f.seek(offset)
          f.truncate(f.tell())

    handler_xml = logging.FileHandler(results_xml, encoding=encoding)
    handler_xml.addFilter(MsgFilter())
    handler_xml.setFormatter(XMLFormatter(indicatorsfmt))

    logger.addHandler(handler_xml)

  if hits_file:
    if os.path.exists(hits_file):
      os.rename(hits_file, hits_file + '.' + strftime("%Y%m%d%H%M%S", localtime()))

  while True:

    pname, action, args = queue.get()

    if action == 'quit':
      if xml_file or log_dir:
        with open(results_xml, 'a') as f:
          f.write('</results>\n<stop utc=%s local=%s/>\n</root>\n' % (xmlquoteattr(strfutctime()), xmlquoteattr(strflocaltime())))
      break

    elif action == 'headers':

      logger.info(' '*77)
      logger.info('headers', extra=dict((n, n) for n in names))
      logger.info('-'*77)

    elif action == 'result':

      typ, resp, candidate, num = args

      results = [(name, value) for (name, _), value in zip(indicatorsfmt, resp.indicators())]
      results += [('candidate', candidate), ('num', num), ('mesg', str(resp)), ('target', resp.str_target())]

      if typ == 'fail':
        logger.error(None, extra=dict(results))
      else:
        logger.info(None, extra=dict(results))

    elif action == 'save_response':

      resp, num = args

      if log_dir:
        filename = '%d_%s' % (num, '-'.join(map(str, resp.indicators())))
        with open('%s.txt' % os.path.join(log_dir, filename), 'wb') as f:
          f.write(resp.dump())

    elif action == 'save_hit':
      if hits_file:
        with open(hits_file, 'ab') as f:
          f.write(b(args[0] +'\n'))

    elif action == 'setLevel':
      logger.setLevel(args[0])

    else:  # 'warn', 'info', 'debug'
      getattr(logger, action)(args[0], extra={'pname': pname})

# }}}



def b(x):
  if isinstance(x, bytes):
    return x
  else:
    return x.encode('ISO-8859-1', errors='ignore')

def B(x):
  if isinstance(x, str):
    return x
  else:
    return x.decode('ISO-8859-1', errors='ignore')

notfound = []

# utils {{{
def expand_path(s):
    return os.path.expandvars(os.path.expanduser(s))

def strfutctime():
  return strftime("%Y-%m-%d %H:%M:%S", gmtime())

def strflocaltime():
  return strftime("%Y-%m-%d %H:%M:%S %Z", localtime())


def build_logdir(opt_dir, opt_auto, assume_yes):
    if opt_auto:
      return create_time_dir(opt_dir or '/tmp/petitor', opt_auto)
    elif opt_dir:
      return create_dir(opt_dir, assume_yes)
    else:
      return None

def create_dir(top_path, assume_yes):
  top_path = os.path.abspath(top_path)
  if os.path.isdir(top_path):
    files = os.listdir(top_path)
    if files:
      if assume_yes or input("Directory '%s' is not empty, do you want to wipe it ? [Y/n]: " % top_path) != 'n':
        for root, dirs, files in os.walk(top_path):
          if dirs:
            print("Directory '%s' contains sub-directories, safely aborting..." % root)
            sys.exit(0)
          for f in files:
            os.unlink(os.path.join(root, f))
          break
  else:
    os.mkdir(top_path)
  return top_path

def create_time_dir(top_path, desc):
  now = localtime()
  date, time = strftime('%Y-%m-%d', now), strftime('%H%M%S', now)
  top_path = os.path.abspath(top_path)
  date_path = os.path.join(top_path, date)
  time_path = os.path.join(top_path, date, time + '_' + desc)

  if not os.path.isdir(top_path):
    os.makedirs(top_path)
  if not os.path.isdir(date_path):
    os.mkdir(date_path)
  if not os.path.isdir(time_path):
    os.mkdir(time_path)

  return time_path

def pprint_seconds(seconds, fmt):
  return fmt % reduce(lambda x, y: divmod(x[0], y) + x[1:], [(seconds,), 60, 60])

def repr23(s):
  if all(True if 0x20 <= ord(c) < 0x7f else False for c in s):
    return s

  return repr(s.encode('latin1'))[1:]


def md5hex(plain):
  return hashlib.md5(plain).hexdigest()

def sha1hex(plain):
  return hashlib.sha1(plain).hexdigest()

def html_unescape(s):
  return html.unescape(s)

def count_lines(filename):
  with open(filename, 'rb') as f:
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read

    buf = read_f(buf_size)
    while buf:
      lines += buf.count(b'\n')
      buf = read_f(buf_size)

    return lines

# I rewrote itertools.product to avoid memory over-consumption when using large wordlists
def product(xs, *rest):
  if len(rest) == 0:
    for x in xs:
      yield [x]
  else:
    for head in xs:
      for tail in product(*rest):
        yield [head] + tail

class chain:
  def __init__(self, *iterables):
    self.iterables = iterables

  def __iter__(self):
    for iterable in self.iterables:
      for element in iterable:
        yield element

class FileIter:
  def __init__(self, filename):
    self.filename = filename

  def __iter__(self):
    return open(self.filename, 'rb')

def padhex(d):
  x = '%x' % d
  return '0' * (len(x) % 2) + x

# These are examples. You can easily write your own iterator to fit your needs.
# Or using the PROG keyword, you can call an external program such as:
#   - seq(1) from coreutils
#   - http://hashcat.net/wiki/doku.php?id=maskprocessor
#   - john -stdout -i
# For example:
# $ ./dummy_test data=PROG0 0='seq 1 80'
# $ ./dummy_test data=PROG0 0='mp64.bin ?l?l?l',$(mp64.bin --combination ?l?l?l)
class RangeIter:

  def __init__(self, typ, rng, random=None):
    if typ not in ['hex', 'int', 'float', 'letters', 'lower', 'lowercase', 'upper', 'uppercase']:
      raise ValueError('Incorrect range type %r' % typ)

    if typ in ('hex', 'int', 'float'):

      m = re.match('(-?[^-]+)-(-?[^-]+)$', rng) # 5-50 or -5-50 or 5--50 or -5--50
      if not m:
        raise ValueError('Unsupported range %r' % rng)

      mn = m.group(1)
      mx = m.group(2)

      if typ in ('hex', 'int'):

        mn = int(mn, 16 if '0x' in mn else 10)
        mx = int(mx, 16 if '0x' in mx else 10)

        if typ == 'hex':
          fmt = padhex
        elif typ == 'int':
          fmt = '%d'

      elif typ == 'float':
        mn = Decimal(mn)
        mx = Decimal(mx)

      if mn > mx:
        step = -1
      else:
        step = 1

    elif typ == 'letters':
      charset = [c for c in string.ascii_letters]

    elif typ in ('lower', 'lowercase'):
      charset = [c for c in string.ascii_lowercase]

    elif typ in ('upper', 'uppercase'):
      charset = [c for c in string.ascii_uppercase]

    def zrange(start, stop, step, fmt):
      x = start
      while x != stop+step:

        if callable(fmt):
          yield fmt(x)
        else:
          yield fmt % x
        x += step

    def letterrange(first, last, charset):
      for k in range(len(last)):
        for x in product(*[chain(charset)]*(k+1)):
          result = ''.join(x)
          if first:
            if first != result:
              continue
            else:
              first = None
          yield result
          if result == last:
            return

    if typ == 'float':
      precision = max(len(str(x).partition('.')[-1]) for x in (mn, mx))

      fmt = '%%.%df' % precision
      exp = 10**precision
      step *= Decimal(1) / exp

      self.generator = zrange, (mn, mx, step, fmt)
      self.size = int(abs(mx-mn) * exp) + 1

      def random_generator():
        while True:
          yield fmt % (Decimal(random.randint(mn*exp, mx*exp)) / exp)

    elif typ in ('hex', 'int'):
      self.generator = zrange, (mn, mx, step, fmt)
      self.size = abs(mx-mn) + 1

      def random_generator():
        while True:
          yield fmt % random.randint(mn, mx)

    else: # letters, lower, upper
      def count(f):
        total = 0
        i = 0
        for c in f[::-1]:
          z = charset.index(c) + 1
          total += (len(charset)**i)*z
          i += 1
        return total + 1

      first, last = rng.split('-')
      self.generator = letterrange, (first, last, charset)
      self.size = count(last) - count(first) + 1

    if random:
      self.generator = random_generator, ()
      self.size = maxint

  def __iter__(self):
    fn, args = self.generator
    return fn(*args)

  def __len__(self):
    return self.size

class ProgIter:
  def __init__(self, prog):
    self.prog = prog

  def __iter__(self):
    p = subprocess.Popen(self.prog.split(' '), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.stdout

class Progress:
  def __init__(self):
    self.current = ''
    self.done_count = 0
    self.hits_count = 0
    self.skip_count = 0
    self.fail_count = 0
    self.seconds = [1]*25 # avoid division by zero early bug condition

class TimeoutError(Exception):
  pass

def ignore_ctrlc():
  signal.signal(signal.SIGINT, signal.SIG_IGN)

def handle_alarm():
  signal.signal(signal.SIGALRM, raise_timeout)

def raise_timeout(signum, frame):
  if signum == signal.SIGALRM:
    raise TimeoutError('timed out')

def enable_alarm(timeout):
  signal.alarm(timeout)

def disable_alarm():
  signal.alarm(0)

# SyncManager.start(initializer) only available since python2.7
class MyManager(SyncManager):
  @classmethod
  def _run_server(cls, registry, address, authkey, serializer, writer, initializer=None, initargs=()):
    ignore_ctrlc()
    super(MyManager, cls)._run_server(registry, address, authkey, serializer, writer)

def ppstr(s):
  if isinstance(s, bytes):
    s = B(s)
  if not isinstance(s, str):
    s = str(s)
  return s.rstrip('\r\n')

def flatten(l):
  r = []
  for x in l:
    if isinstance(x, (list, tuple)):
      r.extend(map(ppstr, x))
    else:
      r.append(ppstr(x))
  return r

def parse_query(qs, keep_blank_values=False, encoding='utf-8', errors='replace'):
  '''Same as urllib.parse.parse_qsl but without replacing '+' with ' '
  '''
  pairs = [s2 for s1 in qs.split('&') for s2 in s1.split(';')]
  r = []

  for name_value in pairs:
    if not name_value:
      continue
    nv = name_value.split('=', 1)
    if len(nv) != 2:
      if keep_blank_values:
        nv.append('')
      else:
        continue
    if len(nv[1]) or keep_blank_values:
      name = unquote(nv[0], encoding=encoding, errors=errors)
      value = unquote(nv[1], encoding=encoding, errors=errors)
      r.append((name, value))
  return r

# }}}

# Controller {{{
class Controller:

  builtin_actions = (
    ('ignore', 'do not report'),
    ('retry', 'try payload again'),
    ('skip', 'stop testing the same keyword value'),
    ('free', 'stop testing the same option value'),
    ('quit', 'terminate execution now'),
    )

  available_encodings = {
    'hex': (lambda s: B(hexlify(s)), 'encode in hexadecimal'),
    'unhex': (lambda s: B(unhexlify(s)), 'decode from hexadecimal'),
    'b64': (lambda s: B(b64encode(b(s))), 'encode in base64'),
    'md5': (md5hex, 'hash in md5'),
    'sha1': (sha1hex, 'hash in sha1'),
    'url': (quote_plus, 'url encode'),
    }

  def expand_key(self, arg):
    yield arg.split('=', 1)

  def find_file_keys(self, value):
    return map(int, re.findall(r'FILE(\d)', value))

  def find_net_keys(self, value):
    return map(int, re.findall(r'NET(\d)', value))

  def find_combo_keys(self, value):
    return [map(int, t) for t in re.findall(r'COMBO(\d)(\d)', value)]

  def find_module_keys(self, value):
    return map(int, re.findall(r'MOD(\d)', value))

  def find_range_keys(self, value):
    return map(int, re.findall(r'RANGE(\d)', value))

  def find_prog_keys(self, value):
    return map(int, re.findall(r'PROG(\d)', value))

  def usage_parser(self, name):
    class MyHelpFormatter(IndentedHelpFormatter):
      def format_epilog(self, epilog):
        return epilog

      def format_heading(self, heading):
        if self.current_indent == 0 and heading == 'Options':
          heading = 'Global options'
        return "%*s%s:\n" % (self.current_indent, "", heading)

      def format_usage(self, usage):
        return '%s\nUsage: %s\n' % (__banner__, usage)

    available_actions = self.builtin_actions + self.module.available_actions
    available_conditions = self.module.Response.available_conditions

    usage = '''%%prog <module-options ...> [global-options ...]

Examples:
  %s''' % '\n  '.join(self.module.usage_hints)

    usage += '''

Module options:
%s ''' % ('\n'.join('  %-14s: %s' % (k, v) for k, v in self.module.available_options))

    epilog = '''
Syntax:
 -x actions:conditions

    actions    := action[,action]*
    action     := "%s"
    conditions := condition=value[,condition=value]*
    condition  := "%s"
''' % ('" | "'.join(k for k, v in available_actions),
       '" | "'.join(k for k, v in available_conditions))

    epilog += '''
%s

%s
''' % ('\n'.join('    %-12s: %s' % (k, v) for k, v in available_actions),
       '\n'.join('    %-12s: %s' % (k, v) for k, v in available_conditions))

    epilog += '''
For example, to ignore all redirects to the home page:
... -x ignore:code=302,fgrep='Location: /home.html'

 -e tag:encoding

    tag        := any unique string (eg. T@G or _@@_ or ...)
    encoding   := "%s"

%s''' % ('" | "'.join(k for k in self.available_encodings),
        '\n'.join('    %-12s: %s' % (k, v) for k, (f, v) in self.available_encodings.items()))

    epilog += '''

For example, to encode every password in base64:
... host=10.0.0.1 user=admin password=_@@_FILE0_@@_ -e _@@_:b64

Please read the README inside for more examples and usage information.
'''

    parser = OptionParser(usage=usage, prog=name, epilog=epilog, version=__banner__, formatter=MyHelpFormatter())

    exe_grp = OptionGroup(parser, 'Execution')
    exe_grp.add_option('-x', dest='actions', action='append', default=[], metavar='arg', help='actions and conditions, see Syntax below')
    exe_grp.add_option('--start', dest='start', type='int', default=0, metavar='N', help='start from offset N in the product of all payload sets')
    exe_grp.add_option('--stop', dest='stop', type='int', default=None, metavar='N', help='stop at offset N')
    exe_grp.add_option('--resume', dest='resume', metavar='r1[,rN]*', help='resume previous run')
    exe_grp.add_option('-e', dest='encodings', action='append', default=[], metavar='arg', help='encode everything between two tags, see Syntax below')
    exe_grp.add_option('-C', dest='combo_delim', default=':', metavar='str', help="delimiter string in combo files (default is ':')")
    exe_grp.add_option('-X', dest='condition_delim', default=',', metavar='str', help="delimiter string in conditions (default is ',')")
    exe_grp.add_option('--allow-ignore-failures', dest='allow_ignore_failures', action='store_true', help="failures cannot be ignored with -x (this is by design to avoid false negatives) this option overrides this safeguard")
    exe_grp.add_option('-y', dest='assume_yes', action='store_true', help="automatically answer yes for all questions")

    opt_grp = OptionGroup(parser, 'Optimization')
    opt_grp.add_option('--rate-limit', dest='rate_limit', type='float', default=0, metavar='N', help='wait N seconds between each attempt (default is 0)')
    opt_grp.add_option('--timeout', dest='timeout', type='int', default=0, metavar='N', help='wait N seconds for a response before retrying payload (default is 0)')
    opt_grp.add_option('--max-retries', dest='max_retries', type='int', default=4, metavar='N', help='skip payload after N retries (default is 4) (-1 for unlimited)')
    opt_grp.add_option('-t', '--threads', dest='num_threads', type='int', default=10, metavar='N', help='number of threads (default is 10)')
    opt_grp.add_option('--groups', dest='groups', default='', metavar='', help="default is to iterate over the cartesian product of all payload sets, use this option to iterate over sets simultaneously instead (aka pitchfork), see syntax inside (default is '0,1..n')")

    log_grp = OptionGroup(parser, 'Logging')
    log_grp.add_option('-l', dest='log_dir', metavar='DIR', help="save output and response data into DIR ")
    log_grp.add_option('-L', dest='auto_log', metavar='SFX', help="automatically save into DIR/yyyy-mm-dd/hh:mm:ss_SFX (DIR defaults to '/tmp/petitor')")
    log_grp.add_option('-R', dest='runtime_file', metavar='FILE', help="save output to FILE")
    log_grp.add_option('--csv', dest='csv_file', metavar='FILE', help="save CSV results to FILE")
    log_grp.add_option('--xml', dest='xml_file', metavar='FILE', help="save XML results to FILE")
    log_grp.add_option('--hits', dest='hits_file', metavar='FILE', help="save found candidates to FILE")

    dbg_grp = OptionGroup(parser, 'Debugging')
    dbg_grp.add_option('-d', '--debug', dest='debug', action='store_true', help='enable debug messages')
    dbg_grp.add_option('--auto-progress', dest='auto_progress', type='int', default=0, metavar='N', help='automatically display progress every N seconds')

    parser.option_groups.extend([exe_grp, opt_grp, log_grp, dbg_grp])

    return parser

  def parse_usage(self, argv):
    parser = self.usage_parser(argv[0])
    opts, args = parser.parse_args(argv[1:])

    if not len(args) > 0:
      parser.print_usage()
      print('ERROR: wrong usage. Please read the README inside for more information.')
      sys.exit(2)

    return opts, args

  def __init__(self, module, argv):
    self.thread_report = []
    self.thread_progress = []

    self.payload = {}
    self.iter_keys = {}
    self.iter_groups = {}
    self.enc_keys = []

    self.module = module

    opts, args = self.parse_usage(argv)

    self.combo_delim = opts.combo_delim
    self.condition_delim = opts.condition_delim
    self.rate_limit = opts.rate_limit
    self.timeout = opts.timeout
    self.max_retries = opts.max_retries
    self.num_threads = opts.num_threads
    self.start, self.stop = opts.start, opts.stop
    self.allow_ignore_failures = opts.allow_ignore_failures
    self.auto_progress = opts.auto_progress
    self.auto_progress_next = None

    self.resume = [int(i) for i in opts.resume.split(',')] if opts.resume else None

    manager = MyManager()
    manager.start()

    self.ns = manager.Namespace()
    self.ns.actions = {}
    self.ns.free_list = []
    self.ns.skip_list = []
    self.ns.paused = False
    self.ns.quit_now = False
    self.ns.start_time = 0
    self.ns.total_size = 1

    log_queue = multiprocessing.Queue()

    logsvc = multiprocessing.Process(name='LogSvc', target=process_logs, args=(log_queue, module.Response.indicatorsfmt, argv, build_logdir(opts.log_dir, opts.auto_log, opts.assume_yes), opts.runtime_file, opts.csv_file, opts.xml_file, opts.hits_file))
    logsvc.daemon = True
    logsvc.start()

    global logger
    logger = Logger(log_queue)

    if opts.debug:
      logger.setLevel(logging.DEBUG)
    else:
      logger.setLevel(logging.INFO)

    wlists = {}
    kargs = []
    for arg in args: # ('host=NET0', '0=10.0.0.0/24', 'user=COMBO10', 'password=COMBO11', '1=combos.txt', 'name=google.MOD2', '2=TLD')
      logger.debug('arg: %r' % arg)
      for k, v in self.expand_key(arg):
        logger.debug('k: %s, v: %s' % (k, v))

        if k.isdigit():
          wlists[k] = v

        else:
          if v.startswith('@'):
            p = expand_path(v[1:])
            with open(p, 'rb') as f:
              v = B(f.read())

          kargs.append((k, v))

    iter_vals = [v for k, v in sorted(wlists.items())]

    logger.debug('kargs: %s' % kargs) # [('host', 'NET0'), ('user', 'COMBO10'), ('password', 'COMBO11'), ('name', 'google.MOD2')]
    logger.debug('iter_vals: %s' % iter_vals) # ['10.0.0.0/24', 'combos.txt', 'TLD']

    for k, v in kargs:

      for e in opts.encodings:
        meta, enc = e.split(':')
        if re.search(r'{0}.+?{0}'.format(meta), v):
          self.enc_keys.append((k, meta, self.available_encodings[enc][0]))

      for i in self.find_file_keys(v):
        if i not in self.iter_keys:
          self.iter_keys[i] = ('FILE', iter_vals[i], [])
        self.iter_keys[i][2].append(k)

      else:
        for i in self.find_net_keys(v):
          if i not in self.iter_keys:
            self.iter_keys[i] = ('NET', iter_vals[i], [])
          self.iter_keys[i][2].append(k)

        else:
          for i, j in self.find_combo_keys(v):
            if i not in self.iter_keys:
              self.iter_keys[i] = ('COMBO', iter_vals[i], [])
            self.iter_keys[i][2].append((j, k))

          else:
            for i in self.find_module_keys(v):
              if i not in self.iter_keys:
                self.iter_keys[i] = ('MOD', iter_vals[i], [])
              self.iter_keys[i][2].append(k)

            else:
              for i in self.find_range_keys(v):
                if i not in self.iter_keys:
                  self.iter_keys[i] = ('RANGE', iter_vals[i], [])
                self.iter_keys[i][2].append(k)

              else:
                for i in self.find_prog_keys(v):
                  if i not in self.iter_keys:
                    self.iter_keys[i] = ('PROG', iter_vals[i], [])
                  self.iter_keys[i][2].append(k)

                else:
                  self.payload[k] = v

    if self.iter_keys:
      if not opts.groups:
        # default is to iterate over the cartesian product of all payload sets
        opts.groups = ','.join(map(str, self.iter_keys))

      for i, g in enumerate(opts.groups.split(':')):
        ks = list(map(int, g.split(',')))
        for k in ks:
          if k not in self.iter_keys:
            raise ValueError('Unknown keyword number %r' % k)

        self.iter_groups[i] = sorted(ks)

    logger.debug('iter_groups: %s' % self.iter_groups) # {0: [0, 1], 1: [2]}
    logger.debug('iter_keys: %s' % self.iter_keys) # [(0, ('NET', '10.0.0.0/24', ['host'])), (1, ('COMBO', 'combos.txt', [(0, 'user'), (1, 'password')])), (2, ('MOD', 'TLD', ['name']))]
    logger.debug('enc_keys: %s' % self.enc_keys) # [('password', 'ENC', hex), ('header', 'B64', b64encode), ...
    logger.debug('payload: %s' % self.payload) # {'host': 'NET0', 'user': 'COMBO10', 'password': 'COMBO11', 'name': 'google.MOD2'}

    self.iter_groups = sorted(self.iter_groups.items())
    self.iter_keys = sorted(self.iter_keys.items())
    self.available_actions = [k for k, _ in self.builtin_actions + self.module.available_actions]
    self.module_actions = [k for k, _ in self.module.available_actions]

    for x in opts.actions:
      self.update_actions(x)

    logger.debug('actions: %s' % self.ns.actions)

  def update_actions(self, arg):
    ns_actions = self.ns.actions

    actions, conditions = arg.split(':', 1)
    for action in actions.split(','):

      conds = [c.split('=', 1) for c in conditions.split(self.condition_delim)]

      if '=' in action:
        name, opts = action.split('=')
      else:
        name, opts = action, None

      if name not in self.available_actions:
        raise ValueError('Unsupported action %r' % name)

      if name not in ns_actions:
        ns_actions[name] = []

      ns_actions[name].append((conds, opts))

    self.ns.actions = ns_actions

  def lookup_actions(self, resp):
    actions = {}
    for action, conditions in self.ns.actions.items():
      for condition, opts in conditions:
        for key, val in condition:
          if key[-1] == '!':
            if resp.match(key[:-1], val):
              break
          else:
            if not resp.match(key, val):
              break
        else:
          actions[action] = opts
    return actions

  def should_free(self, payload):
    # free_list: [[('host', '10.0.0.1')], [('user', 'anonymous')], [('host', '10.0.0.7'),('user','test')], ...
    for l in self.ns.free_list:
      for k, v in l:
        if payload[k] != v:
          break
      else:
        return True

    return False

  def register_free(self, payload, opts):
    self.ns.free_list += [[(k, payload[k]) for k in opts.split('+')]]
    logger.debug('free_list updated: %s' % self.ns.free_list)

  def should_skip(self, prod):
    # skip_list: [[(0, '10.0.0.1')], [(1, 'anonymous')], [(0, '10.0.0.7'), (1, 'test')], ...
    for l in self.ns.skip_list:
      for k, v in l:
        if prod[k] != v:
          break
      else:
        return True

    return False

  def register_skip(self, prod, opts):
    self.ns.skip_list += [[(k, prod[k]) for k in map(int, opts.split('+'))]]
    logger.debug('skip_list updated: %s' % self.ns.skip_list)

  def fire(self):
    logger.info('Starting %s at %s' % (__banner__, strftime('%Y-%m-%d %H:%M %Z', localtime())))

    try:
      self.start_threads()
      self.monitor_progress()
    except KeyboardInterrupt:
      pass
    except:
      logging.exception(sys.exc_info()[1])
    finally:
      self.ns.quit_now = True

    try:
      # waiting for reports enqueued by consumers to be flushed
      while True:
        active = multiprocessing.active_children()
        self.report_progress()
        if not len(active) > 2: # SyncManager and LogSvc
          break
        logger.debug('active: %s' % active)
        sleep(.1)
    except KeyboardInterrupt:
      pass

    if self.ns.total_size >= maxint:
      total_size = -1
    else:
      total_size = self.ns.total_size

    total_time = time() - self.ns.start_time

    hits_count = sum(p.hits_count for p in self.thread_progress)
    done_count = sum(p.done_count for p in self.thread_progress)
    skip_count = sum(p.skip_count for p in self.thread_progress)
    fail_count = sum(p.fail_count for p in self.thread_progress)

    speed_avg = done_count / total_time

    self.show_final()

    logger.info('Hits/Done/Skip/Fail/Size: %d/%d/%d/%d/%d, Avg: %d r/s, Time: %s' % (
      hits_count, done_count, skip_count, fail_count, total_size, speed_avg,
      pprint_seconds(total_time, '%dh %dm %ds')))

    if done_count < total_size:
      resume = []
      for i, p in enumerate(self.thread_progress):
        c = p.done_count + p.skip_count
        if self.resume:
          if i < len(self.resume):
            c += self.resume[i]
        resume.append(str(c))

      logger.info('To resume execution, pass --resume %s' % ','.join(resume))

    logger.quit()
    while len(multiprocessing.active_children()) > 1:
      sleep(.1)

  def push_final(self, resp): pass
  def show_final(self): pass

  def start_threads(self):

    task_queues = [multiprocessing.Queue(maxsize=10000) for _ in range(self.num_threads)]

    # consumers
    for num in range(self.num_threads):
      report_queue = multiprocessing.Queue(maxsize=1000)
      t = multiprocessing.Process(name='Consumer-%d' % num, target=self.consume, args=(task_queues[num], report_queue, logger.queue))
      t.daemon = True
      t.start()
      self.thread_report.append(report_queue)
      self.thread_progress.append(Progress())

    # producer
    t = multiprocessing.Process(name='Producer', target=self.produce, args=(task_queues, logger.queue))
    t.daemon = True
    t.start()

  def produce(self, task_queues, log_queue):

    ignore_ctrlc()

    global logger
    logger = Logger(log_queue)

    def abort(msg):
      logger.warn(msg)
      self.ns.quit_now = True

    psets = {}
    for k, (t, v, _) in self.iter_keys:

      pset = []
      size = 0

      if t in ('FILE', 'COMBO'):
        for name in v.split(','):
          for fpath in sorted(glob.iglob(expand_path(name))):
            if not os.path.isfile(fpath):
              return abort("No such file '%s'" % fpath)

            pset.append(FileIter(fpath))
            size += count_lines(fpath)

      elif t == 'NET':
        pset = [IP(n, make_net=True) for n in v.split(',')]
        size = sum(len(subnet) for subnet in pset)

      elif t == 'MOD':
        elements, size = self.module.available_keys[v]()
        pset = [elements]

      elif t == 'RANGE':
        for r in v.split(','):
          typ, opt = r.split(':', 1)

          try:
            ri = RangeIter(typ, opt)
            size += len(ri)
            pset.append(ri)
          except ValueError as e:
            return abort("Invalid range '%s' of type '%s', %s" % (opt, typ, e))

      elif t == 'PROG':
        m = re.match(r'(.+),(\d+)$', v)
        if m:
          prog, size = m.groups()
        else:
          prog, size = v, maxint

        logger.debug('prog: %s, size: %s' % (prog, size))

        pset = [ProgIter(prog)]
        size = int(size)

      else:
        return abort('Incorrect keyword %r' % t)

      psets[k] = chain(*pset), size

    logger.debug('payload sets: %r' % psets)

    zipit = []
    if not psets:
      total_size = 1
      zipit.append([''])

    else:
      group_sizes = {}
      for i, ks in self.iter_groups:
        group_sizes[i] = reduce(mul, (size for _, size in [psets[k] for k in ks]))

      logger.debug('group_sizes: %s' % group_sizes)

      total_size = max(group_sizes.values())
      biggest, _ = max(group_sizes.items(), key=itemgetter(1))

      for i, ks in self.iter_groups:

        r = []
        for k in ks:
          pset, _ = psets[k]
          r.append(pset)

        it = product(*r)
        if i != biggest:
          it = cycle(it)

        zipit.append(it)

    logger.debug('zipit: %s' % zipit)
    logger.debug('total_size: %d' % total_size)

    if self.stop and total_size > self.stop:
      total_size = self.stop - self.start
    else:
      total_size -= self.start

    if self.resume:
      total_size -= sum(self.resume)

    self.ns.total_size = total_size
    self.ns.start_time = time()

    logger.headers()

    count = 0
    for pp in islice(zip(*zipit), self.start, self.stop):

      if self.ns.quit_now:
        break

      pp = flatten(pp)
      logger.debug('pp: %s' % pp)

      prod = [''] * len(pp)
      for _, ks in self.iter_groups:
        for k in ks:
          prod[k] = pp.pop(0)

      if self.resume:
        idx = count % len(self.resume)
        off = self.resume[idx]

        if count < off * len(self.resume):
          #logger.debug('Skipping %d %s, resume[%d]: %s' % (count, ':'.join(prod), idx, self.resume[idx]))
          count += 1
          continue

      while True:
        if self.ns.quit_now:
          break

        try:
          cid = count % self.num_threads
          task_queues[cid].put_nowait(prod)
          break
        except Full:
          sleep(.1)

      count += 1

    if not self.ns.quit_now:
      for q in task_queues:
        q.put(None)

    logger.debug('producer done')

    while True:
      if self.ns.quit_now:
        for q in task_queues:
          q.cancel_join_thread()
        break
      sleep(.5)

    logger.debug('producer exits')

  def consume(self, task_queue, report_queue, log_queue):

    ignore_ctrlc()
    handle_alarm()

    global logger
    logger = Logger(log_queue)

    module = self.module()

    def shutdown():
      if hasattr(module, '__del__'):
        module.__del__()
      logger.debug('consumer done')

    while True:
      if self.ns.quit_now:
        return shutdown()

      try:
        prod = task_queue.get_nowait()
      except Empty:
        sleep(.1)
        continue

      if prod is None:
        return shutdown()

      payload = self.payload.copy()

      for i, (t, _, keys) in self.iter_keys:
        if t == 'FILE':
          for k in keys:
            payload[k] = payload[k].replace('FILE%d' % i, prod[i])
        elif t == 'NET':
          for k in keys:
            payload[k] = payload[k].replace('NET%d' % i, prod[i])
        elif t == 'COMBO':
          for j, k in keys:
            payload[k] = payload[k].replace('COMBO%d%d' % (i, j), prod[i].split(self.combo_delim, max(j for j, _ in keys))[j])
        elif t == 'MOD':
          for k in keys:
            payload[k] = payload[k].replace('MOD%d' % i, prod[i])
        elif t == 'RANGE':
          for k in keys:
            payload[k] = payload[k].replace('RANGE%d' % i, prod[i])
        elif t == 'PROG':
          for k in keys:
            payload[k] = payload[k].replace('PROG%d' % i, prod[i])

      for k, m, e in self.enc_keys:
        payload[k] = re.sub(r'{0}(.+?){0}'.format(m), lambda m: e(b(m.group(1))), payload[k])

      logger.debug('product: %s' % prod)
      prod_str = ':'.join(prod)

      if self.should_free(payload):
        logger.debug('skipping')
        report_queue.put(('skip', prod_str, None, 0))
        continue

      if self.should_skip(prod):
        logger.debug('skipping')
        report_queue.put(('skip', prod_str, None, 0))
        continue

      try_count = 0
      start_time = time()

      while True:

        while self.ns.paused and not self.ns.quit_now:
          sleep(1)

        if self.ns.quit_now:
          return shutdown()

        if self.rate_limit > 0:
          sleep(self.rate_limit)

        if try_count <= self.max_retries or self.max_retries < 0:

          actions = {}
          try_count += 1

          logger.debug('payload: %s [try %d/%d]' % (payload, try_count, self.max_retries+1))

          try:
            enable_alarm(self.timeout)
            resp = module.execute(**payload)

            disable_alarm()
          except:
            disable_alarm()

            mesg = '%s %s' % sys.exc_info()[:2]
            logger.debug('caught: %s' % mesg)

            #logging.exception(sys.exc_info()[1])

            resp = self.module.Response('xxx', mesg, timing=time()-start_time)

            if hasattr(module, 'reset'):
              module.reset()

            sleep(try_count * .1)
            continue

        else:
          actions = {'fail': None}

        actions.update(self.lookup_actions(resp))
        report_queue.put((actions, prod_str, resp, time() - start_time))

        for name in self.module_actions:
          if name in actions:
            getattr(module, name)(**payload)

        if 'free' in actions:
          self.register_free(payload, actions['free'])
          break

        if 'skip' in actions:
          self.register_skip(prod, actions['skip'])
          break

        if 'fail' in actions:
          break

        if 'quit' in actions:
          return shutdown()

        if 'retry' in actions:
          continue

        break

  def monitor_progress(self):
    # loop until SyncManager, LogSvc and Producer are the only children left alive
    while len(multiprocessing.active_children()) > 3 and not self.ns.quit_now:
      self.report_progress()
      self.monitor_interaction()

  def report_progress(self):
    for i, pq in enumerate(self.thread_report):
      p = self.thread_progress[i]

      while True:

        try:
          actions, current, resp, seconds = pq.get_nowait()
          #logger.info('actions reported: %s' % '+'.join(actions))

        except Empty:
          break

        if actions == 'skip':
          p.skip_count += 1
          continue

        if self.resume:
          offset = p.done_count + self.resume[i]
        else:
          offset = p.done_count

        offset = (offset * self.num_threads) + i + 1 + self.start

        p.current = current
        p.seconds[p.done_count % len(p.seconds)] = seconds

        if 'quit' in actions:
          self.ns.quit_now = True

        if 'fail' in actions:
          if not self.allow_ignore_failures or 'ignore' not in actions:
            logger.result('fail', resp, current, offset)

        elif 'ignore' not in actions:
          logger.result('hit', resp, current, offset)

        if 'fail' in actions:
          p.fail_count += 1

        elif 'retry' in actions:
          continue

        elif 'ignore' not in actions:
          p.hits_count += 1

          logger.save_response(resp, offset)
          logger.save_hit(current)

          self.push_final(resp)

        p.done_count += 1

  def monitor_interaction(self):

    def read_command():
      i, _, _ = select([sys.stdin], [], [], .1)
      if not i:
        return None
      command = i[0].readline().strip()

      return command

    command = read_command()

    if command is None:
      if self.auto_progress == 0:
        return

      if self.ns.paused:
        self.auto_progress_next = None
        return

      if self.auto_progress_next is None:
        self.auto_progress_next = time() + self.auto_progress
        return

      if time() < self.auto_progress_next:
        return

      self.auto_progress_next = None
      command = ''

    if command == 'h':
      logger.info('''Available commands:
       h       show help
       <Enter> show progress
       d/D     increase/decrease debug level
       p       pause progress
       f       show verbose progress
       x arg   add monitor condition
       a       show all active conditions
       q       terminate execution now
       ''')

    elif command == 'q':
      self.ns.quit_now = True

    elif command == 'p':
      self.ns.paused = not self.ns.paused
      logger.info(self.ns.paused and 'Paused' or 'Unpaused')

    elif command == 'd':
      logger.setLevel(logging.DEBUG)

    elif command == 'D':
      logger.setLevel(logging.INFO)

    elif command == 'a':
      logger.info(repr(self.ns.actions))

    elif command.startswith('x') or command.startswith('-x'):
      _, arg = command.split(' ', 1)
      try:
        self.update_actions(arg)
      except ValueError:
        logger.warn('usage: x actions:conditions')

    else: # show progress

      thread_progress = self.thread_progress
      num_threads = self.num_threads
      total_size = self.ns.total_size

      total_count = sum(p.done_count+p.skip_count for p in thread_progress)
      speed_avg = num_threads / (sum(sum(p.seconds) / len(p.seconds) for p in thread_progress) / num_threads)
      if total_size >= maxint:
        etc_time = 'inf'
        remain_time = 'inf'
      else:
        remain_seconds = (total_size - total_count) / speed_avg
        remain_time = pprint_seconds(remain_seconds, '%02d:%02d:%02d')
        etc_seconds = datetime.now() + timedelta(seconds=remain_seconds)
        etc_time = etc_seconds.strftime('%H:%M:%S')

      logger.info('Progress: {0:>3}% ({1}/{2}) | Speed: {3:.0f} r/s | ETC: {4} ({5} remaining) {6}'.format(
        total_count * 100 // total_size,
        total_count,
        total_size,
        speed_avg,
        etc_time,
        remain_time,
        self.ns.paused and '| Paused' or ''))

      if command == 'f':
        for i, p in enumerate(thread_progress):
          total_count = p.done_count + p.skip_count
          logger.info(' {0:>3}: {1:>3}% ({2}/{3}) {4}'.format(
            '#%d' % (i+1),
            int(100*total_count/(1.0*total_size/num_threads)),
            total_count,
            total_size/num_threads,
            p.current))

# }}}

# Response_Base {{{
def match_range(size, val):
  if '-' in val:
    size_min, size_max = val.split('-')

    if not size_min and not size_max:
      raise ValueError('Invalid interval')

    elif not size_min: # size == -N
      return size <= float(size_max)

    elif not size_max: # size == N-
      return size >= float(size_min)

    else:
      size_min, size_max = float(size_min), float(size_max)
      if size_min >= size_max:
        raise ValueError('Invalid interval')

      return size_min <= size <= size_max

  else:
    return size == float(val)

class Response_Base:

  available_conditions = (
    ('code', 'match status code'),
    ('size', 'match size (N or N-M or N- or -N)'),
    ('time', 'match time (N or N-M or N- or -N)'),
    ('mesg', 'match message'),
    ('fgrep', 'search for string in mesg'),
    ('egrep', 'search for regex in mesg'),
    )

  indicatorsfmt = [('code', -5), ('size', -4), ('time', 7)]

  def __init__(self, code, mesg, timing=0, trace=None):
    self.code = code
    self.mesg = mesg
    self.time = timing.time if isinstance(timing, Timing) else timing
    self.size = len(mesg)
    self.trace = trace

  def indicators(self):
    return self.code, self.size, '%.3f' % self.time

  def __str__(self):
    return self.mesg

  def match(self, key, val):
    return getattr(self, 'match_'+key)(val)

  def match_code(self, val):
    return re.match('%s$' % val, str(self.code))

  def match_size(self, val):
    return match_range(self.size, val)

  def match_time(self, val):
    return match_range(self.time, val)

  def match_mesg(self, val):
    return val == self.mesg

  def match_fgrep(self, val):
    return val in self.mesg

  def match_egrep(self, val):
    return re.search(val, self.mesg)

  def dump(self):
    return b(self.trace or str(self))

  def str_target(self):
    return ''

class Timing:
  def __enter__(self):
    self.t1 = time()
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    self.time = time() - self.t1

# }}}

# TCP_Cache {{{
class TCP_Connection:
  def __init__(self, fp, banner=None):
    self.fp = fp
    self.banner = banner

  def close(self):
    self.fp.close()

class TCP_Cache:

  available_actions = (
    ('reset', 'close current connection in order to reconnect next time'),
    )

  available_options = (
    ('persistent', 'use persistent connections [1|0]'),
    )

  def __init__(self):
    self.cache = {} # '10.0.0.1:22': ('root', conn1), '10.0.0.2:22': ('admin', conn2),
    self.curr = None

  def __del__(self):
    for _, (_, c) in self.cache.items():
      c.close()
    self.cache.clear()

  def bind(self, host, port, *args, **kwargs):

    hp = '%s:%s' % (host, port)
    key = ':'.join(map(str, args))

    if hp in self.cache:
      k, c = self.cache[hp]

      if key == k:
        self.curr = hp, k, c
        return c.fp, c.banner

      else:
        c.close()
        del self.cache[hp]

    self.curr = None

    logger.debug('connect')
    conn = self.connect(host, port, *args, **kwargs)

    self.cache[hp] = (key, conn)
    self.curr = hp, key, conn

    return conn.fp, conn.banner

  def reset(self, **kwargs):
    if self.curr:
      hp, _, c = self.curr

      c.close()
      del self.cache[hp]

      self.curr = None

# }}}

if hasattr(pycurl, 'PRIMARY_PORT'):
  proxytype_mapping = {
    'http': pycurl.PROXYTYPE_HTTP,
    'socks4': pycurl.PROXYTYPE_SOCKS4,
    'socks4a': pycurl.PROXYTYPE_SOCKS4A,
    'socks5': pycurl.PROXYTYPE_SOCKS5,
    'socks5_with_hostname': pycurl.PROXYTYPE_SOCKS5_HOSTNAME,
  }
else:
  # PRIMARY_PORT available since libcurl-7.21.0 and all PROXY_* since libcurl-7.18
  # PRIMARY_PORT and all PROXY_* available since pycurl-7.19.5.1
  notfound.append('libcurl')


class Response_HTTP(Response_Base):

  indicatorsfmt = [('code', -4), ('size:clen', -13), ('time', 6)]

  def __init__(self, code, response, timing=0, trace=None, content_length=-1, target={}):
    Response_Base.__init__(self, code, response, timing, trace=trace)
    self.content_length = content_length
    self.target = target

  def indicators(self):
    return self.code, '%d:%d' % (self.size, self.content_length), '%.3f' % self.time

  def __str__(self):
    lines = re.findall('^(HTTP/.+)$', self.mesg, re.M)
    if lines:
      return lines[-1].rstrip('\r')
    else:
      return self.mesg

  def match_clen(self, val):
    return match_range(self.content_length, val)

  def match_egrep(self, val):
    return re.search(val, self.mesg, re.M)

  def str_target(self):
    return ' '.join('%s=%s' % (k, xmlquoteattr(str(v))) for k, v in self.target.items())

  available_conditions = Response_Base.available_conditions
  available_conditions += (
    ('clen', 'match Content-Length header (N or N-M or N- or -N)'),
    )

class HTTPRequestParser(BaseHTTPRequestHandler):
  def __init__(self, fd):
    self.rfile = fd
    self.error = None
    self.body = None

    command, path, version = B(self.rfile.readline()).split()
    self.raw_requestline = b('%s %s %s' % (command, path, 'HTTP/1.1' if version.startswith('HTTP/2') else version))

    self.parse_request()
    self.request_version = version

    if self.command == 'POST':
      self.body = B(self.rfile.read(-1)).rstrip('\r\n')

      if 'Content-Length' in self.headers:
        del self.headers['Content-Length']

  def send_error(self, code, message):
    self.error = message

class Controller_HTTP(Controller):

  def expand_key(self, arg):
    key, val = arg.split('=', 1)
    if key == 'raw_request':

      with open(val, 'rb') as fd:
        r = HTTPRequestParser(fd)

      if r.error:
        raise ValueError('Failed to parse file %r as a raw HTTP request' % val, r.error)

      opts = {}

      if r.path.startswith('http'):
        opts['url'] = r.path
      else:
        _, _, opts['path'], opts['params'], opts['query'], opts['fragment'] = urlparse(r.path)
        opts['host'] = r.headers['Host']

      opts['header'] = str(r.headers)
      opts['method'] = r.command
      opts['body'] = r.body

      for key, val in opts.items():
        if val:
          yield (key, val)

    else:
      yield (key, val)

class HTTP_fuzz(TCP_Cache):
  '''Brute-force HTTP'''

  usage_hints = [
    '''%prog url=http://10.0.0.1/FILE0 0=paths.txt -x ignore:code=404 -x ignore,retry:code=500''',
    '''%prog url=http://10.0.0.1/manager/html user_pass=COMBO00:COMBO01 0=combos.txt -x ignore:code=401''',
    '''%prog url=http://10.0.0.1/phpmyadmin/index.php method=POST'''
    ''' body='pma_username=root&pma_password=FILE0&server=1&lang=en' 0=passwords.txt follow=1'''
    """ accept_cookie=1 -x ignore:fgrep='Cannot log in to the MySQL server'""",
    ]

  available_options = (
    ('url', 'target url (scheme://host[:port]/path?query)'),
    ('body', 'body data'),
    ('header', 'use custom headers'),
    ('method', 'method to use [GET|POST|HEAD|...]'),
    ('raw_request', 'load request from file'),
    ('scheme', 'scheme [http|https]'),
    ('auto_urlencode', 'automatically perform URL-encoding [1|0]'),
    ('pathasis', 'retain sequences of /../ or /./ [0|1]'),
    ('user_pass', 'username and password for HTTP authentication (user:pass)'),
    ('auth_type', 'type of HTTP authentication [basic | digest | ntlm]'),
    ('follow', 'follow any Location redirect [0|1]'),
    ('max_follow', 'redirection limit [5]'),
    ('accept_cookie', 'save received cookies to issue them in future requests [0|1]'),
    ('proxy', 'proxy to use (host:port)'),
    ('proxy_type', 'proxy type [http|socks4|socks4a|socks5]'),
    ('resolve', 'hostname to IP address resolution to use (hostname:IP)'),
    ('ssl_cert', 'client SSL certificate file (cert+key in PEM format)'),
    ('timeout_tcp', 'seconds to wait for a TCP handshake [10]'),
    ('timeout', 'seconds to wait for a HTTP response [20]'),
    ('before_urls', 'comma-separated URLs to query before the main request'),
    ('before_header', 'use a custom header in the before_urls request'),
    ('before_egrep', 'extract data from the before_urls response to place in the main request'),
    ('after_urls', 'comma-separated URLs to query after the main request'),
    ('max_mem', 'store no more than N bytes of request+response data in memory [-1 (unlimited)]'),
    )
  available_options += TCP_Cache.available_options

  Response = Response_HTTP

  def connect(self, host, port, scheme):
    fp = pycurl.Curl()
    fp.setopt(pycurl.SSL_VERIFYPEER, 0)
    fp.setopt(pycurl.SSL_VERIFYHOST, 0)
    fp.setopt(pycurl.HEADER, 1)
    fp.setopt(pycurl.USERAGENT, 'Mozilla/5.0')
    fp.setopt(pycurl.NOSIGNAL, 1)

    return TCP_Connection(fp)

  @staticmethod
  def perform_fp(fp, method, url, header='', body=''):
    #logger.debug('perform: %s' % url)
    fp.setopt(pycurl.URL, url)

    if method == 'GET':
      fp.setopt(pycurl.HTTPGET, 1)

    elif method == 'POST':
      fp.setopt(pycurl.POST, 1)
      fp.setopt(pycurl.POSTFIELDS, body)

    elif method == 'HEAD':
      fp.setopt(pycurl.NOBODY, 1)

    else:
      fp.setopt(pycurl.CUSTOMREQUEST, method)

    headers = [h.strip('\r') for h in header.split('\n') if h]
    fp.setopt(pycurl.HTTPHEADER, headers)

    fp.perform()

  def execute(self, url=None, host=None, port='', scheme='http', path='/', params='', query='', fragment='', body='',
    header='', method='GET', auto_urlencode='1', pathasis='0', user_pass='', auth_type='basic',
    follow='0', max_follow='5', accept_cookie='0', proxy='', proxy_type='http', resolve='', ssl_cert='', timeout_tcp='10', timeout='20', persistent='1',
    before_urls='', before_header='', before_egrep='', after_urls='', max_mem='-1'):

    if url:
      scheme, host, path, params, query, fragment = urlparse(url)
      del url

    if host:
      if ':' in host:
        host, port = host.split(':')

    if resolve:
      resolve_host, resolve_ip = resolve.split(':', 1)
      if port:
        resolve_port = port
      else:
        resolve_port = 80

      resolve = '%s:%s:%s' % (resolve_host, resolve_port, resolve_ip)

    if proxy_type in proxytype_mapping:
      proxy_type = proxytype_mapping[proxy_type]
    else:
      raise ValueError('Invalid proxy_type %r' % proxy_type)

    fp, _ = self.bind(host, port, scheme)

    fp.setopt(pycurl.PATH_AS_IS, int(pathasis))
    fp.setopt(pycurl.FOLLOWLOCATION, int(follow))
    fp.setopt(pycurl.MAXREDIRS, int(max_follow))
    fp.setopt(pycurl.CONNECTTIMEOUT, int(timeout_tcp))
    fp.setopt(pycurl.TIMEOUT, int(timeout))
    fp.setopt(pycurl.PROXY, proxy)
    fp.setopt(pycurl.PROXYTYPE, proxy_type)

    if resolve:
      fp.setopt(pycurl.RESOLVE, [resolve])

    def noop(buf): pass
    fp.setopt(pycurl.WRITEFUNCTION, noop)

    def debug_func(t, s):
      if max_mem > 0 and trace.tell() > max_mem:
        return 0

      if t not in (pycurl.INFOTYPE_HEADER_OUT, pycurl.INFOTYPE_DATA_OUT, pycurl.INFOTYPE_TEXT, pycurl.INFOTYPE_HEADER_IN, pycurl.INFOTYPE_DATA_IN):
        return 0

      s = B(s)

      if t in (pycurl.INFOTYPE_HEADER_OUT, pycurl.INFOTYPE_DATA_OUT):
        trace.write(s)

      elif t == pycurl.INFOTYPE_TEXT and 'upload completely sent off' in s:
        trace.write('\n\n')

      elif t in (pycurl.INFOTYPE_HEADER_IN, pycurl.INFOTYPE_DATA_IN):
        trace.write(s)
        response.write(s)

    max_mem = int(max_mem)
    response, trace = StringIO(), StringIO()

    fp.setopt(pycurl.DEBUGFUNCTION, debug_func)
    fp.setopt(pycurl.VERBOSE, 1)

    if user_pass:
      fp.setopt(pycurl.USERPWD, user_pass)
      if auth_type == 'basic':
        fp.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_BASIC)
      elif auth_type == 'digest':
        fp.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_DIGEST)
      elif auth_type == 'ntlm':
        fp.setopt(pycurl.HTTPAUTH, pycurl.HTTPAUTH_NTLM)
      else:
        raise ValueError('Incorrect auth_type %r' % auth_type)

    if ssl_cert:
      fp.setopt(pycurl.SSLCERT, ssl_cert)

    if accept_cookie == '1':
      fp.setopt(pycurl.COOKIEFILE, '')
      # warning: do not pass a Cookie: header into HTTPHEADER if using COOKIEFILE as it will
      # produce requests with more than one Cookie: header
      # and the server will process only one of them (eg. Apache only reads the last one)

    if before_urls:
      for before_url in before_urls.split(','):
        self.perform_fp(fp, 'GET', before_url, before_header)

      if before_egrep:
        for be in before_egrep.split('|'):
          mark, regex = be.split(':', 1)
          val = re.search(regex, response.getvalue(), re.M).group(1)

          if auto_urlencode == '1':
            val = html_unescape(val)
            val = quote(val)

          header = header.replace(mark, val)
          query = query.replace(mark, val)
          body = body.replace(mark, val)

      response = StringIO()

    if auto_urlencode == '1':
      path = quote(path)
      query = urlencode(parse_query(query, True))
      body = urlencode(parse_query(body, True))

    if port:
      host = '%s:%s' % (host, port)

    url = urlunparse((scheme, host, path, params, query, fragment))
    self.perform_fp(fp, method, url, header, body)

    target = {}
    target['ip'] = fp.getinfo(pycurl.PRIMARY_IP)
    target['port'] = fp.getinfo(pycurl.PRIMARY_PORT)
    target['hostname'] = host

    for h in header.split('\n'):
      if ': ' in h:
        k, v = h.split(': ', 1)
        if k.lower() == 'host':
          target['vhost'] = v.rstrip('\r')
          break

    if after_urls:
      for after_url in after_urls.split(','):
        self.perform_fp(fp, 'GET', after_url)

    http_code = fp.getinfo(pycurl.HTTP_CODE)
    content_length = fp.getinfo(pycurl.CONTENT_LENGTH_DOWNLOAD)
    response_time = fp.getinfo(pycurl.TOTAL_TIME) - fp.getinfo(pycurl.PRETRANSFER_TIME)

    if persistent == '0':
      self.reset()

    return self.Response(http_code, response.getvalue(), response_time, trace.getvalue(), content_length, target)

# }}}

# Dummy Test {{{
def generate_tst():
  return ['prd', 'dev'], 2

class Dummy_test:
  '''Testing module'''

  usage_hints = (
    """%prog data=_@@_RANGE0_@@_ 0=hex:0x00-0xff -e _@@_:unhex""",
    """%prog data=RANGE0 0=int:10-0""",
    """%prog data=PROG0 0='seq -w 10 -1 0'""",
    """%prog data=PROG0 0='mp64.bin -i ?l?l?l',$(mp64.bin --combination -i ?l?l?l)""",
    )

  available_options = (
    ('data', 'data to test'),
    ('data2', 'data2 to test'),
    ('delay', 'fake random delay'),
    )
  available_actions = ()

  available_keys = {
    'TST': generate_tst,
    }

  Response = Response_Base

  def execute(self, data, data2='', delay='1'):
    code, mesg = 0, '%s / %s' % (data, data2)
    with Timing() as timing:
      sleep(random.randint(0, int(delay)*1000)/1000.0)

    return self.Response(code, mesg, timing)

# }}}

# modules {{{
modules = [
  ('http_fuzz', (Controller_HTTP, HTTP_fuzz)),
  ('dummy_test', (Controller, Dummy_test)),
  ]

dependencies = {
  'pycurl': [('http_fuzz', 'rdp_gateway'), 'http://pycurl.io/', '7.43.0'],
  'libcurl': [('http_fuzz', 'rdp_gateway'), 'https://curl.haxx.se/', '7.58.0'],
  'python': [('http_fuzz',), 'Petitor requires Python 3.6 or above'],
  }

# }}}

# main {{{
if __name__ == '__main__':
  set_start_method('fork')
  multiprocessing.freeze_support()

  def show_usage():
    print(__banner__)
    print('''Usage: petitor.py module --help

Available modules:
%s''' % '\n'.join('  + %-13s : %s' % (k, v[1].__doc__) for k, v in modules))

    sys.exit(2)

  available = dict(modules)
  name = os.path.basename(sys.argv[0]).lower()

  if name not in available:
    if len(sys.argv) == 1:
      show_usage()

    name = os.path.basename(sys.argv[1]).lower()
    if name not in available:
      show_usage()

    del sys.argv[0]

  # start
  ctrl, module = available[name]
  powder = ctrl(module, [name] + sys.argv[1:])
  powder.fire()

