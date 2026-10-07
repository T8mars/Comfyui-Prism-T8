"""Capture uv's native transfer counters without exposing its terminal redraws.

Linux uses a private pseudo-terminal for child output only. The existing process
group, cancellation, environment, proxy and package resolver remain in control.
Other platforms retain the ordinary log stream and phase reporting.
"""
import codecs
import errno
import json
import os
from pathlib import Path
import re
import select
import threading
import time

from .terminal_ui import ESCAPES, clean

TRANSFER = re.compile(r'([\w.+-]+)\s+[-=#>━─█░ ]+\s+([\d.]+)\s*(B|KiB|MiB|GiB)\s*/\s*([\d.]+)\s*(B|KiB|MiB|GiB)')
UNITS = {'B': 1, 'KiB': 2**10, 'MiB': 2**20, 'GiB': 2**30}


def transfer_counter(line):
    match = TRANSFER.fullmatch(clean(line))
    if not match:
        return None
    name, done, done_unit, total, total_unit = match.groups()
    try:
        done, total = round(float(done) * UNITS[done_unit]), round(float(total) * UNITS[total_unit])
    except (ValueError, OverflowError):
        return None
    if total <= 0 or done < 0:
        return None
    return name, min(done, total), total


def is_uv_install(command):
    return (len(command) > 1 and Path(str(command[0])).name in ('uv', 'uv.exe') and
            (list(map(str, command[1:3])) in (['pip', 'install'], ['python', 'install'])
             or str(command[1]) == 'venv'))


class PackageOutput:
    def __init__(self, command, log, env, *, enabled=None):
        self.log, self.env, self.stdout = log, env, log
        self.enabled = os.name == 'posix' and (is_uv_install(command) if enabled is None else enabled)
        self.master = self.slave = None
        self.thread = None
        self.error = None
        self.previous = {}
        self.stop = threading.Event()

    def __enter__(self):
        if self.enabled:
            import fcntl
            import pty
            import struct
            import termios
            try:
                self.master, self.slave = pty.openpty()
                # Enough rows for concurrent uv downloads; no user terminal is
                # resized or attached to the child's input.
                fcntl.ioctl(self.slave, termios.TIOCSWINSZ, struct.pack('HHHH', 128, 140, 0, 0))
            except OSError:
                self.close_fds()
            else:
                self.stdout = self.slave
                self.env = dict(self.env, TERM='xterm-256color', UV_NO_PROGRESS='false')
                self.env.pop('JPY_SESSION_NAME', None)  # uv's notebook mode suppresses byte counters
        return self

    def spawned(self):
        if self.master is not None:
            os.close(self.slave)
            self.slave = None
            self.thread = threading.Thread(target=self.read, name='package-progress', daemon=True)
            self.thread.start()

    def emit(self, line):
        text = ''.join(c for c in ESCAPES.sub('', line) if c.isprintable() or c == '\t').rstrip()
        if not text.strip():
            return
        counter = transfer_counter(text)
        if counter:
            name, done, total = counter
            now = time.monotonic()
            prior = self.previous.get(name)
            if prior and now - prior[0] < .5:
                return
            rate = ((done - prior[1]) / (now - prior[0])
                    if prior and total == prior[2] and done >= prior[1] else 0.)
            self.previous[name] = (now, done, total)
            text = json.dumps({'event': 'download_progress', 'description': name,
                               'done_bytes': done, 'total_bytes': total,
                               'bytes_per_second': rate, 'counter_source': 'uv-terminal',
                               'scope': 'Current file; uv byte counters are rounded for display'})
        self.log.write(text + '\n')
        self.log.flush()

    def read(self):
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        pending = ''
        try:
            while not self.stop.is_set():
                if not select.select([self.master], [], [], .2)[0]:
                    continue
                try:
                    chunk = os.read(self.master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:  # PTY writer exited
                        break
                    raise
                if not chunk:
                    break
                rows = (pending + decoder.decode(chunk)).replace('\r', '\n').split('\n')
                pending = rows.pop()
                for row in rows:
                    self.emit(row)
                # Guard against an invalid child that never writes line breaks.
                if len(pending) > 65536:
                    self.log.write(clean(pending) + '\n')
                    self.log.flush()
                    pending = ''
            self.emit(pending + decoder.decode(b'', final=True))
        except BaseException as error:
            self.error = error

    def close_fds(self):
        for name in ('slave', 'master'):
            fd = getattr(self, name)
            if fd is not None:
                os.close(fd)
                setattr(self, name, None)

    def __exit__(self, exc_type, exc, tb):
        if self.slave is not None:
            os.close(self.slave)
            self.slave = None
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.stop.set()
                self.thread.join(timeout=1)
        self.close_fds()
        if self.error is not None and exc_type is None:
            raise RuntimeError('Could not retain package installation output') from self.error
