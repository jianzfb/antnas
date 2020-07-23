# -*- coding: UTF-8 -*-
# @Time    : 2020-05-29 17:41
# @File    : utils.py
# @Author  : jian<jian@mltalker.com>
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from datetime import datetime
from binascii import b2a_hex
import errno
import hashlib
from hmac import compare_digest
import os
import socket
from threading import Thread
import uuid
import warnings
import re
import sys
if sys.version > '3':
    PY3 = True
else:
    PY3 = False


def new_token(*args, **kwargs):
    """generator for new random tokens

    For now, just UUIDs.
    """
    return uuid.uuid4().hex


def hash_token(token, salt=8, rounds=16384, algorithm='sha512'):
    """hash a token, and return it as `algorithm:salt:hash`

    If `salt` is an integer, a random salt of that many bytes will be used.
    """
    h = hashlib.new(algorithm)
    if isinstance(salt, int):
        salt = b2a_hex(os.urandom(salt))
    if isinstance(salt, bytes):
        bsalt = salt
        salt = salt.decode('utf8')
    else:
        bsalt = salt.encode('utf8')
    btoken = token.encode('utf8', 'replace')
    h.update(bsalt)
    for i in range(rounds):
        h.update(btoken)
    digest = h.hexdigest()
    
    return "{algorithm}:{rounds}:{salt}:{digest}".format(**locals())


def compare_token(compare, token):
    """compare a token with a hashed token

    uses the same algorithm and salt of the hashed token for comparison
    """
    algorithm, srounds, salt, _ = compare.split(':')
    hashed = hash_token(token, salt=salt, rounds=int(srounds), algorithm=algorithm).encode('utf8')
    compare = compare.encode('utf8')
    if compare_digest(compare, hashed):
        return True
    return False


def url_path_join(*pieces):
    """Join components of url into a relative url

    Use to prevent double slash when joining subpath. This will leave the
    initial and final / in place

    Copied from notebook.utils.url_path_join
    """
    initial = pieces[0].startswith('/')
    final = pieces[-1].endswith('/')
    stripped = [s.strip('/') for s in pieces]
    result = '/'.join(s for s in stripped if s)
    
    if initial:
        result = '/' + result
    if final:
        result = result + '/'
    if result == '//':
        result = '/'
    
    return result


def timestamp():
  now_time = datetime.now()
  if PY3:
    return now_time.timestamp()
  else:
    epoch = datetime.utcfromtimestamp(0)
    total_seconds = (now_time - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds