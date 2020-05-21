#!/usr/bin/python3
import requests
import os
from pip._internal.operations.freeze import freeze

str = '=='
libname = "blivet"
for lib in freeze(local_only=True):

    loc = lib.find(str, 0, len(lib))
    libname = lib[0:loc]
    actual = lib[loc + 2:len(lib)]
    print('Checking ' + libname + '...')
    request = "https://pypi.python.org/pypi/" + libname + "/json"
    try:
        res = requests.get(request).json()
    except:
        print(libname + ' cannot be updated!')
    latest = res["info"]["version"]

    if (actual == latest):
        print(libname + ' is up-to-date!')
    else:
        os.system('pip3 install ' + libname + ' --upgrade --user')
        print(lib + " updated")
