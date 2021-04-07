import json
import subprocess


def daphne(args, cwd='../daphne'):
    proc = subprocess.run(['lein', 'run'] + args,
                          capture_output=True, cwd=cwd)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)
