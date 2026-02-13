# cuda-gdb scrupt to find and load most fresh cuda coredump
# run witn gdb --command=do.py
from pathlib import Path
import re

# enum nvcore file in path dir, get most fresh
def get_fresh(path):
 res = ''
 res_ctime = 0
 p = Path(path)
 reg = re.compile(r"^core_.*nvcudmp$")
 for f in p.iterdir():
  if f.is_file() and reg.match(f.name):
   if res == '':
    res = f.absolute()
    res_ctime = f.stat().st_ctime
   else:
    ctime = f.stat().st_ctime
    if ctime > res_ctime:
     res = f.absolute()
 return res

# main
res = get_fresh('.')
s = str(res)
if s != '':
 cmd = "target cudacore " + s
 gdb.execute(cmd)
 # show cuda kernels
 gdb.execute("info cuda kernels")
 # where
 gdb.execute("where")
