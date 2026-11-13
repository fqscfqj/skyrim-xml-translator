# -*- coding: utf-8 -*-
"""Launch the built exe briefly and confirm it starts without crashing."""
import subprocess, sys, time, os

exe = os.path.abspath("dist/SkyrimXMLTranslator.exe")
print("launching:", exe)
p = subprocess.Popen([exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
time.sleep(12)
rc = p.poll()
if rc is None:
    print("RESULT: process still running after 12s -> started OK")
    p.terminate()
    try:
        p.wait(timeout=10)
    except Exception:
        p.kill()
    print("terminated cleanly")
else:
    out, err = p.communicate()
    print(f"RESULT: process exited early with code {rc}")
    print("stdout:", out.decode("utf-8", "replace")[:2000])
    print("stderr:", err.decode("utf-8", "replace")[:2000])
    sys.exit(1)
