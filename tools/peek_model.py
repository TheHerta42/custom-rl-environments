"""Peek at the num_timesteps recorded inside each SB3 model zip."""
import json
import sys
import zipfile

for p in sys.argv[1:]:
    try:
        with zipfile.ZipFile(p) as z:
            data = [n for n in z.namelist() if n.endswith("data")]
            if data:
                j = json.loads(z.read(data[0]))
                print(f"{p.split('/')[-1]:22} num_timesteps={j.get('num_timesteps')}")
            else:
                print(f"{p.split('/')[-1]:22} (no data entry; entries={z.namelist()[:6]})")
    except Exception as exc:  # noqa: BLE001
        print(f"{p}: {type(exc).__name__}: {exc}")
