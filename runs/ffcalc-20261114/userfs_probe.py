#!/usr/bin/env python3
"""Replicate ffprobe.ko output via sysfs BAR0 mmap (READ-ONLY) so that the
stock `python3 -m ffcalc scan` pipeline runs unchanged on hosts where the
kmod path is unnecessary. Produces collected/bar0map.tsv + probe_<bdf>.txt.
Run from the repo root on the card host:  python3 userfs_probe.py [outdir]
"""
import io
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ffcalc.capture import open_bar0, rd  # noqa: E402

MAP = os.path.join(HERE, "ffcalc", "data", "ga100-fuses.json")


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "collected"
    with io.open(MAP, encoding="utf-8") as f:
        m = json.load(f)

    uuid = {}
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=pci.bus_id,uuid",
         "--format=csv,noheader"]).decode()
    for line in smi.strip().splitlines():
        b, u = line.split(",")
        b = b.strip().lower().replace("00000000:", "0000:")
        uuid[b] = u.strip()

    os.makedirs(outdir, exist_ok=True)
    with io.open(os.path.join(outdir, "bar0map.tsv"), "w") as tsv:
        for bdf in sorted(uuid):
            with io.open("/sys/bus/pci/devices/%s/resource" % bdf) as rf:
                bar0 = int(rf.readline().split()[0], 16)
            _bdf, mm = open_bar0(bdf)
            fn = os.path.join(
                outdir, "probe_%s.txt" % bdf.replace(":", "_").replace(".", "_"))
            with io.open(fn, "w") as f:
                def emit(name, off):
                    f.write("FFPROBE %s %08x %08x\n"
                            % (name, bar0 + off, rd(mm, "0x%x" % off)))
                for key, loc in sorted(m["fields"].items()):
                    for kind in ("status", "fused"):
                        if loc.get(kind):
                            emit("%s.%s" % (key, kind), int(loc[kind], 0))
                for key, spec in sorted(m["bit_fields"].items()):
                    emit("bitfield." + key, int(spec["reg"], 0))
                for key, addr in sorted(m["defective"].items()):
                    emit("defective." + key, int(addr, 0))
                for key, addr in sorted(m["info"].items()):
                    emit("info." + key, int(addr, 0))
            mm.close()
            tsv.write("%s\t0x%08x\t%s\n" % (bdf, bar0, uuid[bdf]))
            print("probed %s -> %s (%s)" % (bdf, fn, uuid[bdf]))


if __name__ == "__main__":
    main()
