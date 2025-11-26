import os
import re
from collections import defaultdict

def parse_cpuinfo():
    with open("/proc/cpuinfo", "r") as f:
        cpuinfo = f.read()

    cpus = cpuinfo.strip().split("\n\n")
    cores = []

    for cpu in cpus:
        info = {}
        for line in cpu.strip().splitlines():
            if ":" in line:
                key, val = map(str.strip, line.split(":", 1))
                info[key] = val
        cores.append(info)

    return cores

def classify_cores(cpuinfo):
    # Group cores by model name + cache size as a heuristic
    core_groups = defaultdict(list)

    for info in cpuinfo:
        cpu_id = int(info.get("processor", -1))
        model = info.get("model name", "unknown")
        cache = info.get("cache size", "unknown")
        key = f"{model}-{cache}"
        core_groups[key].append(cpu_id)

    # Sort groups by number of CPUs or alphabetically
    groups = list(core_groups.values())
    if len(groups) == 1:
        return groups[0], []  # only one core type
    else:
        # Assume group with more cores is E-core group (as on Intel CPUs)
        groups.sort(key=lambda g: max(g))  # or by avg CPU ID
        return groups[-1], groups[0]  # p_cores, e_cores

def main():
    cpuinfo = parse_cpuinfo()
    if not cpuinfo:
        print("Failed to parse /proc/cpuinfo")
        return

    p_cores, e_cores = classify_cores(cpuinfo)
    print(f"Performance cores: {sorted(p_cores)}")
    print(f"Efficiency cores: {sorted(e_cores)}")

if __name__ == "__main__":
    main()
