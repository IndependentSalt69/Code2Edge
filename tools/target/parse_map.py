#!/usr/bin/env python3
"""
Parse GNU LD / Zephyr linker map files for STM32U585 builds.
Extracts exact section sizes (.text, .rodata, .data, .bss), Flash footprint, and SRAM footprint.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MAP_PATH = REPO_ROOT / "build" / "benchmark_harness" / "benchmark_harness.ino.map"


def parse_map_file(map_path: Path) -> dict:
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_path}")

    sections = {}
    with open(map_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Match primary top-level output sections:
    # Pattern: ^(\.[a-zA-Z0-9_\.]+)\s+0x[0-9a-fA-F]+\s+(0x[0-9a-fA-F]+|[0-9]+)
    pattern = re.compile(
        r"^(\.[a-zA-Z0-9_\.]+)\s+0x[0-9a-fA-F]+\s+(0x[0-9a-fA-F]+|[0-9]+)",
        re.MULTILINE
    )

    for match in pattern.finditer(content):
        sec_name = match.group(1)
        size_raw = match.group(2)
        size_bytes = int(size_raw, 16) if size_raw.startswith("0x") else int(size_raw)

        # Ignore sub-sections (e.g. .text.fn_name) if top-level section already captured
        if sec_name in [".text", ".rodata", ".data", ".bss", ".noinit", ".init_array", ".fini_array", ".llext.rodata.noreloc"]:
            sections[sec_name] = size_bytes

    text_bytes = sections.get(".text", 0)
    rodata_bytes = sections.get(".rodata", 0)
    data_bytes = sections.get(".data", 0)
    bss_bytes = sections.get(".bss", 0)
    llext_rodata = sections.get(".llext.rodata.noreloc", 0)

    # Core Flash = .text + .rodata + .data
    flash_core_bytes = text_bytes + rodata_bytes + data_bytes
    flash_total_bytes = flash_core_bytes + llext_rodata

    # Core SRAM = .data + .bss
    sram_used_bytes = data_bytes + bss_bytes

    return {
        "map_file": str(map_path),
        "sections": sections,
        "text_bytes": text_bytes,
        "rodata_bytes": rodata_bytes,
        "llext_rodata_bytes": llext_rodata,
        "data_bytes": data_bytes,
        "bss_bytes": bss_bytes,
        "flash_core_bytes": flash_core_bytes,
        "flash_total_bytes": flash_total_bytes,
        "sram_used_bytes": sram_used_bytes,
        "target_limits": {
            "flash_capacity_bytes": 2097152,  # 2 MB
            "sram_capacity_bytes": 804864     # 786 KB
        },
        "headroom": {
            "flash_headroom_bytes": 2097152 - flash_total_bytes,
            "sram_headroom_bytes": 804864 - sram_used_bytes
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Extract exact Flash and SRAM section usage from GCC/Zephyr map files.")
    parser.add_argument("--map", type=Path, default=DEFAULT_MAP_PATH, help="Path to .map file")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    try:
        data = parse_map_file(args.map)
    except Exception as e:
        print(f"Error parsing map file: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
    else:
        print("================================================================")
        print(" Linker Map Memory Analysis: STM32U585")
        print(f" Source: {data['map_file']}")
        print("================================================================")
        print(" Linker Section Breakdown:")
        print(f"   .text (Code):            {data['text_bytes']:>8} bytes ({data['text_bytes']/1024:.2f} KB)")
        print(f"   .rodata (Constants):     {data['rodata_bytes']:>8} bytes ({data['rodata_bytes']/1024:.2f} KB)")
        if data['llext_rodata_bytes'] > 0:
            print(f"   .llext.rodata (Zephyr):  {data['llext_rodata_bytes']:>8} bytes ({data['llext_rodata_bytes']/1024:.2f} KB)")
        print(f"   .data (Initialized RAM): {data['data_bytes']:>8} bytes ({data['data_bytes']/1024:.2f} KB)")
        print(f"   .bss (Zero-Init RAM):    {data['bss_bytes']:>8} bytes ({data['bss_bytes']/1024:.2f} KB)")
        print("----------------------------------------------------------------")
        print(" Calculated Footprint:")
        print(f"   Application Flash:       {data['flash_total_bytes']:>8} bytes ({data['flash_total_bytes']/1024:.2f} KB)")
        print(f"     [Flash Headroom:       {data['headroom']['flash_headroom_bytes']:>8} bytes ({data['headroom']['flash_headroom_bytes']/1024:.2f} KB)]")
        print(f"   Application SRAM:        {data['sram_used_bytes']:>8} bytes ({data['sram_used_bytes']/1024:.2f} KB)")
        print(f"     [SRAM Headroom:        {data['headroom']['sram_headroom_bytes']:>8} bytes ({data['headroom']['sram_headroom_bytes']/1024:.2f} KB)]")
        print("================================================================")


if __name__ == "__main__":
    main()
