"""Convert xTranslator .sst dictionary files to glossary JSONv2 import package.

SST binary layout (reverse-engineered from With Light dictionary 1.6):
  header: b'SSU8'/'SSU9' + version bytes + u32 byte-len + utf-16-le esp name
  records: repeating
    - 8 bytes: ASCII record type (e.g. INFONAM1, DIALFULL, BOOKDESC, ...)
    - 14 bytes meta: u16 f1, u16 f2, u32 formid, u16 f3, u32 src_byte_len
    - src: utf-16-le, src_byte_len bytes
    - u32 dst_byte_len + dst utf-16-le
    - 9 bytes trailer: u8 status + u32 src_id + u32 dst_id
      (status observed 00/01/02; kept as provenance in rich meta "sst_status")

Output: JSONv2 envelope compatible with src/rag/glossary_import.py:
  {"format_version": 1, "source": ..., "created_at": ..., "terms": [
     {"term": en, "translation": zh, "domain": ..., "pos": ...,
      "source": "<file>/<TYPE>", "note": "formid=0x..."}, ...]}

Dedup policy: same EN text appearing in several .sst files keeps the first
occurrence (stable file order); duplicates counted and reported. Empty EN or
empty ZH entries are skipped and counted.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import struct
import sys
from pathlib import Path

TYPE_RE = re.compile(rb'^[A-Z0-9_]{8}$')
# Unanchored variant for finditer() scanning (^$ anchors never match mid-buffer).
TYPE_SCAN_RE = re.compile(rb'[A-Z0-9_]{8}')

# Map SST record types to coarse domain labels for the sidecar.
DOMAIN_MAP = {
    'WEAP': 'weapon', 'ARMO': 'armor', 'AMMO': 'ammo', 'ALCH': 'alchemy',
    'INGR': 'ingredient', 'BOOK': 'book', 'SCRL': 'scroll', 'SPEL': 'spell',
    'MGEF': 'magic_effect', 'ENCH': 'enchanting', 'PERK': 'perk',
    'SHOU': 'shout', 'RACE': 'race', 'CLAS': 'class', 'FACT': 'faction',
    'QUST': 'quest', 'DIAL': 'dialogue', 'INFO': 'dialogue', 'CELL': 'location',
    'WRLD': 'location', 'LCTN': 'location', 'REFR': 'object', 'FURN': 'furniture',
    'DOOR': 'door', 'LIGH': 'light', 'FLOR': 'flora', 'TREE': 'tree',
    'NPC_': 'npc', 'AVIF': 'actor_value', 'MESG': 'message', 'GMST': 'setting',
    'WOOP': 'word_of_power', 'SNCT': 'sound', 'KEYM': 'key', 'MISC': 'misc',
    'CONT': 'container', 'ACTI': 'activator', 'TACT': 'talking_activator',
    'HDPT': 'headpart', 'APPA': 'apparatus', 'CLFM': 'class', 'HAZD': 'hazard',
    'PROJ': 'projectile', 'EXPL': 'explosion', 'EYES': 'eyes', 'REGN': 'region',
    'BPTD': 'body_part', 'SLGM': 'soul_gem', 'WATR': 'water', 'COLL': 'collision',
    'LSCR': 'load_screen', 'INFOR': 'dialogue',
}


def domain_for(rectype: str) -> str:
    return DOMAIN_MAP.get(rectype[:4], 'general')


def parse_sst(path: Path):
    """Yield (rectype, formid, status, src, dst) tuples.

    Supports both variants observed in the wild:
    - SSU9: header + repeating [8B type][u16 f1][u16 f2][u32 formid][u16 f3]
      [u32 src_bytes][src utf-16-le][u32 dst_bytes][dst utf-16-le][9B trailer].
    - SSU8: header `SSU8` + 8 unknown bytes, then repeating
      [u16 preamble][8B type][u32 f1][u32 formid][u16 f2][u32 src_bytes]
      [src utf-16-le][u32 dst_bytes][dst utf-16-le] (no trailer).
    """
    b = bytes(path.read_bytes())
    if len(b) < 12 or b[:3] != b'SSU':
        raise ValueError(f'Not an SST file: {path} (bad magic)')
    total = len(b)
    if b[:4] == b'SSU8':
        # SSU8: 9-byte file header (magic + 5 unknown bytes), then repeating
        #   [9B header: u8 status? + u32 seq? + u16 x? + u16 y?]
        #   [8B type][u32 f1][u32 formid][u16 f2]
        #   [u32 src_bytes][src utf-16-le][u32 dst_bytes][dst utf-16-le]
        # Records follow each other with no gap.
        off = 9
        while off + 9 + 8 + 4 + 4 + 2 + 4 <= total:
            rectype = b[off + 9:off + 17]
            if not TYPE_RE.match(rectype):
                break
            try:
                f1 = struct.unpack_from('<I', b, off + 17)[0]
                fid = struct.unpack_from('<I', b, off + 21)[0]
                f2 = struct.unpack_from('<H', b, off + 25)[0]
                slen = struct.unpack_from('<I', b, off + 27)[0]
            except Exception:
                break
            if slen % 2 or slen > 10_000_000:
                break
            s0 = off + 31
            if s0 + slen + 4 > total:
                break
            try:
                tlen = struct.unpack_from('<I', b, s0 + slen)[0]
            except Exception:
                break
            if tlen % 2 or tlen > 10_000_000:
                break
            t0 = s0 + slen + 4
            if t0 + tlen > total:
                break
            try:
                src = b[s0:s0 + slen].decode('utf-16-le')
                dst = b[t0:t0 + tlen].decode('utf-16-le')
            except Exception:
                break
            status = b[off]
            yield rectype.decode('ascii'), fid, status, src, dst
            off = t0 + tlen
        return
    # header: magic(4) + 4 unknown bytes + one or more pascal utf-16-le
    # esp names, each [u8?? actually 1-byte-aligned u32 byte-len][name bytes].
    # Observed: byte12 == 0x00 padding, then u24/u32 LE length at 13?
    # Empirically: name starts at offset 13, length = u32 LE at 12 >> 8?
    # Simplest robust approach: scan for first valid record instead of
    # trusting header fields (records have strong magic: 8B type + sane lens).
    off = 12
    total = len(b)
    # Skip the esp-name header: it is utf-16-le text; find first plausible
    # record by scanning for TYPE + sane lengths.
    start = None
    for cand in TYPE_SCAN_RE.finditer(b, off):
        o = cand.start()
        if o + 22 > total:
            continue
        try:
            slen = struct.unpack_from('<I', b, o + 18)[0]
        except Exception:
            continue
        if slen % 2 or slen > 2_000_000 or slen == 0:
            continue
        s0 = o + 22
        if s0 + slen + 4 > total:
            continue
        try:
            tlen = struct.unpack_from('<I', b, s0 + slen)[0]
        except Exception:
            continue
        if tlen % 2 or tlen > 2_000_000:
            continue
        t0 = s0 + slen + 4
        if t0 + tlen > total:
            continue
        # validate decodability of a prefix
        try:
            b[s0:s0 + min(slen, 40)].decode('utf-16-le')
            if tlen:
                b[t0:t0 + min(tlen, 40)].decode('utf-16-le')
        except Exception:
            continue
        # validate that a record plausibly follows (next type or EOF trailer)
        end = t0 + tlen
        nxt = b[end:end + 17]
        if end < total and not (
            TYPE_RE.match(b[end:end + 8])
            or (len(nxt) >= 17 and TYPE_RE.match(nxt[9:17]))
        ):
            continue
        start = o
        break
    if start is None:
        raise ValueError(f'No records found in {path}')
    off = start
    tombstones = 0
    while off + 22 <= total:
        rawtype = b[off:off + 8]
        # xTranslator marks deleted entries by overwriting the record type
        # with b'********' while keeping the body layout intact: skip the
        # body but do not emit the entry, then keep walking.
        tombstone = rawtype == b'********'
        if not tombstone and not TYPE_RE.match(rawtype):
            break
        try:
            f1, f2, fid, f3 = struct.unpack_from('<HHIH', b, off + 8)
            slen = struct.unpack_from('<I', b, off + 18)[0]
        except Exception:
            break
        if slen % 2 or slen > 10_000_000:
            break
        s0 = off + 22
        if s0 + slen + 4 > total:
            break
        try:
            tlen = struct.unpack_from('<I', b, s0 + slen)[0]
        except Exception:
            break
        if tlen % 2 or tlen > 10_000_000:
            break
        t0 = s0 + slen + 4
        if t0 + tlen + 9 > total + 9:  # last record may miss trailer
            break
        try:
            src = b[s0:s0 + slen].decode('utf-16-le')
            dst = b[t0:t0 + tlen].decode('utf-16-le')
        except Exception:
            break
        end = t0 + tlen
        # trailer is 9 bytes except possibly at EOF
        off = end + 9 if end + 9 <= total else total
        if tombstone:
            tombstones += 1
            continue
        status = b[end] if end < total else 0
        yield rawtype.decode('ascii'), fid, status, src, dst


def convert(src_dir: Path, out_path: Path, dry_run: bool = False):
    files = sorted(src_dir.glob('*.sst'))
    if not files:
        raise SystemExit(f'No .sst files in {src_dir}')
    merged: dict[str, dict] = {}
    per_file_counts: dict[str, int] = {}
    dup = 0
    skipped_empty = 0
    for fp in files:
        n = 0
        for rectype, fid, status, src, dst in parse_sst(fp):
            en = (src or '').strip()
            zh = (dst or '').strip()
            if not en or not zh:
                skipped_empty += 1
                continue
            if en in merged:
                dup += 1
                continue
            merged[en] = {
                'term': en,
                'translation': zh,
                'domain': domain_for(rectype),
                'source': f'{fp.name}/{rectype}',
                'note': f'formid=0x{fid:08X} sst_status={status}',
            }
            n += 1
        per_file_counts[fp.name] = n
        print(f'{fp.name}: {n} new terms')
    print(f'TOTAL unique: {len(merged)}, duplicates skipped: {dup}, empty skipped: {skipped_empty}')
    if dry_run:
        return
    payload = {
        'format_version': 1,
        'source': f'xTranslator SST dictionary ({src_dir.name})',
        'created_at': datetime.date.today().isoformat(),
        'terms': list(merged.values()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f'WROTE {out_path} ({size_mb:.1f} MB)')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Convert xTranslator .sst dictionaries to JSONv2')
    ap.add_argument('src_dir', help='folder containing .sst files')
    ap.add_argument('-o', '--out', default='glossary_xtranslator.json',
                    help='output JSON path (default: glossary_xtranslator.json)')
    ap.add_argument('--dry-run', action='store_true', help='only count, do not write')
    args = ap.parse_args(argv)
    convert(Path(args.src_dir), Path(args.out), dry_run=args.dry_run)


if __name__ == '__main__':
    main()
