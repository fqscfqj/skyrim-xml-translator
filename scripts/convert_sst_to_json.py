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


def _valid_ssu8_record_at(b: bytes, off: int, total: int):
    if off + 9 + 8 + 4 + 4 + 2 + 4 > total:
        return None
    rectype = b[off + 9:off + 17]
    if not TYPE_RE.match(rectype) and rectype != b'********':
        return None
    try:
        slen = struct.unpack_from('<I', b, off + 27)[0]
    except Exception:
        return None
    if slen % 2 or slen > 10_000_000:
        return None
    s0 = off + 31
    if s0 + slen + 4 > total:
        return None
    try:
        tlen = struct.unpack_from('<I', b, s0 + slen)[0]
    except Exception:
        return None
    if tlen % 2 or tlen > 10_000_000:
        return None
    t0 = s0 + slen + 4
    if t0 + tlen > total:
        return None
    return rectype, s0, slen, t0, tlen


def _scan_ssu8_start(b: bytes, total: int):
    limit = min(total, 64 * 1024)
    for cand in TYPE_SCAN_RE.finditer(b, 0, limit):
        type_off = cand.start()
        off = type_off - 9
        if off < 9 or off + 31 > total:
            continue
        if _valid_ssu8_record_at(b, off, total) is not None:
            return off
    return None


def parse_sst(path: Path):
    """Yield (rectype, formid, status, src, dst) tuples.

    Supports both variants observed in the wild:
    - SSU9: header + repeating [8B type][u16 f1][u16 f2][u32 formid][u16 f3]
      [u32 src_bytes][src utf-16-le][u32 dst_bytes][dst utf-16-le][9B trailer].
    - SSU8: 9-byte file header, then repeating [9B header][8B type]
      [u32 f1][u32 formid][u16 f2][u32 src_bytes][src][u32 dst_bytes][dst];
      next head at end, next type at end + 9. Bad records resync.
    """
    b = bytes(path.read_bytes())
    if len(b) < 12 or b[:3] != b'SSU':
        raise ValueError(f'Not an SST file: {path} (bad magic)')
    total = len(b)
    if b[:4] == b'SSU8':
        off = _scan_ssu8_start(b, total)
        if off is None:
            raise ValueError(f'No records found in {path}')
        bad_records = 0
        tombstones = 0
        while off + 9 + 8 + 4 <= total:
            valid = _valid_ssu8_record_at(b, off, total)
            if valid is None:
                bad_records += 1
                off += 1
                nxt = TYPE_SCAN_RE.search(b, off, min(total, off + 4096))
                if nxt is None:
                    break
                cand_off = nxt.start() - 9
                off = cand_off if cand_off > off else off
                continue
            rectype, s0, slen, t0, tlen = valid
            try:
                fid = struct.unpack_from('<I', b, off + 21)[0]
            except Exception:
                bad_records += 1
                off += 1
                continue
            try:
                src = b[s0:s0 + slen].decode('utf-16-le') if slen else ""
                dst = b[t0:t0 + tlen].decode('utf-16-le') if tlen else ""
            except Exception:
                bad_records += 1
                off += 1
                continue
            end = t0 + tlen
            next_type_off = end + 9
            _ = next_type_off
            status = b[off]
            off = end
            if rectype == b'********':
                tombstones += 1
                continue
            yield rectype.decode('ascii'), fid, status, src, dst
        return
    # header: magic(4) + 4 unknown bytes + one or more pascal utf-16-le
    # esp names, each [u8?? actually 1-byte-aligned u32 byte-len][name bytes].
    # Observed: byte12 == 0x00 padding, then u24/u32 LE length at 13?
    # Empirically: name starts at offset 13, length = u32 LE at 12 >> 8?
    # Simplest robust approach: scan for first valid record instead of
    # trusting header fields (records have strong magic: 8B type + sane lens).
    off = 12
    total = len(b)
    # Skip header by scanning first 64KB for TYPE + sane lengths.
    start = None
    scan_limit = min(total, 64 * 1024)
    for cand in TYPE_SCAN_RE.finditer(b, off, scan_limit):
        o = cand.start()
        if o + 22 > total:
            continue
        try:
            slen = struct.unpack_from('<I', b, o + 18)[0]
        except Exception:
            continue
        if slen % 2 or slen > 2_000_000:
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
    bad_records = 0
    while off + 22 <= total:
        rawtype = b[off:off + 8]
        tombstone = rawtype == b'********'
        if not tombstone and not TYPE_RE.match(rawtype):
            bad_records += 1
            off += 1
            continue
        try:
            f1, f2, fid, f3 = struct.unpack_from('<HHIH', b, off + 8)
            slen = struct.unpack_from('<I', b, off + 18)[0]
        except Exception:
            bad_records += 1
            off += 1
            continue
        if slen % 2 or slen > 10_000_000:
            bad_records += 1
            off += 1
            continue
        s0 = off + 22
        if s0 + slen + 4 > total:
            bad_records += 1
            off += 1
            continue
        try:
            tlen = struct.unpack_from('<I', b, s0 + slen)[0]
        except Exception:
            bad_records += 1
            off += 1
            continue
        if tlen % 2 or tlen > 10_000_000:
            bad_records += 1
            off += 1
            continue
        t0 = s0 + slen + 4
        if t0 + tlen + 9 > total + 9:
            bad_records += 1
            off += 1
            continue
        try:
            src = b[s0:s0 + slen].decode('utf-16-le') if slen else ""
            dst = b[t0:t0 + tlen].decode('utf-16-le') if tlen else ""
        except Exception:
            bad_records += 1
            off += 1
            continue
        end = t0 + tlen
        off = end + 9 if end + 9 <= total else total
        if tombstone:
            tombstones += 1
            continue
        status = b[end] if end < total else 0
        yield rawtype.decode('ascii'), fid, status, src, dst


def convert(src_dir: Path, out_path: Path, dry_run: bool = False, max_rows: int = 0):
    files = sorted(
        [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() == ".sst"],
        key=lambda p: p.name.lower(),
    ) if src_dir.is_dir() else []
    if not files:
        raise SystemExit(f'No .sst files in {src_dir}')
    merged: dict[str, dict] = {}
    per_file_counts: dict[str, int] = {}
    dup = 0
    skipped_empty = 0
    limited = 0
    max_rows = max(0, int(max_rows or 0))
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
            if max_rows and len(merged) >= max_rows:
                limited += 1
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
    print(f'TOTAL unique: {len(merged)}, duplicates skipped: {dup}, empty skipped: {skipped_empty}, limited: {limited}')
    if dry_run:
        return
    payload = {
        'format_version': 1,
        'source': f'xTranslator SST dictionary ({src_dir.name})',
        'created_at': datetime.date.today().isoformat(),
        'terms': list(merged.values()),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import os as _os
    import tempfile as _tempfile
    tmp_fd, tmp_path = _tempfile.mkstemp(
        dir=str(out_path.parent), prefix=out_path.name + ".", suffix=".tmp")
    try:
        with _os.fdopen(tmp_fd, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
            f.flush()
            try:
                _os.fsync(f.fileno())
            except Exception:
                pass
        _os.replace(tmp_path, out_path)
    finally:
        try:
            if _os.path.exists(tmp_path):
                _os.remove(tmp_path)
        except Exception:
            pass
    size_mb = out_path.stat().st_size / 1_048_576
    print(f'WROTE {out_path} ({size_mb:.1f} MB)')


def main(argv=None):
    ap = argparse.ArgumentParser(description='Convert xTranslator .sst dictionaries to JSONv2')
    ap.add_argument('src_dir', help='folder containing .sst files')
    ap.add_argument('-o', '--out', default='glossary_xtranslator.json',
                    help='output JSON path (default: glossary_xtranslator.json)')
    ap.add_argument('--dry-run', action='store_true', help='only count, do not write')
    ap.add_argument('--max-rows', type=int, default=0,
                    help='max terms to keep (0=unlimited, same as glossary import)')
    args = ap.parse_args(argv)
    convert(Path(args.src_dir), Path(args.out), dry_run=args.dry_run, max_rows=args.max_rows)


if __name__ == '__main__':
    main()
