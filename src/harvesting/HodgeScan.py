"""
HodgeScan.py — Phase 1 Topology Scanner

Exhaustively queries the Kreuzer-Skarke database for all polytopes satisfying
the Standard Model constraint |h11 - h21| = 3 (3 particle generations, χ = ±6).

Instead of scanning by h11 and filtering, this script queries each qualifying
(h11, h21) pair directly using fetch_polytopes(h11=X, h21=Y, lattice="N").
This is dramatically faster and more accurate than the scan approach.

NO triangulations are generated. Hodge number checks only.

Output: data/hodge_scan_results.json
    {
        "qualifying_pairs": {
            "(13,16)": 117,
            "(15,18)": 553,
            ...
        },
        "total_polytopes_found": 1234,
        "scanned_pairs": [...],
        "scan_params": { ... }
    }

Feed results into DeepSpaceHarvester.py to do the expensive triangulation
step only on confirmed qualifying pairs.

NOTE: fetch_polytopes() with a high limit may still not return ALL polytopes
for a given Hodge pair if the KS server caps responses. Per CYTools docs:
"it may happen that fewer polytopes than requested are returned even though
more exist." We use limit=10000 as a practical ceiling and document this.
"""

import os
import json
import argparse
import time
from cytools import fetch_polytopes

KS_H11_MIN = 1
KS_H11_MAX = 491

def get_qualifying_pairs(h11_min, h11_max):
    """Generate all (h11, h21) pairs where |h11 - h21| = 3."""
    pairs = []
    for h11 in range(h11_min, h11_max + 1):
        for h21 in [h11 + 3, h11 - 3]:
            if h21 >= 1:  # h21 must be positive
                pairs.append((h11, h21))
    return sorted(pairs)


def _save(output_path, qualifying_pairs, scanned_pairs, total_found, args):
    results = {
        "qualifying_pairs": qualifying_pairs,
        "scanned_pairs": [list(p) for p in scanned_pairs],
        "total_polytopes_found": total_found,
        "scan_params": {
            "h11_min": args.h11_min,
            "h11_max": args.h11_max,
            "limit_per_pair": args.limit,
            "constraint": "|h11-h21|=3",
            "lattice": "N",
            "note": "limit per pair may not capture all polytopes per CYTools docs"
        }
    }
    with open(output_path, 'w') as f:
        import json
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Direct Hodge pair query: find all |h11-h21|=3 polytopes in the KS database."
    )
    parser.add_argument('--h11_min', type=int, default=KS_H11_MIN,
                        help=f"Start of h11 scan range (default: {KS_H11_MIN})")
    parser.add_argument('--h11_max', type=int, default=KS_H11_MAX,
                        help=f"End of h11 scan range (default: {KS_H11_MAX})")
    parser.add_argument('--limit', type=int, default=10000,
                        help="Max polytopes to fetch per Hodge pair (default: 10000). "
                             "Per CYTools docs, results may be capped by the KS server "
                             "even if more exist. Increase to check for more.")
    parser.add_argument('--output', type=str, default="data/hodge_scan_results.json",
                        help="Output path for scan results (default: data/hodge_scan_results.json)")
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to existing scan results JSON to resume from.")

    args = parser.parse_args()

    all_pairs = get_qualifying_pairs(args.h11_min, args.h11_max)

    print(f"🔭 HodgeScan: Direct KS Hodge Pair Query")
    print(f"📐 Constraint: |h11 - h21| = 3 (χ = ±6)")
    print(f"🌌 h11 range: {args.h11_min} to {args.h11_max}")
    print(f"🔢 Total (h11, h21) pairs to query: {len(all_pairs)}")
    print(f"📦 Limit per pair: {args.limit}")
    print()

    # Resume support
    qualifying_pairs = {}
    scanned_pairs = set()
    total_found = 0

    if args.resume and os.path.exists(args.resume):
        with open(args.resume, 'r') as f:
            prev = json.load(f)
        qualifying_pairs = prev.get("qualifying_pairs", {})
        scanned_pairs = set(tuple(p) for p in prev.get("scanned_pairs", []))
        total_found = sum(qualifying_pairs.values())
        print(f"♻️  Resuming from {args.resume}")
        print(f"   Already scanned: {len(scanned_pairs)} pairs, {total_found} polytopes found")
        print()

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else "data", exist_ok=True)

    start_time = time.time()
    pairs_done = 0

    for h11, h21 in all_pairs:
        if (h11, h21) in scanned_pairs:
            pairs_done += 1
            continue

        try:
            polys = fetch_polytopes(
                h11=h11,
                h21=h21,
                lattice="N",
                limit=args.limit,
                as_list=True
            )
            count = len(polys)
        except Exception as e:
            print(f"  ⚠️  ({h11},{h21}): fetch failed — {e}")
            scanned_pairs.add((h11, h21))
            pairs_done += 1
            continue

        scanned_pairs.add((h11, h21))
        pairs_done += 1

        if count > 0:
            key = f"({h11},{h21})"
            chi = 2 * (h11 - h21)
            qualifying_pairs[key] = count
            total_found += count

            cap_warning = " ⚠️  MAY BE CAPPED" if count >= args.limit else ""
            print(f"  ✅ ({h11:3d},{h21:3d})  χ={chi:+d}  {count:6,} polytopes{cap_warning}")

        # Progress every 50 pairs
        if pairs_done % 50 == 0:
            elapsed = time.time() - start_time
            remaining = len(all_pairs) - pairs_done
            rate = pairs_done / max(elapsed, 1)
            eta_seconds = remaining / max(rate, 0.001)
            eta_str = time.strftime("%Hh %Mm %Ss", time.gmtime(eta_seconds))
            print(f"     [{pairs_done}/{len(all_pairs)} pairs] {total_found:,} qualifying polytopes found | ETA: {eta_str}")

            # Save progress
            _save(args.output, qualifying_pairs, scanned_pairs, total_found, args)

    # Final save
    _save(args.output, qualifying_pairs, scanned_pairs, total_found, args)

    elapsed_total = time.time() - start_time

    print("\n" + "="*55)
    print(f"✅ HODGE SCAN COMPLETE")
    print(f"h11 Range:                  {args.h11_min} — {args.h11_max}")
    print(f"Pairs Queried:              {len(scanned_pairs):,}")
    print(f"Qualifying Pairs Found:     {len(qualifying_pairs):,}")
    print(f"Total Polytopes Found:      {total_found:,}")
    print(f"Runtime:                    {time.strftime('%Hh %Mm %Ss', time.gmtime(elapsed_total))}")
    print()
    print(f"Qualifying Pairs (sorted by count):")
    for pair, count in sorted(qualifying_pairs.items(), key=lambda x: -x[1]):
        h11, h21 = map(int, pair.strip("()").split(","))
        chi = 2 * (h11 - h21)
        cap = " ⚠️ may be capped" if count >= args.limit else ""
        print(f"   {pair:12s}  χ={chi:+d}   {count:6,} polytopes{cap}")
    print()
    print(f"Results saved to:           {args.output}")
    print("="*55)
    print()
    print("Next step: pass results to DeepSpaceHarvester.py")
    print("Each polytope above yields many universes via triangulation.")
    if any(v >= args.limit for v in qualifying_pairs.values()):
        print()
        print("⚠️  WARNING: Some pairs hit the fetch limit. The true polytope")
        print(f"   count for those pairs may exceed {args.limit:,}. Re-run with")
        print(f"   --limit=50000 to check.")