"""
Evaluation harness. Ingests the 6 starter documents (3 Delhivery + 3 macro) and runs
comparison, then prints a summary of relations found, so the 4 required cases
(corroborates / contradicts / reconcilable / unrelated) can be manually reviewed.

Usage:
    python -m backend.eval               # process all 6 docs + compare
    python -m backend.eval --fresh-db    # wipe factloom.db first

Requires GEMINI_API_KEY (and optionally GROQ_API_KEY) in .env, and network access to
generativelanguage.googleapis.com / api.groq.com. This performs real LLM calls and will
consume API quota — each document is chunked into ~8-page windows, so a 100-page PDF is
roughly 12-13 extraction calls plus a handful of comparison calls.
"""
import os
import sys
import glob

from backend import store
from backend.extract import process_pdf
from backend.compare import compare_new_facts

DATASETS = [
    "data/starter-datasets/delhivery/*.pdf",
    "data/starter-datasets/india-macroeconomy/*.pdf",
]


def main():
    if "--fresh-db" in sys.argv:
        db_path = store.DB_PATH
        if os.path.exists(db_path):
            os.remove(db_path)
        print(f"removed {db_path}")

    store.init_db()

    pdf_paths = []
    for pattern in DATASETS:
        pdf_paths.extend(sorted(glob.glob(pattern)))

    print(f"processing {len(pdf_paths)} documents...\n")
    for path in pdf_paths:
        print(f"--- {os.path.basename(path)} ---")
        facts = process_pdf(path)
        print(f"  extracted {len(facts)} facts")
        compare_new_facts(facts)

    relations = store.get_all_relations()
    print(f"\n{len(relations)} total relations found:\n")
    by_type = {}
    for r in relations:
        by_type.setdefault(r["relation_type"], []).append(r)

    for rel_type in ("corroborates", "contradicts", "reconcilable"):
        rels = by_type.get(rel_type, [])
        print(f"=== {rel_type} ({len(rels)}) ===")
        for r in rels[:5]:
            fa = store.get_fact(r["fact_a_id"])
            fb = store.get_fact(r["fact_b_id"])
            print(f"  [{fa['document_id']} p{fa['page_no']}] {fa['entity']} {fa['metric']}={fa['value']}{fa['unit'] or ''}"
                  f"  <->  [{fb['document_id']} p{fb['page_no']}] {fb['entity']} {fb['metric']}={fb['value']}{fb['unit'] or ''}")
            print(f"    reason={r['reason']} confidence={r['confidence']}  {r['explanation']}")
        print()

    print("Note: 'unrelated' facts are not stored as relations by design (see compare.py) —")
    print("verify Case 4 by checking that dissimilar-metric facts never appear above.")


if __name__ == "__main__":
    main()
