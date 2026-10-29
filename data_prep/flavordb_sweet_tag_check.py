"""Count how FlavorDB uses the descriptor "sweet", from the raw Pyrfume FlavorDB behaviour table.

Supports the Methods statement that the FlavorDB flavour percept "sweet" is a default tag rather than a
sensory annotation and was therefore excluded before consolidation (consolidate_flavor_datasets.py keeps the
odour percept "sweet"). Descriptors are parsed exactly as in that script: split on semicolons, stripped and
lower-cased.

Usage: python flavordb_sweet_tag_check.py --behavior <path to flavordb_behavior.csv> [--out counts.csv]
The Pyrfume FlavorDB archive (https://github.com/pyrfume/pyrfume-data, folder flavordb) provides the input.
"""
import argparse
import os

import pandas as pd


def parse(text):
    if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
        return set()
    return {t.strip().lower() for t in text.split(";") if t.strip()}


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--behavior", default=os.path.join(here, "..", "data", "raw", "pyrfume", "flavordb_behavior.csv"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    beh = pd.read_csv(args.behavior)
    flavour = beh["Flavor Percepts"].apply(parse)
    odour = beh["Odor Percepts"].apply(parse)
    f_sweet = flavour.apply(lambda s: "sweet" in s)
    o_sweet = odour.apply(lambda s: "sweet" in s)
    f_only = f_sweet & (flavour.apply(len) == 1) & (odour.apply(len) == 0)
    rows = [
        ("FlavorDB compounds in the behaviour table", len(beh)),
        ('flavour percept contains "sweet"', int(f_sweet.sum())),
        ('flavour percept is exactly "sweet"', int(flavour.apply(lambda s: s == {"sweet"}).sum())),
        ('flavour "sweet" with no other flavour or odour descriptor', int(f_only.sum())),
        ('odour percept contains "sweet" (retained)', int(o_sweet.sum())),
        ("both flavour and odour tagged sweet", int((f_sweet & o_sweet).sum())),
        ('either percept contains "sweet"', int((f_sweet | o_sweet).sum())),
        ("no flavour and no odour descriptor at all", int(((flavour.apply(len) == 0) & (odour.apply(len) == 0)).sum())),
        ("second most frequent flavour percept string", beh["Flavor Percepts"].value_counts().index[1]),
        ("its count", int(beh["Flavor Percepts"].value_counts().iloc[1])),
    ]
    out = pd.DataFrame(rows, columns=["quantity", "value"])
    print(out.to_string(index=False))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
