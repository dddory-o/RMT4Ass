"""Checks that every DOI in references_doi.csv resolves and prints the registered title (Crossref / DataCite) so you can compare it with the reference.
Needs internet. Run: python check_refs.py      Fix or remove any line marked MISMATCH or NOT FOUND before submission.
"""
import csv, json, urllib.request, urllib.parse

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ref-check/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

def lookup(doi):
    q = urllib.parse.quote(doi, safe="")
    try:
        t = get("https://api.crossref.org/works/" + q)["message"]
        return "crossref", (t.get("title") or [""])[0]
    except Exception:
        pass
    try:
        t = get("https://api.datacite.org/dois/" + q)["data"]["attributes"]
        return "datacite", (t.get("titles") or [{}])[0].get("title", "")
    except Exception:
        return None, ""

bad = 0
for row in csv.DictReader(open("references_doi.csv", encoding="utf-8")):
    if not row["doi"]:
        print("no DOI   ", row["key"]); continue
    src, title = lookup(row["doi"])
    if not src:
        bad += 1; print("NOT FOUND", row["key"], row["doi"]); continue
    words = [w for w in title.lower().replace(":", " ").split() if len(w) > 3][:4]
    ok = all(w in row["reference"].lower() for w in words)
    bad += (not ok)
    print(("ok       " if ok else "MISMATCH "), row["key"], row["doi"], "|", title[:80])
print("problems:", bad)
