"""Scarica generi MusicBrainz per i top-N artisti Last.fm (per play-count) → ~18 macro-generi.
Rate-limit 1.1s/req, checkpoint (riprende da artist_genre_map.tsv), certifi SSL.
Uso: fetch_mb_genres.py <N>
"""
import json, ssl, sys, time, urllib.request, urllib.error
from pathlib import Path

OUT = Path("/Users/lucaaliberti/Downloads/xsage-clean/outputs_results/lastfm")
PC = OUT / "artist_playcounts.tsv"; MAP = OUT / "artist_genre_map.tsv"
UA = "XSAGE-research/0.1 (academic; lucaaliberti898@gmail.com)"
try:
    import certifi; CTX = ssl.create_default_context(cafile=certifi.where())
except Exception:
    CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE

MACRO = [
    ("metal", ["metal", "metalcore", "grindcore"]),
    ("punk", ["punk", "hardcore", "emo"]),
    ("hip hop", ["hip hop", "rap", "trap", "grime"]),
    ("electronic", ["electronic", "techno", "house", "trance", "edm", "idm", "drum and bass",
                     "dubstep", "synth", "electro", "ambient", "downtempo", "breakbeat", "trip hop"]),
    ("jazz", ["jazz", "bebop", "swing", "fusion"]),
    ("classical", ["classical", "baroque", "orchestral", "opera"]),
    ("blues", ["blues"]),
    ("country", ["country", "americana", "bluegrass"]),
    ("folk", ["folk", "singer-songwriter", "acoustic"]),
    ("reggae", ["reggae", "ska", "dub", "dancehall"]),
    ("r&b/soul", ["soul", "r&b", "rnb", "funk", "motown", "disco"]),
    ("world", ["world", "latin", "afro", "reggaeton", "k-pop", "j-pop", "bossa", "flamenco", "celtic"]),
    ("experimental", ["experimental", "noise", "avant-garde", "industrial", "drone"]),
    ("pop", ["pop"]),
    ("rock", ["rock", "indie", "grunge", "shoegaze", "psychedelic", "garage", "britpop", "alternative"]),
]

def to_macro(genres):
    # genres ordinati per voti desc: mappa il PRIMO (primario) che cade in un bucket
    for g in genres:
        gl = g.lower()
        for macro, kws in MACRO:
            if any(k in gl for k in kws): return macro
    return genres[0] if genres else "other"

def fetch(mbid):
    url = f"https://musicbrainz.org/ws/2/artist/{mbid}?inc=genres+tags&fmt=json"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            d = json.load(urllib.request.urlopen(req, context=CTX, timeout=20))
            gs = sorted(d.get("genres", []), key=lambda x: -x.get("count", 0))
            names = [g["name"] for g in gs]
            if not names:
                ts = sorted(d.get("tags", []), key=lambda x: -x.get("count", 0))
                names = [t["name"] for t in ts][:5]
            return names
        except urllib.error.HTTPError as e:
            if e.code == 503: time.sleep(3 + attempt * 2); continue
            return None
        except Exception:
            time.sleep(2); continue
    return None

def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    done = set()
    if MAP.exists():
        for ln in MAP.read_text(encoding="utf-8").splitlines()[1:]:
            if ln: done.add(ln.split("\t")[0])
    rows = PC.read_text(encoding="utf-8").splitlines()[1:N + 1]
    fresh = not MAP.exists()
    f = MAP.open("a", encoding="utf-8")
    if fresh: f.write("mbid\tname\tplays\tmacro_genre\traw_genres\n")
    n_ok = n_none = 0
    for i, ln in enumerate(rows):
        _, mbid, name, plays, _ = ln.split("\t")
        if mbid in done: continue
        names = fetch(mbid)
        if names is None: n_none += 1; macro = "UNKNOWN"; raw = ""
        else: n_ok += 1; macro = to_macro(names); raw = "|".join(names[:5])
        f.write(f"{mbid}\t{name}\t{plays}\t{macro}\t{raw}\n"); f.flush()
        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{len(rows)}] ok={n_ok} none={n_none} last={name}->{macro}", flush=True)
        time.sleep(1.1)
    f.close()
    print(f"FATTO: {n_ok} con genere, {n_none} falliti, su {len(rows)} nuovi (top-{N})")

if __name__ == "__main__":
    sys.exit(main())
