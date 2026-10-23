#!/usr/bin/env python3
"""
expand.py -- the "planner".

Turns the raw link list into many small, balanced jobs so that a 45 GB album
never lands on one runner:

  1. every link is listed with gallery-dl (nothing is downloaded) to learn how
     many files it contains and, when the site says so, how big they are
  2. the files are cut into chunks (CHUNK_GB each, at most CHUNK_MAX_ITEMS files)
  3. every chunk becomes one parallel job that downloads only its slice
     (gallery-dl --range a-b) and cleans up after itself

Env: INPUT_LINKS, CHUNK_GB (8), CHUNK_MAX_ITEMS (40), UNKNOWN_ITEM_MB (150),
     COOKIES, CREATE_RELEASES (1/0), GITHUB_RUN_NUMBER, GITHUB_REPOSITORY, GITHUB_SHA
"""
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

GB = 1024 ** 3
MB = 1024 ** 2
CHUNK_BYTES = float(os.getenv("CHUNK_GB", "8")) * GB
MAX_ITEMS = int(os.getenv("CHUNK_MAX_ITEMS", "40"))
UNKNOWN = int(float(os.getenv("UNKNOWN_ITEM_MB", "150")) * MB)
MAX_JOBS = 256          # GitHub matrix limit
MAX_ASSETS = 800        # a release holds at most 1000 files
RUN = os.getenv("GITHUB_RUN_NUMBER", "0")
IMG_X = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".avif", ".heic"}
VID_X = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".ts", ".wmv", ".3gp"}
AUD_X = {".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma"}
TITLE_KEYS = ("album_name", "album_title", "title", "album", "gallery", "folder", "name", "subcategory_name")
DIRECT_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".avif", ".heic",
              ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".flv", ".ts", ".wmv", ".3gp",
              ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma"}


def read_links():
    raw = os.environ.get("INPUT_LINKS", "").strip()
    if not raw:
        lines = []
        if Path("links.txt").exists():
            for line in open("links.txt", encoding="utf-8"):
                line = line.split(" #")[0].strip()
                if line and not line.startswith("#"):
                    lines.append(line)
        raw = "\n".join(lines)
    links = []
    for tok in re.split(r"\s+", raw):
        for u in re.split(r",(?=https?://)", tok):
            u = u.strip().strip(",")
            if u and u not in links:
                links.append(u)
    return links


def target_of(url):
    ext = Path(urlparse(url).path).suffix.lower()
    return "direct://" + url if ext in DIRECT_EXT else url


def list_items(url):
    """Returns {"sizes": [...], "kinds": [...], "title": str} (size 0 = unknown), or None if listing failed."""
    cmd = ["gallery-dl", "-J", "--retries", "4"]
    if Path("cookies.txt").exists():
        cmd += ["--cookies", "cookies.txt"]
    try:
        p = subprocess.run(cmd + [target_of(url)], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=900)
        data = json.loads(p.stdout)
    except Exception as e:
        print(f"::warning::could not list {url} ({e.__class__.__name__}); it will run as one job")
        return None
    sizes, kinds, title = [], [], ""
    for m in data:
        if not (isinstance(m, list) and m):
            continue
        meta = next((x for x in m[1:] if isinstance(x, dict)), {})
        if not title:  # album name lives in the directory / file metadata
            for key in TITLE_KEYS:
                v = meta.get(key)
                if isinstance(v, str) and v.strip():
                    title = v.strip()
                    break
        if m[0] == 3:  # 3 = file URL message
            size = 0
            for key in ("size", "filesize", "file_size", "bytes"):
                v = meta.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    size = int(v)
                    break
            ext = "." + str(meta.get("extension") or "").lower().lstrip(".")
            if ext == ".":
                ext = Path(urlparse(str(m[1])).path).suffix.lower()
            kinds.append("IMG" if ext in IMG_X else "VID" if ext in VID_X
                         else "AUD" if ext in AUD_X else "FILE")
            sizes.append(size)
    if not sizes:
        return None
    if not title:
        title = Path(urlparse(url).path).name or urlparse(url).netloc
    return {"sizes": sizes, "kinds": kinds, "title": re.sub(r"\s+", " ", title)}


def label_of(info):
    """IMG / VID / AUD / FILE: what most of the album is (by bytes, else by count)."""
    weight = {}
    for size, k in zip(info["sizes"], info["kinds"]):
        weight[k] = weight.get(k, 0) + (size or UNKNOWN)
    return max(weight, key=weight.get)


def chunk(sizes, cap, max_items):
    out, start, acc, n = [], 1, 0, 0
    for i, s in enumerate(sizes, 1):
        s = s or UNKNOWN
        if n and (acc + s > cap or n >= max_items):
            out.append((start, i - 1))
            start, acc, n = i, 0, 0
        acc += s
        n += 1
    if n:
        out.append((start, len(sizes)))
    return out


def plan(links, listed):
    cap, max_items = CHUNK_BYTES, MAX_ITEMS
    while True:
        per_link = {}
        for idx, url in enumerate(links, 1):
            info = listed.get(url)
            per_link[idx] = chunk(info["sizes"], cap, max_items) if info else None
        total = sum(len(c) if c else 1 for c in per_link.values())
        if total <= MAX_JOBS or cap > 40 * GB:
            return per_link, cap
        cap, max_items = cap * 1.5, max_items * 2  # too many jobs -> make chunks bigger


def precreate_releases(include, tags):
    """Create every release up front, in a fixed order, so GitHub lists them in order.

    GitHub sorts releases by creation time (newest first). If each fetch job created its
    own release, the order would depend on which job finished first -> scrambled numbers.
    So the releases are created here, highest number first, lowest last: the lowest
    number ends up on top of the list. Releases that stay empty are deleted later by
    number_releases.py.
    """
    if os.getenv("CREATE_RELEASES", "1") != "1":
        return
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    sha = os.environ.get("GITHUB_SHA", "")
    if not repo:
        return
    order = []
    for e in include:
        if e["tag"] not in order:
            order.append(e["tag"])
    for tag in reversed(order):
        cmd = ["gh", "release", "create", tag, "--repo", repo,
               "--title", tags.get(tag, tag), "--notes", ""]
        if sha:
            cmd += ["--target", sha]
        ok = False
        for _ in range(3):
            if subprocess.run(cmd, capture_output=True, text=True).returncode == 0:
                ok = True
                break
            time.sleep(3)
        if not ok:
            print(f"::warning::could not pre-create release {tag}; its fetch job will create it")
        time.sleep(1.2)  # release timestamps have 1 second resolution


def main():
    if os.getenv("COOKIES"):
        Path("cookies.txt").write_text(os.environ["COOKIES"])
    links = read_links()
    with ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(list_items, links))
    listed = dict(zip(links, results))
    per_link, cap = plan(links, listed)
    if cap != CHUNK_BYTES:
        print(f"::notice::many files -> chunk size raised to {cap / GB:.0f} GB to stay under {MAX_JOBS} jobs")

    include, rows, tags = [], [], {}   # tags: release tag -> release title
    for idx, url in enumerate(links, 1):
        chunks = per_link[idx]
        if not chunks:  # listing failed / empty -> one plain job, fetch.py reports what happens
            include.append({"job": f"{idx}", "idx": idx, "part": "", "range": "", "url": url,
                            "tag": f"run-{RUN}-{idx}"})
            rows.append((idx, f"FILE | {Path(urlparse(url).path).name or url}", url, "?", "?", 1))
            tags[f"run-{RUN}-{idx}"] = f"FILE | {Path(urlparse(url).path).name or url}"[:100]
            continue
        info = listed[url]
        sizes = info["sizes"]
        title = f"{label_of(info)} | {info['title']}"
        per_chunk_tag = len(sizes) > MAX_ASSETS
        for k, (a, b) in enumerate(chunks, 1):
            part = f"p{k:02d}" if len(chunks) > 1 else ""
            rng = f"{a}-{b}" if len(chunks) > 1 else ""
            tag = f"run-{RUN}-{idx}" + (f"-{part}" if per_chunk_tag and part else "")
            include.append({"job": f"{idx}-{part}" if part else f"{idx}", "idx": idx,
                            "part": part, "range": rng, "url": url, "tag": tag})
            if tag not in tags:
                tags[tag] = (f"{title} (part {k}/{len(chunks)})" if per_chunk_tag else title)[:100]
        known = sum(sizes)
        unknown = sum(1 for s in sizes if not s)
        est = f"{known / GB:.1f} GB" + (f" + {unknown} unknown" if unknown else "")
        rows.append((idx, title, url, len(sizes), est, len(chunks)))

    if len(include) > MAX_JOBS:
        print(f"::warning::{len(include)} jobs, GitHub allows {MAX_JOBS}; using the first {MAX_JOBS}")
        include = include[:MAX_JOBS]

    # Pass the title along to the fetch jobs (they create the release only if it is missing).
    for e in include:
        e["title"] = tags.get(e["tag"], e["tag"])
    precreate_releases(include, tags)

    summary = ["## Plan", "| # | Release | Link | Files | Size | Jobs |", "|---|---|---|---|---|---|"]
    summary += [f"| {i} | {t.replace('|', chr(92) + '|')} | {u} | {n} | {s} | {j} |" for i, t, u, n, s, j in rows]
    text = "\n".join(summary)
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text + "\n")
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"matrix={json.dumps({'include': include})}\n")
            f.write(f"count={len(include)}\n")
    print(f"{len(include)} job(s) queued")


if __name__ == "__main__":
    main()
