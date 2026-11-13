#!/usr/bin/env python3
"""Compile every LaTeX document in the repository to PDF.

    python build.py                      build everything that is out of date
    python build.py Lecture3 midterm     build only these (and what they depend on)
    python build.py --force              recompile everything, even if up to date
    python build.py --list               show the documents and the build order
    python build.py --clean              delete auxiliary files (keeps the PDFs)

A document is any .tex file with a \\documentclass. A target is a document name
(lecture3, practice1_answers), a .tex path, or a directory (Lecture3, Exams).

Documents cite each other with xr-hyper (\\externaldocument), which reads the .aux file
of the other document, so the script reads these lines, compiles every document after
the documents it cites, and compiles independent documents in parallel. Each document is
compiled with latexmk (pdflatex), which reruns until the references are stable and skips
documents whose sources have not changed. Book/make_book.py is run before Book/book.tex.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {"__pycache__", "node_modules", "venv"}

# Commands run (from ROOT) before a document is compiled.
PRE_STEPS = {
    "Book/book.tex": [sys.executable, "Book/make_book.py"],
}

LATEXMK = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
           "-file-line-error", "-synctex=1"]

print_lock = threading.Lock()


def say(*args):
    with print_lock:
        print(*args, flush=True)


def rel(path):
    return path.relative_to(ROOT).as_posix()


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


class Doc:
    def __init__(self, path):
        self.path = path
        self.name = rel(path)
        self.text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        self.deps = set()  # documents whose .aux files this one reads

    def __repr__(self):
        return self.name


def find_documents():
    docs = {}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in SKIP_DIRS)
        for f in sorted(filenames):
            if f.endswith(".tex"):
                doc = Doc(Path(dirpath) / f)
                if re.search(r"^\s*\\documentclass", doc.text, re.M):
                    docs[doc.path.resolve()] = doc
    for doc in docs.values():
        for name in re.findall(r"\\externaldocument\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}", doc.text):
            name = name.strip()
            target = (doc.path.parent / (name if name.endswith(".tex") else name + ".tex")).resolve()
            if target in docs:
                doc.deps.add(docs[target])
            else:
                say(f"warning: {doc.name} cites {name}, which is not a document in this repository")
    return sorted(docs.values(), key=lambda d: d.name)


def build_order(docs):
    """Topological order: every document after the documents it cites."""
    order, done, visiting = [], set(), set()

    def visit(doc, chain):
        if doc in done:
            return
        if doc in visiting:
            cycle = " -> ".join(d.name for d in chain[chain.index(doc):] + [doc])
            sys.exit(f"error: circular \\externaldocument references: {cycle}")
        visiting.add(doc)
        for dep in sorted(doc.deps, key=lambda d: d.name):
            visit(dep, chain + [doc])
        visiting.discard(doc)
        done.add(doc)
        order.append(doc)

    for doc in docs:
        visit(doc, [])
    return order


def select(docs, targets, with_deps=True):
    """The documents named by the targets, plus (with_deps) everything they depend on."""
    by_stem = {}
    for doc in docs:
        by_stem.setdefault(doc.path.stem, []).append(doc)
    chosen = set()
    for t in targets:
        stem = t[:-4] if t.endswith(".tex") else t
        path = (Path.cwd() / t).resolve()
        if stem in by_stem:
            chosen.update(by_stem[stem])
        elif path.is_dir():
            found = [d for d in docs if path in d.path.resolve().parents]
            if not found:
                sys.exit(f"error: no LaTeX documents in {t}")
            chosen.update(found)
        elif path.is_file():
            found = [d for d in docs if d.path.resolve() == path]
            if not found:
                sys.exit(f"error: {t} is not a LaTeX document (no \\documentclass)")
            chosen.update(found)
        else:
            sys.exit(f"error: unknown target {t!r} (try --list)")
    stack = list(chosen) if with_deps else []
    while stack:
        for dep in stack.pop().deps:
            if dep not in chosen:
                chosen.add(dep)
                stack.append(dep)
    return [d for d in docs if d in chosen]


def tex_errors(doc, latexmk_output):
    """The error messages of a failed compilation."""
    log = doc.path.with_suffix(".log")
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.exists() else []
    out = []
    for i, line in enumerate(lines):
        if line.startswith("!") or re.search(r"\.\w+:\d+: ", line):
            out.extend(lines[i:i + 3])
            if len(out) >= 15:
                break
    if not out:
        out = latexmk_output.strip().splitlines()[-15:]
    return out


def undefined_refs(doc):
    log = doc.path.with_suffix(".log")
    if not log.exists():
        return []
    text = log.read_text(encoding="utf-8", errors="replace")
    return sorted(set(re.findall(r"Reference `([^']+)' on page \d+ undefined", text)))


def compile_doc(doc, force):
    start = time.time()
    if doc.name in PRE_STEPS:
        step = subprocess.run(PRE_STEPS[doc.name], cwd=ROOT, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, errors="replace")
        if step.returncode != 0:
            return False, (step.stdout + step.stderr).strip().splitlines()[-15:], time.time() - start
    cmd = LATEXMK + (["-g"] if force else []) + [doc.path.name]
    run = subprocess.run(cmd, cwd=doc.path.parent, stdin=subprocess.DEVNULL,
                         capture_output=True, text=True, errors="replace")
    output = run.stdout + run.stderr
    if run.returncode != 0:
        return False, tex_errors(doc, output), time.time() - start
    note = "built" if re.search(r"Run number \d+ of rule", output) else "up to date"
    refs = undefined_refs(doc)
    if refs:
        shown = ", ".join(refs[:5]) + (", ..." if len(refs) > 5 else "")
        note += f"; {len(refs)} undefined reference(s): {shown}"
    return True, note, time.time() - start


def build(docs, jobs, force):
    status = {}  # doc -> "ok" | "failed" | "skipped"
    pending = list(docs)
    running = {}
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        while pending or running:
            for doc in list(pending):
                if any(status.get(d) in ("failed", "skipped") for d in doc.deps):
                    pending.remove(doc)
                    status[doc] = "skipped"
                    failed = sorted(d.name for d in doc.deps
                                    if status.get(d) in ("failed", "skipped"))
                    say(f"  skip   {doc.name}  (depends on {', '.join(failed)}, which failed)")
                elif all(status.get(d) == "ok" for d in doc.deps):
                    pending.remove(doc)
                    running[pool.submit(compile_doc, doc, force)] = doc
            if not running:
                break
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                doc = running.pop(future)
                ok, info, seconds = future.result()
                status[doc] = "ok" if ok else "failed"
                pdf = rel(doc.path.with_suffix(".pdf"))
                if ok:
                    say(f"  ok     {pdf}  ({info}, {seconds:.1f}s)")
                else:
                    say(f"  FAILED {doc.name}  ({seconds:.1f}s)\n"
                        + "\n".join("         | " + line for line in info)
                        + f"\n         full log: {rel(doc.path.with_suffix('.log'))}")
    return status


def main():
    parser = argparse.ArgumentParser(
        description="Compile every LaTeX document in the repository to PDF.")
    parser.add_argument("targets", nargs="*",
                        help="document names, .tex files or directories (default: everything)")
    parser.add_argument("-f", "--force", action="store_true",
                        help="recompile even if the PDF is up to date")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count() or 4,
                        help="number of documents compiled in parallel (default: %(default)s)")
    parser.add_argument("--list", action="store_true",
                        help="list the documents in build order and exit")
    parser.add_argument("--clean", action="store_true",
                        help="delete auxiliary files with latexmk -c (the PDFs are kept)")
    args = parser.parse_args()

    if not shutil.which("latexmk"):
        sys.exit("error: latexmk not found on PATH (it comes with TeX Live and MiKTeX)")

    docs = find_documents()
    build_order(docs)  # fail early on circular references
    if args.targets:
        docs = select(docs, args.targets, with_deps=not args.clean)
    order = build_order(docs)

    if args.list:
        for doc in order:
            deps = sorted(d.name for d in doc.deps)
            print(doc.name + (f"   <- {', '.join(deps)}" if deps else ""))
        return 0

    if args.clean:
        for doc in order:
            subprocess.run(["latexmk", "-c", doc.path.name], cwd=doc.path.parent,
                           stdin=subprocess.DEVNULL, capture_output=True)
            say(f"  clean  {doc.name}")
        return 0

    start = time.time()
    say(f"Compiling {len(order)} document(s) with up to {args.jobs} in parallel...")
    status = build(order, max(1, args.jobs), args.force)
    failed = [d.name for d in order if status.get(d) == "failed"]
    skipped = [d.name for d in order if status.get(d) == "skipped"]
    ok = len(order) - len(failed) - len(skipped)
    say(f"\n{ok} ok, {len(failed)} failed, {len(skipped)} skipped in {time.time() - start:.1f}s")
    return 1 if failed or skipped else 0


if __name__ == "__main__":
    sys.exit(main())
