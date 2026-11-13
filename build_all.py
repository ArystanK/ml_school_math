"""Compile every LaTeX document of the course and build all PDFs.

Usage (from any directory):

    python build_all.py            build everything that changed
    python build_all.py --force    rebuild everything from scratch (latexmk -g)
    python build_all.py --only practice4 --only book
                                   build only these documents (and what they depend on)
    python build_all.py --list     print the build order and exit
    python build_all.py --clean    delete the auxiliary files (keeps the PDFs)

Every .tex file with a \\documentclass is a document, except the generated chapters of the
book. Documents that cite other documents with \\externaldocument (xr-hyper) are compiled
after them, so the order follows the dependencies automatically, also for new files:

    lecture3 -> lecture4 -> appendix_measure -> appendix_topology
    practiceN -> practiceN_answers,   midterm -> midterm_solutions,   final -> final_solutions

Before the book, Book/make_book.py regenerates its chapters from the lecture sources.
Each document is compiled with latexmk, which reruns pdflatex until the cross-references are
stable and skips documents whose sources (including the .aux files of the documents they
cite) have not changed. If a document fails, the documents that depend on it are skipped,
and the script ends with exit code 1.
"""
import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOOK = ROOT / "Book" / "book.tex"
MAKE_BOOK = ROOT / "Book" / "make_book.py"
SKIP_DIRS = {ROOT / "Book" / "chapters"}

COMMENT = re.compile(r"(?<!\\)%.*")
DOCUMENTCLASS = re.compile(r"^\s*\\documentclass", re.M)
EXTERNAL = re.compile(r"\\externaldocument\s*(?:\[[^\]]*\])?\s*\{([^}]*)\}")
LOG_ERROR = re.compile(r"^(!.*|.*:\d+: .*)$")
LOG_WARNINGS = {
    "undefined references": re.compile(r"There were undefined references|Reference `[^']*' on page \d+ undefined"),
    "multiply defined labels": re.compile(r"There were multiply-defined labels"),
    "undefined citations": re.compile(r"Citation `[^']*' on page \d+ undefined"),
}

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")


def strip_comments(text):
    return "\n".join(COMMENT.sub("", line) for line in text.splitlines())


def find_documents():
    """All LaTeX documents in the repository, with the documents they depend on."""
    docs = {}
    for path in sorted(ROOT.rglob("*.tex")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        if any(skip in path.parents for skip in SKIP_DIRS):
            continue
        text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        if not DOCUMENTCLASS.search(text):
            continue
        deps = []
        for name in EXTERNAL.findall(text):
            dep = (path.parent / name.strip()).resolve()
            if dep.suffix != ".tex":
                dep = dep.with_name(dep.name + ".tex")
            deps.append(dep)
        docs[path.resolve()] = deps
    return docs


def build_order(docs):
    """Sort the documents so that every document comes after the documents it cites."""
    order, state = [], {}

    def visit(doc, chain):
        if state.get(doc) == "done":
            return
        if state.get(doc) == "active":
            cycle = chain[chain.index(doc):] + [doc]
            sys.exit("Circular \\externaldocument references: " + " -> ".join(rel(d) for d in cycle))
        state[doc] = "active"
        for dep in docs[doc]:
            if dep not in docs:
                sys.exit(f"{rel(doc)} cites {rel(dep)}, which does not exist")
            visit(dep, chain + [doc])
        state[doc] = "done"
        order.append(doc)

    def priority(doc):
        # Lectures first, the book (generated from the lectures) last.
        top = doc.relative_to(ROOT).parts[0]
        return (2 if doc == BOOK else 0 if top.startswith("Lecture") else 1, rel(doc))

    for doc in sorted(docs, key=priority):
        visit(doc, [])
    return order


def with_dependencies(selected, docs):
    needed, stack = set(), list(selected)
    while stack:
        doc = stack.pop()
        if doc not in needed:
            needed.add(doc)
            stack.extend(docs[doc])
    return needed


def rel(path):
    return path.relative_to(ROOT).as_posix()


def log_problems(doc):
    """The first error lines and the warnings of a document's log file."""
    log = doc.with_suffix(".log")
    if not log.exists():
        return [], []
    text = log.read_text(encoding="utf-8", errors="replace")
    errors = [line for line in text.splitlines() if LOG_ERROR.match(line)][:5]
    warnings = [name for name, pattern in LOG_WARNINGS.items() if pattern.search(text)]
    return errors, warnings


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace")


def compile_document(doc, force):
    if doc == BOOK:
        result = run([sys.executable, MAKE_BOOK.name], MAKE_BOOK.parent)
        if result.returncode != 0:
            return False, [f"make_book.py failed:"] + result.stdout.strip().splitlines()[-5:], []
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-file-line-error"]
    if force:
        cmd.append("-g")
    result = run(cmd + [doc.name], doc.parent)
    retried = False
    if result.returncode != 0:
        # An interrupted or simultaneous compile can leave a truncated .aux, .toc or .out file,
        # which then breaks every later run. Clear them once and try again from scratch.
        run(["latexmk", "-c", doc.name], doc.parent)
        result = run(cmd + ["-g", doc.name], doc.parent)
        retried = True
    errors, warnings = log_problems(doc)
    if result.returncode != 0 and not errors:
        errors = result.stdout.strip().splitlines()[-5:]
    if retried and result.returncode == 0:
        warnings = ["needed a clean rebuild (stale auxiliary files)"] + warnings
    return result.returncode == 0, errors, warnings


def clean(order):
    for doc in order:
        run(["latexmk", "-c", doc.name], doc.parent)
        print(f"  cleaned  {rel(doc)}")


def main():
    parser = argparse.ArgumentParser(description="Compile every LaTeX document of the course.")
    parser.add_argument("--force", action="store_true", help="rebuild everything (latexmk -g)")
    parser.add_argument("--only", action="append", metavar="NAME",
                        help="build only documents whose path contains NAME, with their dependencies")
    parser.add_argument("--list", action="store_true", help="print the build order and exit")
    parser.add_argument("--clean", action="store_true", help="delete auxiliary files, keep the PDFs")
    args = parser.parse_args()

    if shutil.which("latexmk") is None:
        sys.exit("latexmk was not found on the PATH; install TeX Live or MiKTeX with latexmk.")

    docs = find_documents()
    order = build_order(docs)
    if args.only:
        selected = [doc for doc in order if any(name in rel(doc) for name in args.only)]
        if not selected:
            sys.exit("No document matches " + ", ".join(args.only))
        needed = with_dependencies(selected, docs)
        order = [doc for doc in order if doc in needed]

    if args.list:
        for i, doc in enumerate(order, 1):
            deps = ", ".join(rel(d) for d in docs[doc])
            print(f"{i:3}. {rel(doc)}" + (f"   (after {deps})" if deps else ""))
        return
    if args.clean:
        clean(order)
        return

    print(f"Building {len(order)} documents" + (" from scratch" if args.force else "") + "\n")
    failed, warned, start = [], [], time.time()
    for doc in order:
        broken = [dep for dep in docs[doc] if dep in failed]
        if broken:
            failed.append(doc)
            print(f"  SKIPPED  {rel(doc)}  (depends on failed {', '.join(rel(d) for d in broken)})")
            continue
        t = time.time()
        ok, errors, warnings = compile_document(doc, args.force)
        status = "ok" if ok else "FAILED"
        print(f"  {status:7}  {rel(doc)}  ({time.time() - t:.0f} s)")
        for line in errors if not ok else []:
            print(f"             {line}")
        if ok and warnings:
            warned.append(doc)
            print(f"             warning: {', '.join(warnings)}")
        if not ok:
            failed.append(doc)

    print(f"\n{len(order) - len(failed)} of {len(order)} documents built in {time.time() - start:.0f} s"
          + (f", {len(warned)} with warnings" if warned else "") + ".")
    if failed:
        print("Failed or skipped: " + ", ".join(rel(d) for d in failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
