"""Generate the chapters of the book from the lecture sources.

Run this script after editing a lecture, then compile book.tex with pdflatex three times
(for the table of contents and the cross-references):

    python make_book.py
    pdflatex book.tex && pdflatex book.tex && pdflatex book.tex

The lectures themselves are not changed. For every lecture, the script writes
chapters/<name>.tex with the body of the lecture (between \\begin{document} and
\\end{document}) as one chapter, and chapters/macros.tex with the macros of all
lecture preambles. Labels get a prefix per lecture (L1-, L2-, ..., and A-, B- for the
appendices), so that labels such as sec:exercises, which every lecture has, do not clash.
"""
import re
from pathlib import Path

BOOK = Path(__file__).resolve().parent
ROOT = BOOK.parent
OUT = BOOK / "chapters"

# (source, label prefix, chapter file). The appendices are lettered A and B in the book, as in
# their own section numbers, and their label prefixes are these letters.
CHAPTERS = [
    ("Lecture1/lecture1.tex", "L1-", "lecture1.tex"),
    ("Lecture2/lecture2.tex", "L2-", "lecture2.tex"),
    ("Lecture3/lecture3.tex", "L3-", "lecture3.tex"),
    ("Lecture4/lecture4.tex", "L4-", "lecture4.tex"),
    ("Lecture3/appendix_topology.tex", "A-", "appendix_topology.tex"),
    ("Lecture4/appendix_measure.tex", "B-", "appendix_measure.tex"),
]
PREFIXES = tuple(prefix for _, prefix, _ in CHAPTERS)

# Wording that only makes sense in the separate lecture notes. The replacements are made in this
# order, so a phrase must come before the shorter phrases that it contains.
REWORDING = {
    "Lecture1/lecture1.tex": [
        ("this is proved in Lecture~2", "this is proved in Theorem~\\ref{L2-thm:orthogonal}"),
    ],
    "Lecture2/lecture2.tex": [
        ("with $k$-means (Lecture~4)", "with $k$-means (Section~\\ref{L4-sec:latent})"),
    ],
    "Lecture3/lecture3.tex": [
        ("(by the proposition on coercive functions in the separate appendix on topology, differential topology and functional analysis)",
         "(by Proposition~\\ref{A-prop:coercive})"),
        ("Lecture~4 proves that SGD converges", "Theorem~\\ref{L4-thm:sgd} proves that SGD converges"),
        ("(this is proved in the separate appendix on topology, differential topology and functional analysis)",
         "(Theorem~\\ref{A-thm:extreme-value})"),
        ("Its proof, which uses the mean value theorem in each coordinate, is given in the separate appendix on topology, differential topology and functional analysis.",
         "Its proof, which uses the mean value theorem in each coordinate, is given in Proposition~\\ref{A-prop:C1}."),
        ("The first part is proved in the separate appendix on topology, differential topology and functional analysis, with the implicit function theorem.",
         "The first part is Theorem~\\ref{A-thm:kkt-proof}, which is proved with the implicit function theorem."),
        ("(see the separate appendix on topology, differential topology and functional analysis)",
         "(Corollary~\\ref{A-cor:parameter-integrals})"),
        ("the separate appendix on topology, differential topology and functional analysis",
         "Appendix~\\ref{A-chapter} on topology, differential topology and functional analysis"),
        ("The separate appendix carries out this proof.",
         "Theorem~\\ref{A-thm:picard-lindelof} carries out this proof."),
        ("the Fubini--Tonelli theorem in the appendix to Lecture~4",
         "the Fubini--Tonelli theorem, Theorem~\\ref{B-thm:fubini-tonelli}"),
        ("the appendix to Lecture~4 proves this with the dominated convergence theorem",
         "Example~\\ref{B-ex:escaping-mass} proves this with the dominated convergence theorem"),
    ],
    "Lecture4/lecture4.tex": [
        ("(by the proposition on uniform continuity in the appendix to Lecture~3)",
         "(by Proposition~\\ref{A-prop:uniform-continuity})"),
        ("the likelihood ratio test in the appendix to this lecture",
         "the likelihood ratio test of Example~\\ref{B-ex:martingales}"),
        ("The appendix to this lecture", "Appendix~\\ref{B-chapter}"),
        ("the appendix to this lecture", "Appendix~\\ref{B-chapter}"),
    ],
    "Lecture3/appendix_topology.tex": [
        ("the appendix to Lecture~4", "Appendix~\\ref{B-chapter}"),
    ],
    "Lecture4/appendix_measure.tex": [
        ("the appendix to Lecture~3 proves it with the Riesz representation theorem",
         "Theorem~\\ref{A-thm:radon-nikodym-proof} proves it with the Riesz representation theorem"),
        ("the appendix to Lecture~3", "Appendix~\\ref{A-chapter}"),
    ],
}

# Running heads for chapters whose titles do not fit into the page header.
RUNNING_HEAD = {
    "Lecture3/appendix_topology.tex": "Topology and Functional Analysis",
}


def in_comment(text, i):
    """Whether position i of text lies in a comment."""
    line = text[text.rfind("\n", 0, i) + 1:i]
    return re.search(r"(?<!\\)%", line) is not None


def group_end(text, i):
    """Index after the brace group that starts at text[i] == '{', skipping comments."""
    depth = 0
    while True:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "%":
            i = text.index("\n", i)
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1


def definitions(preamble):
    """The \\newcommand and \\DeclareMathOperator statements of a preamble, as (name, text) pairs."""
    text = preamble
    result = []
    for m in re.finditer(r"\\(newcommand|DeclareMathOperator\*?)\s*", text):
        if in_comment(text, m.start()):
            continue
        i = m.end()
        if text[i] == "{":
            j = group_end(text, i)
            name = text[i + 1:j - 1].strip()
        else:
            name = re.match(r"\\[a-zA-Z]+", text[i:]).group(0)
            j = i + len(name)
        while True:  # optional [number of arguments] and [default]
            k = len(text[j:]) - len(text[j:].lstrip())
            if text[j + k] != "[":
                break
            j = text.index("]", j + k) + 1
        j += len(text[j:]) - len(text[j:].lstrip())
        j = group_end(text, j)
        result.append((name, text[m.start():j]))
    return result


def title(preamble):
    t = re.search(r"\\title\{([^}]*)\}", preamble).group(1)
    return re.sub(r"^Lecture \d+(, Appendix)?:\s*", "", t)


def chapter(source, prefix):
    text = (ROOT / source).read_text(encoding="utf-8")
    begin = text.index("\\begin{document}")
    preamble = text[:begin]
    body = text[begin + len("\\begin{document}"):text.index("\\end{document}")]
    body = re.sub(r"^\s*\\(maketitle|tableofcontents)\s*\n", "", body, flags=re.M)
    body = re.sub(r"\\label\{([^}]*)\}", lambda m: "\\label{%s%s}" % (prefix, m.group(1)), body)
    # References to other chapters (the appendix to Lecture 3 refers to Lecture 3 as L3-...) keep their prefix.
    body = re.sub(r"\\ref\{([^}]*)\}",
                  lambda m: "\\ref{%s}" % (m.group(1) if m.group(1).startswith(PREFIXES) else prefix + m.group(1)),
                  body)
    # In the book, "Theorem~\ref{L3-...} of Lecture~3" already reads "Theorem 3.10.6", and
    # "Theorem~\ref{B-...} of the appendix to Lecture~4" reads "Theorem B.3.2".
    body = re.sub(r"(\\ref\{L(\d)-[^}]*\}) of Lecture~\2", r"\1", body)
    body = re.sub(r"(\\ref\{[AB]-[^}]*\}) of the appendix to Lecture~\d", r"\1", body)
    for old, new in REWORDING.get(source, []):
        if old in body:
            body = body.replace(old, new)
        else:
            print(f"warning: '{old}' not found in {source}")
    return preamble, (
        f"% Generated by make_book.py from {source}; edit the lecture, not this file.\n"
        f"\\chapter{{{title(preamble)}}}\\label{{{prefix}chapter}}\n"
        + (f"\\chaptermark{{{RUNNING_HEAD[source]}}}\n" if source in RUNNING_HEAD else "")
        + "% The group keeps settings such as \\emergencystretch inside the chapter.\n"
        "\\begingroup\n" + body.strip() + "\n\\endgroup\n")


def main():
    OUT.mkdir(exist_ok=True)
    macros = {}
    for source, prefix, name in CHAPTERS:
        preamble, tex = chapter(source, prefix)
        (OUT / name).write_text(tex, encoding="utf-8", newline="\n")
        for macro, definition in definitions(preamble):
            if macro not in macros:
                macros[macro] = definition
            elif macros[macro] != definition:
                print(f"warning: {macro} is defined differently in {source}; keeping the first definition")
        print(f"{source} -> chapters/{name}")
    (OUT / "macros.tex").write_text(
        "% Generated by make_book.py from the lecture preambles; edit the lectures, not this file.\n"
        + "\n".join(macros.values()) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(macros)} macros -> chapters/macros.tex")


if __name__ == "__main__":
    main()
