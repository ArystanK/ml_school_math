# Mathematics for Machine Learning

Lecture notes, practice problems, sample exams and the syllabus of the course **Mathematics for Machine Learning**. The course covers the mathematics behind the machine learning, deep learning and reinforcement learning courses that follow it: linear algebra, vector calculus and optimization, probability and statistics.

**Start with the [syllabus](Syllabus/syllabus.pdf).** It has the schedule, the sections of the notes for every lecture, the homework, and the grading rules.

## Materials

| | Lecture notes | Practice problems | Answers and hints |
|---|---|---|---|
| Lecture 1: Linear Algebra | [lecture1.pdf](Lecture1/lecture1.pdf) | [practice1.pdf](Lecture1/practice1.pdf) | [practice1_answers.pdf](Lecture1/practice1_answers.pdf) |
| Lecture 2: Analytic Geometry | [lecture2.pdf](Lecture2/lecture2.pdf) | [practice2.pdf](Lecture2/practice2.pdf) | [practice2_answers.pdf](Lecture2/practice2_answers.pdf) |
| Lecture 3: Vector Calculus and Optimization | [lecture3.pdf](Lecture3/lecture3.pdf) | [practice3.pdf](Lecture3/practice3.pdf) | [practice3_answers.pdf](Lecture3/practice3_answers.pdf) |
| Lecture 4: Probability and Statistics | [lecture4.pdf](Lecture4/lecture4.pdf) | [practice4.pdf](Lecture4/practice4.pdf) | [practice4_answers.pdf](Lecture4/practice4_answers.pdf) |
| Appendix A: Topology, Differential Topology and Functional Analysis | [appendix_topology.pdf](Lecture3/appendix_topology.pdf) | | |
| Appendix B: Measure-Theoretic Probability | [appendix_measure.pdf](Lecture4/appendix_measure.pdf) | | |

- **[The book](Book/book.pdf)** contains all four lectures and both appendices in one volume, with one table of contents and working cross-references. The section numbers are the same as in the separate notes: §4.8 is Section 8 of Lecture 4.
- The four "Lectures" of the notes are not the same as the 11 lectures of the course. The syllabus says which sections each lecture covers, and which sections are self-study.
- Every lecture of the notes ends with key takeaways, exercises and programming exercises.
- The appendices are optional reading, for students who want the rigorous foundations: they prove the results of Lecture 3 that need topology, and they build probability on measure theory.

## Homework (60% of the grade)

- The homework consists of the **exercises and programming exercises marked "Homework."** in red at the end of the lecture notes: six exercises and one or two programming exercises per lecture, 66 exercises and 16 programming exercises in total. The syllabus lists them under each lecture.
- The other exercises are optional practice.
- Programming exercises use **Python 3 with NumPy and Matplotlib** (`pip install numpy matplotlib`; SciPy is useful for checking results). Implement the methods yourself, and use library functions only to check your results.
- Start on the homework right after each lecture, not just before the deadline.

## Practice problems (not graded)

- The practice problems follow the sections of the notes. Problems marked **Mandatory** cover the core skills: do them after each lecture. Problems marked ⋆ are harder or more surprising.
- The answers and hints are in a separate file. Try each problem before you look.

## Exams (40% of the grade)

- **Midterm** (90 minutes), after Lecture 6, on Lectures 1–6 of the course: linear algebra and analytic geometry.
- **Final** (90 minutes), after Lecture 11, on Lectures 7–11: calculus, optimization, probability and statistics, using the linear algebra where needed.
- Each exam has **five problems of 20 points** and a **bonus problem** for up to 10 extra points. The score is capped at 100.
- A one-page cheatsheet is handed out with each exam. No other notes and no calculators are allowed; the problems give the numerical values you need.
- To prepare, use:
  - the **sample exams with solutions**: [sample midterm](Exams/sample_midterm.pdf) ([solutions](Exams/sample_midterm_solutions.pdf)) and [sample final](Exams/sample_final.pdf) ([solutions](Exams/sample_final_solutions.pdf)). The real exams have the same format and level, but different problems;
  - the **approximate cheatsheets**: [midterm](Exams/approximate_midterm_cheatsheet.pdf) and [final](Exams/approximate_final_cheatsheet.pdf). The cheatsheet handed out at the exam is a shorter selection from these, so learn to work with all of them.

**To pass the course, you need at least 70% of the total grade.**

## How to work with the notes

- **Before a lecture**, skim the sections listed for it in the syllabus.
- **After a lecture**, reread the sections, do the mandatory practice problems, then the homework.
- **Self-study sections** (listed in the syllabus) are part of the course. The homework covers them, and the exams may include them. The self-study topics of Lectures 3–4 are discussed in a practice session before the midterm.
- **Notation:** vectors of ℝⁿ are bold columns, and matrices are capital letters. The introductions of Lectures 1, 3 and 4 explain the notation; Lecture 2 uses that of Lecture 1.
- The notes prove almost every result. On a first reading, it is fine to skip a long proof and come back to it; the examples and the key takeaways show what matters.

## Building the PDFs yourself (optional)

All PDFs are included, so you do not need to compile anything. If you want to rebuild them from the LaTeX sources, you need a TeX distribution with `latexmk` (TeX Live or MiKTeX) and Python 3. Then run, from the repository root:

```
python build_all.py             # build everything that changed
python build_all.py --only book  # build one document (and what it depends on)
python build_all.py --list       # show all documents in build order
```

The book is generated from the lecture sources by `Book/make_book.py`, which `build_all.py` runs automatically.

## Repository layout

```
Syllabus/       syllabus
Lecture1/       Lecture 1: notes, practice problems, answers
Lecture2/       Lecture 2: notes, practice problems, answers
Lecture3/       Lecture 3 and Appendix A
Lecture4/       Lecture 4 and Appendix B
Book/           all notes in one volume, and the script that generates it
Exams/          sample exams with solutions, approximate cheatsheets
build_all.py    compiles every document
```

## Found a mistake?

If you find an error or an unclear passage in the notes, please tell your instructor, with the file, the section and what you think is wrong.
