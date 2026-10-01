# Related work

Where each sub-project sits against the work it builds on, and what it does
that the cited work does not. BibTeX for every entry is in
[`references.bib`](references.bib), each checked against its DOI or arXiv
record.

## collapse: quantum models of classical time series

**Computational mechanics** (Crutchfield & Young 1989,
`crutchfield1989inferring`) defines the causal states of a process and its
statistical complexity `C_μ`. **CSSR** (Shalizi & Shalizi 2004,
`shalizi2004cssr`) estimates them from data by merging histories with a
statistical test. This project uses the same test, after float equality was
shown to merge nothing (see [`DECISIONS.md`](DECISIONS.md)).

**Quantum causal states** (Gu, Wiesner, Rieper & Vedral 2012,
`gu2012quantum`) showed that non-orthogonal quantum states can store a
process's past with memory `C_q < C_μ`. That is a statement about
*representation* built from a known process. Here it is checked on
*estimated* causal states of a real series, and kept strictly apart from
prediction (claims C1 and C2 in [`HYPOTHESES.md`](HYPOTHESES.md)).

**Hidden quantum Markov models** (Monras, Beige & Wiesner 2011,
`monras2011hqmm`; Srinivasan, Gordon & Boots 2018, `srinivasan2018hqmm`)
define and learn Kraus-operator models of sequences, and report cases where
an HQMM beats an HMM. What this project adds is the comparison at **matched
free-parameter count**, over at least five seeds, on one temporal split
declared up front. On Seattle precipitation the HQMM does not earn its
parameters, and that null result is the headline.

**Leggett–Garg inequalities** (Leggett & Garg 1985,
`leggett1985macrorealism`; review by Emary, Lambert & Nori 2014,
`emary2014lgi`) test macrorealism with invasive measurements at different
times. This project shows why that test says nothing about a passively
recorded series: with an aligned estimator `K3 ≤ 1` is an identity, so a
"violation" can only be an estimator bug.

## lawlearn: learning the law from trajectories

**Hamiltonian and Lagrangian neural networks** (Greydanus, Dzamba &
Yosinski 2019, `greydanus2019hnn`; Cranmer et al. 2020, `cranmer2020lnn`)
build conservation structure into the network. **SINDy** (Brunton, Proctor &
Kutz 2016, `brunton2016sindy`) recovers equations by sparse regression over
a function library. An LNN was tried here and failed: `L` is fixed only up to
scale, an additive constant and a total time derivative, and the optimiser
drifted along the scale gauge. This project instead learns the *canonical
momentum* `p(v)` and the potential `V(x)`, which needs first derivatives only
and no Hessian inversion, and then fixes the remaining gauges explicitly
([`lawlearn/`](lawlearn/)). Its question is narrower and measured: can a physical constant
(the speed of light) be read out of slow motion alone, what error floor does
the readout itself impose, and does the physics structure make the network
extrapolate? It does not; only the extracted formula does (claim C4b).
