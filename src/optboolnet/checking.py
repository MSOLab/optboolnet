import os
import re
import tempfile
from nusmv import NuSMV

# NuSMV reserved keywords that cannot be used as variable identifiers.
# Variables whose names collide are prefixed with "v" (e.g. MAX -> vMAX).
_NUSMV_RESERVED = frozenset({
    # Module-level
    "MODULE", "DEFINE", "MDEFINE", "CONSTANTS", "VAR", "IVAR", "FROZENVAR",
    "ASSIGN", "TRANS", "INIT", "INVAR", "SPEC", "CTLSPEC", "LTLSPEC",
    "PSLSPEC", "COMPUTE", "INVARSPEC", "FAIRNESS", "JUSTICE", "COMPASSION",
    "ISA", "CONSTRAINT", "SIMPWFF", "CTLWFF", "LTLWFF", "PSLWFF", "COMPWFF",
    # COMPUTE operators (most likely to appear as bio variable names)
    "MAX", "MIN",
    # Set / type operators
    "IN", "UNION",
    # Types
    "BOOLEAN", "INTEGER", "REAL", "WORD", "WORD1", "BOOL",
    "SIGNED", "UNSIGNED", "ARRAY", "OF",
    # Built-in functions
    "COUNT", "EXTEND", "RESIZE", "SIZEOF", "TOINT", "SWCONST",
    # Case expression
    "CASE", "ESAC",
    # Temporal / path
    "NEXT", "SELF", "PROCESS",
    # Boolean literals
    "TRUE", "FALSE",
    # CTL / LTL single-letter operators that NuSMV reserves
    "EBF", "EBG", "ABF", "ABG",
    # Lowercase variants that NuSMV also reserves
    "mod", "union", "in", "xor", "xnor", "case", "esac", "next", "init",
    "process", "array", "of", "boolean", "integer", "real", "word", "self",
    "count", "extend", "resize", "sizeof", "toint", "signed", "unsigned",
})


def _nusmv_var(n):
    if isinstance(n, int):
        return "x%d" % n
    s = str(n)
    return ("v" + s) if s in _NUSMV_RESERVED else s


def _sanitize_smv_expr(expr):
    """Replace reserved NuSMV keywords used as identifiers in a string expression."""
    return re.sub(
        r'\b([A-Za-z_][A-Za-z0-9_]*)\b',
        lambda m: ("v" + m.group(1)) if m.group(1) in _NUSMV_RESERVED else m.group(1),
        expr,
    )


def _nusmv_model(bn, control=None, update_mode="synchronous"):
    """
    bn: minibn.CNFBooleanNetwork
    control: Control
    update_mode: synchronous, asynchronous
    """

    dom = bn.vars_list
    udom = ["u%s" % n for n in dom]
    var = _nusmv_var

    lines = ["MODULE main"]
    lines.append("VAR")
    for i in dom:
        lines.append("%s: boolean;" % var(i))
    lines.append("ASSIGN")
    for i in dom:
        if update_mode == "synchronous":
            lines.append("next(%s) := f%s;" % (var(i), i))
        else:
            lines.append("next(%s) := {%s, f%s};" % (var(i), var(i), i))

    lines.append("DEFINE")
    if control is None:
        control = {}
    for n in bn.vars_list:
        if n in control:
            lines.append(f"f{n} := {'TRUE' if control[n] else 'FALSE'};")
            continue
        clauses = bn.items_clause(n)
        if not clauses:
            lines.append(f"f{n} := FALSE;")
        elif len(clauses) == 1 and not clauses[0].args:
            lines.append(f"f{n} := TRUE;")
        else:

            def smv_or(clause):
                neg = [f"!{var(m)}" for m in clause.neg_literals]
                pos = [f"{var(m)}" for m in clause.pos_literals]
                expr = " | ".join(neg + pos)
                if len(neg + pos) > 1:
                    expr = f"({expr})"
                return expr

            smv_and = " & ".join((smv_or(clause) for clause in clauses))
            lines.append(f"f{n} := {smv_and};")

    if update_mode != "synchronous":
        lines.append(
            "FIXEDPOINTS := %s;" % (" & ".join(["%s = f%s" % (var(i), i) for i in dom]))
        )
        lines.append("TRANS")
        lines.append("  FIXEDPOINTS")
        if update_mode == "general":
            for i in dom:
                lines.append("| next(%s) != %s" % (var(i), var(i)))
        elif update_mode == "asynchronous":
            for i in dom:
                freeze = " & ".join(
                    ["next({0})={0}".format(var(j)) for j in dom if i != j]
                )
                freeze = " & %s" % freeze if freeze else ""
                lines.append("| next({0})!={0}{1}".format(var(i), freeze))
        lines.append(";")
    return "\n".join(lines) + "\n"


def _nusmv_state(dstate):
    def _expr(n, v):
        return f"{'!' if not v else ''}{_nusmv_var(n)}"

    return " & ".join((_expr(n, v) for n, v in dstate.items()))


def _nusmv_run(nusmv_input, smvfile, with_counterexample=False):
    """Write nusmv_input, invoke NuSMV, and return stdout as a string.

    with_counterexample: if True, omit -dcx so NuSMV includes the trace.
    """
    tmp_smvfile = smvfile is None
    if tmp_smvfile:
        _, smvfile = tempfile.mkstemp(suffix=".smv")
    mc = None
    try:
        with open(smvfile, "w") as fp:
            fp.write(nusmv_input)
        mc = NuSMV(smvfile)
        if with_counterexample:
            mc.opts.pop("dcx", None)
        return mc.check_output()
    finally:
        mc = None  # release NuSMV handles before unlinking (required on Windows)
        if tmp_smvfile:
            try:
                os.unlink(smvfile)
            except PermissionError:
                pass


def _nusmv_alltrue(nusmv_input, smvfile):
    output = _nusmv_run(nusmv_input, smvfile)
    return all(
        line.split()[-1] == "true"
        for line in output.split("\n")
        if line.startswith("-- specification ")
    )


def _parse_loop_length(output):
    """Return the number of states in the loop section of a NuSMV counterexample.

    NuSMV marks the start of the loop with '-- Loop starts here --'.
    Falls back to the total state count in the trace if no loop marker is found
    (e.g. fixed-point attractors where NuSMV omits the marker).
    """
    in_loop = False
    loop_count = 0
    total_count = 0
    for line in output.split("\n"):
        if "Loop starts here" in line:
            in_loop = True
        elif "-> State:" in line:
            total_count += 1
            if in_loop:
                loop_count += 1
    return loop_count if loop_count > 0 else (total_count if total_count > 0 else None)


def nusmv_check_attractor(
    bn, attractor, control=None, update_mode="synchronous", smvfile=None
):
    """
    Returns true if attractor is indeed an attractor of the bn

    bn: CNFBooleanNetwork
    attractor: Attractor
    control: Control
    update_mode: synchronous, asynchronous, general
    smvfile: if None use a temporary file
    """
    dstate = dict(zip(sorted(bn.vars_list), attractor.value_list[0]))
    dstate_smv = _nusmv_state(dstate)
    nusmv_input = _nusmv_model(bn, control=control, update_mode=update_mode)
    nusmv_input += f"INIT {dstate_smv};\n"
    nusmv_input += f"CTLSPEC AG EF ({dstate_smv});"
    return _nusmv_alltrue(nusmv_input, smvfile)


def nusmv_check_phenotype(bn, control=None, update_mode="synchronous", smvfile=None):
    """
    Returns true if all the attractors have p=1 constantly

    bn: CNFBooleanNetwork
    control: Control
    update_mode: synchronous, asynchronous, general
    smvfile: if None, uses a temporary file
    """
    nusmv_input = _nusmv_model(bn, control=control, update_mode=update_mode)
    nusmv_input += f"CTLSPEC EF AG {_sanitize_smv_expr(bn.phenotype)};"
    return _nusmv_alltrue(nusmv_input, smvfile)


def nusmv_check_phenotype_full(bn, control=None, update_mode="synchronous", smvfile=None):
    """
    Like nusmv_check_phenotype but also returns the cycle length of the
    counterexample attractor when the check fails.

    Uses LTLSPEC F G <phenotype> instead of CTLSPEC EF AG <phenotype>.
    For synchronous (deterministic) BNs the two are logically equivalent,
    but LTL counterexamples are guaranteed to be lassos with a
    '-- Loop starts here --' marker, making the attractor length parseable.

    Returns:
        (ok, loop_len): ok is True iff all attractors satisfy the phenotype;
        loop_len is the attractor cycle length from the counterexample trace,
        or None when ok is True.

    bn: CNFBooleanNetwork
    control: Control
    update_mode: synchronous (recommended), asynchronous, general
    smvfile: if None, uses a temporary file
    """
    nusmv_input = _nusmv_model(bn, control=control, update_mode=update_mode)
    nusmv_input += f"LTLSPEC F G {_sanitize_smv_expr(bn.phenotype)};"
    output = _nusmv_run(nusmv_input, smvfile, with_counterexample=True)
    ok = all(
        line.split()[-1] == "true"
        for line in output.split("\n")
        if line.startswith("-- specification ")
    )
    return ok, (None if ok else _parse_loop_length(output))
