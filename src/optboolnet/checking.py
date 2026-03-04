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


def _nusmv_control_constraints(control, allowed_vars=None):
    """Return INIT/INVAR constraints to lock controlled nodes, if any."""
    if not control:
        return ""

    allowed = set(allowed_vars) if allowed_vars is not None else None
    terms = [
        f"{'' if value else '!'}{_nusmv_var(name)}"
        for name, value in sorted(control.items(), key=lambda kv: str(kv[0]))
        if (allowed is None or name in allowed)
    ]
    if not terms:
        return ""
    expr = " & ".join(terms)
    return f"INIT {expr};\nINVAR {expr};\n"


def _bool_from_expr_value(value):
    s = str(value).strip().upper()
    if s in ["TRUE", "1"]:
        return True
    if s in ["FALSE", "0"]:
        return False
    if isinstance(value, (int, bool)):
        return bool(value)
    raise ValueError(f"Cannot convert propagated constant '{value}' to boolean")


def _preprocess_bn_with_mpbn(bn, control):
    """Apply control-driven constant propagation and return a reduced BN.

    The returned control is empty because the control values are already applied
    directly to the reduced transition functions.
    """
    import mpbn

    reduced = mpbn.MPBooleanNetwork(bn)
    reduced.simplify(in_place=True)

    if control is None:
        control = {}
    for k, v in control.items():
        if k in reduced:
            reduced[k] = 1 if v else 0

    reduced.propagate_constants()
    constants = {k: _bool_from_expr_value(v) for k, v in reduced.constants().items()}

    # Remove constants except phenotype (keep it explicit for CTL/LTL specification).
    for k in list(constants.keys()):
        if k != bn.phenotype:
            reduced.pop(k, None)

    if bn.phenotype in constants and bn.phenotype not in reduced:
        reduced[bn.phenotype] = 1 if constants[bn.phenotype] else 0

    reduced_bn = bn.__class__(reduced, bn._control_config, to_cnf=True)
    return reduced_bn, {}


def _phenotype_spec_clause(phenotype_expr, property_variant):
    temporal_expr = _phenotype_temporal_expr(phenotype_expr, property_variant)
    spec_kw = "CTLSPEC" if property_variant == "ctl_ef_ag" else "LTLSPEC"
    return f"{spec_kw} {temporal_expr};"


def _phenotype_temporal_expr(phenotype_expr, property_variant):
    if property_variant == "ctl_ef_ag":
        return f"EF AG {phenotype_expr}"
    if property_variant == "ltl_fg":
        return f"F G {phenotype_expr}"
    raise ValueError(
        "Unsupported property_variant '{}'. Supported: ctl_ef_ag, ltl_fg".format(
            property_variant
        )
    )


def _nusmv_spec_truths(output):
    return [
        line.split()[-1] == "true"
        for line in output.split("\n")
        if line.startswith("-- specification ")
    ]


def _control_param_vars(dom):
    return {n: (f"__lk{idx}", f"__cv{idx}") for idx, n in enumerate(dom)}


def _nusmv_control_assignment(control, dom, ctrl_var_map):
    terms = []
    for n in dom:
        lock_var, val_var = ctrl_var_map[n]
        if n in control:
            terms.append(lock_var)
            terms.append(val_var if control[n] else f"!{val_var}")
        else:
            terms.append(f"!{lock_var}")
    return " & ".join(terms)


def _nusmv_model_param_controls(
    bn, update_mode="synchronous", constrain_controlled_vars=True
):
    """NuSMV model with parameterized controls via FROZENVAR lock/value pairs.

    This builder currently supports synchronous updates.
    """
    if update_mode != "synchronous":
        raise NotImplementedError(
            "Parameterized-control batch checking currently supports synchronous update mode only"
        )

    dom = bn.vars_list
    var = _nusmv_var
    ctrl_var_map = _control_param_vars(dom)

    lines = ["MODULE main"]
    lines.append("VAR")
    for i in dom:
        lines.append(f"{var(i)}: boolean;")

    lines.append("FROZENVAR")
    for n in dom:
        lock_var, val_var = ctrl_var_map[n]
        lines.append(f"{lock_var}: boolean;")
        lines.append(f"{val_var}: boolean;")

    lines.append("ASSIGN")
    for i in dom:
        lock_var, val_var = ctrl_var_map[i]
        lines.append(f"next({var(i)}) := case {lock_var}: {val_var}; TRUE: f{i}; esac;")

    lines.append("DEFINE")
    for n in dom:
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

    if constrain_controlled_vars:
        lock_terms = []
        for n in dom:
            lock_var, val_var = ctrl_var_map[n]
            lock_terms.append(f"(!{lock_var} | ({var(n)} = {val_var}))")
        lines.append(f"INVAR {' & '.join(lock_terms)};")

    return "\n".join(lines) + "\n", ctrl_var_map


def _nusmv_run(nusmv_input, smvfile, with_counterexample=False, nusmv_opts=None):
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
        if nusmv_opts:
            for opt_name, enabled in nusmv_opts.items():
                if not isinstance(enabled, bool):
                    raise ValueError(
                        f"NuSMV option '{opt_name}' must be bool, got {type(enabled).__name__}"
                    )
                if enabled:
                    mc.opts[opt_name] = True
                else:
                    mc.opts.pop(opt_name, None)
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


def _nusmv_alltrue(nusmv_input, smvfile, nusmv_opts=None):
    output = _nusmv_run(nusmv_input, smvfile, nusmv_opts=nusmv_opts)
    return all(_nusmv_spec_truths(output))


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
    bn,
    attractor,
    control=None,
    update_mode="synchronous",
    smvfile=None,
    nusmv_opts=None,
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
    return _nusmv_alltrue(nusmv_input, smvfile, nusmv_opts=nusmv_opts)


def nusmv_check_phenotype(
    bn,
    control=None,
    update_mode="synchronous",
    smvfile=None,
    property_variant="ctl_ef_ag",
    constrain_controlled_vars=False,
    nusmv_opts=None,
    preprocess_propagation=False,
):
    """
    Returns true if all the attractors have p=1 constantly

    bn: CNFBooleanNetwork
    control: Control
    update_mode: synchronous, asynchronous, general
    smvfile: if None, uses a temporary file
    property_variant:
        - "ctl_ef_ag" (default): CTLSPEC EF AG phenotype
        - "ltl_fg": LTLSPEC F G phenotype
    constrain_controlled_vars:
        If True, adds INIT/INVAR constraints for controlled nodes to reduce the
        state space explored by NuSMV.
    nusmv_opts:
        Optional dict of boolean NuSMV command-line flags, e.g.
        {"dynamic": True, "reorder": True}.
    preprocess_propagation:
        If True, preprocess the BN with MPBN constant propagation under the given
        control assignment, then run model checking on the reduced network.
    """
    eval_bn = bn
    eval_control = control if control is not None else {}
    if preprocess_propagation:
        eval_bn, eval_control = _preprocess_bn_with_mpbn(bn, eval_control)

    phenotype_expr = _sanitize_smv_expr(eval_bn.phenotype)
    nusmv_input = _nusmv_model(eval_bn, control=eval_control, update_mode=update_mode)
    if constrain_controlled_vars:
        nusmv_input += _nusmv_control_constraints(
            eval_control, allowed_vars=eval_bn.vars_list
        )
    nusmv_input += _phenotype_spec_clause(phenotype_expr, property_variant)
    return _nusmv_alltrue(nusmv_input, smvfile, nusmv_opts=nusmv_opts)


def nusmv_check_phenotype_full(
    bn,
    control=None,
    update_mode="synchronous",
    smvfile=None,
    nusmv_opts=None,
    preprocess_propagation=False,
):
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
    eval_bn = bn
    eval_control = control if control is not None else {}
    if preprocess_propagation:
        eval_bn, eval_control = _preprocess_bn_with_mpbn(bn, eval_control)

    nusmv_input = _nusmv_model(eval_bn, control=eval_control, update_mode=update_mode)
    nusmv_input += f"LTLSPEC F G {_sanitize_smv_expr(eval_bn.phenotype)};"
    output = _nusmv_run(
        nusmv_input,
        smvfile,
        with_counterexample=True,
        nusmv_opts=nusmv_opts,
    )
    ok = all(
        _nusmv_spec_truths(output)
    )
    return ok, (None if ok else _parse_loop_length(output))


def nusmv_check_phenotype_batch(
    bn,
    controls,
    update_mode="synchronous",
    smvfile=None,
    property_variant="ctl_ef_ag",
    constrain_controlled_vars=True,
    nusmv_opts=None,
):
    """Batch phenotype checks for multiple controls in one NuSMV invocation.

    Returns a list of booleans in the same order as `controls`.
    """
    controls = list(controls)
    if not controls:
        return []

    phenotype_expr = _sanitize_smv_expr(bn.phenotype)
    temporal_expr = _phenotype_temporal_expr(phenotype_expr, property_variant)
    spec_kw = "CTLSPEC" if property_variant == "ctl_ef_ag" else "LTLSPEC"

    nusmv_input, ctrl_var_map = _nusmv_model_param_controls(
        bn,
        update_mode=update_mode,
        constrain_controlled_vars=constrain_controlled_vars,
    )
    for ctrl in controls:
        ctrl_expr = _nusmv_control_assignment(ctrl, bn.vars_list, ctrl_var_map)
        nusmv_input += f"{spec_kw} ({ctrl_expr}) -> ({temporal_expr});\n"

    output = _nusmv_run(nusmv_input, smvfile, nusmv_opts=nusmv_opts)
    results = _nusmv_spec_truths(output)
    if len(results) != len(controls):
        raise RuntimeError(
            f"Unexpected number of spec results: got {len(results)}, expected {len(controls)}"
        )
    return results
