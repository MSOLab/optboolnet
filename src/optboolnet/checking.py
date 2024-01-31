import os
import tempfile
from nusmv import NuSMV


def _nusmv_var(n):
    if isinstance(n, int):
        return "x%d" % n
    return n


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


def _nusmv_alltrue(nusmv_input, smvfile):
    tmp_smvfile = smvfile is None
    if tmp_smvfile:
        _, smvfile = tempfile.mkstemp(suffix=".smv")
    try:
        with open(smvfile, "w") as fp:
            fp.write(nusmv_input)
        mc = NuSMV(smvfile)
        return mc.alltrue()
    finally:
        if tmp_smvfile:
            os.unlink(smvfile)


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
    nusmv_input += f"CTLSPEC EF AG {bn.phenotype};"
    return _nusmv_alltrue(nusmv_input, smvfile)
