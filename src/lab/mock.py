"""Deterministic mock backend. Content is templated, not intelligent: it exercises the pipeline,
the toolkit (tool calls hit the real network) and the validators. Judgements are random by design,
so mock rankings carry no information."""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Any, Callable

from pydantic import BaseModel

from lab.llm import BackendResult
from lab.toolkit import ToolCapExceeded

STEP_TEMPLATES = {
    "read": dict(name="Read the exchange field", roles=["anthropologist", "agronomist"], tools=["get_metrics", "list_motifs"],
                 action="Read global and local metrics, list the motifs and note which ones matter.",
                 out=("list of motifs", "motif ids with one line each")),
    "persp": dict(name="Take actor perspectives", roles=["socio-economist", "anthropologist"], tools=["get_perspective"],
                  action="Take the perspective of at least two actors, one non-human or marginal, and note what each gives and is owed.",
                  out=("perspective notes", "one paragraph per actor")),
    "themes": dict(name="Name the deep themes", roles=["philosopher", "designer"], tools=["get_evidence"],
                   action="State up to three themes shared by the actors, each grounded in evidence ids from S1 and S2.",
                   out=("theme list", "theme label plus evidence ids")),
    "analog": dict(name="Search structural analogues", roles=["informatics specialist", "designer"], tools=["find_analogues"],
                   action="Match the strongest motif and theme against the analogue catalogue and choose one analogue.",
                   out=("analogue choice", "analogue id and why")),
    "oper": dict(name="Try a network operator", roles=["mathematician", "agronomist"], tools=["apply_operators"],
                 action="Apply one operator that closes an open loop and read the consequence report.",
                 out=("consequence note", "who gains, who loses, new obligation")),
    "invent": dict(name="Try an invented operator", roles=["mathematician", "designer"], tools=["apply_edits"],
                   action="Describe an operator of your own in prose and perform it with apply_edits.",
                   out=("operator log", "edits and consequences")),
    "ambig": dict(name="Balance local and global as appropriate", roles=["agronomist", "mathematician"], tools=["get_metrics"],
                  action="Somehow balance local and global readings as appropriate.",
                  out=("balance note", "where they disagree")),
    "frame": dict(name="Form and state the frame", roles=["designer", "anthropologist"], tools=[],
                  action="Combine the chosen analogue and the consequence note into a frame of the form 'If the problem situation is approached as if it is X, then Y'. Use S1 to S4.",
                  out=("frame statement", "one sentence in Dorst's form")),
}
RECIPES = [["read", "persp", "themes", "analog", "frame"], ["read", "oper", "persp", "frame"],
           ["read", "themes", "analog", "oper", "frame"], ["persp", "read", "invent", "frame"],
           ["read", "ambig", "analog", "frame"], ["read", "persp", "analog", "oper", "invent", "frame"]]
NAMES = ["Exchange ring reframing", "Perspective walk reframing", "Motif and theme pairing", "Operator handhold reframing",
         "Local-global balance reframing", "Analogue-first reframing", "Debt and gift reframing", "Boundary shifting reframing",
         "Mediator insertion reframing"]


def _h(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def blurb(rng: random.Random, n: int, topic: str = "the method") -> str:
    bank = ("step exchange reciprocity actor perspective motif theme frame operator loop debt gift network local global "
            "analogue evidence trace paradox boundary mediator scale obligation return share seed grain").split()
    return f"Mock note on {topic}: " + " ".join(rng.choice(bank) for _ in range(n)) + "."


class MockBackend:
    def __init__(self, seed: int = 1, people: dict | None = None, merge_pairs: list[list[str]] | None = None):
        self.seed = seed
        self.people = people or {}
        self.merge_pairs = merge_pairs or []

    # ---------------------------------------------------------------- entry
    def run(self, *, task: str, model: str, prompt: str, ctx: dict[str, Any], schema: type[BaseModel],
            tool_specs: list[dict[str, Any]], call_tool: Callable[[str, dict[str, Any]], Any] | None,
            persona_id: str | None, attempt: int) -> BackendResult:
        key = hashlib.sha256(json.dumps(ctx, sort_keys=True, default=str).encode()).hexdigest()[:10]
        rng = random.Random(_h(self.seed, task, persona_id, key))
        tool_trace: list[dict[str, Any]] = []
        handler = getattr(self, f"_t_{task}", None)
        if handler is None:
            raise NotImplementedError(f"mock has no handler for task {task!r}")
        data = handler(ctx=ctx, rng=rng, pid=persona_id, tools=tool_specs, call=call_tool)
        text = json.dumps(data, default=str)
        return BackendResult(text, tool_trace, len(prompt) // 4, len(text) // 4)

    # ---------------------------------------------------------------- text tasks
    def _txt(self, task: str, n: int = 50):
        def f(ctx, rng, pid, **_):
            return {"text": blurb(rng, n, f"{task} by {pid}")}
        return f

    _t_lens_note = lambda self, **k: self._txt("lens note", 70)(**k)  # noqa: E731
    _t_lens_synthesis = lambda self, **k: self._txt("lens synthesis", 60)(**k)  # noqa: E731
    _t_method_draft = lambda self, **k: self._txt("method draft", 80)(**k)  # noqa: E731
    _t_method_response = lambda self, **k: self._txt("method response", 90)(**k)  # noqa: E731
    _t_aar_designers = lambda self, **k: self._txt("designers' answers", 40)(**k)  # noqa: E731
    _t_aar_executors = lambda self, **k: self._txt("executors' answers", 40)(**k)  # noqa: E731
    _t_aar_ethnographer = lambda self, **k: self._txt("ethnographer's answers", 40)(**k)  # noqa: E731
    _t_executor_notes = lambda self, **k: self._txt("executor notes", 60)(**k)  # noqa: E731
    _t_ethnography = lambda self, **k: self._txt("ethnography", 70)(**k)  # noqa: E731
    _t_harvest = lambda self, **k: self._txt("harvest", 90)(**k)  # noqa: E731
    _t_fixation_note = lambda self, **k: self._txt("fixation", 30)(**k)  # noqa: E731
    _t_digest = lambda self, **k: self._txt("digest", 120)(**k)  # noqa: E731
    _t_reflection = lambda self, **k: self._txt("reflection", 80)(**k)  # noqa: E731
    _t_implementation_notes = lambda self, **k: self._txt("implementation notes", 80)(**k)  # noqa: E731
    _t_field_questions = lambda self, **k: self._txt("field questions", 60)(**k)  # noqa: E731

    def _t_leakage_probe(self, **_):
        return {"text": "unknown"}

    def _t_extract_network(self, **_):
        return {"nodes": [], "edges": []}

    def _t_network_review(self, ctx, pid, **_):
        return {"edits": [], "notes": f"Mock review by {pid} for {ctx.get('role')}: no changes proposed."}

    def _t_digest_check(self, **_):
        return {"ok": True, "issues": []}

    # ---------------------------------------------------------------- methods
    def _protocol(self, rng: random.Random, seed_key: str) -> dict[str, Any]:
        k = _h(seed_key) % len(RECIPES)
        steps = []
        for i, key in enumerate(RECIPES[k], 1):
            t = STEP_TEMPLATES[key]
            steps.append({"id": f"S{i}", "name": t["name"], "roles": t["roles"], "inputs": ["case/context.md", "network"] + (["S1"] if i > 1 else []) + ([f"S{i-1}"] if i > 2 else []),
                          "action": t["action"], "tools": t["tools"], "output": {"type": t["out"][0], "schema_hint": t["out"][1]},
                          "done_when": "the output type is filled and cites at least one evidence id", "max_calls": 3})
        name = NAMES[_h(seed_key, "n") % len(NAMES)]
        return {"name": name, "purpose": blurb(rng, 30, "purpose and claim"), "network_stance": ["uses", "extends", "uses"][_h(seed_key) % 3],
                "stance_reason": "Exchange structure gives handholds that theme lists alone lack.",
                "representations": "The exchange network, perspective notes and a motif list.", "steps": steps,
                "how_themes": "Themes are read off open loops and conflicting valencies, then checked against perspectives.",
                "how_perspectives": "Ego views for at least one non-human and one marginal human actor, labelled as hypotheses.",
                "how_analogues": "Catalogue search by theme and motif match, then a human choice.",
                "how_scales": "Every operator is read locally per actor and globally; disagreements are flagged.",
                "how_paradox": "State the paradox once, then set it aside during the leap.",
                "how_frame": "Frame in Dorst's form, tied to a motif, an analogue and an operator result.",
                "human_role": "Breeding team members run steps and overrule computed steps.",
                "stopping_rules": "Stop when one frame cites evidence from three steps or the budget is spent.",
                "failure_modes": "Thin network, fixation on the first analogue, false agency for non-humans.",
                "budget_calls": min(30, 3 * len(steps) + 4), "example_frame": None}

    def _t_method_protocol(self, ctx, rng, pid, **_):
        return self._protocol(rng, ctx.get("draft_a", "") + str(pid))

    def _t_merge_methods(self, ctx, rng, pid, **_):
        first = next(iter(ctx["protocols"].values()))
        return {**first, "name": "Merged reframing route"}

    def _t_revise_stuck(self, ctx, rng, pid, **_):
        p = json.loads(json.dumps(ctx["protocol"]))
        log = []
        for s in p["steps"]:
            if re.search(r"somehow|as appropriate|\[\[FAIL\]\]", s["action"], re.I):
                s["action"] = re.sub(r"somehow |\[\[FAIL\]\]", "", s["action"], flags=re.I).replace("as appropriate", "using the local-versus-global report")
                log.append({"change": f"clarified {s['id']} after stuck point", "ledger_ids": []})
        p["example_frame"] = "If the problem situation is approached as if it is a shared exchange ring, then unreturned data becomes the design object."
        log.append({"change": "added example frame from Trial 1", "ledger_ids": []})
        return {"protocol": p, "changelog": log}

    def _t_revision(self, ctx, rng, pid, **_):
        p = json.loads(json.dumps(ctx["protocol"]))
        items = [i for i in ctx["ledger_items_with_responses"] if i["response"] in ("adopt", "adapt")]
        rng.shuffle(items)
        log = []
        for it in items[:3]:
            p["how_analogues"] = p["how_analogues"].split(" [")[0] + f" [café: {it['type']} from {it['giver']}]"
            log.append({"change": f"{it['response']}ed {it['type']} from {it['giver']}", "ledger_ids": [it["id"]]})
        if not log:
            log.append({"change": "no change: every item declined", "ledger_ids": []})
        return {"protocol": p, "changelog": log}

    # ---------------------------------------------------------------- execution
    def _t_exec_step(self, ctx, rng, pid, tools, call, **_):
        step = ctx["step"]
        names = [t["name"] for t in tools]
        seen: dict[str, Any] = {}
        motifs = nodes = None
        summary = []

        def use(name: str, args: dict[str, Any]) -> Any:
            if call is None or name not in names:
                return None
            try:
                return call(name, args)
            except ToolCapExceeded:
                return None
            except Exception as exc:  # tool errors are part of the trace
                summary.append(f"{name} error: {str(exc)[:60]}")
                return None

        if "list_motifs" in names:
            motifs = use("list_motifs", {})
            loops = [e for x in (motifs or []) if x["pattern"] == "open_reciprocity_loop" for e in x["edges"]]
            summary.append(f"{len(motifs or [])} motifs; open loop edges: {', '.join(loops[:4])}")
        if "get_metrics" in names:
            g = use("get_metrics", {"scope": "global"})
            summary.append(f"density {g and g.get('density')}, reciprocity {g and g.get('reciprocity_generalised')}")
            nodes = list((use("get_metrics", {"scope": "local"}) or {}).keys())
        if "get_perspective" in names:
            ids = nodes or [a["id"] for a in ctx.get("network_actors", [])[:2]] or ["missing_actor"]
            for a in ids[:2]:
                r = use("get_perspective", {"actor_id": a})
                summary.append(f"perspective {a}: owes {len((r or {}).get('open_obligations', []))}")
        if "find_analogues" in names:
            r = use("find_analogues", {"motif_pattern": "open_reciprocity_loop", "k": 3})
            summary.append("top analogue " + str((r or [{}])[0].get("analogue")))
        if "get_evidence" in names:
            eid = (motifs or [{"edges": ["e001"]}])[0].get("edges") or ["e001"]
            use("get_evidence", {"id": eid[0]})
            summary.append(f"evidence for {eid[0]}")
        if "apply_operators" in names:
            earlier = " ".join(ctx.get("outputs_of_earlier_steps_you_may_use", {}).values())
            found = re.search(r"open loop edges: (e\d+)", earlier)
            if found:
                r = use("apply_operators", {"chain": [{"op": "close_loop", "args": {"edge": found.group(1),
                                                       "counter": {"contribution": "results returned", "flow_type": "data"}}}]})
                summary.append("closed loop; affected " + str((r or {}).get("affected_perspectives")))
        if "apply_edits" in names:
            ns = nodes or [a["id"] for a in ctx.get("network_actors", [])]
            if ns:
                r = use("apply_edits", {"edits": [{"op": "update_node", "id": ns[0], "fields": {"classification": "mock reclassification"}}]})
                summary.append("edited " + ns[0])
        out = f"{step['name']}: " + "; ".join(summary or ["no tools used"]) + "."
        interp = "Assumed 'balance' means listing disagreements only." if re.search(r"somehow|as appropriate", step["action"], re.I) else None
        stuck = "[[FAIL]]" in step["action"]
        hyp = [{"text": f"Simulated: {ctx['role_assignment'] and list(ctx['role_assignment'])[0]} holders may read this step differently.", "label": "hypothesis"}]
        return {"output": out, "interpretation": interp, "stuck": stuck, "hypotheses": hyp}

    def _t_frames(self, ctx, rng, pid, **_):
        outs = ctx["step_outputs"]
        want = min(int(ctx.get("frames_wanted", 3)), 3)
        analog = ["a rotating savings ring", "a seed lending library", "a mutualist network", "a gift exchange ring"]
        frames = []
        for i in range(max(1, want)):
            frames.append({"statement": f"If the problem situation is approached as if it is {analog[rng.randrange(4)]}, then unreturned contributions become the design object.",
                           "sees": blurb(rng, 25, "what the frame lets the team see"), "evidence_steps": list(outs)[:3]})
        return {"frames": frames, "best": rng.randrange(len(frames))}

    def _t_aar(self, ctx, rng, pid, **_):
        ids = [x for x in ctx["valid_citation_ids"] if re.fullmatch(r"S\d+", x)]
        interps = [x for x in ctx["valid_citation_ids"] if x.startswith("interp:")]
        ts = ctx["trace_summary"]
        worked = ids[: max(1, len(ids) // 2)]
        idle = [i for i in ids[len(worked):-1]] or []
        stuck = [{"claim": "Step was unclear and the executor recorded an assumption", "cite": c} for c in interps]
        stuck += [{"claim": "Step produced no valid output twice", "cite": s} for s in ts["stuck"]]
        cmp_ = None
        if "trial1_aar" in ctx:
            cmp_ = "Compared with Trial 1: stuck points listed there " + ("persist." if stuck else "did not recur.")
        return {"steps_worked": [{"claim": "Produced output that later steps used", "cite": s} for s in worked],
                "steps_no_work": [{"claim": "Output not used by any later step", "cite": s} for s in idle],
                "stuck_points": stuck, "surprises": [{"claim": "Tool output pointed at an actor the step did not plan for", "cite": "trace:1"}],
                "frame_quality": "Mock: frames are templated, so quality is not informative.",
                "method_quality": "Mock: step structure ran end to end within budget.",
                "attribution": "Method: step wording. Case: toy network. Executors: none (mock).",
                "proposed_changes": [{"claim": "Tighten the wording of the last step", "cite": ids[-1]}], "comparison": cmp_}

    # ---------------------------------------------------------------- café
    def _t_cafe_items(self, ctx, rng, pid, **_):
        mk = lambda t: blurb(rng, 25, f"{t} from {pid} on table {ctx['table']}")  # noqa: E731
        d: dict[str, Any] = {"items": [{"type": "addition", "text": mk("addition")}, {"type": "challenge", "text": mk("challenge")},
                                       {"type": "graft", "text": mk("graft")}],
                             "must_produce_in": "addition", "reversal_of": None}
        if pid == "PV":
            d["reversal_of"] = "the central step"
            d["items"][1]["text"] = "Reverse the central step: start from the frame and work back to the network."
        return d

    def _t_reception(self, ctx, rng, pid, **_):
        out = []
        for it in ctx["items"]:
            out.append({"item_id": it["id"], "decision": rng.choices(["adopt", "adapt", "decline"], [4, 3, 3])[0],
                        "reason": "Mock reason tied to the protocol."})
        return {"responses": out}

    def _t_merge_check(self, **_):
        return {"pairs": self.merge_pairs}

    def _t_merge_agree(self, **_):
        return {"agree": True, "reason": "Mock: components overlap."}

    # ---------------------------------------------------------------- ranking
    def _t_judge(self, ctx, rng, pid, **_):
        show = ctx["show_output"]
        labels = sorted(set(re.findall(r"OPTION ([A-Z])\b", show)))
        order = labels[:]
        random.Random(_h(self.seed, "judge", pid, ctx["block"], ctx["criterion"])).shuffle(order)
        return {"block": ctx["block"], "criterion": ctx["criterion"],
                "options": {l: {"strengths": [f"mock strength {l}"], "weaknesses": [f"mock weakness {l}"]} for l in labels},
                "reasoning": "Mock random judgement.", "ranking": order}

    def _t_summarise(self, ctx, rng, pid, **_):
        text = ctx["mentions"]
        out: dict[str, Any] = {}
        cur = None
        for ln in text.splitlines():
            m = re.match(r"## (\S+) \(", ln)
            if m:
                cur = m.group(1)
                out[cur] = {"strengths": [], "weaknesses": []}
                continue
            m = re.match(r"\s*\[(b\d+-[sw]\d+)\] (strength|weakness):", ln)
            if m and cur:
                key = "strengths" if m.group(2) == "strength" else "weaknesses"
                if len(out[cur][key]) < 1:
                    out[cur][key].append({"theme": f"mock {m.group(2)} theme", "mentions": [m.group(1)]})
        return {"criteria": out}

    def _t_difference_column(self, ctx, rng, pid, **_):
        rows = ctx["rows"]
        return {"rows": {r: f"Mock: {r.replace('_', ' ')} for {ctx['method']}" for r in rows}}

    def _t_statement(self, ctx, rng, pid, **_):
        sel = ctx["selection"]["selected"]
        d = {"text": blurb(rng, 60, "which differences matter"), "weakness_of": sel[rng.randrange(len(sel))]}
        if pid in ("PL", "SY"):
            d["special_check"] = "Mock check: no incompatible steps found."
        return d

    def _t_difference_map(self, ctx, rng, pid, **_):
        sel = ctx["selected"]
        tags = ["complementary", "compatible", "in_tension", "exclusive"]
        pairs = []
        k = 0
        for i, a in enumerate(sel):
            for b in sel[i + 1:]:
                pairs.append({"a": f"{a}:how_themes", "b": f"{b}:how_themes", "tag": tags[k % 4], "reason": "Mock reason.", "sources": ["difference_matrix"]})
                k += 1
        if pairs and not any(p["tag"] == "exclusive" for p in pairs):
            pairs[-1]["tag"] = "exclusive"
        return {"pairs": pairs}

    def _t_composite(self, ctx, rng, pid, **_):
        sel = list(ctx["selected_methods"])
        base = json.loads(json.dumps(ctx["selected_methods"][sel[0]]))
        base["name"] = "Modular composite reframing route"
        mods = [{"name": f"Module {s['id']}: {s['name']}", "step_ids": [s["id"]], "inputs": s["inputs"][:1] or ["network"],
                 "outputs": [s["output"]["type"]], "source_methods": sel[:2] if i % 2 == 0 else sel[:1],
                 "ledger_items": [x["id"] for x in ctx["ledger"] if x["table"] == sel[0]][:1], "trial_evidence": [f"{sel[0]} {s['id']}"]}
                for i, s in enumerate(base["steps"])]
        res, opn = [], []
        for i, p in enumerate((ctx["difference_map"]["pairs"])):
            if p["tag"] == "exclusive":
                (res if i % 2 == 0 else opn).append(f"{p['a']} vs {p['b']}: " + ("kept the first" if i % 2 == 0 else "left to the human team"))
        return {"protocol": base, "modules": mods, "failed_modules": [{"claim": "Mock: an idle step was dropped", "cite": f"{sel[0]} S2"}],
                "open_choices": opn, "resolutions": res}

    def _t_objection(self, ctx, rng, pid, **_):
        return {"text": blurb(rng, 30, f"objection from {pid}")}

    def _t_integrator_answer(self, ctx, rng, pid, **_):
        return {"replies": [{"member": m, "reply": "Mock reply.", "resolved": rng.random() < 0.7} for m in ctx["objections"]], "revised": None}

    def _t_final_reflection(self, ctx, rng, pid, **_):
        return {k: blurb(rng, 40, k) for k in ("generalises", "depends_on_case_A", "lab_process", "toolkit", "next_run")}

    def _t_revise_composite(self, ctx, rng, pid, **_):
        c = json.loads(json.dumps(ctx["composite"]))
        c["protocol"]["example_frame"] = "If the problem situation is approached as if it is a shared exchange ring, then unreturned contributions become the design object."
        return c
