"""Phase 1: opening lens notes, independent (spec 9)."""

from __future__ import annotations

from lab.run import Run
from lab.schemas import Text

BRIEF = """Dorst's route. Map a wide field of players. Distil the deep themes they share. Look for other contexts where those
themes were handled well. Make a creative leap to a frame of the form "If the problem situation is approached as if it is X, then Y".
The challenge. Improve on theme mapping and analogue search as the engine of reframing.
Hunches: (1) reasoning with networks could help; (2) actors, human and non-human, exchange: they make contributions and hold
expectations, not always with human-like intentions, and receive and reciprocate (Mauss); (3) networks have local and global
properties and both matter; (4) every relevant actor has a different perspective.
Starting ideas (offered, not imposed): exchange network with reciprocity states; typed expectations; motifs as structural twins of
themes; explicit network operators as handholds; checks from several perspectives and scales; setting the paradox aside after stating it.
A method may reject the network hunch if it says why. Anchors: Mauss, Mary Douglas, Latour, Bourdieu, Hutchins, network science, Dorst, Schon."""


def run_phase(run: Run) -> bool:
    field_map = run.p("field", "A", "field_map.md").read_text()
    dorst = run.cfg.resolve("inputs/dorst_notes.md")
    dorst_txt = dorst.read_text() if dorst and dorst.exists() else ""
    notes = {}
    for pid in run.core_ids:  # members never see each other's notes
        f = run.p("lenses", f"{pid}.md")
        if f.exists():
            notes[pid] = f.read_text()
            continue
        t = run.llm.call(task="lens_note", schema=Text, persona=run.persona(pid),
                         context={"brief": BRIEF, "dorst_notes": dorst_txt, "case_A_field_map": field_map},
                         check=run.cap_words(600)).obj.text
        run.write_text(f, t + "\n")
        notes[pid] = t
    syn = run.llm.call(task="lens_synthesis", schema=Text, persona=run.persona("FA"), context={"notes": notes}).obj.text
    run.write_text(run.p("lenses", "synthesis.md"), syn + "\n")
    return True
