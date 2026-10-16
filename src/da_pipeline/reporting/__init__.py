"""Turning a completed run into something a reader -- or a model -- can use.

The analysis modules leave nineteen CSVs and thirty-odd figures in a directory.
That is the right output for someone who already knows the project, and the
wrong one for everybody else: the findings are real but they are spread across
files, and the several rules for reading them honestly are not written anywhere
the numbers are.

Three things are built here from one shared layer:

* ``findings`` extracts what a run actually found -- the headline numbers, the
  cautions they come with -- from the analysis frames, once, in a form neither
  the report nor the evidence pack has to re-derive;
* ``html_report`` renders that plus every figure into a single self-contained
  file that opens in a browser with nothing beside it;
* ``evidence`` writes the same findings as JSON for a language model, next to a
  prompt that carries the rules for reading them.

The split matters because the two consumers are so different. A person reading
the report can be trusted to notice a caveat in a paragraph. A model given a
table of numbers with no context will confidently report that two conditions
are equivalent because a p-value was large, so the prompt has to say, in
advance and in as many words, that it must not.
"""
