"""What the transcript operations share.

A conversation is a fold over turns: render the shared transcript for
the participant whose turn it is (`text/render`), ask that
participant's model, append what it said (`text/extend`) — so a
conversation is composed from `text/chat` rather than duplicated beside
it. The two operations are files under `ops/text/`; what they both need
to read a transcript is here.

`sees` is a param of the render step, and so sweepable, rather than a
setting on the participant.
"""
