"""Path patching: a sender's effect on a receiver, along the path
between them and no other.

Activation patching (`intervene/patch`) changes what EVERY downstream
component sees, so a head that "matters" may matter only because the
thing it feeds matters. Path patching (Wang et al. 2022; Goldowsky-Dill
et al. 2023) asks the narrower question a circuit claim actually needs:
does what THIS head writes reach THAT head's query — as opposed to
reaching the answer by some other route?

The procedure, per (sender, receiver):

1. the clean prompt, caching every component's output;
2. the corrupt prompt, caching the sender's;
3. the clean prompt again with the sender's output replaced by its
   corrupt value and every component BETWEEN the two frozen at its
   clean value — so the only thing that changed at the receiver's input
   is what arrived along the path — reading the receiver's new output;
4. the clean prompt once more with the receiver's output set to that,
   everything else running normally, which is where the metric is read.

Steps 1 and 2 are done once; each sender costs steps 3 and 4. A
receiver of `logits` needs no step 4: freezing everything after the
sender already leaves only the direct path, and the logits are the
metric.
"""

from __future__ import annotations





