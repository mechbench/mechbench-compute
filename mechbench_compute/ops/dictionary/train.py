from __future__ import annotations

from mechbench_compute.api import (
    In,
    Op,
    Output,
    P,
    Resume,
    items_of,
    read_train_settings,
    train_dictionary,
)

OP = Op(
    name="dictionary/train",
    needs=frozenset({"model.forward", "network:huggingface.co"}),
    resume=Resume("state-restorable"),
    summary=(
        "Train a sparse autoencoder on a model's activations at one layer, streamed from a corpus and "
        "never stored, and emit it as a dictionary that every feature operation reads."
    ),
    description="""\
The corpus is a pinned Hugging Face dataset, read from its parquet files
as it is needed (`corpus`), or a stored records collection bound to
`records`. A source names its `repo`, `config`, `split`, `revision` and
the column to read: `field`, a text column (a document), or `chat`, a
column of conversations, which is rendered through the model's chat
template. Several sources are interleaved by `weight`. Records are read
as documents (`text`), conversations (`messages`) or prompts (`user`).
The order is fixed by `seed`: the files, row groups and rows are read in
a seeded order, and a stored collection in a seeded order each pass.

Documents are tokenized, each beginning with the beginning-of-sequence
token, and packed end to end into windows of `context` tokens. The model
runs each window to the layer read and no further, and the activation
at `point` is kept at every position but the first of a window and the
beginning-of-sequence token, whose activations are outliers by orders of
magnitude. The first `heldout_tokens` of the stream are set aside before
training and never trained on.

Activations fill a buffer of `buffer_tokens`, which is shuffled and
trained through in batches of `batch_tokens` before the next fill. They
are scaled so that their mean squared norm is the width of the space,
the scale fitted on the first buffer; the dictionary emitted has the
scale folded into its weights, so it reads the model's raw activations
and gives the features the trainer saw.

`activation` `batchtopk` keeps the `k` × batch largest pre-activations
across the batch, with an auxiliary loss that teaches the features that
have not fired in the last ten million tokens (or a tenth of the run) to
reconstruct what the others miss; at inference it is a JumpReLU whose
threshold is the running mean of the smallest activation kept.
`jumprelu` learns each feature's threshold through a straight-through
estimate and is pulled toward `l0` features per token. The decoder's
rows are kept at unit length; the weights are float32 and trained with
Adam, warmed up over the first steps and decayed to zero over the last
fifth.

Progress is reported for every buffer batch harvested and every step.
Every `checkpoint_tokens` the training state is saved at the end of a
buffer, with where the corpus stands, so an interrupted run resumes and
finishes as the uninterrupted one would have.

At the end, on the held-out windows, the header reports how well the
dictionary reads the model: the variance it explains, the features per
token, the share of features that never fire and that fire on more than
a tenth of the tokens, and the model's next-token loss with the
dictionary's reconstruction spliced in at the point, beside the loss
with the point zeroed as the scale (`recovered` is one minus their
ratio).

Sparse autoencoders train today. A transcoder or a crosscoder is the
same kind with another derivation, and is refused by name until this
operation trains them.
""",
    params=(
        P("derivation", "string",
          "What to train: a sparse autoencoder, which reads and writes one space.",
          "sae", choices=("sae", "transcoder", "crosscoder")),
        P("layer", "int", "The layer the dictionary reads."),
        P("point", "string",
          "The point the dictionary reads at `layer`: one vector per position.",
          "resid_post"),
        P("width", "int", "How many features.", 16384),
        P("activation", "string",
          "The trainer: BatchTopK (`k` features per token on average across a batch) or JumpReLU (a "
          "learned threshold per feature, pulled toward `l0`).",
          "batchtopk", choices=("batchtopk", "jumprelu")),
        P("k", "int", "Under `batchtopk`: the features kept per token, on average over a batch.", 64),
        P("l0", "float", "Under `jumprelu`: the features per token the sparsity penalty aims at.", 64),
        P("tokens", "int", "The activations to train on, in all.", 200_000_000),
        P("context", "int", "The tokens in one window the model reads.", 1024),
        P("batch_tokens", "int", "The activations in one training step.", 4096),
        P("buffer_tokens", "int",
          "The activations harvested and shuffled together before they are trained on; rounded down to a "
          "whole number of batches.",
          262144),
        P("lr", "float", "Adam's learning rate at its peak.", 2e-4),
        P("checkpoint_tokens", "int",
          "Save the training state at the end of the buffer in which every this many activations "
          "have been trained on.",
          10_000_000),
        P("heldout_tokens", "int",
          "The activations set aside before training, on which the header's fidelity is measured.",
          262144),
        P("corpus", "json",
          "The Hugging Face dataset to read, `{\"hub\": {\"repo\", \"config\", \"split\", \"revision\", "
          "\"field\" | \"chat\", \"files\"}, \"weight\"}`, or a list of them interleaved by weight. "
          "`files` (a glob) replaces the config's own; pin `revision` to a commit for a corpus that "
          "cannot move. Leave it out when `records` is bound.",
          None),
    ),
    inputs=(
        In("records", "records/record",
           "A stored corpus in place of `corpus`: documents (`text`), conversations (`messages`) or prompts "
           "(`user`).",
           many=True, required=False),
    ),
    output=Output("direction/dictionary", collection=True,
                  doc="One item per feature `{id, index, vector, encoder, norm, b_enc, threshold}`, stored as "
                      "shards; the header carries `derivation`, `reads`, `writes`, `model`, `width`, `d_in`, "
                      "`d_out`, `activation` (`jumprelu`, with the `training` that made it and its `k` or "
                      "`l0`), `b_dec`, `trained` (the corpus with each source's commit and documents read, "
                      "the hashes of the first buffer's and the held-out windows' tokens, tokens and steps, "
                      "seed, scale, hyperparameters, the training curve and the compute version) and "
                      "`fidelity` on the held-out tokens (`variance_explained`, `l0`, `mse`, `dead`, "
                      "`dense`, and `loss`: `clean`, `spliced`, `zero`, `delta`, `zero_delta`, "
                      "`recovered`)."),
    example={"model": {"$param": "model"}, "layer": 17, "corpus": [
        {"hub": {"repo": "HuggingFaceFW/fineweb-edu", "config": "sample-10BT", "split": "train",
                 "field": "text"}, "weight": 1},
        {"hub": {"repo": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "chat": "messages"},
         "weight": 1}]},
)


def run(ctx, inputs, params):
    read_train_settings(params)
    model = ctx.model(params.get("model"))
    return train_dictionary(model, items_of(inputs.get("records") or []), params,
                            on_start=ctx.on_start, on_item=ctx.on_item, on_checkpoint=ctx.on_checkpoint,
                            resume_state=ctx.resume_state)
