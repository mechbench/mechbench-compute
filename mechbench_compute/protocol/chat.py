"""The local half of `text/chat`.

`text/chat` serves two tiers from one operation: a provider's endpoint,
which the remote path dispatches, and local weights, which are this. The
operation asks its executor for it rather than loading a model itself.
"""

from __future__ import annotations


class Chat:
    """Chat: see this module's docstring."""

    def _block_chat_local(self, inputs, params, on_item=None, on_start=None,
                          resume_items=None):
        from mechbench_compute import chat as chat_mod

        model = self._model_loaded(params.get("model"))
        records = inputs.get("records") or []
        return chat_mod.run_local(model, params.get("model"), records, params,
                                  inputs=inputs,
                                  on_item=on_item, on_start=on_start,
                                  resume_items=resume_items)
