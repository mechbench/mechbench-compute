from __future__ import annotations


class Chat:
    def _block_chat_local(self, inputs, params, on_item=None, on_start=None,
                          resume_items=None):
        from mechbench_compute import chat as chat_mod

        model = self._model_loaded(params.get("model"))
        records = inputs.get("records") or []
        return chat_mod.run_local(model, params.get("model"), records, params,
                                  inputs=inputs,
                                  on_item=on_item, on_start=on_start,
                                  resume_items=resume_items)
