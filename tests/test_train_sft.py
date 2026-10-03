from __future__ import annotations

import dataclasses

import mlx.core as mx
import numpy as np
import pytest

from mechbench_compute import architectures, lexicon
from mechbench_compute import model as model_mod
from mechbench_compute.adapters.compute_sft_loss import ROWS, compute_sft_loss
from mechbench_compute.adapters.read_sft_items import SftItem, read_sft_items
from mechbench_compute.adapters.record_refused import RecordRefused
from mechbench_compute.architectures import BY_MODEL_TYPE
from mechbench_compute.distill import encode
from mechbench_compute.ops import Context
from mechbench_compute.ops.adapter import train as train_op
from mechbench_compute.protocol import ProtocolExecutor, ProtocolSpec
from tests.tiny_models import MODEL_TYPES, build_tiny_model
from tests.tiny_tokenizer import CHAT_TEMPLATE

TEMPLATE = (
    "{% for m in messages %}<start>{{ 'model' if m['role'] == 'assistant' else m['role'] }} "
    "{{ m['content'] }}<end>{% endfor %}{% if add_generation_prompt %}<start>model {% endif %}")

GLUED = (
    "{% for m in messages %}<start>{{ 'model' if m['role'] == 'assistant' else m['role'] }}"
    "{{ m['content'] }}<end>{% endfor %}{% if add_generation_prompt %}<start>model{% endif %}")

REFUSING = "{{ raise_exception('Conversation roles must alternate user/assistant') }}"

DOCUMENT = {"id": "d", "text": "the cat sat on a mat and the dog ran"}

CONVERSATION = {"id": "c", "messages": [
    {"role": "user", "content": "the cat sat"},
    {"role": "assistant", "content": "a dog ran"},
    {"role": "user", "content": "the mat"},
    {"role": "assistant", "content": "and the cat sat"}]}


def build_model(arch: str = "llama", template: str = TEMPLATE):
    model = build_tiny_model(arch, BY_MODEL_TYPE[arch])
    model.tokenizer.chat_template = template
    return model


def train(model, records, **params):
    return train_op.run(Context(loaded=model), {"records": records},
                        {"model": "tiny", "objective": "sft", "steps": 6, "lr": 0.05, "seed": 3,
                         "lora": {"rank": 2, "alpha": 4}, **params})


def read_trained(model, record, **params) -> tuple[list[str], list[str]]:
    (item,), _ = read_sft_items(model.tokenizer, [record], params)
    names = model.tokenizer.convert_ids_to_tokens(item.ids)
    return ([n for n, t in zip(names, item.trained) if t],
            [n for n, t in zip(names, item.trained) if not t])


class Logits:
    def __init__(self, z):
        self.z = z

    def __call__(self, ids):
        return self.z[None]


class TestDeclaration:
    def test_the_objective_and_its_settings(self):
        op = lexicon.BY_NAME["adapter/train"]
        params = {p.name: p for p in op.params}
        assert params["objective"].default == "decision"
        assert params["objective"].choices == ("decision", "sft")
        assert params["max_tokens"].default == 512
        assert params["truncation"].default == "cut"
        assert params["truncation"].choices == ("cut", "fail")
        assert not params["target"].required
        assert "record" in {f.name for f in params["batch"].fields}
        assert op.output.kind == "adapter/lora" and not op.output.collection
        assert "`messages`" in op.port("records").doc and "`text`" in op.port("records").doc


class TestWhatIsTrained:
    def test_a_document_is_its_text_as_written_and_every_token_after_the_first(self):
        model = build_model()
        (item,), counts = read_sft_items(model.tokenizer, [DOCUMENT], {})
        assert item.ids == encode(model.tokenizer, DOCUMENT["text"])
        assert item.trained == [False] + [True] * (len(item.ids) - 1)
        assert counts == {"n_documents": 1, "n_conversations": 0, "n_tokens": len(item.ids) - 1,
                          "max_tokens": 512, "truncation": "cut", "truncated": 0,
                          "naturalism": True}

    @pytest.mark.parametrize("arch", MODEL_TYPES)
    def test_a_conversation_trains_its_assistant_turns_only(self, arch):
        trained, untrained = read_trained(build_model(arch), CONVERSATION)
        assert trained == ["a", "dog", "ran", "<end>", "and", "the", "cat", "sat", "<end>"]
        assert untrained == ["<start>", "user", "the", "cat", "sat", "<end>", "<start>", "model",
                             "<start>", "user", "the", "mat", "<end>", "<start>", "model"]

    def test_the_gradient_is_zero_at_every_untrained_token(self):
        (item,), _ = read_sft_items(build_model().tokenizer, [CONVERSATION], {})
        z = mx.random.normal((len(item.ids) - 1, 64), key=mx.random.key(0))
        grad = mx.grad(lambda z: compute_sft_loss(Logits(z), [item]))(z)
        rows = np.abs(np.array(grad)).sum(axis=1)
        assert [bool(r > 0) for r in rows] == item.trained[1:]

    def test_the_loss_is_the_mean_cross_entropy_over_the_trained_tokens(self):
        rng = np.random.default_rng(0)
        n = 2 * ROWS + 90
        trained = [False] + [bool(rng.random() < 0.5) for _ in range(n - 1)]
        trained[ROWS + 1:2 * ROWS + 1] = [False] * ROWS
        item = SftItem([int(t) for t in rng.integers(0, 64, n)], trained)
        z = mx.array(rng.normal(size=(n - 1, 64)).astype(np.float32))
        rows = np.array(z, dtype=np.float64)
        lse = np.log(np.exp(rows - rows.max(axis=1, keepdims=True)).sum(axis=1)) + rows.max(axis=1)
        nll = lse - rows[np.arange(n - 1), item.ids[1:]]
        want = nll[np.array(trained[1:])].mean()
        assert float(compute_sft_loss(Logits(z), [item])) == pytest.approx(want, rel=1e-5)

    def test_the_system_joins_the_first_user_message_untrained(self):
        record = {**CONVERSATION, "system": "the dog"}
        trained, untrained = read_trained(build_model(), record)
        assert trained == ["a", "dog", "ran", "<end>", "and", "the", "cat", "sat", "<end>"]
        assert untrained[:7] == ["<start>", "user", "the", "dog", "the", "cat", "sat"]

    def test_messages_make_a_conversation_whatever_text_sits_beside_them(self):
        trained, _ = read_trained(build_model(), {**CONVERSATION, "text": "the mat"})
        assert trained == ["a", "dog", "ran", "<end>", "and", "the", "cat", "sat", "<end>"]

    def test_a_long_record_is_cut_and_counted(self):
        model = build_model()
        items, counts = read_sft_items(model.tokenizer, [DOCUMENT, CONVERSATION],
                                       {"max_tokens": 10})
        assert [len(i.ids) for i in items] == [10, 10]
        assert items[1].trained == [False] * 8 + [True, True]
        assert counts["truncated"] == 1 and counts["n_tokens"] == 9 + 2

    def test_without_the_gate_a_boundary_inside_a_token_is_taken_where_the_text_before_ends(self):
        trained, _ = read_trained(build_model(template=GLUED), CONVERSATION, naturalism=False)
        assert trained == ["dog", "ran", "<end>", "the", "cat", "sat", "<end>"]


class TestTraining:
    @pytest.mark.parametrize("arch", MODEL_TYPES)
    def test_the_loss_falls_on_a_planted_document(self, arch):
        model = build_model(arch)
        items, _ = read_sft_items(model.tokenizer, [DOCUMENT], {})
        before = float(compute_sft_loss(model.lm, items))
        out = train(model, [DOCUMENT], steps=30, lr=0.01, batch={"record": 1})["out"]
        assert out["train"]["final_loss"] < before / 2

    def test_the_same_seed_gives_the_same_adapter(self):
        records = [DOCUMENT, CONVERSATION]
        one = train(build_model(), records)["out"]
        two = train(build_model(), records)["out"]
        other = train(build_model(), records, seed=4)["out"]
        assert one["data"] == two["data"] and one["train"] == two["train"]
        assert other["data"] != one["data"]

    def test_the_result_is_an_adapter_whose_train_names_the_objective(self):
        out = train(build_model(), [DOCUMENT, CONVERSATION], batch={"record": 2})["out"]
        assert out["kind"] == "adapter/lora" and out["format"] == "safetensors"
        assert out["lora"]["rank"] == 2 and out["lora"]["target_modules"] == ["q_proj", "v_proj"]
        assert {k: v for k, v in out["train"].items() if k != "final_loss"} == {
            "objective": "sft", "steps": 6, "lr": 0.05, "seed": 3, "batch": {"record": 2},
            "n_documents": 1, "n_conversations": 1, "n_tokens": 18, "max_tokens": 512,
            "truncation": "cut", "truncated": 0, "naturalism": True}

    def test_keep_checkpoints_keeps_the_adapter_by_step(self):
        out = train(build_model(), [DOCUMENT, CONVERSATION], keep_checkpoints=True,
                    checkpoint_every=2)
        items = out["checkpoints"]["items"]
        assert [i["coords"] for i in items] == [{"step": 2}, {"step": 4}, {"step": 6}]
        assert items[-1]["data"] == out["out"]["data"]
        assert round(items[-1]["loss"], 4) == out["out"]["train"]["final_loss"]
        assert len({i["data"] for i in items}) == 3
        for item in items:
            assert item["kind"] == "adapter/lora"
            assert item["train"] == {k: v for k, v in out["out"]["train"].items()
                                     if k != "final_loss"}
        assert out["checkpoints"]["train"] == out["out"]["train"]

    def test_a_resumed_run_matches_an_unbroken_one(self):
        records = {"records": [DOCUMENT, CONVERSATION]}
        params = {"model": "tiny", "objective": "sft", "steps": 6, "lr": 0.05, "seed": 3,
                  "lora": {"rank": 2, "alpha": 4}, "batch": {"record": 1}, "checkpoint_every": 2}
        states: list[dict] = []

        def interrupt(state):
            states.append(state)
            if state["step"] == 4:
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            train_op.run(Context(loaded=build_model(), on_checkpoint=interrupt), records, params)
        ticks: list[tuple] = []
        resumed = train_op.run(Context(loaded=build_model(), resume_state=states[-1],
                                       on_item=lambda *a: ticks.append(a)), records, params)["out"]
        unbroken = train_op.run(Context(loaded=build_model()), records, params)["out"]
        assert ticks == [(None, None, True)] * 4 + [()] * 2
        assert resumed["data"] == unbroken["data"] and resumed["train"] == unbroken["train"]

    def test_naming_the_decision_objective_changes_nothing(self):
        params = {"model": "tiny", "target": {"uniform": ["cat", "dog"]}, "closer": " mat",
                  "steps": 3, "lr": 1e-2, "seed": 3, "lora": {"rank": 2, "alpha": 4}}
        records = {"records": [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"}]}
        unnamed = train_op.run(Context(loaded=build_model()), records, params)["out"]
        named = train_op.run(Context(loaded=build_model()), records,
                             {**params, "objective": "decision"})["out"]
        assert named == unnamed and "objective" not in named["train"]


class TestRefusals:
    def refuse(self, record, code, *, template=TEMPLATE, **params) -> RecordRefused:
        with pytest.raises(RecordRefused, match=code) as caught:
            read_sft_items(build_model(template=template).tokenizer, [record], params)
        assert caught.value.code == code and caught.value.record == record["id"]
        assert caught.value.issue == {"code": code, "record": record["id"],
                                      "message": str(caught.value).removeprefix(f"{code}: ")}
        return caught.value

    def test_a_record_with_neither_text_nor_messages(self):
        self.refuse({"id": "p", "user": "the cat"}, "NO_TEXT")
        self.refuse({"id": "e", "text": ""}, "NO_TEXT")

    @pytest.mark.parametrize("message, says", [
        ({"index": 0, "participant": "alice", "role_as_seen": {}, "text": "a dog"}, "text/render"),
        ({"role": "system", "content": "the cat"}, "system"),
        ({"role": "tool", "content": "the cat"}, "role"),
        ({"role": "assistant", "content": [{"type": "tool_call", "name": "calc"}]}, "tool_call"),
        ({"role": "assistant"}, "no `content`"),
    ], ids=["transcript", "system", "tool-role", "tool-call", "no-content"])
    def test_a_message_that_is_not_a_text_turn(self, message, says):
        record = {"id": "m", "messages": [{"role": "user", "content": "the cat"}, message]}
        assert says in str(self.refuse(record, "UNREADABLE_MESSAGE"))

    def test_a_system_with_no_user_message_to_join(self):
        record = {"id": "s", "system": "the dog",
                  "messages": [{"role": "assistant", "content": "a cat"}]}
        self.refuse(record, "UNREADABLE_MESSAGE")

    def test_a_conversation_with_no_assistant_turn(self):
        self.refuse({"id": "u", "messages": [{"role": "user", "content": "the cat"}]},
                    "NO_ASSISTANT_TURN")

    def test_a_template_that_does_not_segment_the_turns(self):
        refused = self.refuse(CONVERSATION, "TEMPLATE_UNSEGMENTED", template=CHAT_TEMPLATE)
        assert "message 1" in str(refused)

    def test_a_template_that_refuses_the_conversation(self):
        assert "alternate" in str(self.refuse(CONVERSATION, "TEMPLATE_REFUSED", template=REFUSING))

    def test_a_turn_boundary_inside_a_token(self):
        self.refuse(CONVERSATION, "NATURALISM_VIOLATION", template=GLUED)

    def test_a_record_too_long_under_fail(self):
        assert "max_tokens" in str(self.refuse(DOCUMENT, "TOO_LONG", max_tokens=4,
                                               truncation="fail"))

    def test_a_record_with_nothing_left_to_train(self):
        self.refuse({"id": "one", "text": "cat"}, "NOTHING_TO_TRAIN")
        self.refuse(CONVERSATION, "NOTHING_TO_TRAIN", max_tokens=8)

    @pytest.mark.parametrize("params, inputs, says", [
        ({"objective": "dpo"}, {}, "'decision' or 'sft'"),
        ({"objective": "sft", "target": {"uniform": ["cat"]}}, {}, "`target`"),
        ({"objective": "sft", "closer": " mat", "operator": {"layers": [1]}}, {},
         "`closer`, `operator`"),
        ({"objective": "sft"}, {"anchors": [{"id": "x", "user": "the", "answer": "cat"}]},
         "`anchors`"),
        ({"max_tokens": 64, "target": {"uniform": ["cat"]}}, {}, "`max_tokens`"),
        ({"objective": "sft", "batch": {"target": 3}}, {}, "`record` items"),
        ({"objective": "sft", "batch": {"record": 0}}, {}, "`record` items"),
        ({"objective": "sft", "max_tokens": 1}, {}, "at least 2"),
        ({"objective": "sft", "truncation": "middle"}, {}, "'cut' or 'fail'"),
    ], ids=["objective", "target", "decision-params", "anchors", "max-tokens", "batch-kind",
            "batch-empty", "max-tokens-small", "truncation"])
    def test_a_param_of_the_other_objective(self, params, inputs, says):
        with pytest.raises(ValueError, match="adapter/train") as caught:
            train_op.run(Context(loaded=build_model()), {"records": [DOCUMENT], **inputs},
                         {"model": "tiny", "steps": 1, **params})
        assert says in str(caught.value)


RESULTS = "u/p/results/j_1"


@pytest.fixture
def tiny_hub(monkeypatch, tmp_path):
    from mechbench_compute import bench

    def load(model_id, **_):
        tiny = build_model()
        return tiny._model, tiny._processor

    architecture = dataclasses.replace(architectures.BY_MODEL_TYPE["llama"], load=load)
    monkeypatch.setattr("mechbench_compute.hub.ensure_model",
                        lambda model_id, **_: (model_id, "0" * 40, tmp_path))
    monkeypatch.setattr(model_mod, "_peek_config", lambda _path: {"model_type": "llama"})
    monkeypatch.setattr(architectures, "for_type", lambda _t: architecture)
    store: dict[str, object] = {}

    def emit(path, payload, **kw):
        store[path] = payload
        return {"path": path}

    monkeypatch.setattr(bench, "emit", emit)
    return store


class TestDownstream:
    def test_an_sft_adapter_fuses_into_a_later_read_like_any_other(self, tiny_hub):
        records = [{"id": "a", "user": "the cat sat"}, {"id": "b", "user": "a dog ran"}]
        capture = {"block": "activations/capture",
                   "params": {"model": "tiny/llama", "layers": [2], "position": "last"},
                   "inputs": {"records": records}}
        nodes = [{"id": "train", "block": "adapter/train",
                  "params": {"model": "tiny/llama", "objective": "sft", "steps": 4, "lr": 0.05,
                             "seed": 3, "lora": {"rank": 2, "alpha": 4}},
                  "inputs": {"records": [DOCUMENT, CONVERSATION]}},
                 {"id": "fused", **capture}, {"id": "plain", **capture},
                 {"id": "wrote", "block": "adapter/measure", "params": {}}]
        edges = [{"from": {"node": "train"}, "to": {"node": "fused", "port": "adapter"}},
                 {"from": {"node": "train"}, "to": {"node": "wrote", "port": "adapter"}}]
        outputs = [{"name": n, "from": {"node": n}} for n in ("train", "fused", "plain", "wrote")]
        spec = ProtocolSpec(kind="pipeline", prompt="", model_id=None, extra={
            "graph": {"dataflow": 2, "nodes": nodes, "edges": edges}, "outputs": outputs,
            "resultPath": RESULTS})
        out = ProtocolExecutor().run(spec).payload["outputs"]
        assert out["train"]["kind"] == "adapter/lora"
        assert out["train"]["train"]["objective"] == "sft"
        fused = {i["id"]: i["vector"] for i in out["fused"]["items"]}
        plain = {i["id"]: i["vector"] for i in out["plain"]["items"]}
        assert fused.keys() == plain.keys() and all(fused[k] != plain[k] for k in fused)
        shares = [i["mass_share"] for i in out["wrote"]["items"]]
        assert shares and abs(sum(shares) - 1.0) < 1e-9
