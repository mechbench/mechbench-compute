from __future__ import annotations

WORDS = ("<pad>", "<unk>", "<start>", "<end>", "user", "model", "assistant",
         "the", "cat", "sat", "on", "a", "mat", "and", "dog", "ran")

CHAT_TEMPLATE = (
    "{% for m in messages %}<start>{{ m['role'] }} "
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for c in m['content'] %}{% if c['type'] == 'text' %}{{ c['text'] }}"
    "{% endif %}{% endfor %}{% endif %}<end>{% endfor %}"
    "{% if add_generation_prompt %}<start>model {% endif %}")


def build_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(WORDS)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", unk_token="<unk>",
        bos_token="<start>", eos_token="<end>",
        additional_special_tokens=["<start>", "<end>"], chat_template=CHAT_TEMPLATE)
