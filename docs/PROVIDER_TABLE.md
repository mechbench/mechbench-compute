# Checking the provider table

`mechbench_compute/providers/pricing.py` holds what each provider model
costs, and `providers/features.py` what each one takes: effort levels,
reasoning displays, prompt caching. Every cost compute records, every
budget it holds a run to, and every price the platform shows comes from
these two tables. Providers change them without notice, so they are
checked against the providers' own pages on a schedule, the same way
every time.

## When

- **Before a release, when the gate says so.** `scripts/release.py`
  refuses to publish while any row was checked more than 60 days ago,
  while a promotion has ended with no price after it, or while a model
  is past its shutdown date (`pricing.find_table_problems`).
- **When a provider announces a model, a price change or a retirement.**
- **When `scripts/check_provider_models.py` finds something.**

## Where

Only the provider's own pages. A third-party price list, a blog post or
a model's memory of a price is not a source.

| Provider | Prices | Models, limits, status |
|---|---|---|
| Anthropic | platform.claude.com/docs/en/about-claude/pricing | /about-claude/models/overview, /about-claude/model-deprecations |
| OpenAI | developers.openai.com/api/docs/pricing (also `.md`) | /api/docs/models/`<id>`, /api/docs/deprecations |
| Gemini | ai.google.dev/gemini-api/docs/pricing | /gemini-api/docs/models, /deprecations, /changelog |
| xAI | docs.x.ai/developers/pricing | /developers/models/`<id>`, /release-notes, /migration |
| DeepSeek | api-docs.deepseek.com/quick_start/pricing | /updates, /news, /api/list-models |
| Fireworks | docs.fireworks.ai/serverless/pricing | fireworks.ai/models/`<id>` |

`PROVIDER_PAGES` in `pricing.py` holds the pricing page each row cites.

## What to record for each model

- Its exact API id. A row prices that id, its dated snapshots
  (`-2026-04-23`, `-20251001`, `-0709`) and its `-latest` alias, and
  nothing else. A new model is unpriced until it has a row of its own,
  so no model borrows a neighbour's price.
- The standard rates, in US dollars per million tokens: input, output,
  cache read, cache write (five minutes), cache write (one hour).
- A long-context tier, when the provider has one: the input size above
  which **the whole request** is billed at other rates, and those rates.
- A promotion: the last day it holds (`until`) and the price after it
  (`then`). When the provider does not say what follows, leave `then`
  empty and say so in `note`; the gate stops releases once `until` has
  passed.
- Its status: `current`, `preview`, `legacy`, `deprecated`, or `alias`
  (an id the provider serves as, and bills as, another model; `note`
  names which). A shutdown date when one is announced.
- Anything the table cannot compute, in `note`: an off-peak discount, a
  restriction on who may use the model, pages that disagree.
- `source` and `checked`: the page read, and the day it was read.
- Its features: the effort levels and reasoning displays the model
  takes, and whether it caches on request. Only values a page states.

Batch, flex, priority and regional rates are not recorded: compute
calls the standard tier.

## How

1. Read each provider's pages. Never guess a price or a limit; where a
   page does not say, the row does not say. Where two pages disagree,
   take the pricing page and write the disagreement in `note`.
2. Run `scripts/check_provider_models.py` with the provider keys in
   `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
   `XAI_API_KEY`, `DEEPSEEK_API_KEY` (one missing is skipped). It lists
   each provider's served models and reports what the table prices that
   is no longer served, what is served that the table does not price,
   and shutdown dates that differ (OpenAI's API gives them). xAI's API
   gives prices too, and those are compared outright. The others' APIs
   give ids only, so their prices are read from the pages.
3. Edit the rows. Set `checked` on every row read, changed or not, and
   bump `TABLE_VERSION` to the day.
4. `python -m pytest tests/test_price_table.py`: every row has a source,
   a check date and a known status; aliases and open-ended promotions
   say what they are.
5. Regenerate the generated fixtures and the platform's copy:
   `scripts/dump_provider_fixtures.py`, then
   `scripts/dump_support_ts.py > ../mechbench-models/src/support.generated.ts`.
6. Release notes: a changed price goes under "Changes that alter results
   without raising", naming the models and the old and new rates, since
   recorded costs and budgets move with it.
