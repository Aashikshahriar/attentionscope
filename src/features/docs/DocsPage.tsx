import { useEffect, useState, type ReactNode } from 'react';
import { NavLink, useParams, Navigate } from 'react-router-dom';
import { Tex } from '@/components/ui/Tex';
import { Badge, Callout, cn } from '@/components/ui/primitives';
import { fetchDemoManifest, fetchTokenizerManifest } from '@/services/runs';
import { assetUrl, REPO_URL } from '@/services/paths';
import { APP_INFO } from '@/services/provenance';
import type { DemoManifest, TokenizerManifestEntry } from '@/types/artifacts';
import type { ModelManifest } from '@/types/contracts';

const SECTIONS = [
  ['methodology', 'Methodology & math'],
  ['interpretation', 'Interpreting attention'],
  ['interventions', 'Interventions'],
  ['provenance', 'Models & provenance'],
  ['privacy', 'Privacy'],
  ['accessibility', 'Accessibility'],
  ['limitations', 'Limitations & support'],
] as const;

function H({ children }: { children: ReactNode }) {
  return <h2 className="mt-8 text-lg font-semibold text-fg first:mt-0">{children}</h2>;
}
function P({ children }: { children: ReactNode }) {
  return <p className="mt-3 text-[14.5px] leading-relaxed text-muted">{children}</p>;
}
function L({ children }: { children: ReactNode }) {
  return (
    <ul className="mt-3 list-disc space-y-1.5 pl-5 text-[14.5px] leading-relaxed text-muted">{children}</ul>
  );
}

function Methodology() {
  return (
    <>
      <H>Scaled dot-product attention</H>
      <P>
        For an input X ∈ ℝ<sup>T×d</sup> (after LayerNorm), each head h projects queries, keys and values:
      </P>
      <Tex
        block
      >{String.raw`Q_h = XW_Q^{h},\quad K_h = XW_K^{h},\quad V_h = XW_V^{h},\qquad W^{h}_{Q,K,V}\in\mathbb{R}^{d\times d_k}`}</Tex>
      <Tex
        block
      >{String.raw`S_h = \frac{Q_hK_h^\top}{\sqrt{d_k}} + M,\qquad A_h = \operatorname{softmax}_{\text{row}}(S_h),\qquad O_h = A_hV_h`}</Tex>
      <Tex block>{String.raw`\operatorname{MHA}(X) = [O_1\,\|\,\cdots\,\|\,O_H]\,W_O + b_O`}</Tex>
      <P>
        S<sub>h</sub> are <em>pre-softmax scores</em>; A<sub>h</sub> are{' '}
        <em>post-softmax attention probabilities</em>. The lab shows probabilities by default and, for live
        runs, the scores (toggle “Pre-softmax”). Rows index queries (the position doing the reading) and
        columns index keys (the position being read). Every row of A sums to 1 over the visible keys.
      </P>
      <H>Masks</H>
      <P>
        Decoder models use a causal mask: M<sub>ij</sub> = 0 if j ≤ i and −∞ otherwise, so future keys receive
        exactly zero probability. AttentionScope runs a single unpadded sequence, so the attention (padding)
        mask is all ones. The Q/K/V lab lets you add padding to see how key masking works, including the
        undefined case of a row with no visible key.
      </P>
      <Tex
        block
      >{String.raw`a_{ij} = \frac{\exp(s_{ij}-m_i)}{\sum_{j'\le i}\exp(s_{ij'}-m_i)},\qquad m_i=\max_{j\le i}s_{ij}`}</Tex>
      <P>
        Subtracting the row maximum m<sub>i</sub> leaves the result unchanged and prevents overflow (stable
        softmax).
      </P>
      <H>Architecture details that matter</H>
      <L>
        <li>
          <strong className="text-fg">GPT-2 small</strong>: 12 layers × 12 heads, d<sub>head</sub> = 64,
          learned absolute position embeddings, scale 1/√64, pre-LayerNorm, tied embeddings, GELU (tanh
          approximation).
        </li>
        <li>
          <strong className="text-fg">Pythia-14m</strong> (GPT-NeoX): 6 layers × 4 heads, d<sub>head</sub> =
          32, rotary position embeddings on the first 8 of 32 dimensions of q and k, scale 1/√32,{' '}
          <em>parallel</em> residual (h ← h + Attn(LN₁h) + MLP(LN₂h)), exact GELU, untied unembedding.
        </li>
        <li>
          Both use standard multi-head attention: one K/V head per query head. Grouped-query or multi-query
          models share K/V across query heads; their tensors must not be visualized as if every head had its
          own keys. AttentionScope records the number of KV heads in each manifest and refuses to load a GQA
          checkpoint into the MHA implementation.
        </li>
        <li>
          Attention dropout exists in GPT-2 training but is disabled in evaluation mode; all runs here are
          deterministic.
        </li>
      </L>
      <H>Metrics</H>
      <Tex
        block
      >{String.raw`H_i = -\sum_{j\le i} a_{ij}\log(a_{ij}+\epsilon),\quad \epsilon=10^{-12}\qquad \tilde H_i = H_i/\log(i+1)`}</Tex>
      <L>
        <li>
          Entropy is in nats, summed only over causally visible keys; head-level values average rows i ≥ 1
          (row 0 has one key).
        </li>
        <li>
          Mean attention distance: Σ<sub>j</sub> a<sub>ij</sub>(i − j), averaged over rows.
        </li>
        <li>
          Previous-token, first-token and self scores: mean of a<sub>i,i−1</sub>, a<sub>i,0</sub>, a
          <sub>i,i</sub> over i ≥ 1.
        </li>
        <li>
          Head similarity: 1 − mean Jensen–Shannon divergence (bits) between corresponding rows, or Pearson
          correlation over visible entries.
        </li>
        <li>
          Attention rollout (Abnar &amp; Zuidema, 2020): R = Â<sub>L−1</sub>⋯Â<sub>0</sub>, Â = rownorm(αI +
          (1 − α)·mean
          <sub>h</sub>A<sub>h</sub>). It ignores values, W<sub>O</sub>, MLPs and LayerNorm.
        </li>
        <li>
          Logit lens: the residual stream after each layer passed through the final LayerNorm and unembedding.
        </li>
      </L>
    </>
  );
}

function Interpretation() {
  return (
    <>
      <Callout tone="warn" title="Attention is not explanation">
        Attention weights describe where a head reads from. They do not, by themselves, establish why the
        model produced an output.
      </Callout>
      <L>
        <li>
          A head&apos;s contribution is A<sub>h</sub>V<sub>h</sub>W<sub>O</sub>
          <sup>h</sup>; a large weight on a token whose value vector is small or orthogonal to downstream
          directions has little effect.
        </li>
        <li>
          Many heads park probability on the first token (an “attention sink”), which often acts as a no-op.
          Heavy attention there is not evidence that the first token matters.
        </li>
        <li>
          Information moves across positions through the residual stream over many layers; a token&apos;s
          representation at layer ℓ already mixes earlier tokens, so a key column is not “that word”.
        </li>
        <li>
          Patterns can be diffuse (high entropy) or specialized (e.g. previous-token or induction patterns).
          Labels such as “induction head” describe behavior on inputs like these, verified with metrics or
          interventions — they are hypotheses, not facts read off one heatmap.
        </li>
        <li>
          AttentionScope makes no claims about “reasoning paths”. Visualizations are descriptive;
          interventions provide limited, intervention-specific evidence.
        </li>
      </L>
      <P>
        Further reading: Jain &amp; Wallace (2019), <em>Attention is not Explanation</em>; Wiegreffe &amp;
        Pinter (2019), <em>Attention is not not Explanation</em>; Elhage et al. (2021),{' '}
        <em>A Mathematical Framework for Transformer Circuits</em>; Olsson et al. (2022),{' '}
        <em>In-context Learning and Induction Heads</em>.
      </P>
    </>
  );
}

function Interventions() {
  return (
    <>
      <P>
        Interventions are only offered where AttentionScope controls the forward pass: live Pythia-14m runs.
        Every intervention re-runs all downstream computation.
      </P>
      <H>Head-output ablation</H>
      <Tex
        block
      >{String.raw`O_h \leftarrow 0 \;\Rightarrow\; \operatorname{MHA}(X) = [O_1\|\cdots\|0\|\cdots\|O_H]\,W_O + b_O`}</Tex>
      <P>
        The attention pattern is still computed (and displayed unchanged); only the head&apos;s write into the
        residual stream is removed. The output-projection bias is kept.
      </P>
      <H>Attention-matrix replacement</H>
      <P>
        A<sub>h</sub> is replaced, after the softmax, by a valid causal distribution — uniform over visible
        keys, one-hot on the previous token, or one-hot on position 0 — and O<sub>h</sub> = A′V<sub>h</sub> is
        computed with the original values. Before running, the replacement is validated: correct shape, rows
        summing to 1, no mass on masked positions, no negative entries. The displayed attention for that head
        becomes the replacement.
      </P>
      <H>What is reported</H>
      <L>
        <li>
          Next-token distribution at the last position before and after (same token ids, no sampling, no
          dropout).
        </li>
        <li>
          Δ log P(target), probabilities and logits of the target, and KL(P<sub>base</sub> ‖ P
          <sub>intervened</sub>) in nats.
        </li>
        <li>
          Intervention kind, layer, head, exact location, model revision and validation report — saved with
          experiments and exports.
        </li>
      </L>
      <H>Validation</H>
      <P>
        The implementation is tested against PyTorch hooks on the reference model: zeroing the input slice of
        <code className="mx-1 font-mono text-[13px]">attention.dense</code> and substituting a head&apos;s
        probabilities inside a custom attention function give logits matching ours to &lt; 10<sup>−8</sup>.
        Tests also check that layers before an ablation are bit-identical and that only downstream computation
        changes.
      </P>
      <Callout title="Precomputed GPT-2 ablations">
        Demo artifacts include a zero-ablation of all 144 GPT-2 heads, computed offline with PyTorch forward
        pre-hooks on
        <code className="mx-1 font-mono text-[12px]">attn.c_proj</code>. These are labelled precomputed and
        are not executed in your browser.
      </Callout>
      <Callout tone="warn" title="Scope of evidence">
        Zero and pattern replacement move activations off-distribution. An effect shows that this intervention
        changes this output for this prompt. Stronger claims need distributions of prompts, mean or resample
        ablations and path patching.
      </Callout>
    </>
  );
}

function Provenance() {
  const [demos, setDemos] = useState<DemoManifest | null>(null);
  const [model, setModel] = useState<ModelManifest | null>(null);
  const [toks, setToks] = useState<TokenizerManifestEntry[]>([]);
  useEffect(() => {
    void fetchDemoManifest()
      .then(setDemos)
      .catch(() => undefined);
    void fetchTokenizerManifest()
      .then(setToks)
      .catch(() => undefined);
    void fetch(assetUrl('models/pythia-14m/manifest.json'))
      .then((r) => r.json() as Promise<ModelManifest>)
      .then(setModel)
      .catch(() => undefined);
  }, []);
  const v = model?.validation as { maxAbsLogitDiff?: number; maxAbsAttentionDiff?: number } | undefined;
  return (
    <>
      <P>
        All artifacts are generated by scripts in{' '}
        <code className="font-mono text-[13px]">scripts/generate-demo-artifacts/</code> from pinned model
        revisions. Checksums below are read from the deployed manifests and verified by the app before use.
      </P>
      <H>GPT-2 small (precomputed demos)</H>
      <L>
        <li>Source: openai-community/gpt2 · license MIT · Radford et al. (2019).</li>
        <li>
          Generated with transformers eager attention, float32, CPU, eval mode, deterministic algorithms;
          attention stored as float32 (no quantization).
        </li>
        <li>
          Each artifact records its environment, row-sum and causality checks, and SHA-256 of its tensor.
        </li>
      </L>
      {demos && (
        <div className="mt-3 overflow-x-auto rounded-lg border border-line">
          <table className="w-full text-[12px]">
            <thead className="bg-elevated text-muted">
              <tr>
                <th className="px-2.5 py-1.5 text-left font-medium">demo</th>
                <th className="px-2.5 py-1.5 text-left font-medium">tokens</th>
                <th className="px-2.5 py-1.5 text-left font-medium">attention SHA-256</th>
              </tr>
            </thead>
            <tbody>
              {demos.demos.map((d) => (
                <tr key={d.id} className="border-t border-line">
                  <td className="px-2.5 py-1.5 text-fg">{d.title}</td>
                  <td className="px-2.5 py-1.5 font-mono">{d.tokens}</td>
                  <td className="break-all px-2.5 py-1.5 font-mono text-[11px] text-muted">
                    {d.attentionSha256}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      <H>Pythia-14m (live in-browser model)</H>
      {model ? (
        <L>
          <li>
            Source: {model.source.repo}@<span className="font-mono">{model.source.revision}</span> · license{' '}
            {model.source.license} · {model.source.citation}
          </li>
          <li>
            Weights: float16 copied bit-exactly from the published checkpoint,{' '}
            {(model.weights.byteLength / 1e6).toFixed(1)} MB, SHA-256{' '}
            <span className="break-all font-mono text-[12px]">{model.weights.sha256}</span>.
          </li>
          <li>
            Tokenizer: {model.tokenizer.name}, SHA-256{' '}
            <span className="break-all font-mono text-[12px]">{model.tokenizer.sha256}</span>.
          </li>
          <li>
            Export validation: an independent NumPy float64 forward pass matched exact-float64 transformers to{' '}
            {v?.maxAbsLogitDiff?.toExponential(1)} (logits) and {v?.maxAbsAttentionDiff?.toExponential(1)}{' '}
            (attention). The TypeScript implementation is tested against the same reference.
          </li>
          <li>
            Browser computation: float64 activations, weights decoded exactly; attention stored as float32.
          </li>
        </L>
      ) : (
        <P>Loading manifest…</P>
      )}
      <H>Tokenizers</H>
      <L>
        {toks.map((t) => (
          <li key={t.id}>
            {t.name} — {t.repo}@<span className="font-mono">{t.revision.slice(0, 10)}</span> ({t.license}),
            SHA-256 <span className="break-all font-mono text-[12px]">{t.sha256}</span>
          </li>
        ))}
      </L>
      <H>Reproducibility</H>
      <P>
        Run{' '}
        <code className="font-mono text-[13px]">
          python scripts/generate-demo-artifacts/generate_gpt2_demos.py
        </code>{' '}
        to regenerate the demos (see the scripts README for pinned versions). Exports carry the model
        revision, tokenizer, token ids, source checksums, precision and app version ({APP_INFO.version},
        commit {APP_INFO.commit}).
      </P>
    </>
  );
}

function Privacy() {
  return (
    <>
      <L>
        <li>
          AttentionScope is a static site. There is no backend, no account and no analytics or telemetry.
        </li>
        <li>
          Text you type is tokenized and processed in your browser (Web Workers). It is never sent anywhere.
        </li>
        <li>
          Network requests go only to this site for app code and bundled artifacts (demos, tokenizers, model
          weights). No third-party fonts, CDNs or trackers are used.
        </li>
        <li>
          Settings are stored in localStorage; saved experiments in IndexedDB on your device. Clear them with
          your browser&apos;s site-data controls or the delete buttons.
        </li>
        <li>
          Model weights are cached in Cache Storage only if you switch on “Keep a copy in browser storage”.
        </li>
        <li>
          Share links contain the view configuration only; live-run text is included only if you opt in for
          that link.
        </li>
        <li>
          GitHub Pages, which hosts the site, may keep standard server logs (e.g. IP addresses) under
          GitHub&apos;s privacy statement.
        </li>
      </L>
    </>
  );
}

function Accessibility() {
  return (
    <>
      <L>
        <li>All controls are keyboard-operable with visible focus. A skip link jumps to the main content.</li>
        <li>
          Heatmaps are focusable: arrow keys move the selected query/key, values are announced through a live
          region, and a table view lists every value with row and column headers.
        </li>
        <li>
          The arc view and compare grid are buttons with descriptive labels; charts provide table
          alternatives.
        </li>
        <li>
          Colormaps (Viridis, Cividis, Magma, Greyscale) are perceptually uniform and colorblind-safe; values
          never rely on color alone.
        </li>
        <li>
          Dark and light themes, a high-contrast mode, and a reduced-motion setting (which also follows your
          OS preference).
        </li>
        <li>
          Token text is shown with visible whitespace markers and bidirectional isolation for right-to-left
          scripts.
        </li>
        <li>Automated axe-core checks run in CI on every page. Please report barriers via GitHub issues.</li>
      </L>
    </>
  );
}

function Limitations() {
  return (
    <>
      <L>
        <li>
          No arbitrary Hugging Face models: browser inference is limited to Pythia-14m, the model whose
          forward pass AttentionScope implements and validates.
        </li>
        <li>Live input is limited to 64 tokens; precomputed demos are fixed prompts.</li>
        <li>
          Pythia-14m is tiny; its predictions are often poor. It is suited to studying mechanisms, not
          capabilities.
        </li>
        <li>
          Pre-softmax scores are available for live runs only. Residual-stream norms and logit lens are shown;
          per-neuron MLP inspection, activation patching between prompts, causal tracing and cross-attention
          models are not implemented.
        </li>
        <li>
          Multilingual comparisons cover tokenization only; neither model was trained to handle Bengali, Hindi
          or Korean well.
        </li>
        <li>Single sequences only (batch size 1, no padding).</li>
      </L>
      <H>Browser support</H>
      <P>
        Current Chrome, Edge, Firefox and Safari (2023 or later) with ES2022, Web Workers, module workers, Web
        Crypto and IndexedDB. The live model needs roughly 150 MB of free memory. Pages work offline after
        loading only if the browser has cached them; the model is cached for offline use only with your
        consent.
      </P>
      <P>
        Source and issue tracker:{' '}
        <a className="text-accent underline" href={REPO_URL} target="_blank" rel="noreferrer">
          {REPO_URL}
        </a>
      </P>
    </>
  );
}

const CONTENT: Record<string, () => ReactNode> = {
  methodology: Methodology,
  interpretation: Interpretation,
  interventions: Interventions,
  provenance: Provenance,
  privacy: Privacy,
  accessibility: Accessibility,
  limitations: Limitations,
};

export default function DocsPage() {
  const { section } = useParams();
  if (!section) return <Navigate to="/docs/methodology" replace />;
  const Body = CONTENT[section];
  if (!Body) return <Navigate to="/docs/methodology" replace />;
  const title = SECTIONS.find(([k]) => k === section)?.[1];
  return (
    <div className="mx-auto grid max-w-6xl gap-8 px-4 py-10 md:grid-cols-[14rem_minmax(0,1fr)]">
      <nav aria-label="Documentation" className="md:sticky md:top-20 md:self-start">
        <ul className="flex flex-wrap gap-1 md:flex-col">
          {SECTIONS.map(([k, label]) => (
            <li key={k}>
              <NavLink
                to={`/docs/${k}`}
                className={({ isActive }) =>
                  cn(
                    'block rounded-md px-2.5 py-1.5 text-[13.5px]',
                    isActive ? 'bg-elevated text-fg' : 'text-muted hover:text-fg',
                  )
                }
              >
                {label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>
      <article className="min-w-0">
        <div className="mb-6 flex items-center gap-2">
          <h1 className="text-2xl font-semibold tracking-tight text-fg">{title}</h1>
          {section === 'provenance' && <Badge tone="teal">live data</Badge>}
        </div>
        <Body />
      </article>
    </div>
  );
}
