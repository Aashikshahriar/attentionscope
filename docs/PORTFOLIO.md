# Portfolio integration

## Card (HTML)

```html
<article class="project-card">
  <a href="https://aashikshahriar.github.io/attentionscope/" target="_blank" rel="noopener">
    <img
      src="https://aashikshahriar.github.io/attentionscope/og-image.png"
      alt="AttentionScope: arc diagram of GPT-2 attention"
      width="1200"
      height="630"
      loading="lazy"
    />
  </a>
  <h3>AttentionScope — Inside the Mind of a Transformer</h3>
  <p>
    Browser-based interpretability lab: real GPT-2 attention tensors, a Pythia-14m language model running in a
    Web Worker with head ablations, a live Q/K/V calculator, and reproducible exports. React · TypeScript ·
    Web Workers · validated against PyTorch.
  </p>
  <p>
    <a href="https://aashikshahriar.github.io/attentionscope/" target="_blank" rel="noopener">Live demo</a> ·
    <a href="https://github.com/aashikshahriar/attentionscope" target="_blank" rel="noopener">Source</a>
  </p>
</article>
```

## Markdown

```markdown
### [AttentionScope](https://aashikshahriar.github.io/attentionscope/) — Inside the Mind of a Transformer

Interpretability lab that runs in the browser: real GPT-2 attention, live Pythia-14m inference with head
ablations and attention replacement, Q/K/V calculator, head metrics, rollout and logit lens.
Implementation validated against exact-float64 PyTorch (logits < 1e-8).
**Stack:** React 19, TypeScript, Vite, Web Workers, Tailwind, Playwright · [Source](https://github.com/aashikshahriar/attentionscope)
```

## Résumé bullet

- Built AttentionScope, an open-source Transformer interpretability lab: implemented a GPT-NeoX forward pass and
  byte-level BPE tokenizer in TypeScript (matching PyTorch to 1e-8 and Hugging Face tokenizers exactly), with
  in-browser head ablations, attention metrics and reproducible exports, deployed via GitHub Actions to Pages.

## Embedding a specific view

Link directly to a configuration, e.g. the induction head:
`https://aashikshahriar.github.io/attentionscope/#/lab?demo=induction&l=5&h=5&q=18`
