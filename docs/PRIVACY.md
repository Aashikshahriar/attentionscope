# Privacy

AttentionScope is a static website. It has no backend, accounts, cookies, analytics or telemetry.

- **Your text stays on your device.** Tokenization and model inference run in Web Workers in your browser.
- **Network requests** go only to the site itself (app code, demo artifacts, tokenizers, model weights). No
  third-party fonts, CDNs, trackers or inference APIs are used.
- **Local storage:** display settings in `localStorage`; saved experiments in IndexedDB; model weights in Cache
  Storage only if you enable "Keep a copy in browser storage". Remove them with your browser's site-data controls
  or the in-app delete/clear buttons.
- **Share links** contain the view configuration only. Text from a live run is added only if you switch on
  "Include input text in the link" for that link.
- **Exports** are generated locally and include the input text you analysed; review them before sharing.
- **Hosting:** GitHub Pages may log standard request metadata (such as IP addresses) under the
  [GitHub Privacy Statement](https://docs.github.com/site-policy/privacy-policies/github-general-privacy-statement).
