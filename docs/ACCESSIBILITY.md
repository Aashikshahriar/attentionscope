# Accessibility

Target: WCAG 2.1 AA.

- Keyboard: every control is reachable and operable; visible focus rings; skip link to main content.
- Heatmaps: focusable; arrow keys move the selected query/key, Home/End jump, Esc clears; each move is announced
  via a live region with the exact value. A **Table view** lists every value with row/column headers.
- Arc view: query tokens form a keyboard listbox; per-key weights are listed as text.
- Compare and intervention grids are labelled buttons; every chart has a table alternative.
- Color: Viridis, Cividis, Magma and Greyscale colormaps (perceptually uniform, colorblind-safe); a √ display
  scale; signed effects use a blue–grey–orange diverging scale with numbers printed in each cell.
- Themes: dark, light, system; high-contrast mode; reduced motion (follows the OS setting or can be forced).
- Text: whitespace in tokens is shown with visible markers and announced in words; right-to-left scripts are
  displayed with bidirectional isolation.
- Testing: axe-core (WCAG 2.0/2.1 A and AA rules) runs in CI on the landing page, lab (demo and live model, all
  tabs, light/high-contrast), tokenizers and docs pages.

Known gaps: canvas pixels themselves are not exposed to assistive technology (the table view is the accessible
equivalent); the Q/K/V matrix editors are large grids of number inputs.

Report barriers through GitHub issues.
