import { useEffect } from 'react';
import { NavLink, Outlet, useLocation, Link } from 'react-router-dom';
import { useSettings, resolvedTheme, prefersReducedMotion } from '@/stores/settings';
import { cn } from '@/components/ui/primitives';
import { REPO_URL } from '@/services/paths';
import { APP_INFO } from '@/services/provenance';
import { Logo } from './Logo';
import { SettingsMenu } from './SettingsMenu';

function useApplySettings() {
  const { theme, highContrast, motion } = useSettings();
  useEffect(() => {
    const apply = () => {
      const root = document.documentElement;
      const t = resolvedTheme(theme);
      root.dataset.theme = t;
      root.dataset.contrast = highContrast ? 'high' : 'normal';
      root.dataset.motion = prefersReducedMotion(motion) ? 'reduce' : 'full';
      document
        .querySelector('meta[name="theme-color"]')
        ?.setAttribute('content', t === 'dark' ? '#080B12' : '#F5F7FB');
    };
    apply();
    const mq = [
      window.matchMedia?.('(prefers-color-scheme: light)'),
      window.matchMedia?.('(prefers-reduced-motion: reduce)'),
    ];
    mq.forEach((m) => m?.addEventListener?.('change', apply));
    return () => mq.forEach((m) => m?.removeEventListener?.('change', apply));
  }, [theme, highContrast, motion]);
}

const nav = [
  { to: '/lab', label: 'Lab' },
  { to: '/tokenizers', label: 'Tokenizers' },
  { to: '/docs', label: 'Docs' },
];

export function AppShell() {
  useApplySettings();
  const { pathname } = useLocation();
  const isLab = pathname.startsWith('/lab');
  useEffect(() => {
    const titles: Record<string, string> = {
      '/': 'AttentionScope — Inside the Mind of a Transformer',
      '/lab': 'Lab · AttentionScope',
      '/tokenizers': 'Tokenizers · AttentionScope',
    };
    document.title =
      titles[pathname] ?? (pathname.startsWith('/docs') ? 'Docs · AttentionScope' : 'AttentionScope');
    if (!isLab) window.scrollTo({ top: 0 });
  }, [pathname, isLab]);

  return (
    <div className="flex min-h-dvh flex-col">
      <a
        href="#main"
        onClick={(e) => {
          e.preventDefault();
          document.getElementById('main')?.focus();
        }}
        className="sr-only z-50 rounded-md bg-accent px-3 py-2 text-accent-fg focus:not-sr-only focus:fixed focus:left-3 focus:top-3"
      >
        Skip to content
      </a>
      <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur supports-[backdrop-filter]:bg-bg/70">
        <div className={cn('mx-auto flex h-14 items-center gap-3 px-4', isLab ? 'max-w-none' : 'max-w-7xl')}>
          <Link
            to="/"
            className="flex items-center gap-2 rounded-md text-fg"
            aria-label="AttentionScope home"
          >
            <Logo className="size-7" />
            <span className="hidden font-semibold tracking-tight sm:inline">AttentionScope</span>
          </Link>
          <nav aria-label="Primary" className="ml-2 flex items-center gap-0.5">
            {nav.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                className={({ isActive }) =>
                  cn(
                    'rounded-md px-2.5 py-1.5 text-[13.5px] font-medium transition-colors',
                    isActive ? 'bg-elevated text-fg' : 'text-muted hover:text-fg',
                  )
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-1">
            <a
              href={REPO_URL}
              target="_blank"
              rel="noreferrer"
              className="inline-flex size-9 items-center justify-center rounded-lg text-muted hover:bg-elevated hover:text-fg"
              aria-label="Source code on GitHub (opens in a new tab)"
            >
              <svg aria-hidden viewBox="0 0 24 24" className="size-[18px]" fill="currentColor">
                <path d="M12 .5a11.5 11.5 0 0 0-3.64 22.41c.58.1.79-.25.79-.56v-2c-3.2.7-3.88-1.37-3.88-1.37-.53-1.33-1.28-1.69-1.28-1.69-1.05-.71.08-.7.08-.7 1.16.08 1.77 1.19 1.77 1.19 1.03 1.77 2.7 1.26 3.36.96.1-.75.4-1.26.73-1.55-2.55-.29-5.24-1.28-5.24-5.69 0-1.26.45-2.29 1.19-3.09-.12-.29-.52-1.46.11-3.05 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.17-1.18 3.17-1.18.63 1.59.23 2.76.11 3.05.74.8 1.19 1.83 1.19 3.09 0 4.42-2.7 5.39-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 .31.21.67.8.56A11.5 11.5 0 0 0 12 .5Z" />
              </svg>
            </a>
            <SettingsMenu />
          </div>
        </div>
      </header>
      <main id="main" tabIndex={-1} className="flex-1 outline-none">
        <Outlet />
      </main>
      {!isLab && (
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-8 text-[13px] text-muted sm:flex-row sm:items-center sm:justify-between">
            <p>
              AttentionScope {APP_INFO.version} · MIT licensed · runs entirely in your browser, no text leaves
              your device.
            </p>
            <nav aria-label="Footer" className="flex flex-wrap gap-x-4 gap-y-1">
              <Link className="hover:text-fg" to="/docs/methodology">
                Methodology
              </Link>
              <Link className="hover:text-fg" to="/docs/provenance">
                Provenance
              </Link>
              <Link className="hover:text-fg" to="/docs/privacy">
                Privacy
              </Link>
              <Link className="hover:text-fg" to="/docs/accessibility">
                Accessibility
              </Link>
              <a className="hover:text-fg" href={REPO_URL} target="_blank" rel="noreferrer">
                GitHub
              </a>
            </nav>
          </div>
        </footer>
      )}
    </div>
  );
}
