import { useState, useEffect, ReactNode } from 'react';

interface NavItem { label: string; path: string }
interface NavSection { title: string; items: NavItem[] }

const NAV: NavSection[] = [
  {
    title: 'Getting Started',
    items: [
      { label: 'Introduction', path: '/docs' },
      { label: 'Installation & Quickstart', path: '/docs/getting-started' },
    ],
  },
  {
    title: 'Core Concepts',
    items: [
      { label: 'Method System', path: '/docs/concepts/method-system' },
      { label: 'Loss System', path: '/docs/concepts/loss-system' },
      { label: 'Data Pipeline', path: '/docs/concepts/data-pipeline' },
    ],
  },
  {
    title: 'Guides',
    items: [
      { label: 'Writing a Custom Method', path: '/docs/guides/custom-method' },
      { label: 'Writing a Custom Loss', path: '/docs/guides/custom-loss' },
    ],
  },
  {
    title: 'Recipes',
    items: [
      { label: 'Diffusion Bridge Translation', path: '/docs/recipes/diffusion-bridge' },
    ],
  },
];

interface Props { children: ReactNode; currentRoute: string }

export default function DocsLayout({ children, currentRoute }: Props) {
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => { setMobileOpen(false); }, [currentRoute]);

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      {/* Header */}
      <header className="sticky top-0 z-30 border-b border-white/10 bg-neutral-950/80 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-4">
            <button
              className="md:hidden p-1 text-neutral-400 hover:text-white transition"
              onClick={() => setMobileOpen(v => !v)}
              aria-label="Toggle menu"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {mobileOpen
                  ? <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  : <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                }
              </svg>
            </button>
            <a href="#/">
              <div className="text-xs uppercase tracking-[0.25em] text-neutral-500">AllAxisAI</div>
              <div className="text-lg font-semibold leading-tight">OpenMedAxis</div>
            </a>
            <span className="hidden md:block text-neutral-700">|</span>
            <span className="hidden md:block text-sm text-neutral-400">Docs</span>
          </div>
          <nav className="hidden md:flex items-center gap-5 text-sm text-neutral-300">
            <a href="#/docs" className={`transition hover:text-white ${currentRoute.startsWith('/docs') ? 'text-white' : ''}`}>Documentation</a>
            <a href="#/" className="transition hover:text-white">Home</a>
            <a
              href="https://github.com/AllAxisAI/OpenMedAxis"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2 rounded-xl border border-white/15 bg-white/5 px-4 py-2 transition hover:bg-white/10"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z" />
              </svg>
              GitHub
            </a>
          </nav>
        </div>
      </header>

      <div className="mx-auto max-w-7xl flex">
        {/* Sidebar */}
        <aside className={`
          fixed inset-y-0 left-0 z-20 w-72 border-r border-white/10 bg-neutral-950/98 backdrop-blur
          transition-transform duration-300 pt-[73px] pb-10 overflow-y-auto
          md:sticky md:top-[73px] md:h-[calc(100vh-73px)] md:w-60 md:translate-x-0
          md:bg-transparent md:backdrop-blur-none
          ${mobileOpen ? 'translate-x-0' : '-translate-x-full'}
        `}>
          <nav className="px-5 pt-6 space-y-7">
            {NAV.map((section) => (
              <div key={section.title}>
                <p className="text-[10px] font-semibold uppercase tracking-[0.2em] text-neutral-600 mb-2.5">
                  {section.title}
                </p>
                <ul className="space-y-0.5">
                  {section.items.map((item) => {
                    const active = currentRoute === item.path;
                    return (
                      <li key={item.path}>
                        <a
                          href={`#${item.path}`}
                          className={`block px-3 py-2 rounded-xl text-sm transition-colors leading-snug
                            ${active
                              ? 'bg-white/10 text-white font-medium'
                              : 'text-neutral-400 hover:text-white hover:bg-white/5'
                            }`}
                        >
                          {item.label}
                        </a>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))}
          </nav>
        </aside>

        {/* Mobile overlay */}
        {mobileOpen && (
          <div
            className="fixed inset-0 z-10 bg-black/60 md:hidden"
            onClick={() => setMobileOpen(false)}
          />
        )}

        {/* Content */}
        <main className="flex-1 min-w-0 px-6 md:px-12 py-10 md:py-14 max-w-3xl">
          {children}
        </main>
      </div>
    </div>
  );
}
