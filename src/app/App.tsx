import { lazy, Suspense } from 'react';
import { createHashRouter, RouterProvider } from 'react-router-dom';
import { AppShell } from '@/components/layout/AppShell';
import { RouteError } from '@/components/layout/RouteError';
import { Spinner } from '@/components/ui/primitives';
import { TooltipProvider } from '@/components/ui/controls';
import { LandingPage } from '@/features/landing/LandingPage';

const LabPage = lazy(() => import('@/features/workspace/LabPage'));
const TokenizersPage = lazy(() => import('@/features/tokenizer/TokenizersPage'));
const DocsPage = lazy(() => import('@/features/docs/DocsPage'));
const NotFound = lazy(() => import('@/components/layout/NotFound'));

function Loading() {
  return (
    <div className="flex min-h-[50vh] items-center justify-center gap-3 text-sm text-muted" role="status">
      <Spinner /> Loading…
    </div>
  );
}

const page = (el: React.ReactNode) => <Suspense fallback={<Loading />}>{el}</Suspense>;

// Hash routing keeps every route working on GitHub Pages, including on refresh.
const router = createHashRouter([
  {
    element: <AppShell />,
    errorElement: <RouteError />,
    children: [
      { index: true, element: <LandingPage /> },
      { path: 'lab', element: page(<LabPage />) },
      { path: 'tokenizers', element: page(<TokenizersPage />) },
      { path: 'docs', element: page(<DocsPage />) },
      { path: 'docs/:section', element: page(<DocsPage />) },
      { path: '*', element: page(<NotFound />) },
    ],
  },
]);

export function App() {
  return (
    <TooltipProvider>
      <RouterProvider router={router} />
    </TooltipProvider>
  );
}
