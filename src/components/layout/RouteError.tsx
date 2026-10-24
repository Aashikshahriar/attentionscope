import { isRouteErrorResponse, Link, useRouteError } from 'react-router-dom';
import { Callout } from '@/components/ui/primitives';

export function RouteError() {
  const err = useRouteError();
  const msg = isRouteErrorResponse(err)
    ? `${err.status} ${err.statusText}`
    : err instanceof Error
      ? err.message
      : 'Unknown error';
  // A chunk that fails to load usually means a new version was deployed.
  const stale = /dynamically imported module|Failed to fetch|Importing a module script failed/i.test(msg);
  return (
    <div className="mx-auto max-w-xl space-y-4 px-4 py-16">
      <h1 className="text-xl font-semibold">Something went wrong</h1>
      <Callout tone="danger" title={stale ? 'The app was updated' : 'Error'}>
        {stale ? 'A newer version of AttentionScope is available. Reload the page to continue.' : msg}
      </Callout>
      <div className="flex gap-3 text-sm">
        <button className="text-accent underline" onClick={() => window.location.reload()}>
          Reload
        </button>
        <Link className="text-accent underline" to="/">
          Go home
        </Link>
      </div>
    </div>
  );
}
