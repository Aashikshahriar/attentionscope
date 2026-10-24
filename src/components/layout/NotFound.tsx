import { Link } from 'react-router-dom';

export default function NotFound() {
  return (
    <div className="mx-auto max-w-xl px-4 py-24 text-center">
      <p className="font-mono text-sm text-accent">404</p>
      <h1 className="mt-2 text-2xl font-semibold">This page does not exist</h1>
      <p className="mt-2 text-muted">The link may be outdated.</p>
      <div className="mt-6 flex justify-center gap-4 text-sm">
        <Link className="text-accent underline" to="/">
          Home
        </Link>
        <Link className="text-accent underline" to="/lab">
          Open the lab
        </Link>
      </div>
    </div>
  );
}
