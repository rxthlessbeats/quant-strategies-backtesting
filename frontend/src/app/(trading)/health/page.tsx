import { TopNav } from "@/components/nav";
import Container from "@/components/container";
import { fetchHealth, getApiBaseUrl } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function HealthPage() {
  const apiBase = getApiBaseUrl();
  let status: string | null = null;
  let error: string | null = null;

  try {
    const res = await fetchHealth();
    status = res.status;
  } catch (e) {
    error = e instanceof Error ? e.message : "Unreachable";
  }

  const ok = status === "ok";

  return (
    <>
      <TopNav title="Health" />
      <Container className="py-6">
        <div className="max-w-lg space-y-4">
          <div
            className={`rounded-lg border px-4 py-6 ${
              ok
                ? "border-emerald-500/40 bg-emerald-500/10"
                : "border-destructive/40 bg-destructive/10"
            }`}
          >
            <p className="text-lg font-medium">
              {ok ? "API is healthy" : "API unavailable"}
            </p>
            {status && (
              <p className="mt-1 font-mono text-sm text-muted-foreground">
                status: {status}
              </p>
            )}
            {error && (
              <p className="mt-2 text-sm text-destructive">{error}</p>
            )}
          </div>
          <dl className="space-y-2 text-sm">
            <div>
              <dt className="text-muted-foreground">Base URL</dt>
              <dd className="font-mono">{apiBase}</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">Health endpoint</dt>
              <dd className="font-mono">{apiBase}/health</dd>
            </div>
            <div>
              <dt className="text-muted-foreground">OpenAPI docs</dt>
              <dd>
                <a
                  href={`${apiBase}/docs`}
                  className="font-mono underline hover:text-foreground"
                  target="_blank"
                  rel="noreferrer"
                >
                  {apiBase}/docs
                </a>
              </dd>
            </div>
          </dl>
        </div>
      </Container>
    </>
  );
}
