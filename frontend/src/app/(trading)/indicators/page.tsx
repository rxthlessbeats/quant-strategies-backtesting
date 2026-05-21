import { TopNav } from "@/components/nav";
import Container from "@/components/container";
import { fetchIndicatorCatalog, getApiBaseUrl } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function IndicatorsPage() {
  let items: Awaited<ReturnType<typeof fetchIndicatorCatalog>> = [];
  let error: string | null = null;

  try {
    items = await fetchIndicatorCatalog();
  } catch (e) {
    error = e instanceof Error ? e.message : "Failed to load catalog";
  }

  return (
    <>
      <TopNav title="Indicators" />
      <Container className="py-6">
        {error ? (
          <p className="text-destructive">{error}</p>
        ) : (
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="w-full text-left text-sm">
              <thead className="border-b border-border bg-muted/50">
                <tr>
                  <th className="px-4 py-3 font-medium">ID</th>
                  <th className="px-4 py-3 font-medium">Category</th>
                  <th className="px-4 py-3 font-medium">Default params</th>
                  <th className="px-4 py-3 font-medium">Description</th>
                  <th className="px-4 py-3 font-medium">Query example</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => {
                  const period = item.params.period;
                  const example =
                    period != null ? `${item.id}:${period}` : item.id;
                  return (
                    <tr
                      key={item.id}
                      className="border-b border-border last:border-0"
                    >
                      <td className="px-4 py-3 font-mono">{item.id}</td>
                      <td className="px-4 py-3">{item.category}</td>
                      <td className="px-4 py-3 font-mono text-xs">
                        {JSON.stringify(item.params)}
                      </td>
                      <td className="px-4 py-3 text-muted-foreground">
                        {item.description}
                      </td>
                      <td className="px-4 py-3 font-mono text-xs">{example}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        <p className="mt-4 text-xs text-muted-foreground">
          API: {getApiBaseUrl()}/api/v1/analysis/indicators
        </p>
      </Container>
    </>
  );
}
