import IndicatorLibrary from "@/components/trading/indicator-library";
import { fetchIndicatorCatalog } from "@/lib/api";

export const dynamic = "force-dynamic";

export default async function IndicatorsPage() {
  let items: Awaited<ReturnType<typeof fetchIndicatorCatalog>> = [];
  let error: string | null = null;
  try { items = await fetchIndicatorCatalog(); }
  catch (e) { error = e instanceof Error ? e.message : "Failed to load catalog"; }
  return <IndicatorLibrary items={items} error={error} />;
}
