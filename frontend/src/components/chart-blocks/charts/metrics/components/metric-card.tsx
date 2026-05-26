import { ArrowDownRight, ArrowUpRight } from "lucide-react";
import { chartTitle } from "@/components/primitives";
import { cn } from "@/lib/utils";

export default function MetricCard({
  label,
  price,
  change,
  className,
}: {
  label: string;
  price: number;
  change: number;
  className?: string;
}) {
  return (
    <section className={cn("flex flex-col", className)}>
      <h2 className={cn(chartTitle({ color: "mute", size: "sm" }), "mb-1")}>
        {label}
      </h2>
      <div className="flex items-center gap-2">
        <span className="text-xl font-medium">
          {price.toLocaleString(undefined, {
            maximumFractionDigits: 2,
            minimumFractionDigits: 2,
          })}
        </span>
        <ChangeIndicator change={change} />
      </div>
      <div className="text-xs text-muted-foreground">Compare to last day</div>
    </section>
  );
}

function ChangeIndicator({ change }: { change: number }) {
  return (
    <span
      className={cn(
        "flex items-center rounded-sm px-1 py-0.5 text-xs text-muted-foreground",
        change > 0
          ? "bg-green-50 text-green-500 dark:bg-green-950"
          : "bg-red-50 text-red-500 dark:bg-red-950",
      )}
    >
      {change > 0 ? "+" : ""}
      {(change * 100).toFixed(2)}%
      {change > 0 ? (
        <ArrowUpRight className="ml-0.5 inline-block h-3 w-3" />
      ) : (
        <ArrowDownRight className="ml-0.5 inline-block h-3 w-3" />
      )}
    </span>
  );
}
