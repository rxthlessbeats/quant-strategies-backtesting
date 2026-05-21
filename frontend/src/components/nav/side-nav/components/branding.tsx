import Link from "next/link";
import { CandlestickChart } from "lucide-react";
import { siteConfig } from "@/config/site";

export default function Branding() {
  return (
    <Link
      href="/"
      className="relative my-2 flex flex-col items-center justify-center gap-y-1 px-4 py-4"
    >
      <div className="dot-matrix absolute left-0 top-0 -z-10 h-full w-full" />
      <CandlestickChart className="text-accent-foreground" size={28} />
      <span className="text-sm font-medium text-accent-foreground">
        {siteConfig.title}
      </span>
    </Link>
  );
}
