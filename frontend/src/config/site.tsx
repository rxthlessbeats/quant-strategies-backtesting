import {
  Activity,
  CandlestickChart,
  Gauge,
  ListOrdered,
  type LucideIcon,
} from "lucide-react";

export type SiteConfig = typeof siteConfig;
export type Navigation = {
  icon: LucideIcon;
  name: string;
  href: string;
};

export const siteConfig = {
  title: "Rookie Trader",
  description: "Advanced Interactive Stock charts and technical indicators",
};

export const navigations: Navigation[] = [
  { icon: Gauge, name: "Dashboard", href: "/" },
  { icon: CandlestickChart, name: "Chart", href: "/chart" },
  { icon: ListOrdered, name: "Indicators", href: "/indicators" },
  { icon: Activity, name: "Health", href: "/health" },
];
