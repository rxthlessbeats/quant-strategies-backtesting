export default function TradingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <main className="min-h-0 flex-1">{children}</main>;
}
