import {
  AverageTicketsCreated,
  Conversions,
  CustomerSatisfication,
  Metrics,
  TicketByChannels,
} from "@/components/chart-blocks";
import Container from "@/components/container";

export const dynamic = "force-dynamic";

export default function HomePage() {
  return (
    <>
      <div>
        <Metrics />
        <div className="grid grid-cols-1 divide-y border-b border-border laptop:grid-cols-3 laptop:divide-x laptop:divide-y-0 laptop:divide-border">
          <Container className="py-4 laptop:col-span-2">
            <AverageTicketsCreated />
          </Container>
          <Container className="py-4 laptop:col-span-1">
            <Conversions />
          </Container>
        </div>
        <div className="grid grid-cols-1 divide-y border-b border-border laptop:grid-cols-2 laptop:divide-x laptop:divide-y-0 laptop:divide-border">
          <Container className="py-4 laptop:col-span-1">
            <TicketByChannels />
          </Container>
          <Container className="py-4 laptop:col-span-1">
            <CustomerSatisfication />
          </Container>
        </div>
      </div>
    </>
  );
}
