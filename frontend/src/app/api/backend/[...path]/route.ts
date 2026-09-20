import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const ALLOWED = /^(health|api\/v1\/.+)$/;

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const joined = path.join("/");
  if (!ALLOWED.test(joined)) {
    return NextResponse.json({ detail: "Not found" }, { status: 404 });
  }

  const fetchSite = request.headers.get("sec-fetch-site");
  if (fetchSite !== "same-origin") {
    return NextResponse.json({ detail: "Unauthorized" }, { status: 401 });
  }

  const backend = (process.env.API_URL ?? "http://127.0.0.1:8000").replace(
    /\/$/,
    "",
  );
  const secret = process.env.API_SECRET;
  if (!secret) {
    return NextResponse.json(
      { detail: "API_SECRET is not configured" },
      { status: 503 },
    );
  }

  const url = `${backend}/${joined}${request.nextUrl.search}`;
  const res = await fetch(url, {
    headers: { "X-API-Key": secret },
    cache: "no-store",
  });
  const body = await res.arrayBuffer();
  return new NextResponse(body, {
    status: res.status,
    headers: {
      "content-type": res.headers.get("content-type") ?? "application/json",
    },
  });
}
