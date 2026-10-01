import { gzip } from "node:zlib";
import { promisify } from "node:util";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";
export const maxDuration = 60;

const ALLOWED = /^(health|api\/v1\/.+)$/;
const gzipBody = promisify(gzip);

export async function GET(
  request: NextRequest,
  context: { params: Promise<{ path: string[] }> },
) {
  const { path } = await context.params;
  const joined = path.join("/");
  if (!ALLOWED.test(joined)) {
    return NextResponse.json({ detail: "Not found" }, {
      status: 404, headers: { "cache-control": "no-store" },
    });
  }

  const fetchSite = request.headers.get("sec-fetch-site");
  if (fetchSite !== "same-origin") {
    return NextResponse.json({ detail: "Unauthorized" }, {
      status: 401, headers: { "cache-control": "no-store" },
    });
  }

  const backend = (process.env.API_URL ?? "http://127.0.0.1:8000").replace(
    /\/$/,
    "",
  );
  const secret = process.env.API_SECRET;
  if (!secret) {
    return NextResponse.json(
      { detail: "API_SECRET is not configured" },
      { status: 503, headers: { "cache-control": "no-store" } },
    );
  }

  const url = `${backend}/${joined}${request.nextUrl.search}`;
  try {
    const res = await fetch(url, {
      headers: { "X-API-Key": secret },
      cache: "no-store",
      signal: AbortSignal.timeout(55000),
    });
    const body = await res.arrayBuffer();
    const compressed = body.byteLength >= 1000 && request.headers
      .get("accept-encoding")?.split(",").some(value => value.trim() === "gzip");
    const payload = compressed ? new Uint8Array(await gzipBody(Buffer.from(body))) : body;
    return new NextResponse(payload, {
      status: res.status,
      headers: {
        "content-type": res.headers.get("content-type") ?? "application/json",
        "cache-control": res.ok && joined !== "health" && !request.nextUrl.searchParams.has("force")
          ? "private, max-age=60"
          : "no-store",
        "vary": "Sec-Fetch-Site, Accept-Encoding",
        ...(compressed ? { "content-encoding": "gzip" } : {}),
      },
    });
  } catch {
    return NextResponse.json(
      { detail: "The market data connection is unavailable. Please try again." },
      { status: 503, headers: { "cache-control": "no-store" } },
    );
  }
}
