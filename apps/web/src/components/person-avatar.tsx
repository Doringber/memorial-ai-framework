"use client";

import { useEffect, useState } from "react";
import { apiFetchBlob } from "@/lib/api";

/**
 * Object URL for a person's photo. Photos sit behind the access key, which an
 * <img src> can't send, so the image is fetched as a blob instead.
 * Change `version` to force a reload after uploading a new photo.
 */
export function usePersonAvatar(personId: string, hasAvatar: boolean, version = 0): string | null {
  const key = personId && hasAvatar ? `${personId}:${version}` : null;
  const [loaded, setLoaded] = useState<{ key: string; url: string } | null>(null);

  useEffect(() => {
    if (!key) return;
    let objectUrl: string | null = null;
    let cancelled = false;
    apiFetchBlob(`/api/person/${personId}/avatar`)
      .then((blob) => {
        if (cancelled || !blob) return;
        objectUrl = URL.createObjectURL(blob);
        setLoaded({ key, url: objectUrl });
      })
      .catch(() => undefined);
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [key, personId]);

  return loaded && loaded.key === key ? loaded.url : null;
}

export function Avatar({ src, name, size }: { src: string | null; name: string; size: number }) {
  const style = { width: size, height: size };
  if (src) {
    // eslint-disable-next-line @next/next/no-img-element -- blob URL, not optimizable by next/image
    return <img src={src} alt={name} style={style} className="rounded-full object-cover shrink-0 border border-ink" />;
  }
  return (
    <div
      style={{ ...style, fontSize: size * 0.4 }}
      className="rounded-full bg-warm border border-ink text-ink flex items-center justify-center font-semibold shrink-0"
      aria-label={name}
    >
      {name.trim().charAt(0) || "?"}
    </div>
  );
}
