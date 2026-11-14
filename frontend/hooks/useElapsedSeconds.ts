"use client";

import { useEffect, useState } from "react";

/**
 * Whole seconds elapsed since the hook first mounted while `active`.
 * Ticks once per second only while active; the interval is cleared when
 * `active` flips to false and on unmount. Returns 0 while inactive.
 */
export function useElapsedSeconds(active: boolean): number {
  const [start] = useState(() => Date.now());
  const [now, setNow] = useState(start);

  useEffect(() => {
    if (!active) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [active]);

  return active ? Math.max(0, Math.floor((now - start) / 1000)) : 0;
}
