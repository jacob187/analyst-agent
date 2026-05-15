"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

export default function PostSignupPage() {
  const router = useRouter();

  useEffect(() => {
    const timer = setTimeout(() => router.replace("/"), 500);
    return () => clearTimeout(timer);
  }, [router]);

  return (
    <div className="flex min-h-[calc(100vh-7rem)] flex-col items-center justify-center gap-4">
      <div
        className="h-10 w-10 animate-spin rounded-full border-2 border-muted-foreground border-t-transparent"
        aria-hidden
      />
      <p className="text-sm text-muted-foreground">
        Setting up your account…
      </p>
    </div>
  );
}
