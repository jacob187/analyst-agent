import Link from "next/link";
import { BarChart2 } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-border/40 bg-background py-6">
      <div className="mx-auto flex max-w-7xl flex-col items-center justify-between gap-4 px-4 text-xs text-muted-foreground sm:flex-row sm:px-6">
        <div className="flex items-center gap-2">
          <div className="flex h-5 w-5 items-center justify-center rounded bg-primary/20">
            <BarChart2 className="h-3 w-3 text-primary" />
          </div>
          <span>analyst·agent — Not financial advice</span>
        </div>
        <div className="flex items-center gap-4">
          <Link href="/about" className="hover:text-foreground transition-colors">About</Link>
          <Link href="/settings" className="hover:text-foreground transition-colors">Settings</Link>
          <span>Built with LangGraph</span>
        </div>
      </div>
    </footer>
  );
}
