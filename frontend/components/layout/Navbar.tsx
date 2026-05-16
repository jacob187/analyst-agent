"use client";

import { useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart2, Menu } from "lucide-react";
import { Show, SignInButton, UserButton } from "@/lib/auth";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { cn } from "@/lib/utils";

// Links shown to everyone (signed-out + signed-in).
const PUBLIC_LINKS = [
  { href: "/", label: "Analyze" },
  { href: "/about", label: "About" },
];

// Signed-in-only links. Each lands on a protected route, so we don't show
// them to signed-out visitors — clicking would just bounce them to /sign-in.
const AUTH_LINKS = [
  { href: "/watchlist", label: "Watchlist" },
  { href: "/companies", label: "Companies" },
  { href: "/history", label: "History" },
];

export function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/40 bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-7xl items-center justify-between px-4 sm:px-6">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 font-display">
          <div className="flex h-7 w-7 items-center justify-center rounded-lg bg-primary">
            <BarChart2 className="h-4 w-4 text-primary-foreground" />
          </div>
          <span className="text-sm font-bold tracking-tight">
            analyst<span className="text-primary">·agent</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden items-center gap-1 md:flex">
          {PUBLIC_LINKS.map((link) => (
            <Link
              key={link.href}
              href={link.href}
              className={cn(
                "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                pathname === link.href
                  ? "bg-primary/10 text-primary"
                  : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              {link.label}
            </Link>
          ))}
          <Show when="signed-in">
            {AUTH_LINKS.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className={cn(
                  "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                  pathname === link.href
                    ? "bg-primary/10 text-primary"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                )}
              >
                {link.label}
              </Link>
            ))}
          </Show>
        </nav>

        <div className="flex items-center gap-2">
          <ThemeToggle />
          <Show when="signed-in">
            <Link href="/settings" className="hidden md:block">
              <Button size="sm" className="rounded-full font-medium">
                Settings
              </Button>
            </Link>
            <div className="hidden md:flex items-center">
              <UserButton
                appearance={{ elements: { avatarBox: "h-8 w-8" } }}
              />
            </div>
          </Show>
          <Show when="signed-out">
            <SignInButton mode="modal">
              <Button size="sm" className="hidden md:inline-flex rounded-full font-medium">
                Sign in
              </Button>
            </SignInButton>
          </Show>

          {/* Mobile hamburger */}
          <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="md:hidden h-9 w-9">
                <Menu className="h-4 w-4" />
              </Button>
            </SheetTrigger>
            <SheetContent side="right" className="w-64">
              <SheetHeader>
                <SheetTitle className="flex items-center gap-2 font-display text-left">
                  <div className="flex h-6 w-6 items-center justify-center rounded-md bg-primary">
                    <BarChart2 className="h-3.5 w-3.5 text-primary-foreground" />
                  </div>
                  analyst·agent
                </SheetTitle>
              </SheetHeader>
              <nav className="mt-6 flex flex-col gap-1">
                {PUBLIC_LINKS.map((link) => (
                  <Link
                    key={link.href}
                    href={link.href}
                    onClick={() => setMobileOpen(false)}
                    className={cn(
                      "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                      pathname === link.href
                        ? "bg-primary/10 text-primary"
                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    )}
                  >
                    {link.label}
                  </Link>
                ))}
                <Show when="signed-in">
                  {AUTH_LINKS.map((link) => (
                    <Link
                      key={link.href}
                      href={link.href}
                      onClick={() => setMobileOpen(false)}
                      className={cn(
                        "rounded-md px-3 py-2 text-sm font-medium transition-colors",
                        pathname === link.href
                          ? "bg-primary/10 text-primary"
                          : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      )}
                    >
                      {link.label}
                    </Link>
                  ))}
                </Show>
                <Show when="signed-in">
                  <Link
                    href="/settings"
                    onClick={() => setMobileOpen(false)}
                    className="mt-2 rounded-md px-3 py-2 text-sm font-medium text-muted-foreground hover:bg-muted hover:text-foreground"
                  >
                    Settings
                  </Link>
                  <div className="mt-4 flex items-center gap-2 px-3">
                    <UserButton
                      appearance={{ elements: { avatarBox: "h-8 w-8" } }}
                    />
                    <span className="text-sm text-muted-foreground">Account</span>
                  </div>
                </Show>
                <Show when="signed-out">
                  <SignInButton mode="modal">
                    <Button
                      size="sm"
                      className="mt-4 mx-3 rounded-full font-medium"
                      onClick={() => setMobileOpen(false)}
                    >
                      Sign in
                    </Button>
                  </SignInButton>
                </Show>
              </nav>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </header>
  );
}
