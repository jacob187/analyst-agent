"use client";

import { useState } from "react";
import { Eye, EyeOff, CheckCircle2, AlertCircle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useApiKeys } from "@/hooks/useApiKeys";
import { useModels } from "@/hooks/useModels";
import { PROVIDER_LABELS, type Provider } from "@/lib/constants";
import type { ApiKeys } from "@/types";

export function ApiKeyForm() {
  const { keys, setKeys } = useApiKeys();
  const { models, envKeys } = useModels();
  const [local, setLocal] = useState<ApiKeys | null>(null);
  const [show, setShow] = useState(false);
  const [saved, setSaved] = useState(false);

  // Use local draft or fall back to persisted keys
  const draft = local ?? keys;

  function update(field: keyof ApiKeys, value: string) {
    setLocal({ ...(local ?? keys), [field]: value });
    setSaved(false);
  }

  function handleSave(e: React.FormEvent) {
    e.preventDefault();
    if (local) {
      setKeys(local);
      setLocal(null);
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  }

  const providers: Provider[] = ["google_genai", "openai", "anthropic"];
  const keyField: Record<Provider, keyof ApiKeys> = {
    google_genai: "google_api_key",
    openai: "openai_api_key",
    anthropic: "anthropic_api_key",
  };

  return (
    <form onSubmit={handleSave} className="space-y-6">
      {/* LLM Provider Keys */}
      <Card>
        <CardHeader>
          <CardTitle className="font-display text-base">
            LLM Provider Keys
          </CardTitle>
          <CardDescription>
            At least one provider key is required. Keys never leave your browser.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          {providers.map((provider) => {
            const field = keyField[provider];
            const hasEnv = envKeys?.[provider === "google_genai" ? "google" : provider] ?? false;
            return (
              <div key={provider} className="space-y-1.5">
                <div className="flex items-center justify-between">
                  <Label htmlFor={field}>{PROVIDER_LABELS[provider]}</Label>
                  {hasEnv && (
                    <Badge variant="secondary" className="text-xs">
                      <CheckCircle2 className="mr-1 h-3 w-3 text-primary" />
                      Set on server
                    </Badge>
                  )}
                </div>
                <div className="relative">
                  <Input
                    id={field}
                    type={show ? "text" : "password"}
                    value={draft[field]}
                    onChange={(e) => update(field, e.target.value)}
                    placeholder={hasEnv ? "Overrides server key" : "sk-..."}
                    className="pr-10 font-mono text-sm"
                  />
                  <button
                    type="button"
                    onClick={() => setShow((s) => !s)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    {show ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>
            );
          })}
        </CardContent>
      </Card>

      {/* SEC Header */}
      <Card>
        <CardHeader>
          <CardTitle className="font-display text-base">
            SEC EDGAR User-Agent
            <span className="ml-2 text-xs font-normal text-destructive">
              Required
            </span>
          </CardTitle>
          <CardDescription>
            Required by SEC fair-access policy. Format:{" "}
            <code className="text-xs">Your Name email@example.com</code>
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2">
            {draft.sec_header ? (
              <CheckCircle2 className="h-4 w-4 shrink-0 text-primary" />
            ) : (
              <AlertCircle className="h-4 w-4 shrink-0 text-destructive" />
            )}
            <Input
              id="sec_header"
              value={draft.sec_header}
              onChange={(e) => update("sec_header", e.target.value)}
              placeholder="Jane Doe jane@example.com"
              className="font-mono text-sm"
            />
          </div>
        </CardContent>
      </Card>

      {/* Tavily + Model */}
      <Card>
        <CardHeader>
          <CardTitle className="font-display text-base">
            Optional Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-1.5">
            <div className="flex items-center justify-between">
              <Label htmlFor="tavily_api_key">Tavily API Key</Label>
              {envKeys?.tavily && (
                <Badge variant="secondary" className="text-xs">
                  <CheckCircle2 className="mr-1 h-3 w-3 text-primary" />
                  Set on server
                </Badge>
              )}
            </div>
            <Input
              id="tavily_api_key"
              type={show ? "text" : "password"}
              value={draft.tavily_api_key}
              onChange={(e) => update("tavily_api_key", e.target.value)}
              placeholder="tvly-... (enables web research)"
              className="font-mono text-sm"
            />
          </div>

          {models.length > 0 && (
            <div className="space-y-1.5">
              <Label htmlFor="model_id">LLM Model</Label>
              <select
                id="model_id"
                value={draft.model_id}
                onChange={(e) => update("model_id", e.target.value)}
                className="flex h-9 w-full rounded-md border border-input bg-background px-3 py-1 text-sm shadow-xs transition-colors focus:outline-none focus:ring-1 focus:ring-ring"
              >
                <option value="">Default model</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.display_name}
                    {m.default ? " (default)" : ""}
                    {m.thinking_capable ? " ✦" : ""}
                  </option>
                ))}
              </select>
              <p className="text-xs text-muted-foreground">✦ = supports extended thinking</p>
            </div>
          )}
        </CardContent>
      </Card>

      <Button
        type="submit"
        className="w-full rounded-full font-semibold"
        size="lg"
      >
        {saved ? (
          <>
            <CheckCircle2 className="mr-2 h-4 w-4" /> Saved
          </>
        ) : (
          "Save Settings"
        )}
      </Button>
    </form>
  );
}
