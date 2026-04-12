import { ApiKeyForm } from "@/components/settings/ApiKeyForm";

export default function SettingsPage() {
  return (
    <div className="mx-auto max-w-2xl px-4 py-12 sm:px-6">
      <div className="mb-8">
        <h1 className="font-display text-2xl font-bold sm:text-3xl">
          Settings
        </h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Configure your API keys and model preferences. All keys are stored
          in your browser&apos;s localStorage and sent directly to the backend
          — they never pass through any intermediate server.
        </p>
      </div>
      <ApiKeyForm />
    </div>
  );
}
