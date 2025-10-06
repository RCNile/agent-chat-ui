import { Button } from "@/components/ui/button";
import { useState, useEffect } from "react";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { PanelLeftOpen, PanelLeftClose, Download, Trash2 } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { toast } from "sonner";

interface SavedOutput {
  id: string;
  title: string;
  content: string;
  threadId: string;
  messageId: string;
  timestamp: Date;
}

function SavedOutputItem({
  output,
  onDelete,
}: {
  output: SavedOutput;
  onDelete: (id: string) => void;
}) {
  const [isDeleting, setIsDeleting] = useState(false);

  const handleDelete = async (e: React.MouseEvent) => {
    e.stopPropagation();
    
    if (!window.confirm("Are you sure you want to delete this saved output?")) {
      return;
    }

    setIsDeleting(true);
    try {
      onDelete(output.id);
      toast.success("Saved output deleted");
    } catch (error) {
      console.error("Failed to delete saved output:", error);
      toast.error("Failed to delete saved output");
    } finally {
      setIsDeleting(false);
    }
  };

  const handleExport = (e: React.MouseEvent) => {
    e.stopPropagation();
    navigator.clipboard.writeText(output.content);
    toast.success("Content copied to clipboard");
  };

  return (
    <div className="group relative w-full px-1">
      <Button
        variant="ghost"
        className="w-[280px] items-start justify-start text-left font-normal h-auto py-2"
      >
        <div className="flex-1 min-w-0 pr-12">
          <p className="truncate text-ellipsis text-sm font-medium">{output.title}</p>
          <p className="text-xs text-muted-foreground">
            {output.timestamp.toLocaleDateString('en-GB')} {output.timestamp.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })}
          </p>
        </div>
      </Button>
      <div className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
        <Button
          variant="ghost"
          size="sm"
          className="p-1 h-6 w-6 text-muted-foreground hover:text-foreground"
          onClick={handleExport}
          title="Copy to clipboard"
        >
          <Download className="h-3 w-3" />
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="p-1 h-6 w-6 text-muted-foreground hover:text-destructive"
          onClick={handleDelete}
          disabled={isDeleting}
          title="Delete saved output"
        >
          {isDeleting ? (
            <div className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
          ) : (
            <Trash2 className="h-3 w-3" />
          )}
        </Button>
      </div>
    </div>
  );
}

function SavedOutputsList({
  outputs,
  onDelete,
}: {
  outputs: SavedOutput[];
  onDelete: (id: string) => void;
}) {
  if (outputs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-6">
        <div className="text-gray-400 mb-2">
          <Download className="h-8 w-8 mx-auto" />
        </div>
        <p className="text-sm text-gray-500">No saved outputs yet</p>
        <p className="text-xs text-gray-400 mt-1">
          Save outputs from chat messages to see them here
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {outputs.map((output) => (
        <SavedOutputItem
          key={output.id}
          output={output}
          onDelete={onDelete}
        />
      ))}
    </div>
  );
}

function SavedOutputsLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 5 }).map((_, i) => (
        <Skeleton
          key={`skeleton-${i}`}
          className="h-20 w-full"
        />
      ))}
    </div>
  );
}

export default function SavedOutputs() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [savedOutputsOpen, setSavedOutputsOpen] = useQueryState(
    "savedOutputsOpen",
    parseAsBoolean.withDefault(false),
  );

  const [outputs, setOutputs] = useState<SavedOutput[]>([]);
  const [loading, setLoading] = useState(false);

  // Function to load saved outputs from localStorage
  const loadOutputs = () => {
    try {
      const saved = localStorage.getItem('saved-outputs');
      if (saved) {
        const parsed = JSON.parse(saved);
        const outputsWithDates = parsed.map((output: any) => ({
          ...output,
          timestamp: new Date(output.timestamp)
        }));
        setOutputs(outputsWithDates);
      }
    } catch (error) {
      console.error('Failed to load saved outputs:', error);
    }
  };

  // Load saved outputs from localStorage on mount
  useEffect(() => {
    setLoading(true);
    loadOutputs();
    setLoading(false);
  }, []);

  // Listen for storage changes (when new outputs are saved)
  useEffect(() => {
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'saved-outputs') {
        loadOutputs();
      }
    };

    // Also listen for custom event for same-window updates
    const handleCustomEvent = () => {
      loadOutputs();
    };

    window.addEventListener('storage', handleStorageChange);
    window.addEventListener('saved-outputs-updated', handleCustomEvent);

    return () => {
      window.removeEventListener('storage', handleStorageChange);
      window.removeEventListener('saved-outputs-updated', handleCustomEvent);
    };
  }, []);

  const handleDelete = (id: string) => {
    const updated = outputs.filter(output => output.id !== id);
    setOutputs(updated);
    localStorage.setItem('saved-outputs', JSON.stringify(updated));
  };

  return (
    <>
      <div className="shadow-inner-left hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start gap-6 border-l-[1px] border-slate-300 lg:flex">
        <div className="flex w-full items-center justify-between px-4 pt-1.5">
          <Button
            className="hover:bg-gray-100"
            variant="ghost"
            onClick={() => setSavedOutputsOpen((p) => !p)}
          >
            {savedOutputsOpen ? (
              <PanelLeftClose className="size-5" />
            ) : (
              <PanelLeftOpen className="size-5" />
            )}
          </Button>
          <h1 className="text-xl font-semibold tracking-tight">
            Saved Outputs
          </h1>
        </div>
        {loading ? (
          <SavedOutputsLoading />
        ) : (
          <SavedOutputsList outputs={outputs} onDelete={handleDelete} />
        )}
      </div>
      <div className="lg:hidden">
        <Sheet
          open={!!savedOutputsOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setSavedOutputsOpen(open);
          }}
        >
          <SheetContent
            side="right"
            className="flex lg:hidden"
          >
            <SheetHeader>
              <SheetTitle>Saved Outputs</SheetTitle>
            </SheetHeader>
            <SavedOutputsList outputs={outputs} onDelete={handleDelete} />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
