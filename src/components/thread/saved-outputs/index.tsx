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
import { PanelLeftOpen, PanelLeftClose, Download, Trash2, Pencil, Check, X, BookmarkIcon } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { cn } from "@/lib/utils";

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
  onUpdate,
}: {
  output: SavedOutput;
  onDelete: (id: string) => void;
  onUpdate: (id: string, newTitle: string) => void;
}) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editTitle, setEditTitle] = useState(output.title);

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

  const handleEditStart = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsEditing(true);
    setEditTitle(output.title);
  };

  const handleEditSave = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (!editTitle.trim()) {
      toast.error("Title cannot be empty");
      return;
    }
    onUpdate(output.id, editTitle.trim());
    setIsEditing(false);
    toast.success("Title updated");
  };

  const handleEditCancel = (e: React.MouseEvent) => {
    e.stopPropagation();
    setIsEditing(false);
    setEditTitle(output.title);
  };

  // Use consistent colour for all saved outputs
  const iconColour = 'bg-orange-100 text-orange-700';

  return (
    <div className="group relative w-full px-3">
      {isEditing ? (
        <div className="flex items-start gap-2 rounded-xl border bg-white p-3 shadow-sm" onClick={(e) => e.stopPropagation()}>
          <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-lg", iconColour)}>
            <BookmarkIcon className="h-5 w-5" />
          </div>
          <div className="flex-1 min-w-0">
            <Input
              value={editTitle}
              onChange={(e) => setEditTitle(e.target.value)}
              className="h-8 text-sm mb-1"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  handleEditSave(e as any);
                } else if (e.key === 'Escape') {
                  handleEditCancel(e as any);
                }
              }}
            />
            <p className="text-xs text-muted-foreground">
              {output.timestamp.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' })}
            </p>
          </div>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-green-600 hover:text-green-700"
              onClick={handleEditSave}
            >
              <Check className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
              onClick={handleEditCancel}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      ) : (
        <div className="relative rounded-xl border bg-white p-3 shadow-sm transition-shadow hover:shadow-md">
          <div className="flex items-start gap-3">
            <div className={cn("flex h-10 w-10 shrink-0 items-center justify-center rounded-lg", iconColour)}>
              <BookmarkIcon className="h-5 w-5" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium leading-tight text-gray-900 line-clamp-2 mb-1">
                {output.title}
              </p>
              <p className="text-xs text-muted-foreground">
                {output.timestamp.toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit', year: 'numeric' })}
              </p>
            </div>
          </div>
          <div className="absolute right-2 top-2 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-gray-100"
              onClick={handleEditStart}
              title="Rename"
            >
              <Pencil className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-gray-100"
              onClick={handleExport}
              title="Copy to clipboard"
            >
              <Download className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive hover:bg-red-50"
              onClick={handleDelete}
              disabled={isDeleting}
              title="Delete saved output"
            >
              {isDeleting ? (
                <div className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent" />
              ) : (
                <Trash2 className="h-3.5 w-3.5" />
              )}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

function SavedOutputsList({
  outputs,
  onDelete,
  onUpdate,
}: {
  outputs: SavedOutput[];
  onDelete: (id: string) => void;
  onUpdate: (id: string, newTitle: string) => void;
}) {
  if (outputs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-full text-center p-6">
        <div className="rounded-xl border bg-white p-8 shadow-sm">
          <div className="text-gray-400 mb-3">
            <BookmarkIcon className="h-12 w-12 mx-auto" />
          </div>
          <p className="text-sm font-medium text-gray-700 mb-1">No saved outputs yet</p>
          <p className="text-xs text-gray-500">
            Save outputs from chat messages to see them here
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-3 overflow-y-scroll px-1 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {outputs.map((output) => (
        <SavedOutputItem
          key={output.id}
          output={output}
          onDelete={onDelete}
          onUpdate={onUpdate}
        />
      ))}
    </div>
  );
}

function SavedOutputsLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-3 overflow-y-scroll px-1 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 5 }).map((_, i) => (
        <div key={`skeleton-${i}`} className="w-full px-3">
          <Skeleton className="h-16 w-full rounded-xl" />
        </div>
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

  const handleUpdate = (id: string, newTitle: string) => {
    const updated = outputs.map(output => 
      output.id === id ? { ...output, title: newTitle } : output
    );
    setOutputs(updated);
    localStorage.setItem('saved-outputs', JSON.stringify(updated));
    // Dispatch custom event for other windows
    window.dispatchEvent(new Event('saved-outputs-updated'));
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
          <SavedOutputsList outputs={outputs} onDelete={handleDelete} onUpdate={handleUpdate} />
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
            <SavedOutputsList outputs={outputs} onDelete={handleDelete} onUpdate={handleUpdate} />
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
