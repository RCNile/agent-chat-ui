import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { useThreads } from "@/providers/Thread";
import { Thread } from "@langchain/langgraph-sdk";
import { useEffect, useState } from "react";
import type { MouseEvent } from "react";

import { getContentString } from "../utils";
import { useQueryState, parseAsBoolean } from "nuqs";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import { Separator } from "@/components/ui/separator";
import { PanelRightOpen, PanelRightClose, Trash2, Pencil, Check, X, FileText } from "lucide-react";
import { useMediaQuery } from "@/hooks/useMediaQuery";
import { Input } from "@/components/ui/input";
import { createClient } from "@/providers/client";
import { getApiKey } from "@/lib/api-key";
import { toast } from "sonner";
import { useUploadedDocuments } from "@/hooks/use-uploaded-documents";
import { MultimodalPreview } from "../MultimodalPreview";

function ThreadList({
  threads,
  onThreadClick,
  onThreadDelete,
  onThreadUpdate,
}: {
  threads: Thread[];
  onThreadClick?: (threadId: string) => void;
  onThreadDelete?: (threadId: string) => void;
  onThreadUpdate?: (threadId: string, name: string) => void;
}) {
  const [threadId, setThreadId] = useQueryState("threadId");
  const [apiUrl] = useQueryState("apiUrl");
  const [assistantId] = useQueryState("assistantId");
  const [deletingThreadId, setDeletingThreadId] = useState<string | null>(null);
  const [editingThreadId, setEditingThreadId] = useState<string | null>(null);
  const [editName, setEditName] = useState<string>("");

  const handleThreadClick = (newThreadId: string) => {
    onThreadClick?.(newThreadId);
    if (newThreadId === threadId) return;
    
    // Preserve apiUrl and assistantId in URL when switching threads
    const params = new URLSearchParams(window.location.search);
    if (apiUrl) params.set('apiUrl', apiUrl);
    if (assistantId) params.set('assistantId', assistantId);
    params.set('threadId', newThreadId);
    
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
    
    setThreadId(newThreadId);
  };

  const handleEditStart = (thread: Thread, e: MouseEvent) => {
    e.stopPropagation();
    let currentName = thread.thread_id;
    if (
      typeof thread.values === "object" &&
      thread.values &&
      "messages" in thread.values &&
      Array.isArray(thread.values.messages) &&
      thread.values.messages?.length > 0
    ) {
      const firstMessage = thread.values.messages[0];
      currentName = getContentString(firstMessage.content);
    }
    setEditingThreadId(thread.thread_id);
    setEditName(currentName);
  };

  const handleEditSave = async (threadIdToUpdate: string, e: MouseEvent) => {
    e.stopPropagation();
    
    if (!editName.trim()) {
      toast.error("Thread name cannot be empty");
      return;
    }

    try {
      if (!apiUrl) {
        toast.error("Error", {
          description: "API URL not configured",
          duration: 3000,
        });
        return;
      }

      const client = createClient(apiUrl, getApiKey() ?? undefined);
      await client.threads.update(threadIdToUpdate, {
        metadata: { custom_name: editName.trim() }
      });
      
      toast.success("Thread name updated");
      onThreadUpdate?.(threadIdToUpdate, editName.trim());
      setEditingThreadId(null);
    } catch (error) {
      console.error("Failed to update thread:", error);
      toast.error("Failed to update thread name");
    }
  };

  const handleEditCancel = (e: MouseEvent) => {
    e.stopPropagation();
    setEditingThreadId(null);
    setEditName("");
  };

  const handleDeleteThread = async (threadIdToDelete: string, e: MouseEvent) => {
    e.stopPropagation();
    
    if (!window.confirm("Are you sure you want to delete this thread? This action cannot be undone.")) {
      return;
    }

    setDeletingThreadId(threadIdToDelete);
    
    try {
      if (!apiUrl) {
        toast.error("Error", {
          description: "API URL not configured",
          duration: 3000,
        });
        return;
      }

      const client = createClient(apiUrl, getApiKey() ?? undefined);
      await client.threads.delete(threadIdToDelete);
      
      toast.success("Thread deleted successfully");
      onThreadDelete?.(threadIdToDelete);
      
      // If we deleted the current thread, clear it from URL
      if (threadIdToDelete === threadId) {
        const params = new URLSearchParams(window.location.search);
        if (apiUrl) params.set('apiUrl', apiUrl);
        if (assistantId) params.set('assistantId', assistantId);
        params.delete('threadId');
        
        const newUrl = `${window.location.pathname}?${params.toString()}`;
        window.history.replaceState({}, '', newUrl);
        setThreadId(null);
      }
    } catch (error) {
      console.error("Failed to delete thread:", error);
      toast.error("Failed to delete thread", {
        description: "Please try again",
        duration: 3000,
      });
    } finally {
      setDeletingThreadId(null);
    }
  };

  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {threads.map((t) => {
        let itemText: string = t.metadata?.custom_name || t.thread_id;
        if (!t.metadata?.custom_name &&
          typeof t.values === "object" &&
          t.values &&
          "messages" in t.values &&
          Array.isArray(t.values.messages) &&
          t.values.messages?.length > 0
        ) {
          const firstMessage = t.values.messages[0];
          const contentText = getContentString(firstMessage.content);
          itemText = contentText || t.thread_id;
        }
        const isDeleting = deletingThreadId === t.thread_id;
        const isEditing = editingThreadId === t.thread_id;
        const isCurrentThread = t.thread_id === threadId;
        
        return (
          <div
            key={t.thread_id}
            className="group relative w-full px-1"
          >
            {isEditing ? (
              <div className="flex items-center gap-1 w-[280px] p-2" onClick={(e) => e.stopPropagation()}>
                <Input
                  value={editName}
                  onChange={(e) => setEditName(e.target.value)}
                  className="h-8 text-sm flex-1"
                  autoFocus
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      handleEditSave(t.thread_id, e as any);
                    } else if (e.key === 'Escape') {
                      handleEditCancel(e as any);
                    }
                  }}
                />
                <Button
                  variant="ghost"
                  size="sm"
                  className="p-1 h-6 w-6 text-green-600 hover:text-green-700"
                  onClick={(e) => handleEditSave(t.thread_id, e)}
                >
                  <Check className="h-3 w-3" />
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  className="p-1 h-6 w-6 text-muted-foreground hover:text-foreground"
                  onClick={handleEditCancel}
                >
                  <X className="h-3 w-3" />
                </Button>
              </div>
            ) : (
              <>
                <Button
                  variant="ghost"
                  className={`w-[280px] items-start justify-start text-left font-normal ${
                    isCurrentThread ? "bg-accent" : ""
                  }`}
                  onClick={(e) => {
                    e.preventDefault();
                    handleThreadClick(t.thread_id);
                  }}
                >
                  <p className="truncate text-ellipsis pr-16">{itemText}</p>
                </Button>
                <div className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity flex items-center gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="p-1 h-6 w-6 text-muted-foreground hover:text-foreground"
                    onClick={(e) => handleEditStart(t, e)}
                    title="Rename thread"
                  >
                    <Pencil className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="p-1 h-6 w-6 text-muted-foreground hover:text-destructive"
                    onClick={(e) => handleDeleteThread(t.thread_id, e)}
                    disabled={isDeleting}
                    title="Delete thread"
                  >
                    {isDeleting ? (
                      <div className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
                    ) : (
                      <Trash2 className="h-3 w-3" />
                    )}
                  </Button>
                </div>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ThreadHistoryLoading() {
  return (
    <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
      {Array.from({ length: 30 }).map((_, i) => (
        <Skeleton
          key={`skeleton-${i}`}
          className="h-10 w-[280px]"
        />
      ))}
    </div>
  );
}

function UploadedDocumentsList() {
  const { 
    documents, 
    removeDocument, 
    clearAllDocuments,
    toggleDocumentSelection,
    selectAllDocuments,
    deselectAllDocuments,
  } = useUploadedDocuments();

  if (documents.length === 0) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-4 text-center text-sm text-muted-foreground">
        <FileText className="h-8 w-8 text-gray-300" />
        <p>No documents uploaded yet</p>
      </div>
    );
  }

  return (
    <div className="flex h-full w-full flex-col gap-3">
      <div className="flex items-center justify-between px-4">
        <div className="flex items-center gap-2">
          {documents.length > 0 && (
            <Checkbox
              checked={documents.every(doc => doc.selected ?? false)}
              onCheckedChange={(checked) => {
                if (checked) {
                  selectAllDocuments();
                } else {
                  deselectAllDocuments();
                }
              }}
            />
          )}
          <span className="text-xs text-muted-foreground">
            {documents.length} {documents.length === 1 ? "document" : "documents"}
          </span>
        </div>
        {documents.length > 0 && (
          <Button
            variant="ghost"
            size="sm"
            className="h-6 px-2 text-xs text-muted-foreground hover:text-destructive"
            onClick={() => {
              if (window.confirm("Are you sure you want to clear all uploaded documents?")) {
                clearAllDocuments();
                toast.success("All documents cleared");
              }
            }}
          >
            Clear All
          </Button>
        )}
      </div>
      <div className="flex h-full w-full flex-col items-start justify-start gap-2 overflow-y-scroll px-2 [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent">
        {documents.map((doc) => (
          <div key={doc.id} className="w-full flex items-start gap-2">
            <Checkbox
              checked={doc.selected ?? false}
              onCheckedChange={() => toggleDocumentSelection(doc.id)}
              className="mt-2"
            />
            <div className="flex-1">
              <MultimodalPreview
                block={doc.block}
                removable
                onRemove={() => removeDocument(doc.id)}
                size="sm"
              />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ThreadHistory() {
  const isLargeScreen = useMediaQuery("(min-width: 1024px)");
  const [chatHistoryOpen, setChatHistoryOpen] = useQueryState(
    "chatHistoryOpen",
    parseAsBoolean.withDefault(false),
  );

  const { getThreads, threads, setThreads, threadsLoading, setThreadsLoading } =
    useThreads();

  useEffect(() => {
    if (typeof window === "undefined") return;
    setThreadsLoading(true);
    getThreads()
      .then(setThreads)
      .catch(console.error)
      .finally(() => setThreadsLoading(false));
  }, [getThreads]);

  const handleThreadDelete = (deletedThreadId: string) => {
    // Remove the deleted thread from the local state
    setThreads(prevThreads => prevThreads.filter(t => t.thread_id !== deletedThreadId));
  };

  const handleThreadUpdate = (threadId: string, newName: string) => {
    // Update the thread name in local state
    setThreads(prevThreads => prevThreads.map(t => 
      t.thread_id === threadId 
        ? { ...t, metadata: { ...t.metadata, custom_name: newName } }
        : t
    ));
  };

  return (
    <>
      <div className="shadow-inner-right hidden h-screen w-[300px] shrink-0 flex-col items-start justify-start border-r-[1px] border-slate-300 lg:flex">
        {/* Header */}
        <div className="flex w-full items-center justify-between px-4 pt-1.5 pb-4">
          <Button
            className="hover:bg-gray-100"
            variant="ghost"
            onClick={() => setChatHistoryOpen((p) => !p)}
          >
            {chatHistoryOpen ? (
              <PanelRightOpen className="size-5" />
            ) : (
              <PanelRightClose className="size-5" />
            )}
          </Button>
          <h1 className="text-xl font-semibold tracking-tight">
            Thread History
          </h1>
        </div>
        
        {/* Thread History Section - Top Half */}
        <div className="flex-1 flex flex-col min-h-0 w-full">
          <div className="px-4 pb-2">
            <h2 className="text-sm font-medium text-muted-foreground">Threads</h2>
          </div>
          <div className="flex-1 overflow-hidden">
            {threadsLoading ? (
              <ThreadHistoryLoading />
            ) : (
              <ThreadList 
                threads={threads} 
                onThreadDelete={handleThreadDelete}
                onThreadUpdate={handleThreadUpdate}
              />
            )}
          </div>
        </div>

        <Separator className="my-2" />

        {/* Uploaded Documents Section - Bottom Half */}
        <div className="flex-1 flex flex-col min-h-0 w-full pb-4">
          <div className="px-4 pb-2">
            <h2 className="text-sm font-medium text-muted-foreground">Uploaded Documents</h2>
          </div>
          <div className="flex-1 overflow-hidden">
            <UploadedDocumentsList />
          </div>
        </div>
      </div>
      
      <div className="lg:hidden">
        <Sheet
          open={!!chatHistoryOpen && !isLargeScreen}
          onOpenChange={(open) => {
            if (isLargeScreen) return;
            setChatHistoryOpen(open);
          }}
        >
          <SheetContent
            side="left"
            className="flex flex-col lg:hidden"
          >
            <SheetHeader>
              <SheetTitle>Thread History</SheetTitle>
            </SheetHeader>
            
            {/* Thread History Section */}
            <div className="flex-1 flex flex-col min-h-0 mt-4">
              <h2 className="text-sm font-medium text-muted-foreground px-2 pb-2">Threads</h2>
              <div className="flex-1 overflow-hidden">
                <ThreadList
                  threads={threads}
                  onThreadClick={() => setChatHistoryOpen((o) => !o)}
                  onThreadDelete={handleThreadDelete}
                  onThreadUpdate={handleThreadUpdate}
                />
              </div>
            </div>

            <Separator className="my-4" />

            {/* Uploaded Documents Section */}
            <div className="flex-1 flex flex-col min-h-0">
              <h2 className="text-sm font-medium text-muted-foreground px-2 pb-2">Uploaded Documents</h2>
              <div className="flex-1 overflow-hidden">
                <UploadedDocumentsList />
              </div>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </>
  );
}
